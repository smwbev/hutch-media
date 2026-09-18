"""Invariant tests for the ``hutch_video_edit`` tool (no network; ``requests`` mocked).

Run from a hermes-agent checkout:

    PYTHONPATH=/path/to/hermes-agent python -m pytest tests/ -q
"""

import base64
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_plugin():
    spec = importlib.util.spec_from_file_location("hutch_media_edit_under_test", REPO_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Ctx:
    def __init__(self):
        self.tools = {}

    def register_image_gen_provider(self, provider):
        pass

    def register_video_gen_provider(self, provider):
        self.video = provider

    def register_tool(self, *, name, toolset, schema, handler, check_fn=None, **kw):
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler,
                            "check_fn": check_fn, **kw}


@pytest.fixture()
def plugin(monkeypatch):
    monkeypatch.setenv("HUTCH_API_KEY", "sk-hutch-test-key")
    monkeypatch.setenv("HUTCH_BASE_URL", "https://relay.test.example/v1")
    mod = _load_plugin()
    monkeypatch.setattr(mod, "_relay_model_ids", lambda: ["x-ai/grok-imagine-video",
                                                          "grok-imagine-video-1.5-preview"])
    return mod


@pytest.fixture()
def tool(plugin):
    ctx = _Ctx()
    plugin.register(ctx)
    return ctx.tools["hutch_video_edit"], ctx.video


def _config(monkeypatch, cfg):
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "load_config", lambda: cfg)


def _mp4(tmp_path, name="in.mp4", size=4096):
    p = tmp_path / name
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * (size - 12))
    return p


def _done_relay(monkeypatch, seen):
    """Submit -> request_id; poll -> done with a CDN url (raw xAI status)."""
    submit = MagicMock(status_code=200)
    submit.json.return_value = {"request_id": "edit-1"}
    poll = MagicMock(status_code=200)
    poll.json.return_value = {"status": "done", "progress": 100,
                              "video": {"url": "https://cdn.example/edited.mp4", "duration": 3}}
    import requests
    monkeypatch.setattr(requests, "post",
                        lambda url, **kw: (seen.update(url=url, json=kw.get("json"),
                                                       headers=kw.get("headers")), submit)[1])
    monkeypatch.setattr(requests, "get", lambda url, **kw: (seen.setdefault("gets", []).append(url), poll)[1])
    monkeypatch.setattr("time.sleep", lambda s: None)
    import agent.video_gen_provider as vgp
    monkeypatch.setattr(vgp, "save_url_video", lambda url, prefix: f"/cache/{prefix}_edited.mp4")


def test_tool_registered_in_video_gen_toolset(tool):
    entry, _ = tool
    assert entry["toolset"] == "video_gen"
    assert entry["schema"]["name"] == "hutch_video_edit"
    assert set(entry["schema"]["parameters"]["required"]) == {"prompt", "video_url"}
    assert callable(entry["check_fn"])


def test_check_fn_requires_hutch_provider_and_creds(plugin, tool, monkeypatch):
    entry, _ = tool
    _config(monkeypatch, {"video_gen": {"provider": "hutch"}})
    assert entry["check_fn"]() is True
    _config(monkeypatch, {"video_gen": {"provider": "xai"}})
    assert entry["check_fn"]() is False
    _config(monkeypatch, {"video_gen": {"provider": "hutch"}})
    monkeypatch.delenv("HUTCH_API_KEY")
    assert entry["check_fn"]() is False


def test_local_mp4_is_inlined_and_url_passes_through(tool, monkeypatch, tmp_path):
    entry, _ = tool
    _config(monkeypatch, {"video_gen": {"provider": "hutch"}})
    seen = {}
    _done_relay(monkeypatch, seen)
    src = _mp4(tmp_path)

    out = json.loads(entry["handler"]({"prompt": "add snow", "video_url": str(src)}))
    assert out["success"], out
    assert seen["url"].endswith("/videos/edits")
    # native xAI contract, nothing else (no seconds/input_reference/size/aspect_ratio)
    assert set(seen["json"]) == {"model", "prompt", "video"}
    assert seen["json"]["model"] == "grok-imagine-video"
    expected = "data:video/mp4;base64," + base64.b64encode(src.read_bytes()).decode("ascii")
    assert seen["json"]["video"] == {"url": expected}
    # polled with the SAME bearer, raw 'done' accepted, result saved locally
    assert seen["gets"][0].endswith("/videos/edit-1")
    assert out["video"] == "/cache/hutch_edited.mp4"
    assert out["public_url"] == "https://cdn.example/edited.mp4"
    assert out["source_video"] == str(src)
    assert out["request_id"] == "edit-1"

    out = json.loads(entry["handler"]({"prompt": "add rain",
                                        "video_url": "https://vidgen.x.ai/x/y.mp4"}))
    assert out["success"], out
    assert seen["json"]["video"] == {"url": "https://vidgen.x.ai/x/y.mp4"}


def test_model_allow_list_rejects_before_any_request(tool, monkeypatch, tmp_path):
    entry, _ = tool
    _config(monkeypatch, {"video_gen": {"provider": "hutch"}})
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("must not reach the relay"))
    src = _mp4(tmp_path)
    for bad in ("grok-imagine-video-1.5", "x-ai/grok-imagine-video-1.5-preview",
                "openai/grok-imagine-video", "sora-2"):
        out = json.loads(entry["handler"]({"prompt": "x", "video_url": str(src), "model": bad}))
        assert not out["success"], bad
        assert out["error_type"] == "model_not_editable", bad
        assert out["editable_models"] == ["grok-imagine-video"]


def test_relay_prefixes_accepted_for_edit_model(plugin):
    ok = plugin._video_edit_model_ok
    assert ok("grok-imagine-video") and ok("x-ai/grok-imagine-video")
    assert ok("xai/grok-imagine-video") and ok("grok/grok-imagine-video")
    assert ok("X-AI/Grok-Imagine-Video")
    assert not ok("grok-imagine-video-1.5") and not ok("openai/grok-imagine-video")


def test_default_edit_model_ignores_video_gen_model(tool, monkeypatch, tmp_path):
    """video_gen.model (generation default, typically 1.5) must never leak in
    as the edit model; only video_gen.hutch.edit_model overrides the built-in."""
    entry, _ = tool
    seen = {}
    _done_relay(monkeypatch, seen)
    src = _mp4(tmp_path)

    _config(monkeypatch, {"video_gen": {"provider": "hutch", "model": "grok-imagine-video-1.5"}})
    out = json.loads(entry["handler"]({"prompt": "x", "video_url": str(src)}))
    assert out["success"], out
    assert seen["json"]["model"] == "grok-imagine-video"

    _config(monkeypatch, {"video_gen": {"provider": "hutch", "model": "grok-imagine-video-1.5",
                                        "hutch": {"edit_model": "x-ai/grok-imagine-video"}}})
    out = json.loads(entry["handler"]({"prompt": "x", "video_url": str(src)}))
    assert out["success"], out
    assert seen["json"]["model"] == "x-ai/grok-imagine-video"


def test_relay_not_supported_400_maps_to_model_not_editable(tool, monkeypatch, tmp_path):
    entry, _ = tool
    _config(monkeypatch, {"video_gen": {"provider": "hutch"}})
    resp = MagicMock(status_code=400,
                     text='{"code":"invalid-argument","error":"Video editing is not supported for this model."}')
    import requests
    monkeypatch.setattr(requests, "post", lambda url, **kw: resp)
    out = json.loads(entry["handler"]({"prompt": "x", "video_url": str(_mp4(tmp_path))}))
    assert not out["success"]
    assert out["error_type"] == "model_not_editable"
    assert "not supported for this model" in out["error"]


def test_failed_job_surfaces_relay_error_message(tool, monkeypatch, tmp_path):
    entry, _ = tool
    _config(monkeypatch, {"video_gen": {"provider": "hutch"}})
    submit = MagicMock(status_code=200)
    submit.json.return_value = {"request_id": "edit-9"}
    poll = MagicMock(status_code=200)
    poll.json.return_value = {"status": "failed",
                              "error": {"message": "video must be between 2 and 15 seconds"}}
    import requests
    monkeypatch.setattr(requests, "post", lambda url, **kw: submit)
    monkeypatch.setattr(requests, "get", lambda url, **kw: poll)
    monkeypatch.setattr("time.sleep", lambda s: None)
    out = json.loads(entry["handler"]({"prompt": "x", "video_url": str(_mp4(tmp_path))}))
    assert not out["success"]
    assert out["error_type"] == "provider_error"
    assert "video must be between 2 and 15 seconds" in out["error"]
    assert "edit-9" in out["error"]


def test_oversized_source_is_rejected_without_request(plugin, tool, monkeypatch, tmp_path):
    entry, _ = tool
    _config(monkeypatch, {"video_gen": {"provider": "hutch"}})
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("must not reach the relay"))
    monkeypatch.setattr(plugin, "_VIDEO_EDIT_MAX_BYTES", 2048)  # keep the test cheap
    big = _mp4(tmp_path, "big.mp4", size=8192)
    out = json.loads(entry["handler"]({"prompt": "x", "video_url": str(big)}))
    assert not out["success"]
    assert out["error_type"] == "video_too_large"
    assert "ffmpeg" in out["error"]


def test_invalid_sources_are_rejected_without_request(tool, monkeypatch, tmp_path):
    entry, _ = tool
    _config(monkeypatch, {"video_gen": {"provider": "hutch"}})
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("must not reach the relay"))
    out = json.loads(entry["handler"]({"prompt": "x", "video_url": "/nonexistent/clip.mp4"}))
    assert out["error_type"] == "invalid_input"
    mov = tmp_path / "clip.mov"
    mov.write_bytes(b"x" * 100)
    out = json.loads(entry["handler"]({"prompt": "x", "video_url": str(mov)}))
    assert out["error_type"] == "invalid_input" and ".mp4" in out["error"]
    out = json.loads(entry["handler"]({"prompt": "", "video_url": str(_mp4(tmp_path))}))
    assert out["error_type"] == "invalid_input"


def test_file_safety_guard_runs_before_read(plugin, tool, monkeypatch, tmp_path):
    """The credential-read guard runs FIRST — before existence/size probes —
    so a denied path leaks neither its existence nor its size."""
    entry, _ = tool
    _config(monkeypatch, {"video_gen": {"provider": "hutch"}})
    import agent.file_safety as fs
    seen = []

    def blocked(p):
        seen.append(p)
        raise PermissionError("blocked by file_safety")

    monkeypatch.setattr(fs, "raise_if_read_blocked", blocked)
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("must not reach the relay"))
    out = json.loads(entry["handler"]({"prompt": "x", "video_url": str(_mp4(tmp_path))}))
    assert not out["success"] and "blocked by file_safety" in out["error"]
    # non-existent denied path: guard fires, not the "not an existing file" message
    out = json.loads(entry["handler"]({"prompt": "x", "video_url": str(tmp_path / "ghost.mp4")}))
    assert "blocked by file_safety" in out["error"] and "existing local file" not in out["error"]
    # oversized denied path: guard fires, size is never disclosed
    big = _mp4(tmp_path, "big.mp4", size=64 * 1024)
    monkeypatch.setattr(plugin, "_VIDEO_EDIT_MAX_BYTES", 1024)
    out = json.loads(entry["handler"]({"prompt": "x", "video_url": str(big)}))
    assert "blocked by file_safety" in out["error"] and "MB" not in out["error"]
    assert len(seen) == 3


def test_non_object_json_bodies_never_escape_as_exceptions(tool, monkeypatch, tmp_path):
    """A relay/CDN answering 200 with a JSON list/string/null must yield a
    JSON error (submit) or be skipped as transient (poll) — never an
    AttributeError escaping the handler. The poll path is shared with
    video_generate, so this guards both."""
    entry, video = tool
    _config(monkeypatch, {"video_gen": {"provider": "hutch"}})
    import requests
    monkeypatch.setattr("time.sleep", lambda s: None)
    src = _mp4(tmp_path)

    for weird in (["not", "a", "dict"], "done", None, 42):
        submit = MagicMock(status_code=200)
        submit.json.return_value = weird
        monkeypatch.setattr(requests, "post", lambda url, **kw: submit)
        out = json.loads(entry["handler"]({"prompt": "x", "video_url": str(src)}))
        assert out["success"] is False and out["error_type"] == "provider_error", weird
        assert "non-object" in out["error"]

    # poll: first body is a bare string, second is a proper 'done' dict
    submit = MagicMock(status_code=200)
    submit.json.return_value = {"request_id": "edit-w"}
    bodies = iter(["done", {"status": "done", "video": {"url": "https://cdn.example/w.mp4",
                                                        "duration": 5}}])
    poll = MagicMock(status_code=200)
    poll.json.side_effect = lambda: next(bodies)
    monkeypatch.setattr(requests, "post", lambda url, **kw: submit)
    monkeypatch.setattr(requests, "get", lambda url, **kw: poll)
    import agent.video_gen_provider as vgp
    monkeypatch.setattr(vgp, "save_url_video", lambda url, prefix: url)
    out = json.loads(entry["handler"]({"prompt": "x", "video_url": str(src)}))
    assert out["success"], out
    assert out["duration"] == 5  # terminal poll metadata surfaced
    # same resilience on the shared provider path
    bodies = iter(["done", {"status": "done", "video": {"url": "https://cdn.example/g.mp4"}}])
    res = video.generate("a boat", model="grok-imagine-video")
    assert res["success"], res


def test_poll_fail_fast_message_explains_auth_binding(tool, monkeypatch, tmp_path):
    entry, _ = tool
    _config(monkeypatch, {"video_gen": {"provider": "hutch"}})
    submit = MagicMock(status_code=200)
    submit.json.return_value = {"request_id": "edit-404"}
    poll = MagicMock(status_code=404, text='{"error":"not found"}')
    import requests
    monkeypatch.setattr(requests, "post", lambda url, **kw: submit)
    monkeypatch.setattr(requests, "get", lambda url, **kw: poll)
    monkeypatch.setattr("time.sleep", lambda s: None)
    out = json.loads(entry["handler"]({"prompt": "x", "video_url": str(_mp4(tmp_path))}))
    assert not out["success"]
    assert "auth binding" in out["error"] and "resubmit" in out["error"]


def test_video_provider_advertises_editing(tool):
    _, video = tool
    caps = video.capabilities()
    assert caps["supports_edit"] is True and caps["edit_models"] == ["grok-imagine-video"]
    rows = {r["id"]: r for r in video.list_models()}
    assert "supports editing" in rows["x-ai/grok-imagine-video"]["strengths"]
    assert "supports editing" not in rows["grok-imagine-video-1.5-preview"]["strengths"]
