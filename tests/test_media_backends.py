"""Invariant tests for the hutch media backends (self-contained ABC providers).

Run from a hermes-agent checkout:

    PYTHONPATH=/path/to/hermes-agent python -m pytest tests/ -q -o 'addopts='
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

RELAY_MODELS = [
    "anthropic/claude-fable-5",
    "openai/gpt-image-2", "gpt-image-2",
    "openai/gpt-image-2.5-flare",
    "x-ai/grok-imagine-image", "grok-imagine-image",
    "meta/muse-image-1.0",
    "x-ai/grok-imagine-video", "grok-imagine-video",
    "grok-imagine-video-1.5-preview",
    "meta/muse-voice-transcribe-1.0",
    "acme/video-transcribe-2",       # "video" substring but not a media model
    "acme/image-embedding-large",    # "image" substring but not a media model
]


def _load_plugin():
    spec = importlib.util.spec_from_file_location("hutch_media_under_test", REPO_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Ctx:
    def __init__(self):
        self.image = []
        self.video = []

    def register_image_gen_provider(self, provider):
        self.image.append(provider)

    def register_video_gen_provider(self, provider):
        self.video.append(provider)


@pytest.fixture()
def plugin(monkeypatch):
    monkeypatch.setenv("HUTCH_API_KEY", "sk-hutch-test")
    monkeypatch.setenv("HUTCH_BASE_URL", "https://relay.test.example/v1")
    mod = _load_plugin()
    monkeypatch.setattr(mod, "_relay_model_ids", lambda: list(RELAY_MODELS))
    return mod


@pytest.fixture()
def providers(plugin):
    ctx = _Ctx()
    plugin.register(ctx)
    assert len(ctx.image) == 1 and len(ctx.video) == 1
    return ctx.image[0], ctx.video[0]


def test_providers_subclass_public_abcs(providers):
    from agent.image_gen_provider import ImageGenProvider
    from agent.video_gen_provider import VideoGenProvider
    image, video = providers
    assert isinstance(image, ImageGenProvider)
    assert isinstance(video, VideoGenProvider)
    assert image.name == "hutch" and video.name == "hutch"


def test_image_catalog_filters_media_models(providers):
    image, _ = providers
    ids = {row["id"] for row in image.list_models()}
    assert "openai/gpt-image-2" in ids
    assert "meta/muse-image-1.0" in ids
    # video and text models excluded; duplicate bare spellings collapsed
    assert not any("video" in i for i in ids)
    assert "anthropic/claude-fable-5" not in ids
    assert "gpt-image-2" not in ids  # bare duplicate of openai/gpt-image-2
    assert "acme/image-embedding-large" not in ids  # negative-token filter


def test_video_catalog_filters_video_models(providers):
    _, video = providers
    ids = {row["id"] for row in video.list_models()}
    assert "x-ai/grok-imagine-video" in ids
    assert "grok-imagine-video-1.5-preview" in ids
    assert all("video" in i for i in ids)
    assert "acme/video-transcribe-2" not in ids  # negative-token filter
    assert "meta/muse-voice-transcribe-1.0" not in ids


def test_image_generate_saves_b64_to_cache(providers, monkeypatch, tmp_path):
    image, _ = providers
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"data": [{"b64_json": "aGVsbG8="}]}
    calls = {}

    def fake_post(url, **kwargs):
        calls["url"] = url
        calls["headers"] = kwargs.get("headers")
        return resp

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    out = image.generate("a corgi", "square")
    assert out["success"], out
    assert out["provider"] == "hutch" and out["modality"] == "text"
    assert calls["url"].endswith("/images/generations")
    assert calls["headers"]["Authorization"] == "Bearer sk-hutch-test"
    assert Path(out["image"]).exists()  # b64 saved to the hermes cache


def test_image_edit_routes_to_edits_endpoint(providers, monkeypatch, tmp_path):
    image, _ = providers
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"data": [{"url": "https://relay.test.example/out.png"}]}
    calls = {}

    def fake_post(url, **kwargs):
        calls["url"] = url
        calls["json"] = kwargs.get("json")
        return resp

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    out = image.generate("edit it", "square", image_url="https://x.example/in.png")
    assert out["success"] and out["modality"] == "image"
    assert calls["url"].endswith("/images/edits")
    # Payload shape parsed by EVERY CLIProxyAPI edits branch: the xAI branch
    # (collectXAIImagesFromJSON: images[].image_url string) and the generic
    # fallback (images[].image_url). The former `image: [{type,url}]` shape
    # yields 0 images -> HTTP 400 on both.
    assert calls["json"]["images"] == [{"image_url": "https://x.example/in.png"}]
    assert "image" not in calls["json"]


def test_video_submit_and_poll_same_credentials(providers, monkeypatch, tmp_path):
    _, video = providers
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {"request_id": "req-123", "status": "queued"}
    poll_resp = MagicMock(status_code=200)
    poll_resp.json.return_value = {"status": "completed", "video": {"url": "https://cdn.example/v.mp4"}}
    seen = {"posts": [], "gets": []}

    import requests
    monkeypatch.setattr(
        requests, "post",
        lambda url, **kw: (seen["posts"].append((url, kw.get("headers"), kw.get("json"))), submit_resp)[1])
    monkeypatch.setattr(
        requests, "get",
        lambda url, **kw: (seen["gets"].append((url, kw.get("headers"))), poll_resp)[1])
    monkeypatch.setattr("time.sleep", lambda s: None)
    # The plugin resolves save_url_video via the module attribute at call time,
    # so THIS patch is the seam production actually reads (a closure-captured
    # name would not see it — verified in review).
    saved = {}

    def fake_save(url, prefix):
        saved["url"] = url
        saved["prefix"] = prefix
        local = tmp_path / "v.mp4"
        local.write_bytes(b"vid")
        return local

    import agent.video_gen_provider as vgp
    monkeypatch.setattr(vgp, "save_url_video", fake_save)
    out = video.generate("a rocket", model="grok-imagine-video",
                         duration=8, aspect_ratio="16:9", resolution="720p")
    assert out["success"], out
    # the patched save ran and its local path is what the response carries
    assert saved == {"url": "https://cdn.example/v.mp4", "prefix": "hutch"}
    assert out["video"] == str(tmp_path / "v.mp4")
    assert seen["posts"][0][0].endswith("/videos/generations")
    assert seen["gets"][0][0].endswith("/videos/req-123")
    # SAME Bearer key on submit and poll (CLIProxyAPI auth-binding contract)
    assert seen["posts"][0][1]["Authorization"] == seen["gets"][0][1]["Authorization"]
    # Relay contract on the NATIVE /v1/videos/generations path (forwarded to
    # the xAI executor untranslated): numeric "duration" ("seconds" is parsed
    # only by the /openai/v1/videos translator), and aspect_ratio/resolution
    # must be present or upstream defaults to 720x1280 portrait.
    payload = seen["posts"][0][2]
    assert payload["duration"] == 8
    assert "seconds" not in payload
    assert payload["aspect_ratio"] == "16:9"
    assert payload["resolution"] == "720p"


def test_video_save_failure_falls_back_to_remote_url(providers, monkeypatch):
    _, video = providers
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {
        "request_id": "req-77", "status": "completed",
        "video": {"url": "https://cdn.example/v77.mp4"},
    }

    import requests
    monkeypatch.setattr(requests, "post", lambda url, **kw: submit_resp)
    import agent.video_gen_provider as vgp

    def broken_save(url, prefix):
        raise RuntimeError("disk full")

    monkeypatch.setattr(vgp, "save_url_video", broken_save)
    out = video.generate("a rocket")
    assert out["success"], out
    assert out["video"] == "https://cdn.example/v77.mp4"


def test_video_image_and_references_are_mutually_exclusive(providers, monkeypatch):
    _, video = providers
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {
        "request_id": "req-55", "status": "completed",
        "video": {"url": "https://cdn.example/v55.mp4"},
    }
    seen = {}

    import requests
    monkeypatch.setattr(requests, "post",
                        lambda url, **kw: (seen.update(json=kw.get("json")), submit_resp)[1])
    import agent.video_gen_provider as vgp
    monkeypatch.setattr(vgp, "save_url_video", lambda url, prefix: url)
    out = video.generate("animate", image_url="https://x.example/a.png",
                         reference_image_urls=["https://x.example/b.png"])
    assert out["success"], out
    # upstream 400s on the combination; image_url wins, references dropped.
    # Native-path i2v shape: image.url (input_reference is an OpenAI-ism the
    # native /videos/generations route never translates).
    assert seen["json"]["image"] == {"url": "https://x.example/a.png"}
    assert "input_reference" not in seen["json"]
    assert "reference_images" not in seen["json"]
    assert "reference_image_urls" not in seen["json"]


def test_video_references_only_use_native_shape(providers, monkeypatch):
    _, video = providers
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {
        "request_id": "req-56", "status": "completed",
        "video": {"url": "https://cdn.example/v56.mp4"},
    }
    seen = {}

    import requests
    monkeypatch.setattr(requests, "post",
                        lambda url, **kw: (seen.update(json=kw.get("json")), submit_resp)[1])
    import agent.video_gen_provider as vgp
    monkeypatch.setattr(vgp, "save_url_video", lambda url, prefix: url)
    out = video.generate("blend", reference_image_urls=[
        "https://x.example/b.png", "https://x.example/c.png"])
    assert out["success"], out
    # native xAI shape: reference_images[].url (reference_image_urls is parsed
    # only by the /openai/v1/videos translator)
    assert seen["json"]["reference_images"] == [
        {"url": "https://x.example/b.png"}, {"url": "https://x.example/c.png"}]
    assert "reference_image_urls" not in seen["json"]


def test_video_poll_fails_fast_on_persistent_4xx(plugin, providers, monkeypatch):
    _, video = providers
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {"request_id": "req-401", "status": "queued"}
    poll_resp = MagicMock(status_code=401)
    counts = {"gets": 0}

    import requests
    monkeypatch.setattr(requests, "post", lambda url, **kw: submit_resp)
    monkeypatch.setattr(
        requests, "get",
        lambda url, **kw: (counts.__setitem__("gets", counts["gets"] + 1), poll_resp)[1])
    monkeypatch.setattr("time.sleep", lambda s: None)
    out = video.generate("nope")
    assert not out["success"]
    assert out["error_type"] == "provider_error"
    # bounded by the consecutive-4xx cap, not the 15-minute deadline
    assert counts["gets"] == plugin._POLL_MAX_CONSECUTIVE_4XX


def test_video_failed_status_is_clean_error(providers, monkeypatch):
    _, video = providers
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {"request_id": "req-9", "status": "queued"}
    poll_resp = MagicMock(status_code=200)
    poll_resp.json.return_value = {"status": "failed"}

    import requests
    monkeypatch.setattr(requests, "post", lambda url, **kw: submit_resp)
    monkeypatch.setattr(requests, "get", lambda url, **kw: poll_resp)
    monkeypatch.setattr("time.sleep", lambda s: None)
    out = video.generate("nope")
    assert not out["success"]
    assert out["error_type"] == "provider_error"


def test_unavailable_without_env(plugin, monkeypatch):
    monkeypatch.delenv("HUTCH_API_KEY", raising=False)
    monkeypatch.delenv("HUTCH_BASE_URL", raising=False)
    ctx = _Ctx()
    plugin.register(ctx)
    assert not ctx.image[0].is_available()
    assert not ctx.video[0].is_available()
    out = ctx.image[0].generate("prompt")
    assert not out["success"] and out["error_type"] == "missing_api_key"
