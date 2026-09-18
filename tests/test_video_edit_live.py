"""Live probe: has xAI enabled video editing on models we currently reject?

Not part of the default run. Requires the relay credentials and opts in via
the ``live`` marker:

    HUTCH_API_KEY=... HUTCH_BASE_URL=... \\
    PYTHONPATH=/path/to/hermes-agent python -m pytest tests/test_video_edit_live.py -q -o 'addopts=' -m live

Why this exists: ``_VIDEO_EDIT_MODELS`` is a static allow-list (the relay's
catalog is unstable and the base model answers even when unlisted). xAI does
not announce capability changes — a model that stops answering HTTP 400
"Video editing is not supported for this model" is the only signal. This
probe asserts the EXACT expected refusal for every candidate; any other
outcome (200 with a request_id, a different 4xx, 5xx) fails loudly with the
instruction to verify and extend the allow-list.

The refusal arrives before billing, so the probe is free while nothing has
changed. If xAI ever returns 200, one edit job is billed once — the price of
the signal. Run monthly, or whenever a new Imagine release ships.
"""

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.live

# Models the relay routes to xAI video endpoints but which xAI currently
# refuses to edit. Keep in sync with the relay's isXAIVideosModel set.
_VIDEO_EDIT_CANDIDATES = ("grok-imagine-video-1.5", "grok-imagine-video-1.5-preview")

# Any tiny public mp4 xAI serves; a stale URL still yields the model refusal
# because model validation happens before the input is fetched.
_PROBE_VIDEO_URL = os.environ.get(
    "HUTCH_EDIT_PROBE_VIDEO",
    "https://vidgen.x.ai/xai-vidgen-bucket/probe-placeholder.mp4",
)


def _load_plugin():
    spec = importlib.util.spec_from_file_location("hutch_media_live", REPO_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def plugin():
    if not (os.environ.get("HUTCH_API_KEY") and os.environ.get("HUTCH_BASE_URL")):
        pytest.skip("HUTCH_API_KEY / HUTCH_BASE_URL not set — live probe skipped")
    return _load_plugin()


@pytest.mark.parametrize("model", _VIDEO_EDIT_CANDIDATES)
def test_candidate_model_still_refuses_editing(plugin, model):
    import requests

    assert not plugin._video_edit_model_ok(model), (
        f"{model} is already in _VIDEO_EDIT_MODELS — drop it from _VIDEO_EDIT_CANDIDATES"
    )
    resp = requests.post(
        f"{plugin._base_url()}/videos/edits",
        headers=plugin._headers(),
        json={"model": model, "prompt": "no-op probe", "video": {"url": _PROBE_VIDEO_URL}},
        timeout=60,
    )
    body = resp.text[:400]
    assert resp.status_code == 400 and "not supported for this model" in body, (
        f"xAI may have enabled editing on {model}: HTTP {resp.status_code} {body!r}. "
        f"Verify manually (docs.x.ai …/video/editing) and extend _VIDEO_EDIT_MODELS — OR the "
        f"probe URL is no longer valid (set HUTCH_EDIT_PROBE_VIDEO to a reachable xAI mp4)."
    )


def test_base_model_is_still_accepted_on_edits(plugin):
    """Control: the allow-listed model must NOT be refused for MODEL reasons.

    Measured relay behaviour: a model refusal is a synchronous HTTP 400; for
    an accepted model the submit is ALWAYS 200 + request_id and input
    validation happens asynchronously — the job then fails with "Unable to
    download provided video_url" and nothing is billed (it never started).
    So with the intentionally unreachable probe URL we expect exactly that:
    200 on submit, then a failed job whose reason is about the input, not the
    model. A 'done' status would mean the probe URL is a real video and a job
    was billed — fail loudly so it is fixed rather than repeated.
    """
    import time

    import requests

    resp = requests.post(
        f"{plugin._base_url()}/videos/edits",
        headers=plugin._headers(),
        json={"model": plugin._DEFAULT_VIDEO_EDIT_MODEL, "prompt": "no-op probe",
              "video": {"url": _PROBE_VIDEO_URL}},
        timeout=60,
    )
    body = resp.text[:400]
    assert "not supported for this model" not in body, body
    assert resp.status_code != 405, f"route missing/405: {body}"
    assert resp.status_code == 200, f"unexpected submit answer HTTP {resp.status_code}: {body}"
    request_id = str(resp.json().get("request_id") or "")
    assert request_id, body

    status, reason = "", ""
    for _ in range(12):  # ~1 min; the input-fetch failure lands within seconds
        poll = requests.get(f"{plugin._base_url()}/videos/{request_id}",
                            headers=plugin._headers(), timeout=30)
        if poll.status_code == 200 and isinstance(poll.json(), dict):
            job = poll.json()
            status = str(job.get("status") or "").lower()
            reason = plugin._failure_reason(job) if status in {"failed", "error"} else ""
            if status in {"failed", "error", "done", "expired"}:
                break
        time.sleep(5)
    assert status != "done", (
        f"control probe PRODUCED A BILLABLE VIDEO (request_id={request_id}). HUTCH_EDIT_PROBE_VIDEO "
        f"points to a live video — point it at an unreachable URL for the control."
    )
    assert status in {"failed", "error"}, f"probe job did not settle: status={status!r}"
    assert "not supported for this model" not in reason, reason
    assert "video_url" in reason or "download" in reason.lower(), (
        f"expected an input-fetch failure for the unreachable probe URL, got: {reason!r}"
    )
