"""Invariant tests for the hutch image_gen and video_gen backends.

Run from a hermes-agent checkout (PYTHONPATH=<checkout>):

    PYTHONPATH=/path/to/hermes-agent python -m pytest tests/ -q -o 'addopts='

Each register() is exercised against the REAL bundled openrouter plugin
modules (imported from the checkout), with a fake ctx capturing what gets
registered — proving the reuse mechanism, the credential swap, and that the
bundled providers stay untouched.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_component(name: str):
    """Import the combined plugin module; *name* kept for call-site clarity."""
    spec = importlib.util.spec_from_file_location(
        f"hutch_media_test_{name}", REPO_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bundled(kind: str):
    """Import the bundled openrouter plugin for *kind* under a hermes_plugins-like name."""
    import os
    checkout = None
    for entry in sys.path:
        candidate = Path(entry or ".") / "plugins" / kind / "openrouter" / "__init__.py"
        if candidate.exists():
            checkout = candidate
            break
    assert checkout is not None, f"bundled plugins/{kind}/openrouter not found on PYTHONPATH"
    module_name = f"hermes_plugins.{kind}__openrouter"
    spec = importlib.util.spec_from_file_location(module_name, checkout)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
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
def bundled_image():
    mod = _load_bundled("image_gen")
    yield mod
    sys.modules.pop("hermes_plugins.image_gen__openrouter", None)


@pytest.fixture()
def bundled_video():
    mod = _load_bundled("video_gen")
    yield mod
    sys.modules.pop("hermes_plugins.video_gen__openrouter", None)


def test_image_registers_hutch_compat_provider(bundled_image):
    ctx = _Ctx()
    _load_component("image_gen").register(ctx)
    assert len(ctx.image) == 1
    provider = ctx.image[0]
    assert provider.name == "hutch"
    assert isinstance(provider, bundled_image.OpenRouterCompatImageProvider)
    # Credentials resolve through the hutch runtime, not openrouter's.
    assert provider._runtime_name == "hutch"
    assert provider._model_env_var == "HUTCH_IMAGE_MODEL"


def test_image_without_bundled_module_disables_quietly(monkeypatch):
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if "image_gen" in k and hasattr(sys.modules.get(k, None), "OpenRouterCompatImageProvider")}
    try:
        ctx = _Ctx()
        _load_component("image_gen").register(ctx)
        assert ctx.image == []  # no crash, no bogus registration
    finally:
        sys.modules.update(saved)


def test_video_registers_hutch_subclass_with_hutch_credentials(bundled_video, monkeypatch):
    monkeypatch.setenv("HUTCH_API_KEY", "sk-hutch-test")
    monkeypatch.setenv("HUTCH_BASE_URL", "https://relay.test.example/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-SHOULD-NOT-BE-USED")
    ctx = _Ctx()
    _load_component("video_gen").register(ctx)
    assert len(ctx.video) == 1
    provider = ctx.video[0]
    assert provider.name == "hutch"
    assert isinstance(provider, bundled_video.OpenRouterVideoGenProvider)
    # The credential swap is total: hutch env only, never openrouter's.
    assert provider._api_key() == "sk-hutch-test"
    assert provider._base_url() == "https://relay.test.example/v1"
    assert provider.is_available()


def test_video_unavailable_without_env(bundled_video, monkeypatch):
    monkeypatch.delenv("HUTCH_API_KEY", raising=False)
    monkeypatch.delenv("HUTCH_BASE_URL", raising=False)
    ctx = _Ctx()
    _load_component("video_gen").register(ctx)
    assert ctx.video and not ctx.video[0].is_available()


def test_bundled_openrouter_video_untouched(bundled_video, monkeypatch):
    """Registering hutch must not mutate the bundled provider's credentials."""
    monkeypatch.setenv("HUTCH_API_KEY", "sk-hutch-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-real")
    ctx = _Ctx()
    _load_component("video_gen").register(ctx)
    bundled = bundled_video.OpenRouterVideoGenProvider()
    assert bundled.name == "openrouter"
    assert bundled._api_key() == "sk-openrouter-real"
    assert "openrouter.ai" in bundled._base_url()


def test_video_catalog_synthesized_from_models(bundled_video, monkeypatch):
    """The catalog comes from /models filtered to video slugs — never /videos/models
    (absent on CLIProxyAPI; captured by the :request_id route) and never
    OpenRouter's snapshot fallback."""
    monkeypatch.setenv("HUTCH_API_KEY", "sk-hutch-test")
    monkeypatch.setenv("HUTCH_BASE_URL", "https://relay.test.example/v1")
    ctx = _Ctx()
    _load_component("video_gen").register(ctx)
    provider = ctx.video[0]

    calls = []

    class _Resp:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {"data": [
                {"id": "x-ai/grok-imagine-video"},
                {"id": "anthropic/claude-fable-5"},
                {"id": "grok-imagine-video-1.5-preview"},
            ]}

    import sys as _sys
    requests_mod = __import__("requests")
    def fake_get(url, **kwargs):
        calls.append(url)
        return _Resp()
    monkeypatch.setattr(requests_mod, "get", fake_get)

    catalog = provider._catalog()
    ids = {e["id"] for e in catalog}
    assert ids == {"x-ai/grok-imagine-video", "grok-imagine-video-1.5-preview"}
    assert all("/videos/models" not in u for u in calls), calls
    assert any(u.endswith("/models") for u in calls), calls
