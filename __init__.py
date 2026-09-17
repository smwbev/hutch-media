"""Hutch media backends — relay image and video generation for Hermes Agent.

One installable plugin (``hermes plugins install smwbev/hutch-media``) that
registers two generation backends against the Hutch relay:

- **image**: reuses the bundled OpenRouter-compatible image provider
  (chat-completions image output + the dedicated ``/images`` API) with
  ``runtime_name="hutch"`` — credentials and endpoint resolve through the
  ``hutch`` model provider (install ``smwbev/hutch-provider`` first).
- **video**: subclasses the bundled OpenRouter video provider (async submit +
  ``GET /videos/:request_id`` polling) with ``HUTCH_API_KEY`` /
  ``HUTCH_BASE_URL``; the catalog is synthesized from ``/models`` because
  CLIProxyAPI serves no ``/videos/models`` route.

Catalogs are live — new relay media models appear without a plugin update.
"""

import logging
import os

logger = logging.getLogger(__name__)

def _openrouter_compat_class():
    """The bundled OpenRouterCompatImageProvider class, via its loaded module.

    The bundled image_gen/openrouter plugin is imported by the PluginManager
    under the ``hermes_plugins`` namespace before user plugins load (bundled
    backends are ``load_now``; user plugins come later in discovery order).
    """
    import sys
    for name, module in list(sys.modules.items()):
        if "image_gen" in name and hasattr(module, "OpenRouterCompatImageProvider"):
            return module.OpenRouterCompatImageProvider
    return None


def _register_image(ctx) -> None:
    """Register the hutch image backend, reusing the bundled compat provider."""
    compat_cls = _openrouter_compat_class()
    if compat_cls is None:
        logger.warning(
            "hutch image_gen: bundled openrouter image plugin not loaded — "
            "cannot reuse OpenRouterCompatImageProvider; hutch image backend disabled."
        )
        return
    ctx.register_image_gen_provider(compat_cls(
        provider_name="hutch",
        display_name="Hutch",
        runtime_name="hutch",
        config_key="hutch",
        model_env_var="HUTCH_IMAGE_MODEL",
        supports_image_api=True,
        setup_schema={
            "name": "Hutch (image)",
            "badge": "relay",
            "tag": "gpt-image-2/2.5, Grok Imagine, Muse Image via the Hutch relay; uses HUTCH_API_KEY",
            "env_vars": [{
                "key": "HUTCH_API_KEY", "prompt": "Hutch relay API key", "url": "",
            }],
        },
    ))


def _openrouter_video_class():
    """The bundled OpenRouterVideoGenProvider class, via its loaded module."""
    import sys
    for name, module in list(sys.modules.items()):
        if "video_gen" in name and hasattr(module, "OpenRouterVideoGenProvider"):
            return module.OpenRouterVideoGenProvider
    return None


def _register_video(ctx) -> None:
    """Register the hutch video backend, reusing the bundled OpenRouter provider."""
    base_cls = _openrouter_video_class()
    if base_cls is None:
        logger.warning(
            "hutch video_gen: bundled openrouter video plugin not loaded — "
            "hutch video backend disabled."
        )
        return

    class HutchVideoGenProvider(base_cls):
        """OpenRouter video wire behavior pointed at the Hutch relay."""

        name = "hutch"
        display_name = "Hutch"

        def _api_key(self) -> str:
            return os.environ.get("HUTCH_API_KEY", "").strip()

        def _base_url(self) -> str:
            return os.environ.get("HUTCH_BASE_URL", "").strip().rstrip("/")

        def is_available(self) -> bool:
            return bool(self._api_key() and self._base_url())

        def _catalog(self):
            """Live video catalog synthesized from the relay's ``/models`` list.

            Overridden entirely: the inherited implementation targets
            OpenRouter's ``/videos/models`` (which CLIProxyAPI does not serve —
            a GET there is captured by the ``/videos/:request_id`` polling
            route and 400s) and falls back to a snapshot of OPENROUTER's video
            models when unreachable — meaningless slugs for the relay. The
            relay's single source of truth is ``/models`` (Bearer auth);
            entries are filtered to video-named slugs with permissive defaults
            — the relay validates the actual generation request. Same TTL
            cache slot as the parent.
            """
            import time
            import requests
            if self._catalog_cache and time.monotonic() - self._catalog_cache[1] < 600:
                return self._catalog_cache[0]
            entries = []
            try:
                response = requests.get(
                    f"{self._base_url()}/models",
                    headers={"Authorization": f"Bearer {self._api_key()}"}, timeout=15)
                response.raise_for_status()
                entries = [
                    {"id": str(row.get("id")), "name": str(row.get("id"))}
                    for row in (response.json().get("data") or [])
                    if "video" in str(row.get("id", "")).lower()
                ]
            except Exception as exc:
                logger.debug("hutch video: /models catalog fetch failed: %s", exc)
                return []
            if entries:
                self._catalog_cache = (entries, time.monotonic())
            return entries

    ctx.register_video_gen_provider(HutchVideoGenProvider())


def register(ctx) -> None:
    """Register both hutch media backends."""
    _register_image(ctx)
    _register_video(ctx)
