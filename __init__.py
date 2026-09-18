"""Hutch media backends — relay image and video generation for Hermes Agent.

Self-contained providers built on the public Hermes ABCs
(``agent.image_gen_provider.ImageGenProvider`` /
``agent.video_gen_provider.VideoGenProvider``) and tailored to the actual
CLIProxyAPI surface:

- image: ``POST /images/generations`` (text) and ``POST /images/edits``
  (image-to-image); responses default to ``b64_json`` and are saved into the
  Hermes image cache.
- video: ``POST /videos/generations`` submit + ``GET /videos/:request_id``
  polling (xAI-style async contract; status queued/in_progress → completed).
- catalogs: CLIProxyAPI has no ``/images/models`` or ``/videos/models`` —
  both catalogs are synthesized from the relay's single ``/models`` list.

Credentials: ``HUTCH_API_KEY`` / ``HUTCH_BASE_URL`` (same env contract as the
companion ``hutch`` model provider from smwbev/hutch-provider).
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 180
_POLL_INTERVAL_S = 10.0
_POLL_DEADLINE_S = 900.0
_POLL_MAX_CONSECUTIVE_4XX = 6  # fail fast on non-transient auth/routing errors
_CATALOG_TTL_S = 300.0

# Model-id substrings that must never be classified as media models even when
# the id also contains "image"/"video" (keeps the substring filter honest for
# future relay slugs like "video-transcribe" or "image-embedding").
_NON_MEDIA_TOKENS = ("transcribe", "embed", "caption", "audio", "voice")

# Curated metadata for known relay model families; anything else that looks
# like a media model in /models still appears with generic text.
_IMAGE_FAMILY_META: Dict[str, Dict[str, str]] = {
    "gpt-image": {"strengths": "OpenAI gpt-image family: strong prompt adherence, text rendering"},
    "grok-imagine-image": {"strengths": "xAI Grok Imagine: fast, photorealistic"},
    "muse-image": {"strengths": "Meta Muse Image"},
}
_DEFAULT_IMAGE_MODEL = "gpt-image-2"
_DEFAULT_VIDEO_MODEL = "grok-imagine-video"


def _api_key() -> str:
    return os.environ.get("HUTCH_API_KEY", "").strip()


def _base_url() -> str:
    return os.environ.get("HUTCH_BASE_URL", "").strip().rstrip("/")


def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}


_catalog_cache: Dict[str, Tuple[List[str], float]] = {}


def _relay_model_ids() -> List[str]:
    """All model ids from the relay's ``/models`` (TTL-cached per base URL); [] on failure.

    The cache is keyed by the resolved base URL so two profiles pointing at
    different relays inside one multiplexing gateway never see each other's
    catalog."""
    base = _base_url()
    cached = _catalog_cache.get(base)
    if cached and time.monotonic() - cached[1] < _CATALOG_TTL_S:
        return cached[0]
    try:
        import requests
        response = requests.get(
            f"{base}/models", headers={"Authorization": f"Bearer {_api_key()}"}, timeout=15)
        response.raise_for_status()
        ids = [str(row.get("id") or "") for row in (response.json().get("data") or [])]
        ids = [i for i in ids if i]
    except Exception as exc:
        logger.debug("hutch media: /models catalog fetch failed: %s", exc)
        return []
    if ids:
        _catalog_cache[base] = (ids, time.monotonic())
    return ids


def _config_model(section: str) -> Optional[str]:
    """``<section>_gen.hutch.model`` → ``<section>_gen.model`` from config.yaml."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        scoped = ((cfg.get(f"{section}_gen") or {}).get("hutch") or {}).get("model")
        if isinstance(scoped, str) and scoped.strip():
            return scoped.strip()
        flat = (cfg.get(f"{section}_gen") or {}).get("model")
        if isinstance(flat, str) and flat.strip():
            return flat.strip()
    except Exception:
        pass
    return None


def _family_meta(model_id: str) -> Dict[str, str]:
    base = model_id.rsplit("/", 1)[-1].lower()
    for prefix, meta in _IMAGE_FAMILY_META.items():
        if base.startswith(prefix):
            return meta
    return {}


def _is_media_model(base_id: str, token: str) -> bool:
    """Substring classification with a negative list, so future relay slugs
    like ``muse-voice-transcribe`` or ``llava-video-caption`` never leak into
    the media catalogs."""
    if token not in base_id:
        return False
    return not any(bad in base_id for bad in _NON_MEDIA_TOKENS)


def _inline_image_ref(value: str) -> str:
    """Return a URL / data-URI the relay accepts for an image input.

    Hermes' image/video tools pass LOCAL ABSOLUTE PATHS for cached images
    (the documented ``image_url`` contract); the relay only accepts public
    URLs or ``data:`` URIs and 400s / fails the job on a bare path. Mirror the
    bundled xAI/OpenRouter providers: pass URLs and data URIs through, read a
    local file behind the shared credential-read guard and inline it as a
    base64 data URI.
    """
    ref = (value or "").strip()
    if not ref:
        return ""
    lower = ref.lower()
    if lower.startswith(("http://", "https://", "data:")):
        return ref
    path = Path(ref).expanduser()
    if not path.is_file():
        return ref  # not a local file — let the relay report the real error
    from agent.file_safety import raise_if_read_blocked
    raise_if_read_blocked(str(path))
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    if not mime.startswith("image/"):
        mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


# Hermes canonical aspect names → relay ``size`` (the xAI branch derives
# aspect_ratio from size via xaiImagesAspectRatioFromSize; the OpenAI/codex
# branch forwards size verbatim to gpt-image). Same table as the bundled
# openai image plugin.
_IMAGE_SIZES: Dict[str, str] = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


def _build_image_provider():
    from agent.image_gen_provider import (
        DEFAULT_ASPECT_RATIO, ImageGenProvider, error_response,
        normalize_reference_images, resolve_aspect_ratio, save_b64_image, success_response,
    )

    class HutchImageGenProvider(ImageGenProvider):
        """Relay image generation on CLIProxyAPI's OpenAI-style ``/images`` endpoints."""

        @property
        def name(self) -> str:
            return "hutch"

        @property
        def display_name(self) -> str:
            return "Hutch"

        def is_available(self) -> bool:
            return bool(_api_key() and _base_url())

        def list_models(self) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            seen = set()
            for model_id in _relay_model_ids():
                base = model_id.rsplit("/", 1)[-1].lower()
                if not _is_media_model(base, "image") or "video" in base:
                    continue
                if base in seen:  # relay lists bare + prefixed spellings; keep one
                    continue
                seen.add(base)
                meta = _family_meta(model_id)
                rows.append({
                    "id": model_id, "display": model_id,
                    "strengths": meta.get("strengths", "Relay image model"),
                })
            return rows

        def default_model(self) -> Optional[str]:
            return _config_model("image") or _DEFAULT_IMAGE_MODEL

        def capabilities(self) -> Dict[str, Any]:
            return {"modalities": ["text", "image"], "max_reference_images": 4}

        def get_setup_schema(self) -> Dict[str, Any]:
            return {
                "name": "Hutch (image)",
                "badge": "relay",
                "tag": "gpt-image / Grok Imagine / Muse Image via the Hutch relay; uses HUTCH_API_KEY",
                "env_vars": [
                    {"key": "HUTCH_API_KEY", "prompt": "Hutch relay API key", "url": ""},
                    {"key": "HUTCH_BASE_URL", "prompt": "Hutch relay base URL (…/v1)", "url": ""},
                ],
            }

        def generate(
            self,
            prompt: str,
            aspect_ratio: str = DEFAULT_ASPECT_RATIO,
            *,
            image_url: Optional[str] = None,
            reference_image_urls: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            import requests
            prompt = (prompt or "").strip()
            aspect = resolve_aspect_ratio(aspect_ratio)
            model_id = (kwargs.get("model") or "").strip() or self.default_model() or _DEFAULT_IMAGE_MODEL
            if not prompt:
                return error_response(error="Prompt is required", error_type="invalid_input",
                                      provider=self.name, model=model_id, prompt="", aspect_ratio=aspect)
            if not self.is_available():
                return error_response(error="HUTCH_API_KEY / HUTCH_BASE_URL not configured",
                                      error_type="missing_api_key", provider=self.name,
                                      model=model_id, prompt=prompt, aspect_ratio=aspect)
            sources: List[str] = []
            if image_url:
                sources.append(image_url)
            sources.extend(normalize_reference_images(reference_image_urls) or [])
            modality = "image" if sources else "text"
            try:
                size = _IMAGE_SIZES.get(aspect)
                if sources:
                    # CLIProxyAPI's /images/edits JSON parsing: the xAI branch
                    # (collectXAIImagesFromJSON) accepts images[] items as
                    # strings or objects with url / image_url / image_url.url;
                    # the generic fallback branch reads ONLY images[].image_url.
                    # ``images: [{"image_url": url}]`` is the one shape parsed
                    # by every branch (verified against openai_images_handlers.go).
                    # Local paths are inlined as data URIs (relay rejects paths).
                    payload: Dict[str, Any] = {
                        "model": model_id, "prompt": prompt,
                        "images": [{"image_url": _inline_image_ref(u)} for u in sources[:4]],
                    }
                    endpoint = f"{_base_url()}/images/edits"
                else:
                    payload = {"model": model_id, "prompt": prompt}
                    endpoint = f"{_base_url()}/images/generations"
                if size:
                    # size is the one aspect carrier every relay branch reads
                    # (xAI derives aspect_ratio from it; OpenAI forwards it).
                    payload["size"] = size
                response = requests.post(endpoint, headers=_headers(), json=payload,
                                         timeout=_REQUEST_TIMEOUT)
                if response.status_code >= 400:
                    detail = response.text[:400]
                    return error_response(error=f"HTTP {response.status_code}: {detail}",
                                          error_type="provider_error", provider=self.name,
                                          model=model_id, prompt=prompt, aspect_ratio=aspect)
                data = (response.json().get("data") or [{}])[0]
                if data.get("b64_json"):
                    path = save_b64_image(data["b64_json"], prefix=self.name, extension="png")
                    image = str(path)
                elif data.get("url"):
                    image = data["url"]
                else:
                    return error_response(error="Relay returned neither b64_json nor url",
                                          error_type="provider_error", provider=self.name,
                                          model=model_id, prompt=prompt, aspect_ratio=aspect)
                return success_response(image=image, model=model_id, prompt=prompt,
                                        aspect_ratio=aspect, provider=self.name, modality=modality)
            except Exception as exc:
                return error_response(error=str(exc), error_type=type(exc).__name__,
                                      provider=self.name, model=model_id, prompt=prompt,
                                      aspect_ratio=aspect)

    return HutchImageGenProvider()


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


def _build_video_provider():
    import agent.video_gen_provider as _vgp
    from agent.video_gen_provider import VideoGenProvider, error_response, success_response

    class HutchVideoGenProvider(VideoGenProvider):
        """Relay video generation: async submit + ``GET /videos/:request_id`` polling."""

        @property
        def name(self) -> str:
            return "hutch"

        @property
        def display_name(self) -> str:
            return "Hutch"

        def is_available(self) -> bool:
            return bool(_api_key() and _base_url())

        def list_models(self) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            seen = set()
            for model_id in _relay_model_ids():
                base = model_id.rsplit("/", 1)[-1].lower()
                if not _is_media_model(base, "video"):
                    continue
                if base in seen:
                    continue
                seen.add(base)
                rows.append({
                    "id": model_id, "display": model_id,
                    "strengths": "Relay video model", "modalities": ["text", "image"],
                })
            return rows

        def default_model(self) -> Optional[str]:
            return _config_model("video") or _DEFAULT_VIDEO_MODEL

        def capabilities(self) -> Dict[str, Any]:
            return {
                "modalities": ["text", "image"],
                "aspect_ratios": ["16:9", "9:16", "1:1"],
                "resolutions": ["480p", "720p"],
                "min_duration": 1,
                "max_duration": 15,
                "supports_audio": False,
                "supports_negative_prompt": False,
                "max_reference_images": 3,
            }

        def get_setup_schema(self) -> Dict[str, Any]:
            return {
                "name": "Hutch (video)",
                "badge": "relay",
                "tag": "Grok Imagine video via the Hutch relay; uses HUTCH_API_KEY",
                "env_vars": [
                    {"key": "HUTCH_API_KEY", "prompt": "Hutch relay API key", "url": ""},
                    {"key": "HUTCH_BASE_URL", "prompt": "Hutch relay base URL (…/v1)", "url": ""},
                ],
            }

        def generate(
            self,
            prompt: str,
            *,
            model: Optional[str] = None,
            image_url: Optional[str] = None,
            reference_image_urls: Optional[List[str]] = None,
            duration: Optional[int] = None,
            aspect_ratio: str = "16:9",
            resolution: str = "720p",
            negative_prompt: Optional[str] = None,
            audio: Optional[bool] = None,
            seed: Optional[int] = None,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            import requests
            prompt = (prompt or "").strip()
            model_id = (model or "").strip() or self.default_model() or _DEFAULT_VIDEO_MODEL
            modality = "image" if image_url else "text"
            if not prompt:
                return error_response(error="Prompt is required", error_type="invalid_input",
                                      provider=self.name, model=model_id, prompt="",
                                      aspect_ratio=aspect_ratio)
            if not self.is_available():
                return error_response(error="HUTCH_API_KEY / HUTCH_BASE_URL not configured",
                                      error_type="missing_api_key", provider=self.name,
                                      model=model_id, prompt=prompt, aspect_ratio=aspect_ratio)
            payload: Dict[str, Any] = {"model": model_id, "prompt": prompt}
            if image_url and reference_image_urls:
                # The relay/upstream rejects the combination outright — prefer
                # the explicit i2v source and drop the references with a note.
                logger.warning("hutch video: image_url and reference_image_urls are mutually "
                               "exclusive on the relay; using image_url and ignoring references")
                reference_image_urls = None
            if image_url:
                # NATIVE path contract (/v1/videos/generations →
                # handleXAIVideosNativePost forwards rawJSON untranslated;
                # the xAI executor's normalizeXAIImageRefs passes image.url
                # through as-is). The OpenAI-ism input_reference.image_url is
                # translated ONLY on /openai/v1/videos — not here. Local
                # paths are inlined as data URIs (upstream fails the job on a
                # bare path: "image_url must be base64 or URL"). Inlining can
                # raise (file_safety guard, unreadable file) — keep the
                # VideoGenProvider contract and return error_response.
                try:
                    payload["image"] = {"url": _inline_image_ref(image_url)}
                except Exception as exc:
                    return error_response(error=str(exc), error_type=type(exc).__name__,
                                          provider=self.name, model=model_id, prompt=prompt,
                                          aspect_ratio=aspect_ratio)
            if reference_image_urls:
                try:
                    payload["reference_images"] = [
                        {"url": _inline_image_ref(u)} for u in list(reference_image_urls)[:3]]
                except Exception as exc:
                    return error_response(error=str(exc), error_type=type(exc).__name__,
                                          provider=self.name, model=model_id, prompt=prompt,
                                          aspect_ratio=aspect_ratio)
            if duration:
                # Native xAI dictionary: numeric "duration" ("seconds" is an
                # OpenAI-ism parsed only by the /openai/v1/videos translator).
                payload["duration"] = int(duration)
            # Native xAI form accepts aspect_ratio/resolution directly; without
            # them upstream defaults to 720x1280 portrait.
            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio
            if resolution:
                payload["resolution"] = resolution
            if seed is not None:
                payload["seed"] = int(seed)
            try:
                submit = requests.post(f"{_base_url()}/videos/generations",
                                       headers=_headers(), json=payload, timeout=60)
                if submit.status_code >= 400:
                    return error_response(error=f"HTTP {submit.status_code}: {submit.text[:400]}",
                                          error_type="provider_error", provider=self.name,
                                          model=model_id, prompt=prompt, aspect_ratio=aspect_ratio)
                body = submit.json()
                request_id = str(body.get("request_id") or body.get("id") or "").strip()
                video_url = self._extract_video_url(body)
                poll_error = ""
                if not video_url and request_id:
                    video_url, poll_error = self._poll(request_id)
                if not video_url:
                    reason = poll_error or "Relay returned no video URL"
                    return error_response(error=f"{reason} (request_id={request_id or 'none'})",
                                          error_type="provider_error", provider=self.name,
                                          model=model_id, prompt=prompt, aspect_ratio=aspect_ratio)
                # Relay URLs are often short-lived — persist into the cache.
                # Resolved via the module attribute (not a closure-captured
                # name) so tests can monkeypatch agent.video_gen_provider.
                try:
                    local = str(_vgp.save_url_video(video_url, prefix=self.name))
                except Exception:
                    local = video_url
                return success_response(video=local, model=model_id, prompt=prompt,
                                        modality=modality, aspect_ratio=aspect_ratio,
                                        duration=duration or 0, provider=self.name)
            except Exception as exc:
                return error_response(error=str(exc), error_type=type(exc).__name__,
                                      provider=self.name, model=model_id, prompt=prompt,
                                      aspect_ratio=aspect_ratio)

        @staticmethod
        def _extract_video_url(body: Dict[str, Any]) -> str:
            """Video URL from a submit/poll payload; '' when not (yet) present."""
            for path in (("video", "url"), ("output", "url"), ("url",), ("video_url",)):
                node: Any = body
                for key in path:
                    node = node.get(key) if isinstance(node, dict) else None
                    if node is None:
                        break
                if isinstance(node, str) and node.strip():
                    return node.strip()
            data = body.get("data")
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return str(data[0].get("url") or "").strip()
            return ""

        @staticmethod
        def _failure_reason(body: Dict[str, Any]) -> str:
            """Human-readable failure text from a relay/xAI poll body."""
            err = body.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("detail") or err.get("code")
                if msg:
                    return str(msg)
            if isinstance(err, str) and err.strip():
                return err.strip()
            for key in ("failure_reason", "message", "detail"):
                val = body.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return "Video generation failed on the relay"

        def _poll(self, request_id: str) -> Tuple[str, str]:
            """Poll ``GET /videos/:request_id`` with the SAME key until terminal.

            Returns ``(video_url, error)``: exactly one is non-empty on a
            terminal outcome; both empty means the deadline passed. The relay's
            ``error.message`` on a failed job is surfaced instead of swallowed —
            it carries the actionable cause (e.g. "image_url must be base64 or
            URL"), which otherwise required a manual GET to recover.

            Non-transient 4xx (401/403/404 — e.g. a lost in-memory auth binding
            after a relay restart) fail fast after a few consecutive hits
            instead of spinning to the 15-minute deadline."""
            import requests
            deadline = time.monotonic() + _POLL_DEADLINE_S
            consecutive_4xx = 0
            first = True
            last_status = 0
            while time.monotonic() < deadline:
                if first:
                    first = False  # poll immediately: fast jobs finish in seconds
                else:
                    time.sleep(_POLL_INTERVAL_S)
                try:
                    poll = requests.get(f"{_base_url()}/videos/{request_id}",
                                        headers=_headers(), timeout=30)
                    last_status = poll.status_code
                    if 400 <= poll.status_code < 500:
                        consecutive_4xx += 1
                        logger.debug("hutch video poll %s: HTTP %s (%d consecutive)",
                                     request_id, poll.status_code, consecutive_4xx)
                        if consecutive_4xx >= _POLL_MAX_CONSECUTIVE_4XX:
                            logger.warning("hutch video poll %s: giving up after %d consecutive "
                                           "HTTP %s responses", request_id, consecutive_4xx,
                                           poll.status_code)
                            return "", (f"Polling failed: HTTP {poll.status_code} "
                                        f"x{consecutive_4xx}: {poll.text[:200]}")
                        continue
                    if poll.status_code >= 500:
                        consecutive_4xx = 0  # upstream hiccup, not an auth/routing failure
                        logger.debug("hutch video poll %s: HTTP %s", request_id, poll.status_code)
                        continue
                    consecutive_4xx = 0
                    body = poll.json()
                except Exception as exc:
                    logger.debug("hutch video poll %s failed: %s", request_id, exc)
                    continue
                status = str(body.get("status") or "").lower()
                if status in {"failed", "error", "expired", "cancelled", "canceled"}:
                    return "", self._failure_reason(body)
                url = self._extract_video_url(body)
                if url and status in {"", "completed", "succeeded", "done"}:
                    return url, ""
            return "", f"Timed out after {int(_POLL_DEADLINE_S)}s (last HTTP {last_status})"

    return HutchVideoGenProvider()


def register(ctx) -> None:
    """Register both hutch media backends."""
    ctx.register_image_gen_provider(_build_image_provider())
    ctx.register_video_gen_provider(_build_video_provider())
