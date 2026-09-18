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

    Large files are re-encoded first (see ``_compress_for_inline``): Hermes'
    cache emits 2–3 MB PNGs and the relay rejects a 3 MB+ body on
    ``/videos/generations`` with "length limit exceeded", so the standard
    "generate → animate" flow died on the transport. Small files are inlined
    byte-exact.
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
    raw = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    if not mime.startswith("image/"):
        mime = "image/png"
    if len(raw) > _INLINE_MAX_RAW_BYTES:
        raw, mime = _compress_for_inline(raw, mime)
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


# Inline budget: base64 inflates by 4/3, and the relay's request-body limit
# on /videos/generations bit at ~3 MB of data URI (a 2.28 MB PNG failed, a
# 110 KB JPEG passed). Keep the encoded payload comfortably under 1 MB.
_INLINE_MAX_RAW_BYTES = 768 * 1024        # ~1 MB after base64
_INLINE_MAX_SIDE_PX = 1536
_INLINE_JPEG_QUALITY = 85


def _compress_for_inline(raw: bytes, mime: str) -> Tuple[bytes, str]:
    """Shrink an oversized image for data-URI transport.

    Downscale so the long side is ≤ ``_INLINE_MAX_SIDE_PX`` and re-encode as
    JPEG q85 (alpha flattened onto white — the relay's media models take
    opaque input anyway). If the result is still over budget, step quality
    down. Pillow is a core Hermes dependency; if decoding fails the original
    bytes are returned untouched so the relay reports the real problem.
    """
    try:
        from io import BytesIO
        from PIL import Image, ImageOps
        with Image.open(BytesIO(raw)) as img:
            img.load()
            # Honour EXIF orientation before resizing: the re-encoded JPEG
            # carries no EXIF, so a phone photo would otherwise arrive rotated.
            img = ImageOps.exif_transpose(img) or img
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.getchannel("A"))
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            scale = _INLINE_MAX_SIDE_PX / max(w, h)
            if scale < 1.0:
                img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                 Image.Resampling.LANCZOS)
            quality = _INLINE_JPEG_QUALITY
            while True:
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                out = buf.getvalue()
                if len(out) <= _INLINE_MAX_RAW_BYTES or quality <= 50:
                    break
                quality -= 10
            logger.debug("hutch media: inlined image re-encoded %s→%s bytes (q=%d, %dx%d)",
                         len(raw), len(out), quality, *img.size)
            return out, "image/jpeg"
    except Exception as exc:  # pragma: no cover - defensive: undecodable input
        logger.debug("hutch media: image re-encode skipped (%s); sending original", exc)
        return raw, mime


# Hermes canonical aspect names → relay ``size`` (the xAI branch derives
# aspect_ratio from size via xaiImagesAspectRatioFromSize; the OpenAI/codex
# branch forwards size verbatim to gpt-image). Same table as the bundled
# openai image plugin.
_IMAGE_SIZES: Dict[str, str] = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}

# Orientation sentence appended to text->image prompts. Measured on the relay
# (gpt-image-2, 3 runs each): a neutral prompt returned 1254x1254 for every
# `size` (18/18); with this exact wording the output matched `size` pixel-for-
# pixel (12/12), including prompts whose subject pulls the other way ("macro
# photo" -> portrait, "tall lighthouse" -> landscape). The ratio words matter:
# "wider than tall" alone gave non-standard 1774x887-class sizes; a prefix
# instead of a suffix gave 941x1672. Keep this wording unless re-measured.
_ORIENTATION_HINTS: Dict[str, str] = {
    "portrait": "Vertical portrait orientation, 2:3 aspect ratio.",
    "landscape": "Wide horizontal landscape orientation, 3:2 aspect ratio.",
}


def _is_gpt_image_model(model_id: str) -> bool:
    """True for the OpenAI gpt-image family (``openai/gpt-image-2``,
    ``gpt-image-2.5-sunburst`` …) — the only relay branch measured to ignore
    ``size`` on text->image. Same prefix match as ``_family_meta``."""
    return model_id.rsplit("/", 1)[-1].lower().startswith("gpt-image")


def _join_hint(prompt: str, hint: str) -> str:
    """Append the orientation sentence without doubling terminal punctuation
    (``prompt`` is already stripped by the caller)."""
    body = prompt.rstrip(".")
    tail = "" if body and body[-1] in "!?" else "."
    return f"{body}{tail} {hint}"


def _measure_aspect(image: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    """Canonical aspect name for a LOCAL image file, plus (width, height).

    Returns ``("", None)`` for remote URLs or unreadable files. Ratio bands:
    within 10% of square → ``square``; wider → ``landscape``; taller →
    ``portrait`` (matches how the tool layer names the three options).
    """
    ref = (image or "").strip()
    if not ref or ref.lower().startswith(("http://", "https://", "data:")):
        return "", None
    try:
        from PIL import Image
        with Image.open(ref) as img:
            w, h = img.size
    except Exception:
        return "", None
    if not w or not h:
        return "", None
    ratio = w / h
    if 0.9 <= ratio <= 1.1:
        return "square", (w, h)
    return ("landscape" if ratio > 1 else "portrait"), (w, h)


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
            applied_hint = ""
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
                    # Text->image: the Codex/ChatGPT backend behind gpt-image
                    # ignores the tool's `size` and picks the aspect from the
                    # prompt text (measured: a neutral prompt gave 1254x1254
                    # for every size, 18/18; the same prompt with an
                    # orientation sentence matched size exactly, 12/12). Append
                    # an orientation hint for non-square gpt-image requests.
                    # Gated to that family: the xAI branch derives aspect from
                    # `size` itself (unmeasured for grok/muse — no mutation
                    # there). Edits honour `size` on their own — not hinted.
                    applied_hint = (
                        _ORIENTATION_HINTS.get(aspect, "")
                        if _is_gpt_image_model(model_id) else ""
                    )
                    sent_prompt = _join_hint(prompt, applied_hint) if applied_hint else prompt
                    payload = {"model": model_id, "prompt": sent_prompt}
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
                # The relay drops size/quality for gpt-image on text->image
                # (returns ~square regardless), while honouring it on edits.
                # Report the aspect the caller actually got, and the requested
                # one alongside, so the agent does not describe a square as
                # "portrait".
                actual_aspect, dims = _measure_aspect(image)
                extra: Dict[str, Any] = {"requested_aspect_ratio": aspect}
                if applied_hint:
                    extra["prompt_orientation_hint"] = applied_hint
                if dims:
                    extra["width"], extra["height"] = dims
                if actual_aspect and actual_aspect != aspect and dims:
                    logger.info("hutch image: requested %s, relay returned %s (%sx%s) for %s",
                                aspect, actual_aspect, dims[0], dims[1], model_id)
                return success_response(image=image, model=model_id, prompt=prompt,
                                        aspect_ratio=actual_aspect or aspect,
                                        provider=self.name, modality=modality, extra=extra)
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
                strengths = "Relay video model"
                if _video_edit_model_ok(model_id):
                    strengths += "; supports editing (hutch_video_edit)"
                rows.append({
                    "id": model_id, "display": model_id,
                    "strengths": strengths, "modalities": ["text", "image"],
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
                # Self-documenting (the core tool layer does not read these):
                # editing is a separate tool, hutch_video_edit.
                "supports_edit": True,
                "edit_models": sorted(_VIDEO_EDIT_MODELS),
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
            return _extract_video_url(body)

        @staticmethod
        def _failure_reason(body: Dict[str, Any]) -> str:
            return _failure_reason(body)

        def _poll(self, request_id: str) -> Tuple[str, str]:
            return _poll_video(request_id)

    return HutchVideoGenProvider()


# ---------------------------------------------------------------------------
# Shared video polling (used by the video_gen provider and hutch_video_edit)
# ---------------------------------------------------------------------------


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


def _poll_video(request_id: str, *, final_body: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """Poll ``GET /videos/:request_id`` with the SAME key until terminal.

    Returns ``(video_url, error)``: exactly one is non-empty on a terminal
    outcome; both empty means the deadline passed. When ``final_body`` (a
    caller-owned dict) is given, the terminal poll body is copied into it so
    callers can read metadata such as ``video.duration``. The relay's
    ``error.message`` on a failed job is surfaced instead of swallowed — it
    carries the actionable cause (e.g. "image_url must be base64 or URL"),
    which otherwise required a manual GET to recover.

    Non-transient 4xx fail fast after a few consecutive hits instead of
    spinning to the 15-minute deadline. The relay binds ``request_id`` to an
    upstream account IN MEMORY with a 3h TTL (``videoAuthBindings``), so a
    relay restart or an expired binding turns every poll into a 4xx — the
    error text says so. Statuses are the raw xAI dictionary (``done``), not
    normalised — the native ``/v1/videos/:id`` route does not translate them.
    """
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
                    return "", (f"Polling rejected: HTTP {poll.status_code} x{consecutive_4xx} "
                                f"({poll.text[:200]}) — the relay was restarted or its 3h "
                                f"auth binding for this request expired; resubmit the job")
                continue
            if poll.status_code >= 500:
                consecutive_4xx = 0  # upstream hiccup, not an auth/routing failure
                logger.debug("hutch video poll %s: HTTP %s", request_id, poll.status_code)
                continue
            consecutive_4xx = 0
            body = poll.json()
            if not isinstance(body, dict):
                raise ValueError(f"non-object poll body: {type(body).__name__}")
        except Exception as exc:
            logger.debug("hutch video poll %s failed: %s", request_id, exc)
            continue
        status = str(body.get("status") or "").lower()
        if status in {"failed", "error", "expired", "cancelled", "canceled"}:
            if final_body is not None:
                final_body.update(body)
            return "", _failure_reason(body)
        url = _extract_video_url(body)
        if url and status in {"", "completed", "succeeded", "done"}:
            if final_body is not None:
                final_body.update(body)
            return url, ""
    return "", f"Timed out after {int(_POLL_DEADLINE_S)}s (last HTTP {last_status})"


# ---------------------------------------------------------------------------
# hutch_video_edit — generative video editing on POST /videos/edits
# ---------------------------------------------------------------------------

# The only model xAI documents for editing (docs.x.ai …/video/editing):
# grok-imagine-video-1.5 / -1.5-preview answer HTTP 400 "Video editing is not
# supported for this model" — from xAI, before billing. Static on purpose:
# the relay's /models catalog is unstable (depends on which xAI accounts are
# live), while the base model answers on /videos/edits even when absent from
# the list (built-in in the relay's registry). Extend after re-running the
# live probe in tests/test_video_edit_live.py.
_VIDEO_EDIT_MODELS = frozenset({"grok-imagine-video"})
_VIDEO_EDIT_MODEL_PREFIXES = frozenset({"", "xai", "x-ai", "grok"})  # = relay isXAIVideosModel
_DEFAULT_VIDEO_EDIT_MODEL = "grok-imagine-video"
# Empirical: an 11.3 MB data:video/mp4 body was accepted on /videos/edits
# (the relay has no MaxBytesReader there; the limit sits upstream). 20 MB is a
# soft guard against blind multi-minute uploads, not a measured ceiling.
_VIDEO_EDIT_MAX_BYTES = 20 * 1024 * 1024
# xAI documents 2–15 s only for extension; edits were observed in the same
# band. Outside it we WARN in the result, never block.
_VIDEO_EDIT_DURATION_RANGE = (2.0, 15.0)

# TODO(2026-09-18): hutch_video_extend — POST /videos/extensions returns 405
# on every model. The relay routes it (xaiVideosExtensionsPath) but its
# default upstream for xAI OAuth accounts, cli-chat-proxy.grok.com, does not
# serve the route. Revisit once the relay carries an xAI account with a
# direct API key (using_api: true); the probe is a one-line POST.

HUTCH_VIDEO_EDIT_SCHEMA: Dict[str, Any] = {
    "name": "hutch_video_edit",
    "description": (
        "Edit an existing video with a text instruction via the Hutch relay: add "
        "objects, effects or weather, change style or lighting while keeping the "
        "source motion. Output keeps the source duration (max 720p). Separate from "
        "video_generate because editing is provider-specific. Returns the edited "
        "video as an absolute local file path in `video`."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "What to change in the source video."},
            "video_url": {
                "type": "string",
                "description": (
                    "Source video: an absolute local .mp4 path (e.g. the `video` returned "
                    "by video_generate) or a public HTTPS URL."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional relay model override. Only models that support editing are "
                    "accepted (currently grok-imagine-video)."
                ),
            },
        },
        "required": ["prompt", "video_url"],
    },
}


def _video_edit_model_ok(model_id: str) -> bool:
    """Allow-list check with the relay's own prefix rules (isXAIVideosModel)."""
    raw = (model_id or "").strip().lower()
    prefix, _, base = raw.rpartition("/")
    return base in _VIDEO_EDIT_MODELS and prefix in _VIDEO_EDIT_MODEL_PREFIXES


def _video_edit_default_model() -> str:
    """``video_gen.hutch.edit_model`` → built-in default.

    Deliberately NOT ``video_gen.model``: that is the generation default
    (typically grok-imagine-video-1.5), which cannot edit — inheriting it
    would turn every edit into a pointless 400.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        scoped = ((cfg.get("video_gen") or {}).get("hutch") or {}).get("edit_model")
        if isinstance(scoped, str) and scoped.strip():
            return scoped.strip()
    except Exception:
        pass
    return _DEFAULT_VIDEO_EDIT_MODEL


def _probe_video_duration(path: Path) -> Optional[float]:
    """Duration in seconds via ffprobe when available; None otherwise."""
    import shutil
    import subprocess
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


def _inline_video_ref(value: str) -> Tuple[str, Dict[str, Any]]:
    """Return ``(video_url_for_relay, meta)``; raises ValueError on bad input.

    URLs pass through. A local ``.mp4`` is read behind the shared credential
    guard and inlined as ``data:video/mp4;base64,…`` — a first-class input per
    xAI docs, NOT re-encoded (the relay accepted 11 MB; transcoding would
    degrade the source). ``meta`` carries size/duration warnings for the result.
    """
    ref = (value or "").strip()
    meta: Dict[str, Any] = {}
    if not ref:
        raise ValueError("video_url is required")
    lower = ref.lower()
    if lower.startswith(("http://", "https://", "data:video/")):
        return ref, meta
    path = Path(ref).expanduser()
    # Guard FIRST — before any stat/exists probe — so a denied path leaks
    # neither its existence nor its size through the error text (the guard
    # works on the resolved path and does not need the file to exist).
    from agent.file_safety import raise_if_read_blocked
    raise_if_read_blocked(str(path))
    if not path.is_file():
        raise ValueError(f"video_url is neither an http(s) URL nor an existing local file: {ref}")
    if path.suffix.lower() != ".mp4":
        raise ValueError("only .mp4 sources are accepted by the relay (H.264/H.265/AV1)")
    size = path.stat().st_size
    if size > _VIDEO_EDIT_MAX_BYTES:
        raise ValueError(
            f"source video is {size / 2**20:.1f} MB, above the {_VIDEO_EDIT_MAX_BYTES / 2**20:.0f} MB "
            f"inline limit — shrink it first, e.g. ffmpeg -i in.mp4 -vf scale=-2:720 -crf 28 out.mp4"
        )
    dur = _probe_video_duration(path)
    if dur is not None:
        meta["source_duration"] = round(dur, 2)
        lo, hi = _VIDEO_EDIT_DURATION_RANGE
        if not (lo <= dur <= hi):
            meta["warning"] = (f"source is {dur:.1f}s; the relay's editing model was observed "
                               f"to accept {lo:.0f}–{hi:.0f}s clips")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}", meta


def _check_hutch_video_edit() -> bool:
    """Tool is offered only when hutch is the active video_gen provider and
    the relay credentials resolve — read at call time, never at import."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
    except Exception:
        return False
    section = cfg.get("video_gen") if isinstance(cfg, dict) else None
    if not (isinstance(section, dict) and section.get("provider") == "hutch"):
        return False
    return bool(_api_key() and _base_url())


def _handle_hutch_video_edit(args: Dict[str, Any], **_kw: Any) -> str:
    import json
    import requests
    from tools.registry import tool_error

    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return tool_error("prompt is required", success=False, error_type="invalid_input")
    if not (_api_key() and _base_url()):
        return tool_error("HUTCH_API_KEY / HUTCH_BASE_URL not configured", success=False,
                          error_type="missing_api_key")

    model_id = str(args.get("model") or "").strip() or _video_edit_default_model()
    if not _video_edit_model_ok(model_id):
        return tool_error(
            f"model {model_id!r} does not support video editing; accepted: "
            f"{', '.join(sorted(_VIDEO_EDIT_MODELS))} (optionally prefixed xai/, x-ai/, grok/)",
            success=False, error_type="model_not_editable",
            editable_models=sorted(_VIDEO_EDIT_MODELS),
        )

    source = str(args.get("video_url") or "")
    try:
        video_ref, meta = _inline_video_ref(source)
    except ValueError as exc:
        etype = "video_too_large" if "inline limit" in str(exc) else "invalid_input"
        return tool_error(str(exc), success=False, error_type=etype)
    except Exception as exc:  # file_safety guard, unreadable file
        return tool_error(str(exc), success=False, error_type=type(exc).__name__)

    # Native xAI edit contract on the relay's untranslated /v1/videos/edits:
    # exactly model + prompt + video.url. duration/aspect_ratio/resolution are
    # ignored on edits per xAI docs (output inherits the source), and the
    # OpenAI-isms (input_reference, seconds) are not translated on this route.
    payload = {"model": model_id, "prompt": prompt, "video": {"url": video_ref}}
    try:
        submit = requests.post(f"{_base_url()}/videos/edits", headers=_headers(),
                               json=payload, timeout=60)
    except Exception as exc:
        return tool_error(f"relay request failed: {exc}", success=False,
                          error_type=type(exc).__name__)
    if submit.status_code >= 400:
        text = submit.text[:400]
        etype = "model_not_editable" if "not supported for this model" in text else "provider_error"
        return tool_error(f"HTTP {submit.status_code}: {text}", success=False, error_type=etype,
                          model=model_id)
    try:
        body = submit.json()
    except Exception:
        return tool_error("relay returned a non-JSON submit response", success=False,
                          error_type="provider_error", model=model_id)
    if not isinstance(body, dict):
        return tool_error(f"relay returned a non-object submit response ({type(body).__name__})",
                          success=False, error_type="provider_error", model=model_id)
    request_id = str(body.get("request_id") or body.get("id") or "").strip()
    video_url = _extract_video_url(body)
    poll_error = ""
    final: Dict[str, Any] = dict(body)
    if not video_url and request_id:
        video_url, poll_error = _poll_video(request_id, final_body=final)
    if not video_url:
        return tool_error(f"{poll_error or 'relay returned no video URL'} "
                          f"(request_id={request_id or 'none'})",
                          success=False, error_type="provider_error", model=model_id)

    try:
        local = str(_vgp_module().save_url_video(video_url, prefix="hutch"))
    except Exception:
        local = video_url
    result: Dict[str, Any] = {
        "success": True,
        "video": local,
        "public_url": video_url,
        "source_video": source,
        "model": model_id,
        "prompt": prompt,
        "provider": "hutch",
        "request_id": request_id,
    }
    # xAI reports the output length in the terminal poll body (video.duration).
    vid = final.get("video")
    dur = vid.get("duration") if isinstance(vid, dict) else None
    if isinstance(dur, (int, float)):
        result["duration"] = dur
    result.update(meta)
    return json.dumps(result, ensure_ascii=False)


def _vgp_module():
    """``agent.video_gen_provider`` resolved at call time (monkeypatch seam)."""
    import agent.video_gen_provider as vgp
    return vgp


def register(ctx) -> None:
    """Register both hutch media backends and the video edit tool."""
    ctx.register_image_gen_provider(_build_image_provider())
    ctx.register_video_gen_provider(_build_video_provider())
    # Provider-specific edit workflow, same toolset as video_generate so it is
    # enabled together with it; check_fn gates on video_gen.provider == hutch.
    ctx.register_tool(
        name="hutch_video_edit",
        toolset="video_gen",
        schema=HUTCH_VIDEO_EDIT_SCHEMA,
        handler=_handle_hutch_video_edit,
        check_fn=_check_hutch_video_edit,
        requires_env=[],
        is_async=False,
        description="Edit an existing video by text instruction via the Hutch relay",
        emoji="🎬",
    )
