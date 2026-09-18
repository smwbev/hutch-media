# hutch-media

[Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that wires
Hermes **image and video generation** to the Hutch relay (a private
OpenAI-compatible gateway, CLIProxyAPI-based).

Companion to [`hutch-provider`](https://github.com/smwbev/hutch-provider)
(the inference provider) — install that first: the image backend resolves its
credentials through the `hutch` model provider.

## Install

```bash
hermes plugins install smwbev/hutch-provider   # inference provider (model-provider kind)
hermes plugins install smwbev/hutch-media      # this plugin (image + video backends)
hermes plugins enable hutch-media
```

## Configure

`~/.hermes/.env`:

```bash
HUTCH_BASE_URL=https://relay.example.com/v1   # your relay endpoint (required)
HUTCH_API_KEY=sk-...                          # your relay key (required)
```

`~/.hermes/config.yaml`:

```yaml
image_gen:
  provider: hutch
  hutch:
    surface: images     # CLIProxyAPI has no /images/models probe endpoint;
                        # force the Images API surface for image models
video_gen:
  provider: hutch
```

Restart the gateway / start a new session; `image_generate` and video
generation now go through the relay. Catalogs are read live from the relay's
`/models`, so newly added media models appear without a plugin update.

### Wire-contract notes (verified against CLIProxyAPI sources)

- **Image edits** (`/images/edits`, JSON): payload uses
  `images: [{"image_url": <url>}]` — the one shape parsed by both the xAI
  branch and the generic fallback.
- **Video** (`/v1/videos/generations`, the NATIVE xAI path — the relay
  forwards the JSON to xAI untranslated): numeric `duration`,
  `image: {"url": ...}` for image-to-video, `reference_images: [{"url": ...}]`,
  plus `aspect_ratio`/`resolution` (without them everything defaults to
  720x1280 portrait). The OpenAI-style `seconds`/`input_reference` spellings
  are only translated on `/openai/v1/videos` and are NOT used here.
- `image_url` and `reference_image_urls` are mutually exclusive upstream;
  the plugin prefers `image_url` and logs a warning.
- Polling reuses the SAME Bearer key as submit (relay auth-binding) and fails
  fast after 6 consecutive 4xx responses instead of spinning to the 15-minute
  deadline. The relay keeps that binding **in memory for 3 hours**: after a
  relay restart or past the TTL, polls for an old `request_id` return 4xx —
  the error says so; resubmit the job.

## Video editing — `hutch_video_edit`

A separate tool (same `video_gen` toolset as `video_generate`, offered when
`video_gen.provider: hutch` and the relay credentials resolve) that edits an
existing clip by text instruction: add objects, effects or weather, change
style or lighting while keeping the source motion.

```text
hutch_video_edit(prompt="add gently falling snow and a warm lens flare",
                 video_url="/abs/path/from/video_generate.mp4")   # or a public HTTPS URL
→ {"success": true, "video": "<local mp4>", "public_url": "<relay URL>",
   "source_video": "...", "model": "grok-imagine-video", "source_duration": 6.0}
```

- **Model**: only `grok-imagine-video` supports editing (xAI docs; `-1.5` /
  `-1.5-preview` answer `400 Video editing is not supported for this model`
  from xAI, before billing). The allow-list is static on purpose — the relay's
  `/models` list is unstable, while the base model answers on `/videos/edits`
  even when unlisted. Override via `video_gen.hutch.edit_model`; the
  generation default `video_gen.model` is deliberately **not** inherited.
- **Input**: local `.mp4` is inlined as `data:video/mp4;base64,…` (a
  first-class xAI input) — not re-encoded; the relay accepted 11 MB, the tool
  refuses above 20 MB with a `ffmpeg` hint. Output keeps the source duration
  and dimensions (720p cap). `duration`/`aspect_ratio`/`resolution` are
  ignored on edits by xAI and therefore not sent.
- **Not available: extend.** `POST /videos/extensions` returns 405 on every
  model — the relay routes it, but its default upstream
  (`cli-chat-proxy.grok.com`, used for xAI OAuth accounts) does not serve the
  route. It would need a relay account with a direct xAI API key
  (`using_api: true`). Re-check when the relay's account mix changes.
- **Detector for allow-list drift**: xAI does not announce capability changes.
  `tests/test_video_edit_live.py` (marker `live`, excluded by default) probes
  the rejected models and fails if any stops returning the exact refusal —
  run it monthly or when a new Imagine version ships:

  ```bash
  HUTCH_API_KEY=… HUTCH_BASE_URL=… \
  PYTHONPATH=/path/to/hermes-agent python -m pytest tests/test_video_edit_live.py -q -o 'addopts=' -m live
  ```

  The refusal arrives before billing, so the probe is free until the day it
  isn't — then one edit job is billed once, which is the signal. A weekly
  Hermes cron running the same command and reporting a red result is a good
  fit for an installation owner (not part of the plugin).

## Relay endpoint coverage (probed against CLIProxyAPI)

| Endpoint | Status | Effect |
|---|---|---|
| `POST /images/generations`, `/images/edits` | ✅ | image generation |
| `GET /images/models` | ❌ absent | probe 404s → force `surface: images` (above) |
| `POST /videos`, `/videos/generations` + `GET /videos/:request_id` | ✅ | video submit + poll |
| `POST /videos/edits` | ✅ `grok-imagine-video` only | `hutch_video_edit` |
| `POST /videos/extensions` | ❌ 405 from upstream (`cli-chat-proxy.grok.com`) | extend not offered |
| `GET /videos/models` | ❌ absent (captured by the `:request_id` route) | video catalog synthesized from `/models` (video-named slugs) |
| `POST /audio/transcriptions` | ❌ absent | relay STT models not reachable yet |

## Tests

```bash
PYTHONPATH=/path/to/hermes-agent python -m pytest tests/ -q      # unit (live probes excluded)
```

## License

MIT
