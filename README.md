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
  deadline.

## Relay endpoint coverage (probed against CLIProxyAPI)

| Endpoint | Status | Effect |
|---|---|---|
| `POST /images/generations`, `/images/edits` | ✅ | image generation |
| `GET /images/models` | ❌ absent | probe 404s → force `surface: images` (above) |
| `POST /videos`, `/videos/generations` + `GET /videos/:request_id` | ✅ | video submit + poll |
| `GET /videos/models` | ❌ absent (captured by the `:request_id` route) | video catalog synthesized from `/models` (video-named slugs) |
| `POST /audio/transcriptions` | ❌ absent | relay STT models not reachable yet |

## Tests

```bash
PYTHONPATH=/path/to/hermes-agent python -m pytest tests/ -q -o 'addopts='
```

## License

MIT
