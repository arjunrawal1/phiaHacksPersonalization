# Backend

FastAPI pipeline for `camera roll -> face cluster pick -> clothing extraction -> product lookup`.

Uploads support common phone formats including HEIC/HEIF (via `pillow-heif`).

## Flow

1. `POST /api/jobs` uploads photos and creates a job.
2. Background worker analyzes faces and creates clusters.
3. If `user_name` is provided and Perplexity + AWS are configured, backend attempts auto face-pick using:
   - frequency of each cluster across uploaded photos
   - Rekognition alignment against public online photos for the user name
4. On high confidence, backend skips manual face pick and starts clothing extraction automatically.
5. Otherwise job moves to `awaiting_face_pick`.
6. `POST /api/jobs/{job_id}/select-cluster` starts clothing extraction for that person when manual pick is needed.
7. Items are inserted as `tier="pending"` and async lookup workers promote them to:
   - `exact`
   - `similar`
   - `generic` (no confident match)
8. Web/mobile poll `GET /api/jobs/{job_id}` to watch progress.

## Features

- SQLite-backed jobs/photos/clusters/items.
- FastAPI static media serving at `/media/*`.
- Optional AWS Rekognition clustering (falls back to local OpenCV/NumPy clustering).
- Optional OpenAI extraction + ranking (`gpt-5.4` default).
- Optional SerpAPI Google Lens + Google Shopping candidate lookup.
- Optional Replicate Grounding-DINO box refinement.
- Manual retry: `POST /api/items/{item_id}/refresh`.

## Run

### Docker

```bash
cp .env.example .env
docker compose up --build
```

### Local

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Backend defaults to `http://localhost:8000`.

### Supabase Buckets (Optional, Recommended)

```bash
cd backend
./scripts/setup_supabase_buckets.sh
```

This creates/ensures public buckets: `photos`, `face-tiles`, `clothing-crops`.

## Endpoints

- `GET /health`
- `GET /`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `POST /api/jobs` (`multipart/form-data` with repeated `photos`, optional `user_name`)
- `POST /api/jobs/{job_id}/select-cluster`
- `POST /api/items/{item_id}/refresh`
- `POST /api/styling/render-upload` (`multipart/form-data`: `photo`, optional `identity_face_crop`, optional repeated `reference_images`)
- `GET /api/styling/auto-runs?job_id=<JOB_ID>`
- `POST /api/items/{item_id}/render-model`
- `GET /api/phia/mobile-feed`
- `POST /api/phia/mobile-feed/simulate-session`
- `POST /api/phia/backfill-favorites`

## Env Notes

Use `.env.example` as the source of truth.

- `OPENAI_API_KEY` enables model-based clothing extraction/ranking.
- `OPENAI_IMAGE_TOOL_MODEL` picks the Responses model used for styling renders with the `image_generation` tool (`gpt-5.4` default).
- `GEMINI_API_KEY` enables Nano Banana renders through Gemini API.
- `GEMINI_IMAGE_MODEL` controls Gemini image model id (`gemini-2.5-flash-image` default).
- `PERPLEXITY_API_KEY` enables fetching online reference images by user name.
- `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` are required for Rekognition face clustering/alignment and auto-pick.
- `AUTO_FACE_PICK_*` controls deterministic auto-pick thresholds.
- `SERPAPI_KEY` enables exact/similar shopping candidates.
- `REPLICATE_API_TOKEN` enables optional bbox refinement.
- Preferred: set `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` to mirror `photoAlbumClosetApp` storage behavior. The backend uploads `photos/*`, `face_tiles/*`, and `clothing_crops/*` into public Supabase buckets and uses those public URLs for external model calls.
- Fallback: `PUBLIC_MEDIA_BASE_URL` can be used as a publicly reachable backend URL when external services need to fetch `/media/*`.

## Styling Render Experiment

Turn a camera photo (and optional reference images) into polished model shots.

```bash
curl -X POST "http://localhost:8000/api/styling/render-upload" \
  -F "photo=@/absolute/path/to/item.jpg" \
  -F "identity_face_crop=@/absolute/path/to/face_tile.jpg" \
  -F "style_preset=aesthetic" \
  -F "render_engine=nano_banana" \
  -F "subject_hint=person on left in black polo" \
  -F "variant_count=3" \
  -F "aspect_ratio=portrait" \
  -F "quality=high" \
  -F "scene_hint=soft luxury studio set"
```

Use extracted pipeline items directly:

```bash
curl -X POST "http://localhost:8000/api/items/<ITEM_ID>/render-model" \
  -H "Content-Type: application/json" \
  -d '{
    "style_preset": "streetwear",
    "include_face_tile": true,
    "variant_count": 3,
    "aspect_ratio": "portrait",
    "quality": "high",
    "scene_hint": "golden hour city crosswalk"
  }'
```

## Real Session Simulation (Phia)

Replay mobile-feed calls using real auth from captured production sessions.

```bash
curl -X POST "http://localhost:8000/api/phia/mobile-feed/simulate-session" \
  -H "Content-Type: application/json" \
  -d '{
    "auth": {
      "phia_id": "<REAL_PHIA_ID>",
      "session_cookie": "<REAL_SESSION_COOKIE>",
      "bearer_token": "<REAL_BEARER_TOKEN>",
      "platform": "IOS_APP",
      "platform_version": "2.3.11.362"
    },
    "inherit_default_auth": true,
    "explore_feed_input": {}
  }'
```

You can also call `GET /api/phia/mobile-feed` with override headers (`x-phia-id`, `cookie`, `authorization`, `x-platform`, `x-platform-version`) to replay a specific session.
