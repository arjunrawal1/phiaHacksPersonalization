# Backend

FastAPI pipeline for `camera roll -> face cluster pick -> clothing extraction -> product lookup`.

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

## Endpoints

- `GET /health`
- `GET /`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `POST /api/jobs` (`multipart/form-data` with repeated `photos`, optional `user_name`)
- `POST /api/jobs/{job_id}/select-cluster`
- `POST /api/items/{item_id}/refresh`

## Env Notes

Use `.env.example` as the source of truth.

- `OPENAI_API_KEY` enables model-based clothing extraction/ranking.
- `PERPLEXITY_API_KEY` enables fetching online reference images by user name.
- `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` are required for Rekognition face clustering/alignment and auto-pick.
- `AUTO_FACE_PICK_*` controls deterministic auto-pick thresholds.
- `SERPAPI_KEY` enables exact/similar shopping candidates.
- `REPLICATE_API_TOKEN` enables optional bbox refinement.
- `PUBLIC_MEDIA_BASE_URL` should be a publicly reachable base URL for this backend when external services need to fetch your media.
