# Backend

FastAPI pipeline for `camera roll -> face cluster pick -> clothing extraction -> product lookup`.

## Flow

1. `POST /api/jobs` uploads photos and creates a job.
2. Background worker analyzes faces and creates clusters (`awaiting_face_pick`).
3. `POST /api/jobs/{job_id}/select-cluster` starts clothing extraction for that person.
4. Items are inserted as `tier="pending"` and async lookup workers promote them to:
   - `exact`
   - `similar`
   - `generic` (no confident match)
5. Web/mobile poll `GET /api/jobs/{job_id}` to watch progress.

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
- `POST /api/jobs` (`multipart/form-data` with repeated `photos`)
- `POST /api/jobs/{job_id}/select-cluster`
- `POST /api/items/{item_id}/refresh`

## Env Notes

Use `.env.example` as the source of truth.

- `OPENAI_API_KEY` enables model-based clothing extraction/ranking.
- `SERPAPI_KEY` enables exact/similar shopping candidates.
- `REPLICATE_API_TOKEN` enables optional bbox refinement.
- `PUBLIC_MEDIA_BASE_URL` should be a publicly reachable base URL for this backend when external services need to fetch your media.
