# Backend

FastAPI service for phia.

## Stack

- Python 3.12
- FastAPI + Uvicorn
- pydantic-settings for config
- Docker + docker-compose

## Layout

```
backend/
├── app/
│   ├── api/routes.py     # HTTP routes
│   ├── core/config.py    # Settings loaded from env
│   └── main.py           # FastAPI app factory
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-dev.txt
```

## Run with Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
```

The API is served at http://localhost:8000

- `GET /` → sanity check
- `GET /health` → liveness probe
- `GET /docs` → Swagger UI
- `GET /redoc` → ReDoc

The `app/` directory is mounted into the container and Uvicorn runs with
`--reload`, so edits refresh the server automatically.

## Run locally (without Docker)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

## Configuration

All settings come from environment variables (or `.env`), parsed by
`app/core/config.py`. See `.env.example` for the supported keys.
