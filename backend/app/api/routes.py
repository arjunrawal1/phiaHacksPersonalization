from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"message": "phia backend is running"}
