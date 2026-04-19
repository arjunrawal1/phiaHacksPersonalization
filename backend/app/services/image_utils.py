from __future__ import annotations

import io

from PIL import Image, ImageOps

_HEIF_REGISTERED = False

try:
    from pillow_heif import register_heif_opener  # type: ignore

    register_heif_opener()
    _HEIF_REGISTERED = True
except Exception:
    _HEIF_REGISTERED = False


def decode_to_rgb(payload: bytes) -> Image.Image:
    if not payload:
        raise ValueError("Empty image payload")
    try:
        with Image.open(io.BytesIO(payload)) as raw:
            oriented = ImageOps.exif_transpose(raw)
            return oriented.convert("RGB")
    except Exception as exc:
        hint = (
            "Unsupported image format. If this is HEIC/HEIF, ensure pillow-heif is installed."
            if not _HEIF_REGISTERED
            else "Unsupported or corrupted image format."
        )
        raise ValueError(hint) from exc

