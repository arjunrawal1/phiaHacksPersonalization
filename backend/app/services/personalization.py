from __future__ import annotations

import base64
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from app.core.config import get_settings
from app.services import db


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _json_output_text(data: dict[str, Any]) -> str | None:
    raw = data.get("output_text")
    if raw:
        return str(raw)
    for out in data.get("output", []):
        if not isinstance(out, dict):
            continue
        for content in out.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"])
    return None


def _image_data_url(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with Image.open(path).convert("RGB") as image:
            if max(image.size) > 1400:
                image.thumbnail((1400, 1400))
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=84)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def _cache_key(photos: list[dict[str, Any]], items: list[dict[str, Any]]) -> str:
    photo_parts: list[str] = []
    for p in sorted(photos, key=lambda row: str(row.get("id") or "")):
        photo_parts.append(
            "|".join(
                [
                    _safe_text(p.get("id")),
                    _safe_text(p.get("captured_at_epoch_ms")),
                    _safe_text(p.get("captured_at")),
                    _safe_text(p.get("latitude")),
                    _safe_text(p.get("longitude")),
                ]
            )
        )

    item_parts: list[str] = []
    for i in sorted(items, key=lambda row: str(row.get("id") or "")):
        best = i.get("best_match") if isinstance(i.get("best_match"), dict) else {}
        item_parts.append(
            "|".join(
                [
                    _safe_text(i.get("id")),
                    _safe_text(i.get("description")),
                    _safe_text(i.get("brand_visible")),
                    _safe_text((best or {}).get("title")),
                    _safe_text((best or {}).get("source")),
                ]
            )
        )

    payload = "||".join(photo_parts) + "##" + "||".join(item_parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _photo_context_line(photo: dict[str, Any]) -> str:
    captured_at = _safe_text(photo.get("captured_at"))
    lat = photo.get("latitude")
    lon = photo.get("longitude")
    loc = ""
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        loc = f" location=({float(lat):.4f}, {float(lon):.4f})"
    when = f" captured_at={captured_at}" if captured_at else ""
    return f"Photo {photo.get('id')}{when}{loc}".strip()


def _summarize_photo_with_openai(
    *,
    openai_api_key: str,
    model: str,
    photo: dict[str, Any],
    photo_path: Path,
    item_hints: list[str],
) -> dict[str, Any]:
    data_url = _image_data_url(photo_path)
    if not data_url:
        return {
            "photo_id": str(photo.get("id") or ""),
            "summary": "Outfit photo from camera roll sync.",
            "style_tags": item_hints[:4] or ["everyday"],
            "brand_hints": [],
            "captured_at": photo.get("captured_at"),
            "latitude": photo.get("latitude"),
            "longitude": photo.get("longitude"),
        }

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "style_tags": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6,
            },
            "brand_hints": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
            },
        },
        "required": ["summary", "style_tags", "brand_hints"],
        "additionalProperties": False,
    }

    context = _photo_context_line(photo)
    hints_line = ", ".join(item_hints[:8]) if item_hints else "none"
    prompt = (
        "Analyze this fashion photo for personalization.\n"
        "Use the metadata context (time/location) as one signal for climate/lifestyle.\n"
        "Return concise output only.\n"
        f"Metadata: {context}\n"
        f"Closet hints from extraction: {hints_line}\n"
        "Requirements:\n"
        "- summary: <= 20 words\n"
        "- style_tags: short tags like 'smart casual', 'athleisure', 'minimal'\n"
        "- brand_hints: likely brands or retailer families if visually plausible; otherwise []\n"
    )

    payload = {
        "model": model,
        "reasoning": {"effort": "low"},
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url, "detail": "low"},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "photo_personalization",
                "schema": schema,
                "strict": True,
            }
        },
    }

    res = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=80,
    )
    res.raise_for_status()
    data = res.json()
    raw = _json_output_text(data)
    if not raw:
        raise RuntimeError("Missing model output")
    parsed = json.loads(raw)
    return {
        "photo_id": str(photo.get("id") or ""),
        "summary": _safe_text(parsed.get("summary")) or "Outfit photo from camera roll sync.",
        "style_tags": [str(x).strip() for x in (parsed.get("style_tags") or []) if str(x).strip()],
        "brand_hints": [str(x).strip() for x in (parsed.get("brand_hints") or []) if str(x).strip()],
        "captured_at": photo.get("captured_at"),
        "latitude": photo.get("latitude"),
        "longitude": photo.get("longitude"),
    }


def _aggregate_with_openai(
    *,
    openai_api_key: str,
    model: str,
    job_id: str,
    photo_insights: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "collection_titles": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 6,
            },
            "notifications": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 6,
            },
            "favorite_brands": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "style_summary": {"type": "string"},
        },
        "required": ["collection_titles", "notifications", "favorite_brands", "style_summary"],
        "additionalProperties": False,
    }

    product_titles: list[str] = []
    brand_visible: list[str] = []
    for item in items:
        if item.get("brand_visible"):
            brand_visible.append(_safe_text(item.get("brand_visible")))
        best = item.get("best_match") if isinstance(item.get("best_match"), dict) else None
        if best and best.get("title"):
            product_titles.append(_safe_text(best.get("title")))

    prompt = (
        "You are generating wardrobe personalization outputs.\n"
        "Use the per-photo insights below (already informed by time/location metadata) and closet/product context.\n"
        "Return practical, app-ready outputs.\n"
        f"job_id={job_id}\n"
        f"photo_insights={json.dumps(photo_insights[:40], ensure_ascii=True)}\n"
        f"brand_visible={json.dumps(brand_visible[:30], ensure_ascii=True)}\n"
        f"product_titles={json.dumps(product_titles[:60], ensure_ascii=True)}\n"
        "Rules:\n"
        "- collection_titles: short, specific, user-facing list names\n"
        "- notifications: concise suggestions or reminders tied to their wardrobe patterns\n"
        "- favorite_brands: inferred from evidence only; avoid made-up luxury bias\n"
        "- style_summary: 2-4 sentences, specific and non-generic\n"
    )

    payload = {
        "model": model,
        "reasoning": {"effort": "low"},
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "personalization_rollup",
                "schema": schema,
                "strict": True,
            }
        },
    }

    res = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=80,
    )
    res.raise_for_status()
    data = res.json()
    raw = _json_output_text(data)
    if not raw:
        raise RuntimeError("Missing aggregate output")
    parsed = json.loads(raw)
    return {
        "collection_titles": [str(x).strip() for x in (parsed.get("collection_titles") or []) if str(x).strip()],
        "notifications": [str(x).strip() for x in (parsed.get("notifications") or []) if str(x).strip()],
        "favorite_brands": [str(x).strip() for x in (parsed.get("favorite_brands") or []) if str(x).strip()],
        "style_summary": _safe_text(parsed.get("style_summary")),
    }


def _fallback_rollup(
    *,
    photo_insights: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    tags: list[str] = []
    brands: list[str] = []
    for insight in photo_insights:
        tags.extend([_safe_text(t) for t in (insight.get("style_tags") or [])])
        brands.extend([_safe_text(b) for b in (insight.get("brand_hints") or [])])

    for item in items:
        if item.get("brand_visible"):
            brands.append(_safe_text(item.get("brand_visible")))
        best = item.get("best_match") if isinstance(item.get("best_match"), dict) else None
        if best and best.get("source"):
            brands.append(_safe_text(best.get("source")))

    dedup_tags = [t for t in dict.fromkeys([t for t in tags if t])][:6]
    dedup_brands = [b for b in dict.fromkeys([b for b in brands if b])][:8]

    collection_titles = [
        "Everyday Capsule",
        "Weekend Rotation",
        "Photo-Ready Fits",
    ]
    if dedup_tags:
        collection_titles[0] = f"{dedup_tags[0].title()} Capsule"
        if len(dedup_tags) > 1:
            collection_titles[1] = f"{dedup_tags[1].title()} Rotation"

    notifications = [
        "New outfit matches are ready from your latest sync.",
        "Your closet has strong repeat pieces worth building into a capsule.",
        "Want tighter recommendations? Keep syncing full-body photos in natural light.",
    ]
    style_summary = (
        "Your closet leans toward repeatable, practical outfits with clear category patterns. "
        "Recent photos suggest a consistent personal uniform you can build into a tighter capsule."
    )
    if dedup_tags:
        style_summary = (
            f"Your style signals are strongest around {', '.join(dedup_tags[:3])}. "
            "You tend to repeat versatile pieces and can group them into intentional mini-collections."
        )

    return {
        "collection_titles": collection_titles,
        "notifications": notifications,
        "favorite_brands": dedup_brands,
        "style_summary": style_summary,
    }


def build_personalization_summary(job_id: str, *, force: bool = False) -> dict[str, Any] | None:
    job = db.get_job(job_id)
    if not job:
        return None

    photos = db.list_photos(job_id)
    items = db.list_clothing_items(job_id)
    debug = db.get_job_debug(job_id)
    current_key = _cache_key(photos, items)

    cached = debug.get("personalization_summary") if isinstance(debug, dict) else None
    if (
        not force
        and isinstance(cached, dict)
        and _safe_text(cached.get("cache_key")) == current_key
        and isinstance(cached.get("photo_insights"), list)
    ):
        out = dict(cached)
        out["source"] = "cached"
        return out

    settings = get_settings()
    media_dir = settings.media_dir

    by_photo: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_photo.setdefault(str(item.get("photo_id") or ""), []).append(item)

    photo_insights: list[dict[str, Any]] = []
    source: str = "fallback"

    if settings.openai_api_key:
        try:
            def one(photo: dict[str, Any]) -> dict[str, Any]:
                photo_id = str(photo.get("id") or "")
                hints: list[str] = []
                for item in by_photo.get(photo_id, []):
                    desc = _safe_text(item.get("description"))
                    if desc:
                        hints.append(desc)
                hints = list(dict.fromkeys(hints))[:8]
                rel = _safe_text(photo.get("relative_path"))
                return _summarize_photo_with_openai(
                    openai_api_key=settings.openai_api_key,
                    model=settings.openai_model,
                    photo=photo,
                    photo_path=media_dir / rel if rel else media_dir / "__missing__.jpg",
                    item_hints=hints,
                )

            max_workers = max(1, min(4, len(photos)))
            if max_workers == 1:
                photo_insights = [one(photo) for photo in photos]
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    photo_insights = list(pool.map(one, photos))
            source = "openai"
        except Exception:
            photo_insights = []
            source = "fallback"

    if not photo_insights:
        for photo in photos:
            photo_id = str(photo.get("id") or "")
            hints: list[str] = []
            for item in by_photo.get(photo_id, []):
                desc = _safe_text(item.get("description"))
                if desc:
                    hints.append(desc)
            summary = ", ".join(list(dict.fromkeys(hints))[:3]) if hints else "Outfit photo from camera roll sync."
            photo_insights.append(
                {
                    "photo_id": photo_id,
                    "summary": summary,
                    "style_tags": list(dict.fromkeys(hints))[:4],
                    "brand_hints": [],
                    "captured_at": photo.get("captured_at"),
                    "latitude": photo.get("latitude"),
                    "longitude": photo.get("longitude"),
                }
            )

    rollup: dict[str, Any] | None = None
    if source == "openai":
        try:
            rollup = _aggregate_with_openai(
                openai_api_key=settings.openai_api_key,
                model=settings.openai_model,
                job_id=job_id,
                photo_insights=photo_insights,
                items=items,
            )
        except Exception:
            source = "fallback"

    if rollup is None:
        rollup = _fallback_rollup(photo_insights=photo_insights, items=items)

    summary = {
        "job_id": job_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source,
        "photo_count": len(photos),
        "collection_titles": rollup.get("collection_titles") or [],
        "notifications": rollup.get("notifications") or [],
        "favorite_brands": rollup.get("favorite_brands") or [],
        "style_summary": _safe_text(rollup.get("style_summary")),
        "photo_insights": photo_insights,
        "cache_key": current_key,
    }

    db.patch_job_debug(
        job_id,
        patch={"personalization_summary": summary},
        event={
            "type": "personalization_summary_generated",
            "message": f"Generated personalization summary via {source}",
            "data": {"photo_count": len(photos)},
        },
    )

    return summary
