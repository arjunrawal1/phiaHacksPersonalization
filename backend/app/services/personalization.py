from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from app.core.config import get_settings
from app.services import db

_PERSONALIZATION_CACHE_VERSION = "v4_new_options_notifications"
_GOOGLE_NEARBY_INCLUDED_TYPES = [
    "clothing_store",
    "department_store",
    "shopping_mall",
    "store",
    "park",
    "tourist_attraction",
    "restaurant",
    "cafe",
]


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


def _cache_key(
    photos: list[dict[str, Any]],
    items: list[dict[str, Any]],
    *,
    places_enabled: bool,
) -> str:
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

    payload = (
        f"{_PERSONALIZATION_CACHE_VERSION}|places_enabled={1 if places_enabled else 0}|"
        + "||".join(photo_parts)
        + "##"
        + "||".join(item_parts)
    )
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


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _lookup_nearby_place(
    *,
    google_places_api_key: str,
    latitude: float,
    longitude: float,
    radius_m: int,
) -> dict[str, Any] | None:
    payload = {
        "includedTypes": _GOOGLE_NEARBY_INCLUDED_TYPES,
        "maxResultCount": 5,
        "rankPreference": "DISTANCE",
        "locationRestriction": {
            "circle": {
                "center": {"latitude": float(latitude), "longitude": float(longitude)},
                "radius": float(max(30, min(radius_m, 500))),
            }
        },
    }
    res = requests.post(
        "https://places.googleapis.com/v1/places:searchNearby",
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": google_places_api_key,
            "X-Goog-FieldMask": (
                "places.displayName,places.primaryType,places.formattedAddress,places.location"
            ),
        },
        json=payload,
        timeout=15,
    )
    res.raise_for_status()
    data = res.json()
    places = data.get("places")
    if not isinstance(places, list) or not places:
        return None

    best: dict[str, Any] | None = None
    best_dist = float("inf")
    for row in places:
        if not isinstance(row, dict):
            continue
        p_loc = row.get("location") if isinstance(row.get("location"), dict) else {}
        p_lat = _to_float(p_loc.get("latitude"))
        p_lon = _to_float(p_loc.get("longitude"))
        dist = (
            _haversine_m(latitude, longitude, p_lat, p_lon)
            if p_lat is not None and p_lon is not None
            else 9_999_999.0
        )
        if dist < best_dist:
            best = row
            best_dist = dist

    if not isinstance(best, dict):
        return None

    display = best.get("displayName") if isinstance(best.get("displayName"), dict) else {}
    place_name = _safe_text(display.get("text"))
    if not place_name:
        return None

    return {
        "nearby_place_name": place_name,
        "nearby_place_type": _safe_text(best.get("primaryType")) or None,
        "nearby_place_address": _safe_text(best.get("formattedAddress")) or None,
        "nearby_place_distance_m": None if best_dist >= 9_000_000 else round(best_dist, 1),
    }


def _enrich_photo_insights_with_places(
    *,
    photo_insights: list[dict[str, Any]],
    google_places_api_key: str,
    radius_m: int,
) -> None:
    # Reuse the same lookup for nearby photos to reduce API calls/cost.
    lookup_cache: dict[str, dict[str, Any] | None] = {}
    for insight in photo_insights:
        lat = _to_float(insight.get("latitude"))
        lon = _to_float(insight.get("longitude"))
        if lat is None or lon is None:
            insight["nearby_place_name"] = None
            insight["nearby_place_type"] = None
            insight["nearby_place_address"] = None
            insight["nearby_place_distance_m"] = None
            continue

        key = f"{lat:.3f},{lon:.3f}"
        place = lookup_cache.get(key)
        if key not in lookup_cache:
            try:
                place = _lookup_nearby_place(
                    google_places_api_key=google_places_api_key,
                    latitude=lat,
                    longitude=lon,
                    radius_m=radius_m,
                )
            except Exception:
                place = None
            lookup_cache[key] = place

        if place:
            insight.update(place)
        else:
            insight["nearby_place_name"] = None
            insight["nearby_place_type"] = None
            insight["nearby_place_address"] = None
            insight["nearby_place_distance_m"] = None


def _resolved_place_signals(photo_insights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for insight in photo_insights:
        name = _safe_text(insight.get("nearby_place_name"))
        if not name:
            continue
        key = name.lower()
        row = by_name.get(key)
        if row is None:
            row = {
                "name": name,
                "type": _safe_text(insight.get("nearby_place_type")).replace("_", " ") or None,
                "count": 0,
                "min_distance_m": None,
            }
            by_name[key] = row
        row["count"] = int(row["count"]) + 1
        dist = _to_float(insight.get("nearby_place_distance_m"))
        if dist is not None and (row["min_distance_m"] is None or dist < float(row["min_distance_m"])):
            row["min_distance_m"] = round(dist, 1)

    places = list(by_name.values())
    places.sort(key=lambda x: (-int(x.get("count") or 0), float(x.get("min_distance_m") or 9_999_999)))
    return places


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
            "nearby_place_name": None,
            "nearby_place_type": None,
            "nearby_place_address": None,
            "nearby_place_distance_m": None,
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
        "nearby_place_name": None,
        "nearby_place_type": None,
        "nearby_place_address": None,
        "nearby_place_distance_m": None,
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
    resolved_places = _resolved_place_signals(photo_insights)

    prompt = (
        "You are generating wardrobe personalization outputs.\n"
        "Use the per-photo insights below (already informed by time/location metadata) and closet/product context.\n"
        "Resolved nearby places are high-priority context for vibe and activity inference.\n"
        "Return practical, app-ready outputs.\n"
        f"job_id={job_id}\n"
        f"photo_insights={json.dumps(photo_insights[:40], ensure_ascii=True)}\n"
        f"resolved_places={json.dumps(resolved_places[:20], ensure_ascii=True)}\n"
        f"brand_visible={json.dumps(brand_visible[:30], ensure_ascii=True)}\n"
        f"product_titles={json.dumps(product_titles[:60], ensure_ascii=True)}\n"
        "Rules:\n"
        "- collection_titles: 2-4 words, short/specific/user-facing, place/activity aware when evidence exists\n"
        "- notifications: concise and action-oriented discovery prompts tied to places/activities and wardrobe patterns\n"
        "- notifications must highlight fresh options to browse (e.g., 'check these out', 'explore these picks', 'new styles to try')\n"
        "- notifications must NOT tell the user to re-wear, pack, or repeat current clothes\n"
        "- frame notifications as recommendation/product discovery, not closet reminders\n"
        "- favorite_brands: inferred from evidence only; avoid made-up luxury bias\n"
        "- style_summary: 2-4 sentences, specific and non-generic\n"
        "- do not output generic filler like 'New outfit matches are ready'\n"
        "Style examples (format only):\n"
        "- collection_titles: 'Racing Outfits', 'Duke Gardens Vibes'\n"
        "- notifications: 'Heading to another car show? Check these fresh picks out', "
        "'New options for your next conference look are ready to explore', "
        "'Going go-karting again soon? Check out these new styles'\n"
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
        "Fresh outfit options are ready. Check these picks out.",
        "New styles matched to your recent activities are ready to explore.",
        "Check out these latest recommendations from your recent sync.",
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

    settings = get_settings()
    photos = db.list_photos(job_id)
    items = db.list_clothing_items(job_id)
    debug = db.get_job_debug(job_id)
    current_key = _cache_key(
        photos,
        items,
        places_enabled=bool(settings.google_places_api_key.strip()),
    )

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
                    "nearby_place_name": None,
                    "nearby_place_type": None,
                    "nearby_place_address": None,
                    "nearby_place_distance_m": None,
                }
            )

    if settings.google_places_api_key.strip():
        _enrich_photo_insights_with_places(
            photo_insights=photo_insights,
            google_places_api_key=settings.google_places_api_key,
            radius_m=settings.google_places_nearby_radius_m,
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
