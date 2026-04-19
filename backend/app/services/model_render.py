from __future__ import annotations

import base64
import json
import re
import uuid
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageOps

from app.core.config import Settings
from app.services.image_utils import decode_to_rgb

STYLE_PRESETS: tuple[str, ...] = (
    "aesthetic",
    "editorial",
    "streetwear",
    "studio",
    "lookbook",
    "lifestyle",
    "runway",
)
ASPECT_RATIOS: tuple[str, ...] = ("portrait", "square", "landscape")
QUALITIES: tuple[str, ...] = ("low", "medium", "high", "auto")
INPUT_FIDELITIES: tuple[str, ...] = ("low", "high")
RENDER_ENGINES: tuple[str, ...] = ("openai", "nano_banana")

_ASPECT_TO_SIZE = {
    "portrait": "1024x1536",
    "square": "1024x1024",
    "landscape": "1536x1024",
}

_ASPECT_TO_GEMINI = {
    "portrait": "3:4",
    "square": "1:1",
    "landscape": "4:3",
}

_GEMINI_MODEL_BY_ENGINE = {
    "nano_banana": "gemini-2.5-flash-image",
}

_STYLE_DIRECTIONS = {
    "aesthetic": (
        "Premium fashion lookbook aesthetic similar to modern luxury resort campaigns: "
        "natural daylight, soft neutral palette, understated elegance, and realistic human texture. "
        "Prefer clean architectural or coastal settings with simple backgrounds and intentional styling."
    ),
    "editorial": (
        "Luxury fashion editorial image with cinematic lighting, strong composition, "
        "premium magazine quality, and intentional pose."
    ),
    "streetwear": (
        "Urban street-style photo with natural movement, textured city background, and "
        "authentic candid energy."
    ),
    "studio": (
        "High-end studio fashion photo with clean seamless backdrop, controlled soft lighting, "
        "and product-focused framing."
    ),
    "lookbook": (
        "Modern ecommerce lookbook photo with clear garment visibility, polished styling, "
        "and balanced contrast."
    ),
    "lifestyle": (
        "Natural lifestyle fashion image with warm daylight, believable environment, "
        "and relaxed model posture."
    ),
    "runway": (
        "Runway-inspired fashion frame with dramatic lighting, confident walk pose, and "
        "high-fashion attitude."
    ),
}

_SHOT_VARIATIONS = [
    "full-body lookbook frame with relaxed runway walk and clean garment visibility",
    "full-body standing pose, neutral expression, minimal background, garment drape clearly visible",
    "three-quarter pose with natural movement and realistic posture",
    "half-body editorial frame with subtle candid expression and true-to-life skin texture",
    "front-facing lookbook stance with clean silhouette and realistic body proportions",
    "resort-style editorial pose in natural sunlight with believable shadows and anatomy",
]


class ModelRenderError(RuntimeError):
    pass


class AutoStylingSkip(RuntimeError):
    pass


def auto_generate_for_photo(
    *,
    settings: Settings,
    job_id: str,
    photo_id: str,
    source_photo_relative_path: str,
    face_crop_relative_path: str | None,
    person_boxes: list[tuple[int, int, int, int]] | None = None,
) -> dict[str, Any]:
    source_photo_path = settings.media_dir / source_photo_relative_path.lstrip("/")
    if not source_photo_path.exists():
        raise ModelRenderError(f"Source photo not found: {source_photo_relative_path}")

    if not face_crop_relative_path:
        raise AutoStylingSkip("Missing target face crop")
    face_crop_path = settings.media_dir / face_crop_relative_path.lstrip("/")
    if not face_crop_path.exists():
        raise AutoStylingSkip("Target face crop does not exist")

    source_image = _load_rgb_image(source_photo_path)
    face_image = _load_rgb_image(face_crop_path)
    detected_person_boxes = person_boxes or []
    if not detected_person_boxes:
        raise AutoStylingSkip("No person detected in source photo")

    # Keep candidate count bounded for GPT selection.
    ranked_candidates = sorted(
        detected_person_boxes,
        key=lambda b: max(1, (b[2] - b[0]) * (b[3] - b[1])),
        reverse=True,
    )[:12]

    selection = _select_target_person_with_gpt(
        settings=settings,
        source_image=source_image,
        face_image=face_image,
        person_boxes=ranked_candidates,
    )
    if not selection:
        raise AutoStylingSkip("Failed to identify target person")

    try:
        selected_index = int(selection.get("selected_index"))
    except Exception as exc:
        raise AutoStylingSkip("Invalid person selection index from GPT") from exc

    if selected_index < 0 or selected_index >= len(ranked_candidates):
        raise AutoStylingSkip("Selected person index was out of range")

    body_visible = bool(selection.get("is_body_visible"))
    selection_reason = str(selection.get("reason") or "").strip()
    selected_box = ranked_candidates[selected_index]

    # Relaxed fallback: majority of body visible is enough even when head-to-toe
    # isn't fully present in frame.
    if not body_visible and _is_majority_body_visible(source_image, selected_box):
        body_visible = True
        if selection_reason:
            selection_reason = f"{selection_reason} (accepted by relaxed majority-body visibility rule)"
        else:
            selection_reason = "Accepted by relaxed majority-body visibility rule"

    if not body_visible:
        return {
            "status": "skipped",
            "skip_reason": selection_reason or "Target person's body visibility is too limited",
            "selected_person_crop_path": None,
            "selected_person_bbox": {
                "x1": int(selected_box[0]),
                "y1": int(selected_box[1]),
                "x2": int(selected_box[2]),
                "y2": int(selected_box[3]),
            },
            "gpt_selected_index": selected_index,
            "gpt_selection_reason": selection_reason,
            "body_visible": False,
            "prompt": "",
            "render_id": None,
            "best_variant_index": None,
            "best_reason": "",
            "variants": [],
        }

    person_crop = _crop_with_pad(source_image, selected_box, pad_ratio=0.08)
    person_rel_path = f"styling_person_crops/{job_id}/{photo_id}_{uuid.uuid4().hex[:8]}.jpg"
    person_abs_path = settings.media_dir / person_rel_path
    person_abs_path.parent.mkdir(parents=True, exist_ok=True)
    person_crop.save(person_abs_path, format="JPEG", quality=92)

    render_result = render_model_variants(
        settings=settings,
        source_relative_paths=[person_rel_path],
        style_preset="aesthetic",
        render_engine="nano_banana",
        subject_hint=None,
        custom_prompt=None,
        scene_hint=None,
        variant_count=3,
        aspect_ratio="portrait",
        quality="high",
        input_fidelity="high",
    )
    evaluation = evaluate_variants(
        settings=settings,
        render_id=str(render_result["render_id"]),
    )

    score_by_index = {
        int(row["variant_index"]): row
        for row in (evaluation.get("variants") or [])
        if isinstance(row, dict) and "variant_index" in row
    }
    best_variant_index = int(evaluation.get("best_variant_index") or 1)

    variants: list[dict[str, Any]] = []
    for idx, (prompt, output_path) in enumerate(
        zip(
            render_result.get("prompts_used") or [],
            render_result.get("output_relative_paths") or [],
            strict=False,
        ),
        start=1,
    ):
        score = score_by_index.get(idx) or {}
        variants.append(
            {
                "variant_index": idx,
                "prompt": str(prompt or ""),
                "output_path": str(output_path or ""),
                "realism": score.get("realism"),
                "aesthetic": score.get("aesthetic"),
                "overall": score.get("overall"),
                "justification": str(score.get("justification") or ""),
                "is_best": idx == best_variant_index,
            }
        )

    return {
        "status": "completed",
        "selected_person_crop_path": person_rel_path,
        "selected_person_bbox": {
            "x1": int(selected_box[0]),
            "y1": int(selected_box[1]),
            "x2": int(selected_box[2]),
            "y2": int(selected_box[3]),
        },
        "gpt_selected_index": selected_index,
        "gpt_selection_reason": selection_reason,
        "body_visible": body_visible,
        "prompt": str((render_result.get("prompts_used") or [""])[0]),
        "render_id": str(render_result["render_id"]),
        "best_variant_index": best_variant_index,
        "best_reason": str(evaluation.get("best_reason") or ""),
        "variants": variants,
    }


def save_uploaded_input_image(
    *,
    settings: Settings,
    payload: bytes,
    run_id: str,
    file_stem: str,
) -> str:
    if not payload:
        raise ModelRenderError("Uploaded image payload was empty")

    try:
        image = decode_to_rgb(payload)
        rel_path = f"styling_inputs/{run_id}/{file_stem}.jpg"
        out_path = settings.media_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="JPEG", quality=92)
        image.close()
        return rel_path
    except Exception as exc:
        raise ModelRenderError(f"Invalid image upload for {file_stem}: {exc}") from exc


def render_model_variants(
    *,
    settings: Settings,
    source_relative_paths: list[str],
    style_preset: str,
    render_engine: str,
    subject_hint: str | None,
    custom_prompt: str | None,
    scene_hint: str | None,
    variant_count: int,
    aspect_ratio: str,
    quality: str,
    input_fidelity: str,
) -> dict[str, Any]:
    if not source_relative_paths:
        raise ModelRenderError("At least one source image is required")

    style = _normalize_choice(style_preset, STYLE_PRESETS, "style_preset")
    engine = _normalize_choice(render_engine, RENDER_ENGINES, "render_engine")
    ratio = _normalize_choice(aspect_ratio, ASPECT_RATIOS, "aspect_ratio")
    chosen_quality = _normalize_choice(quality, QUALITIES, "quality")
    fidelity = _normalize_choice(input_fidelity, INPUT_FIDELITIES, "input_fidelity")
    count = max(1, min(int(variant_count), len(_SHOT_VARIATIONS)))

    source_paths: list[Path] = []
    for rel in source_relative_paths:
        safe = (rel or "").lstrip("/").replace("\\", "/")
        if not safe:
            continue
        abs_path = settings.media_dir / safe
        if not abs_path.exists():
            raise ModelRenderError(f"Source image not found: {safe}")
        source_paths.append(abs_path)
    if not source_paths:
        raise ModelRenderError("No valid source images were found")

    render_id = uuid.uuid4().hex
    output_rel_paths: list[str] = []
    prompts_used: list[str] = []

    for idx in range(count):
        shot_hint = _SHOT_VARIATIONS[idx % len(_SHOT_VARIATIONS)]
        prompt = _build_prompt(
            style_preset=style,
            subject_hint=subject_hint,
            custom_prompt=custom_prompt,
            scene_hint=scene_hint,
            shot_hint=shot_hint,
        )
        prompts_used.append(prompt)

        images: list[str]
        if engine == "nano_banana":
            gemini_model = (settings.gemini_image_model or _GEMINI_MODEL_BY_ENGINE["nano_banana"]).strip()
            if not gemini_model:
                gemini_model = _GEMINI_MODEL_BY_ENGINE["nano_banana"]
            data = _call_gemini_image_edit(
                api_key=settings.gemini_api_key,
                model=gemini_model,
                source_paths=source_paths,
                prompt=prompt,
                aspect_ratio=_ASPECT_TO_GEMINI[ratio],
                quality=chosen_quality,
            )
            images = _extract_gemini_generated_images(data)
        else:
            if not settings.openai_api_key:
                raise ModelRenderError("OPENAI_API_KEY is missing")
            data = _call_openai_image_edit(
                api_key=settings.openai_api_key,
                model=(settings.openai_image_tool_model or settings.openai_model),
                source_paths=source_paths,
                prompt=prompt,
                size=_ASPECT_TO_SIZE[ratio],
                quality=chosen_quality,
                input_fidelity=fidelity,
            )
            images = _extract_openai_generated_images(data)

        if not images:
            raise ModelRenderError(
                "The model returned no images for this request. Try another prompt or quality preset."
            )

        raw = _decode_image_b64(images[0])
        out_rel = f"model_renders/{render_id}/{idx + 1:02d}.jpg"
        out_path = settings.media_dir / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(raw)
        output_rel_paths.append(out_rel)

    return {
        "render_id": render_id,
        "style_preset": style,
        "render_engine": engine,
        "prompts_used": prompts_used,
        "output_relative_paths": output_rel_paths,
    }


def _normalize_choice(value: str, allowed: tuple[str, ...], label: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized in allowed:
        return normalized
    choices = ", ".join(allowed)
    raise ModelRenderError(f"Invalid {label}='{value}'. Allowed values: {choices}")


def _build_prompt(
    *,
    style_preset: str,
    subject_hint: str | None,
    custom_prompt: str | None,
    scene_hint: str | None,
    shot_hint: str,
) -> str:
    lines = [
        "put this guy in an aesthetic background. i need an image for a fashion shopping website, "
        "he should be modeling the outfit. needs to be very very aesthetic and realistic. "
        "the background should be realistic, a plain color will suffice. "
        "he should be wearing white air forces.",
    ]
    cleaned_custom = (custom_prompt or "").strip()
    if cleaned_custom:
        lines.append(cleaned_custom)
    return "\n".join(lines)


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _crop_with_pad(
    image: Image.Image,
    bbox_xyxy: tuple[int, int, int, int],
    *,
    pad_ratio: float,
) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox_xyxy
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(round(box_w * pad_ratio))
    pad_y = int(round(box_h * pad_ratio))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(width, x2 + pad_x)
    cy2 = min(height, y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return image.copy()
    return image.crop((cx1, cy1, cx2, cy2))


def _is_majority_body_visible(
    image: Image.Image,
    bbox_xyxy: tuple[int, int, int, int],
) -> bool:
    width, height = image.size
    if width <= 0 or height <= 0:
        return False
    x1, y1, x2, y2 = bbox_xyxy
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    height_ratio = box_h / float(height)
    area_ratio = (box_w * box_h) / float(width * height)
    # Majority-visible threshold: substantial person region, not necessarily full head-to-toe.
    return height_ratio >= 0.55 and area_ratio >= 0.12


def _select_target_person_with_gpt(
    *,
    settings: Settings,
    source_image: Image.Image,
    face_image: Image.Image,
    person_boxes: list[tuple[int, int, int, int]],
) -> dict[str, Any] | None:
    if not settings.openai_api_key:
        raise ModelRenderError("OPENAI_API_KEY is required for target person selection")
    if not person_boxes:
        return None

    schema = {
        "type": "object",
        "properties": {
            "selected_index": {"type": "integer"},
            "is_body_visible": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["selected_index", "is_body_visible", "reason"],
        "additionalProperties": False,
    }
    bbox_lines = [f"{i}: [{b[0]}, {b[1]}, {b[2]}, {b[3]}]" for i, b in enumerate(person_boxes)]

    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "You are selecting the target person from person detections in a fashion photo.\n"
                "Pick the person that best matches the target face image.\n"
                "Then decide if that selected person's MAJORITY OF BODY is visible enough for realistic model generation.\n"
                "Head-to-toe visibility is NOT required. It's okay if feet or some lower leg is cropped.\n"
                "Set is_body_visible=true when most of the torso/upper body and a substantial portion of lower body are visible.\n"
                "Return strict JSON with selected_index, is_body_visible, and reason.\n"
                "Person boxes:\n"
                + "\n".join(bbox_lines)
            ),
        },
        {"type": "input_text", "text": "Target face crop:"},
        {"type": "input_image", "image_url": _image_to_data_url(face_image), "detail": "high"},
        {"type": "input_text", "text": "Full source image:"},
        {"type": "input_image", "image_url": _image_to_data_url(source_image), "detail": "high"},
    ]
    for idx, bbox in enumerate(person_boxes):
        content.append({"type": "input_text", "text": f"Candidate person crop {idx}:"})
        content.append(
            {
                "type": "input_image",
                "image_url": _image_to_data_url(_crop_with_pad(source_image, bbox, pad_ratio=0.0)),
                "detail": "high",
            }
        )

    payload = {
        "model": settings.openai_model or "gpt-5.4",
        "reasoning": {"effort": "low"},
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "target_person_selection",
                "schema": schema,
                "strict": True,
            }
        },
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    if response.status_code >= 300:
        return None
    try:
        data = response.json()
    except Exception:
        return None
    raw = _extract_openai_text(data)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _image_to_data_url(image: Image.Image, *, max_side: int = 1600) -> str:
    work = image.copy()
    width, height = work.size
    if max(width, height) > max_side:
        scale = max_side / float(max(width, height))
        work = work.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    encoded = base64.b64encode(_image_to_jpeg_bytes(work, quality=90)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _image_to_jpeg_bytes(image: Image.Image, *, quality: int) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _call_openai_image_edit(
    *,
    api_key: str,
    model: str,
    source_paths: list[Path],
    prompt: str,
    size: str,
    quality: str,
    input_fidelity: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for source in source_paths:
        content.append(
            {
                "type": "input_image",
                "image_url": _data_url_for_image(source),
                "detail": "high",
            }
        )

    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "tools": [
            {
                "type": "image_generation",
                "action": "edit",
                "size": size,
                "quality": quality,
                "output_format": "jpeg",
                "input_fidelity": input_fidelity,
            }
        ],
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=240,
        )
    except Exception as exc:
        raise ModelRenderError(f"OpenAI request failed: {exc}") from exc

    if response.status_code >= 300:
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                err = body.get("error") or {}
                if isinstance(err, dict):
                    detail = str(err.get("message") or "")
        except Exception:
            detail = ""
        suffix = f" ({detail})" if detail else ""
        raise ModelRenderError(f"OpenAI image generation failed with HTTP {response.status_code}{suffix}")

    try:
        return response.json()
    except Exception as exc:
        raise ModelRenderError(f"OpenAI returned invalid JSON: {exc}") from exc


def _extract_openai_generated_images(response: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in response.get("output", []) or []:
        if item.get("type") != "image_generation_call":
            continue
        result = item.get("result")
        if isinstance(result, str) and result.strip():
            out.append(result.strip())
    return out


def _call_gemini_image_edit(
    *,
    api_key: str,
    model: str,
    source_paths: list[Path],
    prompt: str,
    aspect_ratio: str,
    quality: str,
) -> dict[str, Any]:
    if not api_key:
        raise ModelRenderError("GEMINI_API_KEY is missing")
    if not model:
        raise ModelRenderError("Gemini model id is missing")

    parts: list[dict[str, Any]] = [{"text": prompt}]
    for source in source_paths:
        encoded, mime = _image_base64_and_mime(source)
        parts.append({"inline_data": {"mime_type": mime, "data": encoded}})

    image_size = _gemini_image_size_for_quality(quality)
    image_cfg: dict[str, Any] = {"aspectRatio": aspect_ratio}
    if image_size:
        image_cfg["imageSize"] = image_size

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": image_cfg,
        },
    }

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=240,
        )
    except Exception as exc:
        raise ModelRenderError(f"Gemini request failed: {exc}") from exc

    if response.status_code >= 300:
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                err = body.get("error") or {}
                if isinstance(err, dict):
                    detail = str(err.get("message") or "")
        except Exception:
            detail = ""
        suffix = f" ({detail})" if detail else ""
        raise ModelRenderError(f"Gemini image generation failed with HTTP {response.status_code}{suffix}")

    try:
        return response.json()
    except Exception as exc:
        raise ModelRenderError(f"Gemini returned invalid JSON: {exc}") from exc


def _extract_gemini_generated_images(response: dict[str, Any]) -> list[str]:
    out: list[str] = []
    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        return out
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        content = cand.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline_data = part.get("inlineData") or part.get("inline_data")
            if not isinstance(inline_data, dict):
                continue
            data = inline_data.get("data")
            if isinstance(data, str) and data.strip():
                out.append(data.strip())
    return out


def _gemini_image_size_for_quality(quality: str) -> str | None:
    normalized = (quality or "").strip().lower()
    if normalized == "low":
        return "1K"
    if normalized in {"medium", "high"}:
        return "2K"
    return None


def _decode_image_b64(value: str) -> bytes:
    raw = value.strip()
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw)
    except Exception as exc:
        raise ModelRenderError(f"Invalid generated image payload: {exc}") from exc


def _data_url_for_image(path: Path) -> str:
    encoded, mime = _image_base64_and_mime(path)
    return f"data:{mime};base64,{encoded}"


def _image_base64_and_mime(path: Path) -> tuple[str, str]:
    if not path.exists():
        raise ModelRenderError(f"Missing source image file: {path}")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return encoded, mime


class VariantEvaluationError(RuntimeError):
    pass


def evaluate_variants(
    *,
    settings: Settings,
    render_id: str,
) -> dict[str, Any]:
    """Score each generated variant for the given render_id on realism and aesthetic,
    and pick the best overall."""

    render_id = (render_id or "").strip()
    if not render_id:
        raise VariantEvaluationError("render_id is required")

    render_dir = settings.media_dir / "model_renders" / render_id
    if not render_dir.exists() or not render_dir.is_dir():
        raise VariantEvaluationError(f"No renders found for render_id={render_id}")

    variant_paths = sorted(
        p for p in render_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not variant_paths:
        raise VariantEvaluationError(f"No variant images in {render_dir}")

    if not settings.openai_api_key:
        raise VariantEvaluationError("OPENAI_API_KEY is missing")

    rubric = (
        "You are a senior fashion e-commerce art director. You will receive several "
        "candidate fashion model shots that are meant to be used on a clothing shopping "
        "website. Score each image independently on two dimensions:\n"
        "  - realism (0-10): how believable the person, skin, hands, anatomy, garment "
        "physics, lighting, and background are. Penalize CGI/plastic look, warped hands, "
        "extra fingers, fake skin, or composited backgrounds.\n"
        "  - aesthetic (0-10): how appealing, clean, and on-brand the image is for a "
        "modern luxury/clean fashion shopping site (composition, color, styling, vibe).\n"
        "Use the full 0-10 range. Give a short 1-sentence justification for each.\n"
        "Then choose ONE best variant overall (balance realism and aesthetic, but "
        "never pick an image with broken anatomy).\n\n"
        "Respond ONLY with a valid JSON object, no markdown, matching exactly this shape:\n"
        "{\n"
        '  "variants": [\n'
        '    {"variant_index": <int>, "realism": <number>, "aesthetic": <number>, '
        '"overall": <number>, "justification": <string>}\n'
        "  ],\n"
        '  "best_variant_index": <int>,\n'
        '  "best_reason": <string>\n'
        "}\n"
        "`overall` should reflect your combined judgment for fashion e-commerce usage."
    )

    content: list[dict[str, Any]] = [{"type": "input_text", "text": rubric}]
    indexed_paths: list[tuple[int, Path]] = []
    for path in variant_paths:
        match = re.match(r"^(\d+)", path.stem)
        if match:
            idx = int(match.group(1))
        else:
            idx = len(indexed_paths) + 1
        indexed_paths.append((idx, path))

    indexed_paths.sort(key=lambda x: x[0])

    for idx, path in indexed_paths:
        content.append({"type": "input_text", "text": f"Variant {idx}:"})
        content.append(
            {
                "type": "input_image",
                "image_url": _data_url_for_image(path),
                "detail": "high",
            }
        )

    payload = {
        "model": settings.openai_model or "gpt-5.4",
        "input": [{"role": "user", "content": content}],
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
    except Exception as exc:
        raise VariantEvaluationError(f"OpenAI request failed: {exc}") from exc

    if response.status_code >= 300:
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                err = body.get("error") or {}
                if isinstance(err, dict):
                    detail = str(err.get("message") or "")
        except Exception:
            detail = ""
        suffix = f" ({detail})" if detail else ""
        raise VariantEvaluationError(
            f"Evaluation call failed with HTTP {response.status_code}{suffix}"
        )

    try:
        data = response.json()
    except Exception as exc:
        raise VariantEvaluationError(f"OpenAI returned invalid JSON: {exc}") from exc

    text = _extract_openai_text(data)
    parsed = _parse_evaluation_json(text)

    raw_scores = parsed.get("variants") if isinstance(parsed, dict) else None
    if not isinstance(raw_scores, list) or not raw_scores:
        raise VariantEvaluationError("Evaluator returned no variant scores")

    valid_indices = {idx for idx, _ in indexed_paths}
    normalized: list[dict[str, Any]] = []
    for row in raw_scores:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("variant_index"))
        except (TypeError, ValueError):
            continue
        if idx not in valid_indices:
            continue
        normalized.append(
            {
                "variant_index": idx,
                "realism": _clamp_score(row.get("realism")),
                "aesthetic": _clamp_score(row.get("aesthetic")),
                "overall": _clamp_score(row.get("overall")),
                "justification": str(row.get("justification") or "").strip(),
            }
        )

    if not normalized:
        raise VariantEvaluationError("Evaluator scores did not match any known variants")

    normalized.sort(key=lambda r: r["variant_index"])

    best_idx: int | None = None
    raw_best = parsed.get("best_variant_index") if isinstance(parsed, dict) else None
    try:
        candidate = int(raw_best) if raw_best is not None else None
    except (TypeError, ValueError):
        candidate = None
    if candidate is not None and candidate in valid_indices:
        best_idx = candidate

    if best_idx is None:
        best_row = max(
            normalized,
            key=lambda r: (r["overall"], (r["realism"] + r["aesthetic"]) / 2),
        )
        best_idx = best_row["variant_index"]

    best_reason = ""
    if isinstance(parsed, dict):
        best_reason = str(parsed.get("best_reason") or "").strip()

    return {
        "render_id": render_id,
        "variants": normalized,
        "best_variant_index": best_idx,
        "best_reason": best_reason,
    }


def _extract_openai_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    chunks: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in {"output_text", "text"}:
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
    return "\n".join(chunks).strip()


def _parse_evaluation_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise VariantEvaluationError("Evaluator returned empty response")

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise VariantEvaluationError(f"Evaluator JSON was malformed: {exc}") from exc

    raise VariantEvaluationError("Evaluator did not return JSON")


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0:
        return 0.0
    if score > 10:
        return 10.0
    return round(score, 2)
