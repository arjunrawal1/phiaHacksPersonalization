#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import os
import ssl
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps
from pillow_heif import register_heif_opener
from ultralytics import YOLOWorld

register_heif_opener()

IMAGE_PATHS = [
    "/Users/arjun/Downloads/IMG_0385.HEIC",
    "/Users/arjun/Downloads/IMG_0751.HEIC",
    "/Users/arjun/Downloads/IMG_8444.HEIC",
    "/Users/arjun/Downloads/IMG_9011.heic",
    "/Users/arjun/Downloads/103_0242.JPG",
    "/Users/arjun/Downloads/103_0344.JPEG",
    "/Users/arjun/Downloads/IMG_7144.JPG",
    "/Users/arjun/Downloads/IMG_7501.JPG",
    "/Users/arjun/Downloads/IMG_5496.JPG",
    "/Users/arjun/Downloads/DSC_1427.JPEG",
    "/Users/arjun/Downloads/IMG_6464.PNG",
    "/Users/arjun/Downloads/IMG_3293.DNG",
]

ENV_PATH = Path("/Users/arjun/phiaHacksPersonalization/backend/.env")
OUT_BASE = Path("/Users/arjun/phiaHacksPersonalization/backend/data/person_crops")
REK_CONFIDENCE_THRESHOLD = 88.0
REK_SIMILARITY_THRESHOLD = 80.0


@dataclass
class FaceDet:
    path: Path
    ext_id: str
    face_id: str
    confidence: float
    bbox_norm: dict[str, float]


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env[key] = value
    return env


def load_image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def to_jpeg_bytes(img: Image.Image, quality: int = 94) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def to_rekognition_jpeg_bytes(img: Image.Image, max_bytes: int = 5 * 1024 * 1024) -> bytes:
    """
    Rekognition Image.Bytes max is 5 MB.
    Compress (quality + optional downscale) until payload fits.
    """
    work = img.copy()
    for _ in range(6):
        for q in (92, 88, 84, 80, 76, 72, 68, 64):
            payload = to_jpeg_bytes(work, quality=q)
            if len(payload) <= max_bytes:
                return payload
        # Still too large: downscale and retry.
        w, h = work.size
        work = work.resize((max(512, int(w * 0.85)), max(512, int(h * 0.85))), Image.Resampling.LANCZOS)
    return to_jpeg_bytes(work, quality=60)


def norm_bbox_to_px(b: dict[str, float], w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, int(round(float(b.get("left", 0.0)) * w))))
    y1 = max(0, min(h - 1, int(round(float(b.get("top", 0.0)) * h))))
    x2 = max(0, min(w, int(round((float(b.get("left", 0.0)) + float(b.get("width", 0.0))) * w))))
    y2 = max(0, min(h, int(round((float(b.get("top", 0.0)) + float(b.get("height", 0.0))) * h))))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def fallback_choose_person(face_px: tuple[int, int, int, int], person_boxes: list[tuple[int, int, int, int]]) -> int:
    fx, fy = bbox_center(face_px)
    containing = []
    for i, (x1, y1, x2, y2) in enumerate(person_boxes):
        if x1 <= fx <= x2 and y1 <= fy <= y2:
            area = max(1, (x2 - x1) * (y2 - y1))
            containing.append((area, i))
    if containing:
        containing.sort()
        return containing[0][1]

    dists = []
    for i, b in enumerate(person_boxes):
        cx, cy = bbox_center(b)
        dists.append((((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5, i))
    dists.sort()
    return dists[0][1]


def crop_with_pad(img: Image.Image, box: tuple[int, int, int, int], pad_ratio: float) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    px = int(round(bw * pad_ratio))
    py = int(round(bh * pad_ratio))
    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(w, x2 + px)
    ny2 = min(h, y2 + py)
    return img.crop((nx1, ny1, nx2, ny2))


def to_data_url(img: Image.Image, max_side: int = 1600) -> str:
    img2 = img.copy()
    w, h = img2.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img2 = img2.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    payload = to_jpeg_bytes(img2, quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("utf-8")


def gpt_choose_box(
    openai_key: str,
    model: str,
    full_img: Image.Image,
    face_img: Image.Image,
    person_crops: list[Image.Image],
    person_boxes: list[tuple[int, int, int, int]],
) -> int | None:
    if not openai_key or not person_crops:
        return None

    schema = {
        "type": "object",
        "properties": {
            "selected_index": {"type": "integer"},
            "reason": {"type": "string"},
        },
        "required": ["selected_index", "reason"],
        "additionalProperties": False,
    }

    lines = []
    for i, b in enumerate(person_boxes):
        x1, y1, x2, y2 = b
        lines.append(f"{i}: [{x1}, {y1}, {x2}, {y2}]")

    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "You are selecting which YOLO person box belongs to the SAME person as the reference face. "
                "Return only one selected_index from the candidate list.\n"
                "Candidates with bbox coordinates:\n" + "\n".join(lines)
            ),
        },
        {"type": "input_text", "text": "Reference face:"},
        {"type": "input_image", "image_url": to_data_url(face_img), "detail": "high"},
        {"type": "input_text", "text": "Full image context:"},
        {"type": "input_image", "image_url": to_data_url(full_img), "detail": "high"},
    ]

    for i, crop in enumerate(person_crops):
        content.append({"type": "input_text", "text": f"Candidate person crop index {i}"})
        content.append({"type": "input_image", "image_url": to_data_url(crop), "detail": "high"})

    payload = {
        "model": model,
        "reasoning": {"effort": "low"},
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "person_box_choice",
                "schema": schema,
                "strict": True,
            }
        },
    }

    try:
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("output_text")
        if not raw:
            for out in data.get("output", []):
                for c in out.get("content", []):
                    if c.get("type") == "output_text" and c.get("text"):
                        raw = c["text"]
                        break
                if raw:
                    break
        if not raw:
            return None
        parsed = json.loads(raw)
        idx = int(parsed.get("selected_index"))
        if 0 <= idx < len(person_boxes):
            return idx
        return None
    except Exception:
        return None


def main() -> None:
    env = load_env_file(ENV_PATH)

    aws_region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or "us-east-1"
    aws_access_key_id = env.get("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key = env.get("AWS_SECRET_ACCESS_KEY", "")

    openai_key = env.get("OPENAI_API_KEY", "")
    openai_model = env.get("OPENAI_MODEL") or "gpt-4.1-mini"

    if not aws_access_key_id or not aws_secret_access_key:
        raise RuntimeError("AWS credentials missing in backend/.env")

    out_dir = OUT_BASE / datetime.now().strftime("%Y%m%d_%H%M%S")
    crops_dir = out_dir / "crops"
    debug_dir = out_dir / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Rekognition clustering
    rek = boto3.client(
        "rekognition",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    collection_id = f"person-crops-{uuid.uuid4().hex[:8]}"
    rek.create_collection(CollectionId=collection_id)

    all_faces: list[FaceDet] = []
    image_cache: dict[str, Image.Image] = {}

    try:
        for i, p in enumerate(IMAGE_PATHS):
            path = Path(p)
            if not path.exists():
                continue
            img = load_image_rgb(path)
            image_cache[str(path)] = img
            jpeg = to_rekognition_jpeg_bytes(img)
            ext_id = f"img_{i}"

            try:
                result = rek.index_faces(
                    CollectionId=collection_id,
                    Image={"Bytes": jpeg},
                    ExternalImageId=ext_id,
                    DetectionAttributes=["DEFAULT"],
                    MaxFaces=15,
                    QualityFilter="AUTO",
                )
            except Exception as e:
                print(f"[warn] index_faces failed for {path.name}: {e}")
                continue

            for rec in result.get("FaceRecords", []) or []:
                face = rec.get("Face") or {}
                fid = face.get("FaceId")
                conf = float(face.get("Confidence") or 0.0)
                bbox = face.get("BoundingBox") or {}
                if not fid or conf < REK_CONFIDENCE_THRESHOLD:
                    continue
                all_faces.append(
                    FaceDet(
                        path=path,
                        ext_id=ext_id,
                        face_id=fid,
                        confidence=conf,
                        bbox_norm={
                            "left": float(bbox.get("Left") or 0.0),
                            "top": float(bbox.get("Top") or 0.0),
                            "width": float(bbox.get("Width") or 0.0),
                            "height": float(bbox.get("Height") or 0.0),
                        },
                    )
                )

        if not all_faces:
            raise RuntimeError("Rekognition found no usable faces.")

        parent: dict[str, str] = {f.face_id: f.face_id for f in all_faces}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        own_ids = set(parent.keys())
        for f in all_faces:
            try:
                matches = rek.search_faces(
                    CollectionId=collection_id,
                    FaceId=f.face_id,
                    FaceMatchThreshold=REK_SIMILARITY_THRESHOLD,
                    MaxFaces=50,
                )
            except Exception:
                continue
            for m in matches.get("FaceMatches", []) or []:
                m_face = m.get("Face") or {}
                m_id = m_face.get("FaceId")
                if m_id and m_id in own_ids and m_id != f.face_id:
                    union(f.face_id, m_id)

        grouped: dict[str, list[FaceDet]] = {}
        for f in all_faces:
            grouped.setdefault(find(f.face_id), []).append(f)

        def cluster_key(members: list[FaceDet]) -> tuple[int, int, float]:
            unique_images = len({str(m.path) for m in members})
            return (unique_images, len(members), sum(m.confidence for m in members) / max(1, len(members)))

        clusters = sorted(grouped.values(), key=cluster_key, reverse=True)
        target_cluster = clusters[0]

        faces_by_image: dict[str, FaceDet] = {}
        for f in target_cluster:
            k = str(f.path)
            prev = faces_by_image.get(k)
            if prev is None or f.confidence > prev.confidence:
                faces_by_image[k] = f

        print("[info] target cluster images:", len(faces_by_image), "of", len(IMAGE_PATHS))

    finally:
        try:
            rek.delete_collection(CollectionId=collection_id)
        except Exception:
            pass

    # YOLO-World + GPT selection
    ssl._create_default_https_context = ssl._create_unverified_context
    model = YOLOWorld("yolov8s-world.pt")
    model.set_classes(["person"])

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "output_dir": str(out_dir),
        "target_cluster_image_count": len(faces_by_image),
        "images": [],
    }

    for path_str in IMAGE_PATHS:
        path = Path(path_str)
        rec: dict[str, Any] = {
            "source_path": path_str,
            "status": "skipped",
        }
        if not path.exists():
            rec["reason"] = "missing_file"
            manifest["images"].append(rec)
            continue

        img = image_cache.get(str(path)) or load_image_rgb(path)
        w, h = img.size

        face = faces_by_image.get(str(path))
        if face is None:
            rec["reason"] = "target_face_not_found_in_image"
            manifest["images"].append(rec)
            continue

        face_px = norm_bbox_to_px(face.bbox_norm, w, h)

        yolo_res = model.predict(source=np.array(img), conf=0.05, verbose=False)[0]
        person_boxes: list[tuple[int, int, int, int]] = []
        person_scores: list[float] = []
        for b in yolo_res.boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in b.xyxy.tolist()[0]]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(1, min(w, x2))
            y2 = max(1, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            person_boxes.append((x1, y1, x2, y2))
            person_scores.append(float(b.conf))

        rec["face_bbox_px"] = list(face_px)
        rec["person_boxes_px"] = [list(b) for b in person_boxes]
        rec["person_scores"] = person_scores

        if not person_boxes:
            rec["reason"] = "no_person_boxes"
            manifest["images"].append(rec)
            continue

        person_crops = [crop_with_pad(img, b, pad_ratio=0.0) for b in person_boxes]
        face_img = crop_with_pad(img, face_px, pad_ratio=0.35)

        selected_idx: int | None = None
        selection_source = "fallback"
        if len(person_boxes) > 1:
            selected_idx = gpt_choose_box(
                openai_key=openai_key,
                model=openai_model,
                full_img=img,
                face_img=face_img,
                person_crops=person_crops,
                person_boxes=person_boxes,
            )
            if selected_idx is not None:
                selection_source = "gpt"

        if selected_idx is None:
            selected_idx = fallback_choose_person(face_px, person_boxes)
            selection_source = "fallback"

        chosen_box = person_boxes[selected_idx]
        person_crop = crop_with_pad(img, chosen_box, pad_ratio=0.08)

        crop_name = f"{path.stem}_person_crop.jpg"
        crop_path = crops_dir / crop_name
        person_crop.save(crop_path, format="JPEG", quality=92)

        dbg = img.copy()
        draw = ImageDraw.Draw(dbg)
        for i, box in enumerate(person_boxes):
            color = "yellow" if i != selected_idx else "lime"
            draw.rectangle(box, outline=color, width=5 if i == selected_idx else 3)
            draw.text((box[0] + 6, box[1] + 6), f"p{i}", fill=color)
        draw.rectangle(face_px, outline="red", width=4)
        draw.text((face_px[0] + 6, face_px[1] + 6), "face", fill="red")
        dbg_path = debug_dir / f"{path.stem}_debug.jpg"
        dbg.save(dbg_path, format="JPEG", quality=90)

        rec.update(
            {
                "status": "ok",
                "selection_source": selection_source,
                "selected_person_index": selected_idx,
                "selected_person_box_px": list(chosen_box),
                "crop_path": str(crop_path),
                "debug_path": str(dbg_path),
            }
        )
        manifest["images"].append(rec)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ok_count = sum(1 for i in manifest["images"] if i.get("status") == "ok")
    print(f"[done] saved {ok_count} crops")
    print(f"[done] output dir: {out_dir}")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
