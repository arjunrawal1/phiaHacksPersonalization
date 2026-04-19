from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.core.config import Settings
from app.services import model_render


def _write_image(path: Path, *, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size=size, color=(180, 180, 180))
    img.save(path, format="JPEG", quality=90)


def _base_settings(tmp_path: Path) -> Settings:
    return Settings(
        media_dir=tmp_path,
        sqlite_path=tmp_path / "db.sqlite3",
        data_dir=tmp_path,
        openai_api_key="test-key",
        gemini_api_key="test-key",
    )


def test_auto_generate_skips_when_no_person_detections(tmp_path) -> None:
    settings = _base_settings(tmp_path)
    photo_rel = "photos/job/0000.jpg"
    face_rel = "face_tiles/job/photo.jpg"
    _write_image(tmp_path / photo_rel, size=(900, 1400))
    _write_image(tmp_path / face_rel, size=(220, 220))

    with pytest.raises(model_render.AutoStylingSkip, match="No person detected"):
        model_render.auto_generate_for_photo(
            settings=settings,
            job_id="job",
            photo_id="photo",
            source_photo_relative_path=photo_rel,
            face_crop_relative_path=face_rel,
            person_boxes=[],
        )


def test_auto_generate_skips_when_body_not_visible(tmp_path, monkeypatch) -> None:
    settings = _base_settings(tmp_path)
    photo_rel = "photos/job/0000.jpg"
    face_rel = "face_tiles/job/photo.jpg"
    _write_image(tmp_path / photo_rel, size=(900, 1400))
    _write_image(tmp_path / face_rel, size=(220, 220))

    monkeypatch.setattr(
        model_render,
        "_select_target_person_with_gpt",
        lambda **_: {
            "selected_index": 0,
            "is_body_visible": False,
            "reason": "Body is heavily occluded.",
        },
    )

    result = model_render.auto_generate_for_photo(
        settings=settings,
        job_id="job",
        photo_id="photo",
        source_photo_relative_path=photo_rel,
        face_crop_relative_path=face_rel,
        person_boxes=[(100, 120, 420, 1000)],
    )
    assert result["status"] == "skipped"
    assert "occluded" in str(result["skip_reason"]).lower()


def test_auto_generate_completes_and_returns_scored_variants(tmp_path, monkeypatch) -> None:
    settings = _base_settings(tmp_path)
    photo_rel = "photos/job/0000.jpg"
    face_rel = "face_tiles/job/photo.jpg"
    _write_image(tmp_path / photo_rel, size=(900, 1400))
    _write_image(tmp_path / face_rel, size=(220, 220))

    monkeypatch.setattr(
        model_render,
        "_select_target_person_with_gpt",
        lambda **_: {
            "selected_index": 0,
            "is_body_visible": True,
            "reason": "Strong face and full-body match.",
        },
    )
    monkeypatch.setattr(
        model_render,
        "render_model_variants",
        lambda **_: {
            "render_id": "render-1",
            "prompts_used": ["p1", "p2", "p3"],
            "output_relative_paths": [
                "model_renders/render-1/01.jpg",
                "model_renders/render-1/02.jpg",
                "model_renders/render-1/03.jpg",
            ],
        },
    )
    monkeypatch.setattr(
        model_render,
        "evaluate_variants",
        lambda **_: {
            "render_id": "render-1",
            "variants": [
                {"variant_index": 1, "realism": 7.0, "aesthetic": 7.3, "overall": 7.1, "justification": "ok"},
                {"variant_index": 2, "realism": 9.0, "aesthetic": 8.9, "overall": 9.0, "justification": "best"},
                {"variant_index": 3, "realism": 6.5, "aesthetic": 6.8, "overall": 6.6, "justification": "weak"},
            ],
            "best_variant_index": 2,
            "best_reason": "Best realism and styling.",
        },
    )

    result = model_render.auto_generate_for_photo(
        settings=settings,
        job_id="job",
        photo_id="photo",
        source_photo_relative_path=photo_rel,
        face_crop_relative_path=face_rel,
        person_boxes=[(100, 120, 420, 1000)],
    )

    assert result["status"] == "completed"
    assert result["render_id"] == "render-1"
    assert result["best_variant_index"] == 2
    assert len(result["variants"]) == 3
    assert any(v["is_best"] for v in result["variants"])
    assert (tmp_path / result["selected_person_crop_path"]).exists()


def test_auto_generate_raises_when_render_generation_fails(tmp_path, monkeypatch) -> None:
    settings = _base_settings(tmp_path)
    photo_rel = "photos/job/0000.jpg"
    face_rel = "face_tiles/job/photo.jpg"
    _write_image(tmp_path / photo_rel, size=(900, 1400))
    _write_image(tmp_path / face_rel, size=(220, 220))

    monkeypatch.setattr(
        model_render,
        "_select_target_person_with_gpt",
        lambda **_: {
            "selected_index": 0,
            "is_body_visible": True,
            "reason": "Visible body.",
        },
    )

    def _raise_render(**kwargs):
        raise model_render.ModelRenderError("generation failed")

    monkeypatch.setattr(model_render, "render_model_variants", _raise_render)

    with pytest.raises(model_render.ModelRenderError, match="generation failed"):
        model_render.auto_generate_for_photo(
            settings=settings,
            job_id="job",
            photo_id="photo",
            source_photo_relative_path=photo_rel,
            face_crop_relative_path=face_rel,
            person_boxes=[(100, 120, 420, 1000)],
        )
