from __future__ import annotations

from app.services import db


def test_styling_auto_run_claim_is_idempotent(tmp_path) -> None:
    db.init_db(tmp_path / "phia.db")
    job_id = db.create_job(photo_count=1)
    photo_id = db.create_photo(job_id, "photos/job/0000.jpg", 1200, 1600)

    first = db.claim_styling_auto_run(
        job_id=job_id,
        photo_id=photo_id,
        source_photo_path="photos/job/0000.jpg",
        face_crop_path=f"face_tiles/{job_id}/{photo_id}.jpg",
        trigger_item_id="item-a",
    )
    second = db.claim_styling_auto_run(
        job_id=job_id,
        photo_id=photo_id,
        source_photo_path="photos/job/0000.jpg",
        face_crop_path=f"face_tiles/{job_id}/{photo_id}.jpg",
        trigger_item_id="item-b",
    )

    assert first["is_new"] is True
    assert second["is_new"] is False
    assert first["id"] == second["id"]


def test_styling_auto_run_variants_and_best_pick_persist(tmp_path) -> None:
    db.init_db(tmp_path / "phia.db")
    job_id = db.create_job(photo_count=1)
    photo_id = db.create_photo(job_id, "photos/job/0000.jpg", 1200, 1600)
    claimed = db.claim_styling_auto_run(
        job_id=job_id,
        photo_id=photo_id,
        source_photo_path="photos/job/0000.jpg",
        face_crop_path=f"face_tiles/{job_id}/{photo_id}.jpg",
        trigger_item_id="item-a",
    )
    run_id = str(claimed["id"])

    db.update_styling_auto_run(
        run_id,
        status="completed",
        selected_person_crop_path=f"styling_person_crops/{job_id}/{photo_id}.jpg",
        selected_person_bbox={"x1": 10, "y1": 20, "x2": 120, "y2": 340},
        gpt_selected_index=0,
        gpt_selection_reason="Matched face and body fully visible",
        body_visible=True,
        prompt="prompt-1",
        render_id="render-1",
        best_variant_index=2,
        best_reason="Highest realism + aesthetic",
    )
    db.replace_styling_auto_variants(
        run_id,
        best_variant_index=2,
        variants=[
            {
                "variant_index": 1,
                "prompt": "prompt-1",
                "output_path": "model_renders/render-1/01.jpg",
                "realism": 7.5,
                "aesthetic": 7.1,
                "overall": 7.3,
                "justification": "Good but weaker pose.",
            },
            {
                "variant_index": 2,
                "prompt": "prompt-1",
                "output_path": "model_renders/render-1/02.jpg",
                "realism": 8.8,
                "aesthetic": 8.9,
                "overall": 8.85,
                "justification": "Best lighting and silhouette.",
            },
        ],
    )

    runs = db.list_styling_auto_runs(job_id)
    assert len(runs) == 1
    row = runs[0]
    assert row["status"] == "completed"
    assert row["best_variant_index"] == 2
    assert row["best_reason"] == "Highest realism + aesthetic"
    assert row["selected_person_bbox"] == {"x1": 10, "y1": 20, "x2": 120, "y2": 340}

    variants = db.list_styling_auto_variants(run_id)
    assert [v["variant_index"] for v in variants] == [1, 2]
    assert variants[0]["is_best"] == 0
    assert variants[1]["is_best"] == 1
