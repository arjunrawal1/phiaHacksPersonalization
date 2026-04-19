from __future__ import annotations

from app.services import db
from app.services import pipeline


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


def test_insert_clothing_item_defaults_closet_item_key_to_item_id(tmp_path) -> None:
    db.init_db(tmp_path / "phia.db")
    job_id = db.create_job(photo_count=1)
    photo_id = db.create_photo(job_id, "photos/job/0000.jpg", 900, 1200)
    item_id = db.insert_clothing_item(
        job_id=job_id,
        photo_id=photo_id,
        category="jacket",
        description="navy track jacket",
        colors=["navy", "white"],
        pattern="colorblock",
        style="sporty",
        brand_visible="oracle red bull racing",
        visibility="clear",
        confidence=0.94,
        bounding_box={"x": 0.2, "y": 0.1, "w": 0.4, "h": 0.6},
        crop_path=None,
        tier="pending",
        exact_matches=[],
        similar_products=[],
        phia_products=[],
        best_match=None,
        best_match_confidence=0.0,
    )

    row = db.get_clothing_item(item_id)
    assert row is not None
    assert row["closet_item_key"] == item_id


def test_assign_closet_item_keys_persists_openai_mapping(tmp_path, monkeypatch) -> None:
    db.init_db(tmp_path / "phia.db")
    job_id = db.create_job(photo_count=2)
    photo_a = db.create_photo(job_id, "photos/job/0000.jpg", 900, 1200)
    photo_b = db.create_photo(job_id, "photos/job/0001.jpg", 900, 1200)

    item_a = db.insert_clothing_item(
        job_id=job_id,
        photo_id=photo_a,
        category="jacket",
        description="oracle red bull racing white track jacket",
        colors=["white", "blue"],
        pattern="colorblock",
        style="sporty",
        brand_visible="oracle",
        visibility="clear",
        confidence=0.91,
        bounding_box={"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.6},
        crop_path=None,
        tier="pending",
        exact_matches=[],
        similar_products=[],
        phia_products=[],
        best_match=None,
        best_match_confidence=0.0,
    )
    item_b = db.insert_clothing_item(
        job_id=job_id,
        photo_id=photo_b,
        category="jacket",
        description="oracle red bull racing white jacket",
        colors=["white", "blue"],
        pattern="colorblock",
        style="sporty",
        brand_visible="oracle",
        visibility="clear",
        confidence=0.88,
        bounding_box={"x": 0.2, "y": 0.1, "w": 0.5, "h": 0.6},
        crop_path=None,
        tier="pending",
        exact_matches=[],
        similar_products=[],
        phia_products=[],
        best_match=None,
        best_match_confidence=0.0,
    )

    monkeypatch.setattr(
        pipeline,
        "_dedupe_closet_items_with_openai",
        lambda rows: {item_a: item_a, item_b: item_a},
    )

    mapping, source = pipeline._assign_closet_item_keys(job_id, [item_a, item_b])
    assert source == "openai"
    assert mapping[item_a] == item_a
    assert mapping[item_b] == item_a

    row_a = db.get_clothing_item(item_a)
    row_b = db.get_clothing_item(item_b)
    assert row_a is not None and row_a["closet_item_key"] == item_a
    assert row_b is not None and row_b["closet_item_key"] == item_a
