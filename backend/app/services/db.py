from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DB_PATH: Path | None = None


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _require_db_path() -> Path:
    if _DB_PATH is None:
        raise RuntimeError("Database not initialized")
    return _DB_PATH


def _connect() -> sqlite3.Connection:
    path = _require_db_path()
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    global _DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _DB_PATH = db_path

    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                photo_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS photos (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                captured_at TEXT,
                captured_at_epoch_ms INTEGER,
                latitude REAL,
                longitude REAL,
                location_source TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_photos_job_id ON photos(job_id);

            CREATE TABLE IF NOT EXISTS face_clusters (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                rep_photo_id TEXT NOT NULL,
                rep_bbox TEXT NOT NULL,
                member_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                FOREIGN KEY(rep_photo_id) REFERENCES photos(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_face_clusters_job_id ON face_clusters(job_id);

            CREATE TABLE IF NOT EXISTS face_detections (
                id TEXT PRIMARY KEY,
                photo_id TEXT NOT NULL,
                cluster_id TEXT,
                bbox TEXT NOT NULL,
                confidence REAL NOT NULL,
                feature TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(photo_id) REFERENCES photos(id) ON DELETE CASCADE,
                FOREIGN KEY(cluster_id) REFERENCES face_clusters(id) ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_face_detections_photo_id ON face_detections(photo_id);
            CREATE INDEX IF NOT EXISTS idx_face_detections_cluster_id ON face_detections(cluster_id);

            CREATE TABLE IF NOT EXISTS selected_cluster (
                job_id TEXT PRIMARY KEY,
                cluster_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                FOREIGN KEY(cluster_id) REFERENCES face_clusters(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS clothing_items (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                photo_id TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                colors TEXT NOT NULL,
                pattern TEXT,
                style TEXT,
                brand_visible TEXT,
                visibility TEXT NOT NULL,
                confidence REAL NOT NULL,
                bounding_box TEXT NOT NULL,
                crop_path TEXT,
                tier TEXT NOT NULL DEFAULT 'generic',
                exact_matches TEXT NOT NULL DEFAULT '[]',
                similar_products TEXT NOT NULL DEFAULT '[]',
                phia_products TEXT NOT NULL DEFAULT '[]',
                best_match TEXT,
                best_match_confidence REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                FOREIGN KEY(photo_id) REFERENCES photos(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_clothing_items_job_id ON clothing_items(job_id);
            CREATE INDEX IF NOT EXISTS idx_clothing_items_photo_id ON clothing_items(photo_id);

            CREATE TABLE IF NOT EXISTS job_debug (
                job_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS styling_auto_runs (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                photo_id TEXT NOT NULL UNIQUE,
                trigger_item_id TEXT,
                status TEXT NOT NULL,
                source_photo_path TEXT NOT NULL,
                face_crop_path TEXT,
                selected_person_crop_path TEXT,
                selected_person_bbox TEXT,
                gpt_selected_index INTEGER,
                gpt_selection_reason TEXT,
                body_visible INTEGER,
                prompt TEXT,
                render_id TEXT,
                best_variant_index INTEGER,
                best_reason TEXT,
                skip_reason TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
                FOREIGN KEY(photo_id) REFERENCES photos(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_styling_auto_runs_job_id ON styling_auto_runs(job_id);
            CREATE INDEX IF NOT EXISTS idx_styling_auto_runs_status ON styling_auto_runs(status);

            CREATE TABLE IF NOT EXISTS styling_auto_variants (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                variant_index INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                output_path TEXT NOT NULL,
                realism REAL,
                aesthetic REAL,
                overall REAL,
                justification TEXT,
                is_best INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES styling_auto_runs(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_styling_auto_variants_run_id ON styling_auto_variants(run_id);
            """
        )

        photo_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(photos)").fetchall()
        }
        if "captured_at" not in photo_columns:
            conn.execute("ALTER TABLE photos ADD COLUMN captured_at TEXT")
        if "captured_at_epoch_ms" not in photo_columns:
            conn.execute("ALTER TABLE photos ADD COLUMN captured_at_epoch_ms INTEGER")
        if "latitude" not in photo_columns:
            conn.execute("ALTER TABLE photos ADD COLUMN latitude REAL")
        if "longitude" not in photo_columns:
            conn.execute("ALTER TABLE photos ADD COLUMN longitude REAL")
        if "location_source" not in photo_columns:
            conn.execute("ALTER TABLE photos ADD COLUMN location_source TEXT")
        if "metadata_json" not in photo_columns:
            conn.execute("ALTER TABLE photos ADD COLUMN metadata_json TEXT")

        existing_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(clothing_items)").fetchall()
        }
        if "phia_products" not in existing_columns:
            conn.execute(
                "ALTER TABLE clothing_items ADD COLUMN phia_products TEXT NOT NULL DEFAULT '[]'"
            )


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _json_or_default(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def create_job(photo_count: int) -> str:
    job_id = str(uuid.uuid4())
    now = utc_now_iso()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, status, photo_count, error, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, "uploading", photo_count, None, now, now),
        )
    return job_id


_UNSET = object()


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    error: str | None | object = _UNSET,
) -> None:
    updates: list[str] = ["updated_at = ?"]
    values: list[Any] = [utc_now_iso()]
    if status is not None:
        updates.append("status = ?")
        values.append(status)
    if error is not _UNSET:
        updates.append("error = ?")
        values.append(error)

    values.append(job_id)
    with _connect() as conn:
        conn.execute(
            f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?",
            values,
        )


def set_job_photo_count(job_id: str, photo_count: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET photo_count = ?, updated_at = ? WHERE id = ?",
            (max(0, int(photo_count)), utc_now_iso(), job_id),
        )


def get_job(job_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_dict(row) if row else None


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def create_photo(
    job_id: str,
    relative_path: str,
    width: int | None,
    height: int | None,
    *,
    captured_at: str | None = None,
    captured_at_epoch_ms: int | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    location_source: str | None = None,
    metadata_json: str | None = None,
) -> str:
    photo_id = str(uuid.uuid4())
    now = utc_now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO photos (
                id, job_id, relative_path, width, height,
                captured_at, captured_at_epoch_ms, latitude, longitude, location_source, metadata_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                photo_id,
                job_id,
                relative_path,
                width,
                height,
                captured_at,
                captured_at_epoch_ms,
                latitude,
                longitude,
                location_source,
                metadata_json,
                now,
            ),
        )
    return photo_id


def list_photos(job_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM photos WHERE job_id = ? ORDER BY created_at ASC", (job_id,)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_photo(photo_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
        return _row_to_dict(row) if row else None


def clear_face_analysis(job_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "DELETE FROM face_detections WHERE photo_id IN (SELECT id FROM photos WHERE job_id = ?)",
            (job_id,),
        )
        conn.execute("DELETE FROM face_clusters WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM selected_cluster WHERE job_id = ?", (job_id,))


def insert_face_detection(
    photo_id: str,
    bbox: dict[str, float],
    confidence: float,
    feature: list[float] | None,
) -> str:
    detection_id = str(uuid.uuid4())
    now = utc_now_iso()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO face_detections (id, photo_id, cluster_id, bbox, confidence, feature, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                detection_id,
                photo_id,
                None,
                json.dumps(bbox),
                confidence,
                json.dumps(feature) if feature is not None else None,
                now,
            ),
        )
    return detection_id


def list_face_detections_for_job(job_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT fd.*
            FROM face_detections fd
            JOIN photos p ON p.id = fd.photo_id
            WHERE p.job_id = ?
            ORDER BY fd.created_at ASC
            """,
            (job_id,),
        ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        parsed = _row_to_dict(row)
        parsed["bbox"] = _json_or_default(parsed.get("bbox"), {})
        parsed["feature"] = _json_or_default(parsed.get("feature"), None)
        out.append(parsed)
    return out


def list_face_detections_for_cluster(cluster_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM face_detections WHERE cluster_id = ? ORDER BY created_at ASC",
            (cluster_id,),
        ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        parsed = _row_to_dict(row)
        parsed["bbox"] = _json_or_default(parsed.get("bbox"), {})
        parsed["feature"] = _json_or_default(parsed.get("feature"), None)
        out.append(parsed)
    return out


def insert_face_cluster(
    job_id: str,
    rep_photo_id: str,
    rep_bbox: dict[str, float],
    member_count: int,
) -> str:
    cluster_id = str(uuid.uuid4())
    now = utc_now_iso()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO face_clusters (id, job_id, rep_photo_id, rep_bbox, member_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (cluster_id, job_id, rep_photo_id, json.dumps(rep_bbox), member_count, now),
        )
    return cluster_id


def set_detection_cluster(detection_ids: list[str], cluster_id: str) -> None:
    if not detection_ids:
        return
    placeholders = ",".join(["?"] * len(detection_ids))
    with _connect() as conn:
        conn.execute(
            f"UPDATE face_detections SET cluster_id = ? WHERE id IN ({placeholders})",
            [cluster_id, *detection_ids],
        )


def list_face_clusters(job_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM face_clusters WHERE job_id = ? ORDER BY member_count DESC, created_at ASC",
            (job_id,),
        ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        parsed = _row_to_dict(row)
        parsed["rep_bbox"] = _json_or_default(parsed.get("rep_bbox"), {})
        out.append(parsed)
    return out


def upsert_selected_cluster(job_id: str, cluster_id: str) -> None:
    now = utc_now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO selected_cluster (job_id, cluster_id, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
              cluster_id = excluded.cluster_id,
              created_at = excluded.created_at
            """,
            (job_id, cluster_id, now),
        )


def get_selected_cluster(job_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM selected_cluster WHERE job_id = ?", (job_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None


def _deep_merge(existing: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(existing)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def patch_job_debug(
    job_id: str,
    *,
    patch: dict[str, Any] | None = None,
    event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_now_iso()
    with _connect() as conn:
        row = conn.execute("SELECT payload FROM job_debug WHERE job_id = ?", (job_id,)).fetchone()
        payload = _json_or_default(row["payload"], {}) if row else {}
        if not isinstance(payload, dict):
            payload = {}

        if patch:
            payload = _deep_merge(payload, patch)
        if event:
            events = payload.get("events")
            if not isinstance(events, list):
                events = []
            events.append({"at": now, **event})
            payload["events"] = events[-300:]

        conn.execute(
            """
            INSERT INTO job_debug (job_id, payload, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
              payload = excluded.payload,
              updated_at = excluded.updated_at
            """,
            (job_id, json.dumps(payload), now),
        )
    return payload


def get_job_debug(job_id: str) -> dict[str, Any]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload FROM job_debug WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    if not row:
        return {}
    parsed = _json_or_default(row["payload"], {})
    return parsed if isinstance(parsed, dict) else {}


def clear_clothing_items(job_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM clothing_items WHERE job_id = ?", (job_id,))


def insert_clothing_item(
    *,
    job_id: str,
    photo_id: str,
    category: str,
    description: str,
    colors: list[str],
    pattern: str,
    style: str,
    brand_visible: str | None,
    visibility: str,
    confidence: float,
    bounding_box: dict[str, float],
    crop_path: str | None,
    tier: str,
    exact_matches: list[dict[str, Any]],
    similar_products: list[dict[str, Any]],
    phia_products: list[dict[str, Any]],
    best_match: dict[str, Any] | None,
    best_match_confidence: float,
) -> str:
    item_id = str(uuid.uuid4())
    now = utc_now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO clothing_items (
                id, job_id, photo_id, category, description, colors, pattern, style,
                brand_visible, visibility, confidence, bounding_box, crop_path, tier,
                exact_matches, similar_products, phia_products, best_match, best_match_confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                job_id,
                photo_id,
                category,
                description,
                json.dumps(colors),
                pattern,
                style,
                brand_visible,
                visibility,
                confidence,
                json.dumps(bounding_box),
                crop_path,
                tier,
                json.dumps(exact_matches),
                json.dumps(similar_products),
                json.dumps(phia_products),
                json.dumps(best_match) if best_match else None,
                best_match_confidence,
                now,
            ),
        )
    return item_id


def update_clothing_item(
    item_id: str,
    *,
    tier: str,
    exact_matches: list[dict[str, Any]],
    similar_products: list[dict[str, Any]],
    phia_products: list[dict[str, Any]],
    best_match: dict[str, Any] | None,
    best_match_confidence: float,
    crop_path: str | None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE clothing_items
            SET tier = ?, exact_matches = ?, similar_products = ?, phia_products = ?, best_match = ?,
                best_match_confidence = ?, crop_path = ?
            WHERE id = ?
            """,
            (
                tier,
                json.dumps(exact_matches),
                json.dumps(similar_products),
                json.dumps(phia_products),
                json.dumps(best_match) if best_match else None,
                best_match_confidence,
                crop_path,
                item_id,
            ),
        )


def get_clothing_item(item_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM clothing_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            return None

    parsed = _row_to_dict(row)
    parsed["colors"] = _json_or_default(parsed.get("colors"), [])
    parsed["bounding_box"] = _json_or_default(parsed.get("bounding_box"), {})
    parsed["exact_matches"] = _json_or_default(parsed.get("exact_matches"), [])
    parsed["similar_products"] = _json_or_default(parsed.get("similar_products"), [])
    parsed["phia_products"] = _json_or_default(parsed.get("phia_products"), [])
    parsed["best_match"] = _json_or_default(parsed.get("best_match"), None)
    return parsed


def list_clothing_items(job_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM clothing_items WHERE job_id = ? ORDER BY created_at ASC",
            (job_id,),
        ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        parsed = _row_to_dict(row)
        parsed["colors"] = _json_or_default(parsed.get("colors"), [])
        parsed["bounding_box"] = _json_or_default(parsed.get("bounding_box"), {})
        parsed["exact_matches"] = _json_or_default(parsed.get("exact_matches"), [])
        parsed["similar_products"] = _json_or_default(parsed.get("similar_products"), [])
        parsed["phia_products"] = _json_or_default(parsed.get("phia_products"), [])
        parsed["best_match"] = _json_or_default(parsed.get("best_match"), None)
        out.append(parsed)
    return out


def list_clothing_items_for_photo(photo_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM clothing_items WHERE photo_id = ? ORDER BY created_at ASC",
            (photo_id,),
        ).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        parsed = _row_to_dict(row)
        parsed["colors"] = _json_or_default(parsed.get("colors"), [])
        parsed["bounding_box"] = _json_or_default(parsed.get("bounding_box"), {})
        parsed["exact_matches"] = _json_or_default(parsed.get("exact_matches"), [])
        parsed["similar_products"] = _json_or_default(parsed.get("similar_products"), [])
        parsed["phia_products"] = _json_or_default(parsed.get("phia_products"), [])
        parsed["best_match"] = _json_or_default(parsed.get("best_match"), None)
        out.append(parsed)
    return out


def claim_styling_auto_run(
    *,
    job_id: str,
    photo_id: str,
    source_photo_path: str,
    face_crop_path: str | None,
    trigger_item_id: str | None = None,
) -> dict[str, Any]:
    now = utc_now_iso()
    run_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO styling_auto_runs (
                id, job_id, photo_id, trigger_item_id, status, source_photo_path,
                face_crop_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                job_id,
                photo_id,
                trigger_item_id,
                "queued",
                source_photo_path,
                face_crop_path,
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM styling_auto_runs WHERE photo_id = ?",
            (photo_id,),
        ).fetchone()
    if not row:
        raise RuntimeError("Failed to claim styling auto run")
    parsed = _row_to_dict(row)
    parsed["selected_person_bbox"] = _json_or_default(parsed.get("selected_person_bbox"), None)
    parsed["is_new"] = parsed["id"] == run_id
    return parsed


def get_styling_auto_run(run_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM styling_auto_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    if not row:
        return None
    parsed = _row_to_dict(row)
    parsed["selected_person_bbox"] = _json_or_default(parsed.get("selected_person_bbox"), None)
    return parsed


def get_styling_auto_run_for_photo(photo_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM styling_auto_runs WHERE photo_id = ?",
            (photo_id,),
        ).fetchone()
    if not row:
        return None
    parsed = _row_to_dict(row)
    parsed["selected_person_bbox"] = _json_or_default(parsed.get("selected_person_bbox"), None)
    return parsed


def update_styling_auto_run(
    run_id: str,
    *,
    status: str | None = None,
    trigger_item_id: str | None | object = _UNSET,
    selected_person_crop_path: str | None | object = _UNSET,
    selected_person_bbox: dict[str, int] | None | object = _UNSET,
    gpt_selected_index: int | None | object = _UNSET,
    gpt_selection_reason: str | None | object = _UNSET,
    body_visible: bool | None | object = _UNSET,
    prompt: str | None | object = _UNSET,
    render_id: str | None | object = _UNSET,
    best_variant_index: int | None | object = _UNSET,
    best_reason: str | None | object = _UNSET,
    skip_reason: str | None | object = _UNSET,
    error: str | None | object = _UNSET,
    started_at: str | None | object = _UNSET,
    finished_at: str | None | object = _UNSET,
) -> None:
    updates: list[str] = ["updated_at = ?"]
    values: list[Any] = [utc_now_iso()]
    if status is not None:
        updates.append("status = ?")
        values.append(status)
    if trigger_item_id is not _UNSET:
        updates.append("trigger_item_id = ?")
        values.append(trigger_item_id)
    if selected_person_crop_path is not _UNSET:
        updates.append("selected_person_crop_path = ?")
        values.append(selected_person_crop_path)
    if selected_person_bbox is not _UNSET:
        updates.append("selected_person_bbox = ?")
        values.append(json.dumps(selected_person_bbox) if selected_person_bbox else None)
    if gpt_selected_index is not _UNSET:
        updates.append("gpt_selected_index = ?")
        values.append(gpt_selected_index)
    if gpt_selection_reason is not _UNSET:
        updates.append("gpt_selection_reason = ?")
        values.append(gpt_selection_reason)
    if body_visible is not _UNSET:
        updates.append("body_visible = ?")
        values.append(None if body_visible is None else (1 if body_visible else 0))
    if prompt is not _UNSET:
        updates.append("prompt = ?")
        values.append(prompt)
    if render_id is not _UNSET:
        updates.append("render_id = ?")
        values.append(render_id)
    if best_variant_index is not _UNSET:
        updates.append("best_variant_index = ?")
        values.append(best_variant_index)
    if best_reason is not _UNSET:
        updates.append("best_reason = ?")
        values.append(best_reason)
    if skip_reason is not _UNSET:
        updates.append("skip_reason = ?")
        values.append(skip_reason)
    if error is not _UNSET:
        updates.append("error = ?")
        values.append(error)
    if started_at is not _UNSET:
        updates.append("started_at = ?")
        values.append(started_at)
    if finished_at is not _UNSET:
        updates.append("finished_at = ?")
        values.append(finished_at)

    values.append(run_id)
    with _connect() as conn:
        conn.execute(
            f"UPDATE styling_auto_runs SET {', '.join(updates)} WHERE id = ?",
            values,
        )


def replace_styling_auto_variants(
    run_id: str,
    *,
    variants: list[dict[str, Any]],
    best_variant_index: int | None,
) -> None:
    now = utc_now_iso()
    with _connect() as conn:
        conn.execute("DELETE FROM styling_auto_variants WHERE run_id = ?", (run_id,))
        for row in variants:
            idx = int(row.get("variant_index") or 0)
            conn.execute(
                """
                INSERT INTO styling_auto_variants (
                    id, run_id, variant_index, prompt, output_path, realism, aesthetic,
                    overall, justification, is_best, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    run_id,
                    idx,
                    str(row.get("prompt") or ""),
                    str(row.get("output_path") or ""),
                    row.get("realism"),
                    row.get("aesthetic"),
                    row.get("overall"),
                    str(row.get("justification") or ""),
                    1 if (best_variant_index is not None and idx == best_variant_index) else 0,
                    now,
                ),
            )


def list_styling_auto_runs(job_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM styling_auto_runs WHERE job_id = ? ORDER BY created_at ASC",
            (job_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        parsed = _row_to_dict(row)
        parsed["selected_person_bbox"] = _json_or_default(parsed.get("selected_person_bbox"), None)
        out.append(parsed)
    return out


def list_styling_auto_variants(run_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM styling_auto_variants WHERE run_id = ? ORDER BY variant_index ASC",
            (run_id,),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]
