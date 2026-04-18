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
            """
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


def create_photo(job_id: str, relative_path: str, width: int | None, height: int | None) -> str:
    photo_id = str(uuid.uuid4())
    now = utc_now_iso()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO photos (id, job_id, relative_path, width, height, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (photo_id, job_id, relative_path, width, height, now),
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
                exact_matches, similar_products, best_match, best_match_confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    best_match: dict[str, Any] | None,
    best_match_confidence: float,
    crop_path: str | None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE clothing_items
            SET tier = ?, exact_matches = ?, similar_products = ?, best_match = ?,
                best_match_confidence = ?, crop_path = ?
            WHERE id = ?
            """,
            (
                tier,
                json.dumps(exact_matches),
                json.dumps(similar_products),
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
        parsed["best_match"] = _json_or_default(parsed.get("best_match"), None)
        out.append(parsed)
    return out
