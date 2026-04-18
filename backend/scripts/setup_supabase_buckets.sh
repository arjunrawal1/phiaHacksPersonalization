#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SUPABASE_URL_INPUT="${SUPABASE_URL:-}"
if [[ -z "${SUPABASE_URL_INPUT}" && -f "${ROOT_DIR}/frontend-web/.env.local" ]]; then
  SUPABASE_URL_INPUT="$(awk -F= '/^NEXT_PUBLIC_SUPABASE_URL=/{print $2}' "${ROOT_DIR}/frontend-web/.env.local" | tail -n 1)"
fi
if [[ -z "${SUPABASE_URL_INPUT}" ]]; then
  echo "Missing SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL in frontend-web/.env.local)." >&2
  exit 1
fi

SUPABASE_URL_INPUT="${SUPABASE_URL_INPUT%/}"
PROJECT_REF="$(echo "${SUPABASE_URL_INPUT}" | sed -E 's#https?://([^.]+)\.supabase\.co#\1#')"
if [[ -z "${PROJECT_REF}" || "${PROJECT_REF}" == "${SUPABASE_URL_INPUT}" ]]; then
  echo "Could not parse project ref from SUPABASE_URL=${SUPABASE_URL_INPUT}" >&2
  exit 1
fi

SERVICE_ROLE_KEY_INPUT="${SUPABASE_SERVICE_ROLE_KEY:-}"
if [[ -z "${SERVICE_ROLE_KEY_INPUT}" ]]; then
  KEYS_JSON="$(supabase projects api-keys --project-ref "${PROJECT_REF}" --output json)"
  SERVICE_ROLE_KEY_INPUT="$(python3 - <<'PY' "${KEYS_JSON}"
import json, sys
keys = json.loads(sys.argv[1])
for row in keys:
    name = (row.get("name") or "").lower()
    if "service_role" in name:
        print(row.get("api_key") or "")
        break
PY
)"
fi
if [[ -z "${SERVICE_ROLE_KEY_INPUT}" ]]; then
  echo "Could not resolve SUPABASE_SERVICE_ROLE_KEY for project ${PROJECT_REF}." >&2
  exit 1
fi

create_public_bucket() {
  local bucket_name="$1"
  local status
  local response
  response="$(mktemp)"
  status="$(curl -sS -o "${response}" -w "%{http_code}" \
    -X POST "${SUPABASE_URL_INPUT}/storage/v1/bucket" \
    -H "Authorization: Bearer ${SERVICE_ROLE_KEY_INPUT}" \
    -H "apikey: ${SERVICE_ROLE_KEY_INPUT}" \
    -H "Content-Type: application/json" \
    --data "{\"id\":\"${bucket_name}\",\"name\":\"${bucket_name}\",\"public\":true}")"
  if [[ "${status}" == "200" || "${status}" == "201" || "${status}" == "409" ]]; then
    rm -f "${response}"
    echo "Bucket ${bucket_name}: ready (HTTP ${status})"
    return 0
  fi
  if [[ "${status}" == "400" ]] && grep -qi "already exists\\|duplicate" "${response}"; then
    rm -f "${response}"
    echo "Bucket ${bucket_name}: ready (HTTP ${status})"
    return 0
  fi
  rm -f "${response}"
  echo "Bucket ${bucket_name}: failed (HTTP ${status})" >&2
  return 1
}

create_public_bucket "${SUPABASE_PHOTOS_BUCKET:-photos}"
create_public_bucket "${SUPABASE_FACE_TILES_BUCKET:-face-tiles}"
create_public_bucket "${SUPABASE_CLOTHING_CROPS_BUCKET:-clothing-crops}"

echo "Supabase bucket setup complete for project ${PROJECT_REF}."
