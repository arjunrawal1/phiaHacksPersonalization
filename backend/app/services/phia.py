from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from app.core.config import Settings

ADD_PRODUCT_MUTATION = (
    "mutation Add($phiaId:String!, $collectionId:String!, $productData:ProductDataInput!){ "
    "addProductToCollection(phiaId:$phiaId, collectionId:$collectionId, productData:$productData) }"
)


@dataclass
class PhiaAuth:
    phia_id: str
    session_cookie: str
    bearer_token: str
    platform: str
    platform_version: str
    source: str


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _extract_header_value(raw: str, name: str) -> str:
    pattern = rf"(?im)^{re.escape(name)}:\s*(.+)$"
    match = re.search(pattern, raw)
    return _clean(match.group(1) if match else "")


def _extract_session_value(cookie_header: str) -> str:
    if not cookie_header:
        return ""
    match = re.search(r"(?:^|;\s*)session=([^;]+)", cookie_header)
    if match:
        return _clean(match.group(1))
    if cookie_header.lower().startswith("session="):
        return _clean(cookie_header.split("=", 1)[1].split(";", 1)[0])
    return _clean(cookie_header)


def _extract_bearer_value(authorization_header: str) -> str:
    if not authorization_header:
        return ""
    match = re.match(r"(?i)^Bearer\s+(.+)$", authorization_header.strip())
    if match:
        return _clean(match.group(1))
    return _clean(authorization_header)


def _read_capture_auth(request_file: Path) -> PhiaAuth | None:
    try:
        raw = request_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    phia_id = _extract_header_value(raw, "x-phia-id")
    cookie_header = _extract_header_value(raw, "cookie")
    session_cookie = _extract_session_value(cookie_header)
    auth_header = _extract_header_value(raw, "authorization")
    bearer_token = _extract_bearer_value(auth_header)
    platform = _extract_header_value(raw, "x-platform") or "IOS_APP"
    platform_version = _extract_header_value(raw, "x-platform-version") or "2.3.11.362"

    if not phia_id or not session_cookie:
        return None

    return PhiaAuth(
        phia_id=phia_id,
        session_cookie=session_cookie,
        bearer_token=bearer_token,
        platform=platform,
        platform_version=platform_version,
        source=str(request_file),
    )


def _candidate_capture_dirs(settings: Settings) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    for path_str in settings.phia_capture_dirs:
        p = Path(path_str).expanduser()
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)

    desktop = Path.home() / "Desktop"
    if desktop.exists():
        for p in sorted(desktop.glob("Raw_*.folder"), key=lambda entry: entry.stat().st_mtime, reverse=True):
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(p)

    return out


def _first_capture_auth(settings: Settings) -> PhiaAuth | None:
    for capture_dir in _candidate_capture_dirs(settings):
        if not capture_dir.exists() or not capture_dir.is_dir():
            continue

        request_files = sorted(
            capture_dir.glob("*Request - api.phia.com_v2_graphql.txt"),
            key=lambda entry: entry.stat().st_mtime,
            reverse=True,
        )
        for request_file in request_files:
            auth = _read_capture_auth(request_file)
            if auth:
                return auth
    return None


def _jwt_exp(token: str) -> int | None:
    token = _clean(token)
    if not token:
        return None
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload + padding)
        data = json.loads(decoded.decode("utf-8"))
        exp = data.get("exp")
        if isinstance(exp, int):
            return exp
    except Exception:
        return None
    return None


def _is_token_expired(token: str) -> bool:
    exp = _jwt_exp(token)
    if exp is None:
        return False
    return exp <= int(time.time())


def _is_cert_verify_error(exc: Exception) -> bool:
    message = str(exc)
    return "CERTIFICATE_VERIFY_FAILED" in message or "SSLCertVerificationError" in message


def resolve_phia_auth(settings: Settings) -> PhiaAuth:
    env_auth = PhiaAuth(
        phia_id=_clean(settings.phia_id),
        session_cookie=_extract_session_value(_clean(settings.phia_session_cookie)),
        bearer_token=_extract_bearer_value(_clean(settings.phia_bearer_token)),
        platform=_clean(settings.phia_platform) or "IOS_APP",
        platform_version=_clean(settings.phia_platform_version) or "2.3.11.362",
        source="env",
    )

    if env_auth.phia_id and env_auth.session_cookie:
        return env_auth

    capture_auth = _first_capture_auth(settings)
    if not capture_auth:
        raise RuntimeError(
            "Could not find Phia auth credentials. Set PHIA_* env vars or provide Raw_*.folder captures."
        )

    return PhiaAuth(
        phia_id=env_auth.phia_id or capture_auth.phia_id,
        session_cookie=env_auth.session_cookie or capture_auth.session_cookie,
        bearer_token=env_auth.bearer_token or capture_auth.bearer_token,
        platform=env_auth.platform or capture_auth.platform,
        platform_version=env_auth.platform_version or capture_auth.platform_version,
        source=capture_auth.source if not env_auth.phia_id or not env_auth.session_cookie else "env+capture",
    )


def backfill_favorites(
    product_urls: list[str],
    settings: Settings,
    collection_id: str | None = None,
) -> dict[str, Any]:
    auth = resolve_phia_auth(settings)
    if not auth.phia_id:
        raise RuntimeError("Missing phia_id")
    if not auth.session_cookie:
        raise RuntimeError("Missing session cookie")

    final_collection_id = _clean(collection_id) or _clean(settings.phia_collection_id) or "all_favorites"

    unique_urls: list[str] = []
    seen: set[str] = set()
    for raw in product_urls:
        candidate = _clean(raw)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        unique_urls.append(candidate)

    headers: dict[str, str] = {
        "content-type": "application/json",
        "cookie": f"session={auth.session_cookie}",
        "x-phia-id": auth.phia_id,
        "x-platform": auth.platform or "IOS_APP",
        "x-platform-version": auth.platform_version or "2.3.11.362",
    }
    if auth.bearer_token and not _is_token_expired(auth.bearer_token):
        headers["authorization"] = f"Bearer {auth.bearer_token}"

    results: list[dict[str, Any]] = []
    added_count = 0
    verify_tls = not settings.phia_allow_insecure_tls

    for url in unique_urls:
        if not (url.startswith("http://") or url.startswith("https://")):
            results.append(
                {"product_url": url, "ok": False, "message": "Skipped: product URL must start with http:// or https://"}
            )
            continue

        payload = {
            "operationName": "Add",
            "query": ADD_PRODUCT_MUTATION,
            "variables": {
                "phiaId": auth.phia_id,
                "collectionId": final_collection_id,
                "productData": {"productUrl": url},
            },
        }

        response = None
        try:
            response = requests.post(
                settings.phia_graphql_url,
                headers=headers,
                json=payload,
                timeout=25,
                verify=verify_tls,
            )
        except Exception as exc:
            if verify_tls and _is_cert_verify_error(exc):
                try:
                    response = requests.post(
                        settings.phia_graphql_url,
                        headers=headers,
                        json=payload,
                        timeout=25,
                        verify=False,
                    )
                except Exception as retry_exc:
                    message = (
                        f"Request failed: {exc}; insecure TLS retry also failed: {retry_exc}"
                    )
                    results.append({"product_url": url, "ok": False, "message": message})
                    continue
            else:
                message = f"Request failed: {exc}"
                if "CERTIFICATE_VERIFY_FAILED" in message and verify_tls:
                    message += " (set PHIA_ALLOW_INSECURE_TLS=true if you are behind a trusted interception proxy)"
                results.append({"product_url": url, "ok": False, "message": message})
                continue

        parsed: dict[str, Any] = {}
        try:
            parsed = response.json()
        except Exception:
            parsed = {}

        if response.status_code >= 400:
            detail = parsed.get("message") if isinstance(parsed, dict) else None
            results.append(
                {
                    "product_url": url,
                    "ok": False,
                    "message": f"HTTP {response.status_code}" + (f": {detail}" if detail else ""),
                }
            )
            continue

        gql_errors = parsed.get("errors") if isinstance(parsed, dict) else None
        if gql_errors:
            first_error = gql_errors[0] if isinstance(gql_errors, list) and gql_errors else gql_errors
            message = ""
            if isinstance(first_error, dict):
                message = _clean(str(first_error.get("message") or "GraphQL error"))
            else:
                message = _clean(str(first_error))
            results.append({"product_url": url, "ok": False, "message": message or "GraphQL error"})
            continue

        added = bool(((parsed.get("data") or {}).get("addProductToCollection")))
        if added:
            added_count += 1
            results.append({"product_url": url, "ok": True, "message": f"Added to {final_collection_id}"})
        else:
            results.append({"product_url": url, "ok": False, "message": "Mutation returned false"})

    failed_count = len([r for r in results if not r["ok"]])

    return {
        "phia_id": auth.phia_id,
        "collection_id": final_collection_id,
        "source": auth.source,
        "requested_count": len(product_urls),
        "attempted_count": len(unique_urls),
        "added_count": added_count,
        "failed_count": failed_count,
        "results": results,
    }
