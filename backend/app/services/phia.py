from __future__ import annotations

import base64
import json
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from urllib3.exceptions import InsecureRequestWarning

from app.core.config import Settings

ADD_PRODUCT_MUTATION = (
    "mutation Add($phiaId:String!, $collectionId:String!, $productData:ProductDataInput!){ "
    "addProductToCollection(phiaId:$phiaId, collectionId:$collectionId, productData:$productData) }"
)
FETCH_CURRENT_USER_QUERY = (
    "query SwiftFetchCurrentUser { "
    "currentUser { id phiaId username firstName lastName email imageUrl createdAt updatedAt phoneNumber } "
    "}"
)
FETCH_USER_NOTIFICATIONS_QUERY = (
    "query SwiftGetUserNotifications($phiaId: String!) { "
    "getUserNotifications(phiaId: $phiaId) { notifications { __typename } priceDropAlerts { __typename } "
    "items { __typename } collections { __typename } } }"
)
FETCH_ACTIVE_CAMPAIGN_QUERY = (
    "query FetchActiveCampaign { activeCampaign { id name startDate endDate } }"
)
FETCH_POPULAR_SEARCHES_QUERY = (
    "query SwiftFetchPopularSearches { "
    "popularSearches { isFemale isTrending rank category imgUrl query } "
    "}"
)
FETCH_TRENDING_PRODUCTS_QUERY = (
    "query SwiftFetchTrendingProducts { "
    "trendingProducts { productId brand price viewCount isTrending gender } "
    "}"
)
FETCH_TRENDING_BRANDS_QUERY = (
    "query SwiftFetchTrendingBrands { "
    "trendingBrands { id name imgUrl logoUrl visitCount description } "
    "}"
)
FETCH_EDITORIALS_QUERY = (
    "query SwiftFetchEditorials { editorials { title description order imgUrl createdAt } }"
)
FETCH_CURATED_TYPES_QUERY = (
    "query SwiftFetchCuratedTypes { curatedTypes { name imgUrl } }"
)
FETCH_EXPLORE_FEED_QUERY = (
    "query SwiftFetchExploreFeed($input: ExploreFeedInput!) { exploreFeed(input: $input) { offset hasMore } }"
)
FETCH_COLLECTION_PRODUCTS_QUERY = (
    "query SwiftFetchCollectionProducts($phiaId: String!) { "
    "listCollections(phiaId: $phiaId) { "
    "collections { id name itemCount collectionType coverUrl createdAt updatedAt savedProducts { "
    "id name priceUsd productUrl imgUrl primaryBrandName sourceDisplayName addedToCollectionAt "
    "} } } }"
)
FETCH_BRANDS_QUERY = (
    "query SwiftFetchSavedBrands { brands { id displayName coverImageUrl logoUrl description isTrending } }"
)
PRODUCTS_GOOGLE_SHOPPING_QUERY = (
    "query ProductsGoogleShoppingApi($query: ProductSearchInput!, $filters: ProductSearchFilters!, "
    "$options: ProductSearchOptions!) { products(query: $query, filters: $filters, options: $options) { "
    "numProducts results { id name imgUrl productUrl primaryBrandName sourceDisplayName priceUsd } } }"
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


def _post_json(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int,
    verify: bool,
) -> requests.Response:
    if verify:
        return requests.post(url, headers=headers, json=payload, timeout=timeout, verify=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InsecureRequestWarning)
        return requests.post(url, headers=headers, json=payload, timeout=timeout, verify=False)


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


def resolve_phia_auth_for_session(
    *,
    settings: Settings,
    phia_id: str | None = None,
    session_cookie: str | None = None,
    cookie_header: str | None = None,
    bearer_token: str | None = None,
    authorization_header: str | None = None,
    platform: str | None = None,
    platform_version: str | None = None,
    inherit_default_auth: bool = True,
    source: str = "request_override",
) -> PhiaAuth:
    base = (
        resolve_phia_auth(settings)
        if inherit_default_auth
        else PhiaAuth(
            phia_id="",
            session_cookie="",
            bearer_token="",
            platform=_clean(settings.phia_platform) or "IOS_APP",
            platform_version=_clean(settings.phia_platform_version) or "2.3.11.362",
            source="empty",
        )
    )

    final_phia_id = _clean(phia_id) or base.phia_id

    cookie_candidate = _clean(cookie_header) or _clean(session_cookie)
    final_session_cookie = (
        _extract_session_value(cookie_candidate) if cookie_candidate else base.session_cookie
    )

    auth_header_candidate = _clean(authorization_header) or _clean(bearer_token)
    final_bearer = (
        _extract_bearer_value(auth_header_candidate) if auth_header_candidate else base.bearer_token
    )

    final_platform = _clean(platform) or base.platform or "IOS_APP"
    final_platform_version = _clean(platform_version) or base.platform_version or "2.3.11.362"

    if not final_phia_id:
        raise RuntimeError("Missing phia_id for simulated session")
    if not final_session_cookie:
        raise RuntimeError("Missing session cookie for simulated session")

    return PhiaAuth(
        phia_id=final_phia_id,
        session_cookie=final_session_cookie,
        bearer_token=final_bearer,
        platform=final_platform,
        platform_version=final_platform_version,
        source=source,
    )


def _graphql_headers(auth: PhiaAuth, *, include_bearer: bool) -> dict[str, str]:
    headers: dict[str, str] = {
        "content-type": "application/json",
        "cookie": f"session={auth.session_cookie}",
        "x-phia-id": auth.phia_id,
        "x-platform": auth.platform or "IOS_APP",
        "x-platform-version": auth.platform_version or "2.3.11.362",
    }
    if include_bearer and auth.bearer_token and not _is_token_expired(auth.bearer_token):
        headers["authorization"] = f"Bearer {auth.bearer_token}"
    return headers


def _graphql_post(
    *,
    settings: Settings,
    auth: PhiaAuth,
    payload: dict[str, Any],
    include_bearer: bool,
) -> dict[str, Any]:
    verify_tls = not settings.phia_allow_insecure_tls
    headers = _graphql_headers(auth, include_bearer=include_bearer)
    response = None
    try:
        response = _post_json(
            url=settings.phia_graphql_url,
            headers=headers,
            payload=payload,
            timeout=20,
            verify=verify_tls,
        )
    except Exception as exc:
        if verify_tls and _is_cert_verify_error(exc):
            response = _post_json(
                url=settings.phia_graphql_url,
                headers=headers,
                payload=payload,
                timeout=20,
                verify=False,
            )
        else:
            raise RuntimeError(f"Request failed: {exc}") from exc

    parsed: dict[str, Any] = {}
    try:
        parsed = response.json()
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON response: {exc}") from exc

    if response.status_code >= 400:
        detail = parsed.get("message") if isinstance(parsed, dict) else None
        raise RuntimeError(
            f"HTTP {response.status_code}" + (f": {detail}" if detail else "")
        )

    gql_errors = parsed.get("errors") if isinstance(parsed, dict) else None
    if gql_errors:
        first_error = gql_errors[0] if isinstance(gql_errors, list) and gql_errors else gql_errors
        if isinstance(first_error, dict):
            message = _clean(str(first_error.get("message") or "GraphQL error"))
            raise RuntimeError(message or "GraphQL error")
        raise RuntimeError(_clean(str(first_error)) or "GraphQL error")

    data = parsed.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("GraphQL payload missing data object")
    return data


def _feed_query(
    *,
    settings: Settings,
    auth: PhiaAuth,
    operation_name: str,
    query: str,
    variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"operationName": operation_name, "query": query}
    if variables is not None:
        payload["variables"] = variables
    # Cookie auth is currently the most reliable for session captures.
    return _graphql_post(settings=settings, auth=auth, payload=payload, include_bearer=False)


def _sorted_editorials(editorials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        editorials,
        key=lambda item: (
            item.get("order") is None,
            item.get("order") if isinstance(item.get("order"), int) else 10_000_000,
            item.get("createdAt") or "",
        ),
    )


def fetch_mobile_feed(
    settings: Settings,
    *,
    auth_override: PhiaAuth | None = None,
    explore_feed_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    auth = auth_override or resolve_phia_auth(settings)
    if not auth.phia_id:
        raise RuntimeError("Missing phia_id")
    if not auth.session_cookie:
        raise RuntimeError("Missing session cookie")

    errors: list[str] = []

    def run_query(
        operation_name: str,
        query: str,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return _feed_query(
                settings=settings,
                auth=auth,
                operation_name=operation_name,
                query=query,
                variables=variables,
            )
        except Exception as exc:
            errors.append(f"{operation_name}: {exc}")
            return {}

    current_user_data = run_query("SwiftFetchCurrentUser", FETCH_CURRENT_USER_QUERY)
    notifications_data = run_query(
        "SwiftGetUserNotifications",
        FETCH_USER_NOTIFICATIONS_QUERY,
        variables={"phiaId": auth.phia_id},
    )
    campaign_data = run_query("FetchActiveCampaign", FETCH_ACTIVE_CAMPAIGN_QUERY)
    popular_searches_data = run_query("SwiftFetchPopularSearches", FETCH_POPULAR_SEARCHES_QUERY)
    trending_products_data = run_query("SwiftFetchTrendingProducts", FETCH_TRENDING_PRODUCTS_QUERY)
    trending_brands_data = run_query("SwiftFetchTrendingBrands", FETCH_TRENDING_BRANDS_QUERY)
    editorials_data = run_query("SwiftFetchEditorials", FETCH_EDITORIALS_QUERY)
    curated_types_data = run_query("SwiftFetchCuratedTypes", FETCH_CURATED_TYPES_QUERY)
    explore_feed_data = run_query(
        "SwiftFetchExploreFeed",
        FETCH_EXPLORE_FEED_QUERY,
        variables={"input": explore_feed_input or {}},
    )
    collections_data = run_query(
        "SwiftFetchCollectionProducts",
        FETCH_COLLECTION_PRODUCTS_QUERY,
        variables={"phiaId": auth.phia_id},
    )
    brands_data = run_query("SwiftFetchSavedBrands", FETCH_BRANDS_QUERY)

    current_user = current_user_data.get("currentUser")
    notifications = (notifications_data.get("getUserNotifications") or {}) if notifications_data else {}
    active_campaign = campaign_data.get("activeCampaign")

    popular_searches = popular_searches_data.get("popularSearches") or []
    trending_products = trending_products_data.get("trendingProducts") or []
    trending_brands = trending_brands_data.get("trendingBrands") or []
    editorials = editorials_data.get("editorials") or []
    curated_types = curated_types_data.get("curatedTypes") or []
    explore_feed = explore_feed_data.get("exploreFeed") or {}
    collections = ((collections_data.get("listCollections") or {}).get("collections") or [])
    all_brands = brands_data.get("brands") or []

    brand_image_lookup: dict[str, str] = {}
    for brand in trending_brands:
        name = _clean(str(brand.get("name") or "")).lower()
        image_url = _clean(str(brand.get("imgUrl") or ""))
        if name and image_url and name not in brand_image_lookup:
            brand_image_lookup[name] = image_url

    sorted_editorials = _sorted_editorials([item for item in editorials if isinstance(item, dict)])
    editorial_images = [
        _clean(str(item.get("imgUrl") or ""))
        for item in sorted_editorials
        if _clean(str(item.get("imgUrl") or ""))
    ]

    top_trends: list[dict[str, Any]] = []
    for idx, editorial in enumerate(sorted_editorials[:8]):
        top_trends.append(
            {
                "id": f"editorial-{idx + 1}",
                "title": _clean(str(editorial.get("title") or "")) or "Trending now",
                "image_url": _clean(str(editorial.get("imgUrl") or "")),
                "description": _clean(str(editorial.get("description") or "")),
                "order": editorial.get("order"),
                "created_at": _clean(str(editorial.get("createdAt") or "")),
            }
        )

    normalized_trending_products: list[dict[str, Any]] = []
    for idx, product in enumerate(trending_products):
        brand_name = _clean(str(product.get("brand") or ""))
        fallback_image = editorial_images[idx % len(editorial_images)] if editorial_images else ""
        image_url = brand_image_lookup.get(brand_name.lower()) or fallback_image
        raw_price = _clean(str(product.get("price") or ""))
        normalized_trending_products.append(
            {
                "id": _clean(str(product.get("productId") or f"trending-product-{idx + 1}")),
                "brand": brand_name or "Trending",
                "name": f"{brand_name} pick" if brand_name else "Trending pick",
                "price": raw_price or None,
                "view_count": product.get("viewCount"),
                "gender": _clean(str(product.get("gender") or "")),
                "is_trending": bool(product.get("isTrending")),
                "image_url": image_url or None,
            }
        )

    normalized_trending_brands: list[dict[str, Any]] = []
    for idx, brand in enumerate(trending_brands):
        normalized_trending_brands.append(
            {
                "id": _clean(str(brand.get("id") or f"trending-brand-{idx + 1}")),
                "name": _clean(str(brand.get("name") or "Unknown brand")) or "Unknown brand",
                "image_url": _clean(str(brand.get("imgUrl") or "")) or None,
                "logo_url": _clean(str(brand.get("logoUrl") or "")) or None,
                "description": _clean(str(brand.get("description") or "")) or None,
                "visit_count": brand.get("visitCount"),
            }
        )

    normalized_popular_searches: list[dict[str, Any]] = []
    for idx, item in enumerate(popular_searches):
        normalized_popular_searches.append(
            {
                "id": f"popular-search-{idx + 1}",
                "label": _clean(str(item.get("query") or "")),
                "image_url": _clean(str(item.get("imgUrl") or "")) or None,
                "category": _clean(str(item.get("category") or "")) or None,
                "rank": _clean(str(item.get("rank") or "")) or None,
                "is_trending": bool(item.get("isTrending")),
            }
        )

    normalized_curated_types: list[dict[str, Any]] = []
    for idx, item in enumerate(curated_types):
        normalized_curated_types.append(
            {
                "id": f"curated-type-{idx + 1}",
                "name": _clean(str(item.get("name") or "")),
                "image_url": _clean(str(item.get("imgUrl") or "")) or None,
            }
        )

    normalized_brands: list[dict[str, Any]] = []
    for idx, item in enumerate(all_brands[:40]):
        normalized_brands.append(
            {
                "id": _clean(str(item.get("id") or f"brand-{idx + 1}")),
                "name": _clean(str(item.get("displayName") or "")) or "Brand",
                "image_url": _clean(str(item.get("coverImageUrl") or "")) or None,
                "logo_url": _clean(str(item.get("logoUrl") or "")) or None,
                "description": _clean(str(item.get("description") or "")) or None,
                "is_trending": bool(item.get("isTrending")),
            }
        )

    saved_items: list[dict[str, Any]] = []
    for collection in collections:
        products = collection.get("savedProducts") or []
        for product in products:
            saved_items.append(
                {
                    "id": _clean(str(product.get("id") or "")),
                    "name": _clean(str(product.get("name") or "")) or "Saved item",
                    "brand": _clean(str(product.get("primaryBrandName") or "")) or None,
                    "image_url": _clean(str(product.get("imgUrl") or "")) or None,
                    "product_url": _clean(str(product.get("productUrl") or "")) or None,
                    "price_usd": _clean(str(product.get("priceUsd") or "")) or None,
                    "source_display_name": _clean(str(product.get("sourceDisplayName") or "")) or None,
                    "added_to_collection_at": _clean(str(product.get("addedToCollectionAt") or "")) or None,
                    "collection_id": _clean(str(collection.get("id") or "")) or None,
                    "collection_name": _clean(str(collection.get("name") or "")) or None,
                }
            )

    dedup_saved: list[dict[str, Any]] = []
    seen_saved_ids: set[str] = set()
    for item in saved_items:
        item_id = _clean(str(item.get("id") or ""))
        if not item_id or item_id in seen_saved_ids:
            continue
        seen_saved_ids.add(item_id)
        dedup_saved.append(item)

    return {
        "phia_id": auth.phia_id,
        "source": auth.source,
        "fetched_at_epoch_ms": int(time.time() * 1000),
        "errors": errors,
        "current_user": current_user or None,
        "notifications": {
            "notification_count": len(notifications.get("notifications") or []),
            "price_drop_alert_count": len(notifications.get("priceDropAlerts") or []),
            "saved_items_count": len(notifications.get("items") or []),
            "saved_collections_count": len(notifications.get("collections") or []),
        },
        "active_campaign": active_campaign or None,
        "top_trends": top_trends,
        "trending_products": normalized_trending_products,
        "trending_brands": normalized_trending_brands,
        "popular_searches": normalized_popular_searches,
        "curated_types": normalized_curated_types,
        "search_brands": normalized_brands,
        "saved_items": dedup_saved[:120],
        "saved_collections": collections,
        "explore_feed_meta": {
            "offset": explore_feed.get("offset"),
            "has_more": bool(explore_feed.get("hasMore")),
        },
    }


def products_google_shopping(
    *,
    settings: Settings,
    image_urls: list[str] | None = None,
    scraped_name: str | None = None,
    limit: int = 3,
    auth_override: PhiaAuth | None = None,
) -> list[dict[str, Any]]:
    auth = auth_override or resolve_phia_auth(settings)
    if not auth.phia_id:
        raise RuntimeError("Missing phia_id")
    if not auth.session_cookie:
        raise RuntimeError("Missing session cookie")

    cleaned_urls: list[str] = []
    seen: set[str] = set()
    for raw in image_urls or []:
        candidate = _clean(raw)
        if not candidate or not candidate.startswith(("http://", "https://")):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        cleaned_urls.append(candidate)

    cleaned_name = _clean(scraped_name)
    if not cleaned_urls and not cleaned_name:
        return []

    safe_limit = max(1, min(int(limit), 10))
    query_payload: dict[str, Any] = {
        "phiaId": auth.phia_id,
        "platform": auth.platform or "IOS_APP",
        "currencyCode": "USD",
        "origin": "TEXT_SEARCH",
    }
    if cleaned_name:
        query_payload["scrapedName"] = cleaned_name
    if cleaned_urls:
        query_payload["imageUrls"] = [cleaned_urls[0]]

    variables = {
        "query": query_payload,
        "filters": {
            "isAuthenticated": False,
            "isOnsale": False,
        },
        "options": {
            "currencyType": "USD",
            "includeOnlyScrapedData": False,
            "includeProductsOutsideOfSizeCategoryFilters": False,
            "isBrandPage": False,
            "limit": safe_limit,
            "newOnly": False,
            "offset": 0,
            "searchType": "MULTI_MODAL",
            "sortType": "BEST_MATCH",
            "usedOnly": False,
        },
    }

    data = _feed_query(
        settings=settings,
        auth=auth,
        operation_name="ProductsGoogleShoppingApi",
        query=PRODUCTS_GOOGLE_SHOPPING_QUERY,
        variables=variables,
    )

    products_payload = data.get("products") or {}
    results = products_payload.get("results") or []
    if not isinstance(results, list):
        return []

    out: list[dict[str, Any]] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        raw_price = entry.get("priceUsd")
        price_usd: float | None = None
        if isinstance(raw_price, (int, float)):
            price_usd = float(raw_price)
        elif isinstance(raw_price, str):
            try:
                price_usd = float(raw_price)
            except Exception:
                price_usd = None

        out.append(
            {
                "id": _clean(str(entry.get("id") or "")),
                "name": _clean(str(entry.get("name") or "")),
                "img_url": _clean(str(entry.get("imgUrl") or "")),
                "product_url": _clean(str(entry.get("productUrl") or "")),
                "primary_brand_name": _clean(str(entry.get("primaryBrandName") or "")),
                "source_display_name": _clean(str(entry.get("sourceDisplayName") or "")),
                "price_usd": price_usd,
            }
        )

    return out


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
            response = _post_json(
                url=settings.phia_graphql_url,
                headers=headers,
                payload=payload,
                timeout=25,
                verify=verify_tls,
            )
        except Exception as exc:
            if verify_tls and _is_cert_verify_error(exc):
                try:
                    response = _post_json(
                        url=settings.phia_graphql_url,
                        headers=headers,
                        payload=payload,
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
