"""Truthful adapters for live maritime data sources.

This module never creates vessels, satellite scenes, oil incidents, or port
operations.  A provider being unavailable is returned to the caller as such;
that is safer than making a demo look operational.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
import json
import math
import os
import time
import threading
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import requests

# ── Simple TTL cache for slow external APIs ──────────────────────────────────
_cache_lock = threading.Lock()
_cache_store: Dict[str, Tuple[float, Any]] = {}  # key -> (expiry_ts, data)

def _cached(key: str, ttl_seconds: float, fetcher):
    """Return cached result if fresh, otherwise call fetcher and cache."""
    now = time.monotonic()
    with _cache_lock:
        if key in _cache_store and _cache_store[key][0] > now:
            return _cache_store[key][1]
    result = fetcher()
    with _cache_lock:
        _cache_store[key] = (now + ttl_seconds, result)
    return result


ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"
OPEN_METEO_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WPI_QUERY_URL = (
    "https://vcps.nga.mil/nauticalpubs-feature/rest/services/WPI/"
    "World_Port_Index_Viewer/MapServer/0/query"
)
AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"
REQUEST_HEADERS = {"User-Agent": "OCEAN-SHIELD/1.0 (live-data-adapter)"}

# An in-process history buffer gives the AIS lead scorer actual received pings,
# not interpolated tracks. Production deployments should replace this with a
# time-series store shared by all workers.
_AIS_TRACKS: Dict[int, Deque[Dict[str, Any]]] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime] = None) -> str:
    return (value or _now()).isoformat().replace("+00:00", "Z")


def _string(value: Any, default: str = "") -> str:
    if isinstance(value, list):
        return _string(value[0], default) if value else default
    return str(value).strip() if value not in (None, "") else default


def _attr(attributes: Dict[str, Any], *candidates: str, default: str = "") -> str:
    folded = {str(key).casefold(): value for key, value in attributes.items()}
    for name in candidates:
        value = folded.get(name.casefold())
        if value not in (None, ""):
            return _string(value, default)
    return default


def _point_time(value: Any) -> datetime:
    """Parse an AIS provider timestamp and fall back to receipt time."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = _string(value)
    if text:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00").replace(" UTC", "+00:00"))
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return _now()


def live_provider_status() -> Dict[str, Any]:
    """Expose exact data-source readiness without exposing secret values."""
    return {
        "retrieved_at_utc": _iso(),
        "providers": {
            "satellite_catalogue": {
                "status": "available_without_credentials",
                "provider": "NASA ASF DAAC Search API",
                "processing_note": "Catalogue metadata is public; SAR raster download and processing require an authorized scene source.",
            },
            "met_ocean": {
                "status": "available_without_credentials",
                "provider": "Open-Meteo Marine and Forecast APIs",
                "processing_note": "Returns current model/reanalysis conditions, not in-situ observations.",
            },
            "ais": {
                "status": "configured" if os.getenv("AISSTREAM_API_KEY") else "not_configured",
                "provider": "AISStream WebSocket",
                "required_environment": [] if os.getenv("AISSTREAM_API_KEY") else ["AISSTREAM_API_KEY"],
            },
            "ports": {
                "status": "available_without_credentials",
                "provider": "NGA World Port Index",
                "processing_note": "A monthly refreshed reference catalogue, not a live port-operations feed.",
            },
            "oil_spill_incidents": {
                "status": "configured" if os.getenv("OIL_SPILL_FEED_URL") else "not_configured",
                "provider": "Configured authority GeoJSON/JSON feed",
                "required_environment": [] if os.getenv("OIL_SPILL_FEED_URL") else ["OIL_SPILL_FEED_URL"],
                "processing_note": "No public nationwide authoritative Indian live spill-alert feed is assumed by this application.",
            },
        },
    }


def fetch_live_satellite_passes(
    lat: float,
    lon: float,
    days_back: int = 14,
    platform: str = "SENTINEL-1",
    max_results: int = 20,
) -> Dict[str, Any]:
    """Query real ASF catalogue metadata; never invent an orbital pass."""
    start = (_now().replace(tzinfo=None) - timedelta(days=max(1, min(days_back, 60)))).strftime("%Y-%m-%d")
    params = {
        "platform": platform,
        "intersectsWith": f"POINT({lon:.5f} {lat:.5f})",
        "start": start,
        "output": "jsonlite",
        "maxResults": max(1, min(max_results, 100)),
    }
    response_base = {
        "provider": "NASA Alaska Satellite Facility DAAC Search API",
        "provider_url": "https://docs.asf.alaska.edu/api/",
        "coordinates": {"lat": lat, "lon": lon},
        "retrieved_at_utc": _iso(),
        "is_live_catalogue": False,
        "passes": [],
        "latest_pass": None,
    }
    try:
        response = requests.get(ASF_SEARCH_URL, params=params, headers=REQUEST_HEADERS, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError, OSError) as exc:
        return {
            **response_base,
            "status": "unavailable",
            "message": f"ASF catalogue request failed: {type(exc).__name__}",
        }

    raw_results = payload.get("results", payload if isinstance(payload, list) else [])
    unique_passes: Dict[str, Dict[str, Any]] = {}
    for item in raw_results if isinstance(raw_results, list) else []:
        if not isinstance(item, dict):
            continue
        candidate = {
            "granule_name": _string(item.get("granuleName")),
            "platform": _string(item.get("platform"), "Sentinel-1"),
            "sensor": _string(item.get("sensor"), "C-SAR"),
            "acquisition_time_utc": _string(item.get("startTime")),
            "stop_time_utc": _string(item.get("stopTime")),
            "flight_direction": _string(item.get("flightDirection"), "not supplied"),
            "beam_mode": _string(item.get("beamMode"), "not supplied"),
            "polarization": _string(item.get("polarization"), "not supplied"),
            "browse_url": _string(item.get("browse")),
            "size_mb": item.get("sizeMB"),
            "orbit": _string(item.get("orbit"), "not supplied"),
            "path_number": _string(item.get("pathNumber"), "not supplied"),
            "source_product_url": _string(item.get("url")),
        }
        if not candidate["granule_name"]:
            continue
        prior = unique_passes.get(candidate["granule_name"])
        if prior is None or (candidate["size_mb"] or 0) > (prior["size_mb"] or 0):
            unique_passes[candidate["granule_name"]] = candidate

    def product_priority(scene: Dict[str, Any]) -> tuple[int, str]:
        name = scene["granule_name"]
        # GRD is the usual starting point for spill screening; the remaining
        # source products are still returned, merely sorted after it.
        level = 0 if "GRD" in name else 1 if "SLC" in name else 2 if "RAW" in name else 3
        return level, scene.get("acquisition_time_utc") or ""

    passes = sorted(unique_passes.values(), key=product_priority)
    return {
        **response_base,
        "status": "available",
        "is_live_catalogue": True,
        "passes": passes,
        "latest_pass": passes[0] if passes else None,
        "message": "No matching Sentinel-1 pass was returned for this query." if not passes else "Catalogue query completed.",
    }


def _current_values(payload: Dict[str, Any], fields: Iterable[str]) -> Dict[str, Any]:
    current = payload.get("current") or {}
    if current:
        return {field: current.get(field) for field in fields} | {"time": current.get("time")}
    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        return {field: None for field in fields} | {"time": None}
    # Open-Meteo returns time-ordered data. Use the sample closest to fetch time.
    now = _now().replace(tzinfo=None)
    parsed = []
    for index, raw_time in enumerate(hourly["time"]):
        try:
            parsed.append((abs((datetime.fromisoformat(raw_time).replace(tzinfo=None) - now).total_seconds()), index))
        except (TypeError, ValueError):
            continue
    index = min(parsed)[1] if parsed else 0
    return {field: (hourly.get(field) or [None])[index] if len(hourly.get(field) or []) > index else None for field in fields} | {"time": hourly["time"][index]}


def _wind_components(speed_ms: float, direction_from_degrees: float) -> Dict[str, float]:
    radians = math.radians(direction_from_degrees)
    return {"u_east_ms": round(-speed_ms * math.sin(radians), 3), "v_north_ms": round(-speed_ms * math.cos(radians), 3)}


def _current_components(speed_ms: float, direction_to_degrees: float) -> Dict[str, float]:
    radians = math.radians(direction_to_degrees)
    return {"u_east_ms": round(speed_ms * math.sin(radians), 3), "v_north_ms": round(speed_ms * math.cos(radians), 3)}


def fetch_live_ocean_weather(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch current modelled marine and weather conditions without defaults."""
    marine_fields = ["wave_height", "ocean_current_velocity", "ocean_current_direction"]
    weather_fields = ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]
    marine_payload: Dict[str, Any] = {}
    weather_payload: Dict[str, Any] = {}
    marine_error = weather_error = None
    try:
        marine_response = requests.get(
            OPEN_METEO_MARINE_URL,
            params={"latitude": lat, "longitude": lon, "current": ",".join(marine_fields), "hourly": ",".join(marine_fields), "timezone": "GMT", "cell_selection": "sea"},
            headers=REQUEST_HEADERS,
            timeout=15,
        )
        marine_response.raise_for_status()
        marine_payload = marine_response.json()
    except (requests.RequestException, ValueError, OSError) as exc:
        marine_error = type(exc).__name__
    try:
        weather_response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={"latitude": lat, "longitude": lon, "current": ",".join(weather_fields), "hourly": ",".join(weather_fields), "timezone": "GMT"},
            headers=REQUEST_HEADERS,
            timeout=15,
        )
        weather_response.raise_for_status()
        weather_payload = weather_response.json()
    except (requests.RequestException, ValueError, OSError) as exc:
        weather_error = type(exc).__name__

    marine = _current_values(marine_payload, marine_fields)
    weather = _current_values(weather_payload, weather_fields)
    current = None
    wind = None
    if marine.get("ocean_current_velocity") is not None and marine.get("ocean_current_direction") is not None:
        speed = float(marine["ocean_current_velocity"]) / 3.6  # default Open-Meteo unit: km/h
        direction = float(marine["ocean_current_direction"])
        current = {
            "speed_ms": round(speed, 3),
            "speed_knots": round(speed * 1.94384, 2),
            "heading_degrees": direction,
            **_current_components(speed, direction),
            "observed_or_model_time_utc": marine.get("time"),
        }
    if weather.get("wind_speed_10m") is not None and weather.get("wind_direction_10m") is not None:
        speed = float(weather["wind_speed_10m"]) / 3.6
        direction = float(weather["wind_direction_10m"])
        wind = {
            "speed_ms": round(speed, 3),
            "speed_knots": round(speed * 1.94384, 2),
            "direction_from_degrees": direction,
            "gusts_ms": round(float(weather["wind_gusts_10m"]) / 3.6, 3) if weather.get("wind_gusts_10m") is not None else None,
            **_wind_components(speed, direction),
            "observed_or_model_time_utc": weather.get("time"),
        }
    available = bool(current or wind)
    return {
        "status": "available" if available else "unavailable",
        "provider": "Open-Meteo Marine and Forecast APIs",
        "provider_url": "https://open-meteo.com/en/docs/marine-weather-api",
        "retrieved_at_utc": _iso(),
        "coordinates": {"lat": lat, "lon": lon},
        "is_live_model_data": available,
        "surface_current": current,
        "surface_wind_10m": wind,
        "wave_height_m": marine.get("wave_height"),
        "errors": {"marine": marine_error, "weather": weather_error},
        "notice": "These are current model/reanalysis values selected by Open-Meteo, not a substitute for validated in-situ current measurements.",
    }


def _fetch_wpi_uncached(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
) -> Dict[str, Any]:
    """Raw fetch from NGA World Port Index — called only on cache miss."""
    params = {
        "where": "1=1",
        "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": 3000,
    }
    base = {
        "provider": "NGA World Port Index",
        "provider_url": "https://msi.nga.mil/Publications/WPI",
        "retrieved_at_utc": _iso(),
        "data_classification": "reference catalogue refreshed monthly; not live port operations",
        "is_live_operations": False,
        "ports": [],
    }
    try:
        response = requests.get(WPI_QUERY_URL, params=params, headers=REQUEST_HEADERS, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError, OSError) as exc:
        return {**base, "status": "unavailable", "message": f"NGA WPI request failed: {type(exc).__name__}"}

    ports: List[Dict[str, Any]] = []
    for feature in payload.get("features", []):
        properties = feature.get("properties") or feature.get("attributes") or {}
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "Point":
            coordinates = geometry.get("coordinates") or []
            lon, lat = (coordinates + [None, None])[:2]
        else:
            lon, lat = geometry.get("x"), geometry.get("y")
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        name = _attr(properties, "port_name", "portname", "name", "port", default="Unnamed port")
        ports.append({
            "id": _attr(properties, "port_number", "port_no", "id", "objectid", default=f"{lat:.5f}:{lon:.5f}"),
            "name": name,
            "lat": lat,
            "lon": lon,
            "country": _attr(properties, "country", "country_name"),
            "harbor_size": _attr(properties, "harbor_size", "harbor_size_code"),
            "shelter": _attr(properties, "shelter"),
            "facility_type": _attr(properties, "port_type", "type"),
        })
    return {**base, "status": "available", "ports": ports, "count": len(ports)}


def fetch_world_port_index(
    min_lon: float = 66.0,
    min_lat: float = 5.0,
    max_lon: float = 100.0,
    max_lat: float = 38.0,
) -> Dict[str, Any]:
    """Return WPI records with a 10-minute server-side cache to avoid repeated slow external calls."""
    cache_key = f"wpi:{min_lon}:{min_lat}:{max_lon}:{max_lat}"
    return _cached(cache_key, 600, lambda: _fetch_wpi_uncached(min_lon, min_lat, max_lon, max_lat))


def _incident_from_feature(feature: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    properties = feature.get("properties") or feature.get("attributes") or feature
    geometry = feature.get("geometry") or {}
    if geometry.get("type") == "Point":
        coords = geometry.get("coordinates") or []
        lon, lat = (coords + [None, None])[:2]
    else:
        lat, lon = properties.get("lat"), properties.get("lon")
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    return {
        "id": _attr(properties, "id", "event_id", "incident_id", default=f"{lat:.5f}:{lon:.5f}"),
        "name": _attr(properties, "name", "title", "incident", default="Oil-spill incident"),
        "lat": lat,
        "lon": lon,
        "reported_at_utc": _attr(properties, "reported_at", "timestamp", "date", "time"),
        "severity": _attr(properties, "severity", "alert_level"),
        "status": _attr(properties, "status"),
        "source_url": _attr(properties, "url", "source_url"),
    }


def _fetch_incidents_uncached() -> Dict[str, Any]:
    """Raw incident fetch — called only on cache miss."""
    feed_url = os.getenv("OIL_SPILL_FEED_URL")
    base = {
        "provider": "Authority-configured oil-spill incident feed",
        "retrieved_at_utc": _iso(),
        "is_live": False,
        "incidents": [],
    }
    if not feed_url:
        return {
            **base,
            "status": "not_configured",
            "message": "Set OIL_SPILL_FEED_URL to an authorized GeoJSON or JSON incident feed. No incident marker is fabricated.",
        }
    headers = dict(REQUEST_HEADERS)
    if os.getenv("OIL_SPILL_FEED_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['OIL_SPILL_FEED_TOKEN']}"
    try:
        response = requests.get(feed_url, headers=headers, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {**base, "status": "unavailable", "message": f"Incident-feed request failed: {type(exc).__name__}"}
    features = payload.get("features", payload.get("incidents", payload if isinstance(payload, list) else []))
    incidents = [item for feature in features if isinstance(feature, dict) if (item := _incident_from_feature(feature))]
    return {**base, "status": "available", "is_live": True, "incidents": incidents, "count": len(incidents)}


def fetch_live_oil_spill_incidents() -> Dict[str, Any]:
    """Incident feed with 2-minute cache (not_configured results cached for 30s)."""
    return _cached("incidents", 120, _fetch_incidents_uncached)


def _ais_message_to_point(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    meta = message.get("MetaData") or {}
    report = (message.get("Message") or {}).get("PositionReport") or {}
    try:
        mmsi = int(meta.get("MMSI") or report.get("UserID"))
        lat = float(meta.get("latitude"))
        lon = float(meta.get("longitude"))
    except (TypeError, ValueError):
        return None
    observed_at = _point_time(meta.get("time_utc") or meta.get("time") or message.get("Timestamp"))
    return {
        "mmsi": mmsi,
        "vessel_name": _string(meta.get("ShipName"), f"MMSI-{mmsi}"),
        "lat": lat,
        "lon": lon,
        "sog_knots": round(float(report.get("Sog") or 0.0) * 0.1, 2),
        "cog_degrees": round(float(report.get("Cog") or 0.0) * 0.1, 1),
        "heading": report.get("TrueHeading"),
        "status": _string(report.get("NavigationalStatus"), "not supplied"),
        "observed_at": _iso(observed_at),
    }


def _record_ais_points(points: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now = _now()
    for point in points:
        history = _AIS_TRACKS.setdefault(point["mmsi"], deque(maxlen=240))
        if not history or history[-1].get("observed_at") != point["observed_at"]:
            history.append(point)
    cutoff = now.timestamp() - 2 * 3600
    vessels = []
    for mmsi, history in list(_AIS_TRACKS.items()):
        recent = [point for point in history if _point_time(point["observed_at"]).timestamp() >= cutoff]
        if not recent:
            _AIS_TRACKS.pop(mmsi, None)
            continue
        trajectory = []
        for point in recent:
            timestamp = _point_time(point["observed_at"])
            trajectory.append({
                "observed_at": _iso(timestamp),
                "relative_time_hours": round((timestamp - now).total_seconds() / 3600.0, 4),
                "lat": point["lat"], "lon": point["lon"],
                "sog_knots": point["sog_knots"], "cog_degrees": point["cog_degrees"],
            })
        latest = trajectory[-1]
        vessels.append({
            "mmsi": mmsi,
            "imo": None,
            "vessel_name": recent[-1]["vessel_name"],
            "call_sign": "Not supplied by AIS position report",
            "flag_state": "Not supplied by AIS position report",
            "vessel_type": "Not supplied by AIS position report",
            "trajectory": trajectory,
            "current_position": latest,
            "data_origin": "AISStream live PositionReport",
            "is_live_ais": True,
        })
    return sorted(vessels, key=lambda vessel: vessel["mmsi"])


async def fetch_live_ais_traffic(lat: float, lon: float, radius_nm: float = 25.0, duration_s: float = 8.0) -> Dict[str, Any]:
    """Collect real AIS position reports from AISStream for a bounded time window."""
    api_key = os.getenv("AISSTREAM_API_KEY")
    base = {
        "provider": "AISStream WebSocket",
        "provider_url": "https://aisstream.io/documentation",
        "retrieved_at_utc": _iso(),
        "coordinates": {"lat": lat, "lon": lon},
        "is_live": False,
        "vessels": [],
    }
    if not api_key:
        return {**base, "status": "not_configured", "message": "Set AISSTREAM_API_KEY server-side to receive live AIS reports."}
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        return {**base, "status": "unavailable", "message": "The websockets package is not installed."}

    delta_lat = min(max(radius_nm, 1.0), 120.0) / 60.0
    delta_lon = delta_lat / max(math.cos(math.radians(lat)), 0.1)
    subscription = {
        "APIKey": api_key,
        "BoundingBoxes": [[[lat + delta_lat, lon - delta_lon], [lat - delta_lat, lon + delta_lon]]],
        "FilterMessageTypes": ["PositionReport"],
    }
    points: List[Dict[str, Any]] = []
    deadline = asyncio.get_running_loop().time() + min(max(duration_s, 1.0), 20.0)
    try:
        async with connect(AISSTREAM_URL, open_timeout=10, close_timeout=3) as socket:
            await socket.send(json.dumps(subscription))
            while asyncio.get_running_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=max(0.1, deadline - asyncio.get_running_loop().time()))
                    parsed = json.loads(raw)
                except asyncio.TimeoutError:
                    break
                except (TypeError, ValueError):
                    continue
                point = _ais_message_to_point(parsed)
                if point:
                    points.append(point)
    except Exception as exc:
        return {**base, "status": "unavailable", "message": f"AISStream connection failed: {type(exc).__name__}"}
    vessels = _record_ais_points(points)
    return {
        **base,
        "status": "available" if points else "no_messages",
        "is_live": bool(points),
        "received_position_reports": len(points),
        "vessels": vessels,
        "message": "No AIS position report arrived during the bounded collection window." if not points else "Live AIS position reports received.",
    }
