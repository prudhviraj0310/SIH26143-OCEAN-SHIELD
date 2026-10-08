"""
Satellite Data Acquisition Utilities

Adapted from:
- ImageToDEM (Visualization/DEM2rgb.py) — Google Earth Engine satellite download
- AlgoRise (backend/app/services/detection.py) — Copernicus/ESA search

Provides automated satellite imagery discovery and metadata retrieval
from public APIs for SAR (Sentinel-1) and optical (Sentinel-2) scenes.
"""

from __future__ import annotations

import logging
import urllib.request
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Copernicus Data Space Ecosystem (CDSE) — free, no auth for metadata
COPERNICUS_ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1"


def search_sentinel1_scenes(
    lat: float,
    lon: float,
    *,
    radius_km: float = 50.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_results: int = 10,
    product_type: str = "GRD",
) -> Dict[str, Any]:
    """Search for Sentinel-1 SAR scenes near a coordinate.

    Uses Copernicus Data Space Ecosystem (CDSE) OData API.
    No authentication required for metadata queries.

    Args:
        lat, lon: Center coordinate
        radius_km: Search radius in kilometers
        start_date: Start date (ISO format, default: 30 days ago)
        end_date: End date (ISO format, default: today)
        max_results: Maximum number of results
        product_type: SAR product type (GRD, SLC, OCN)

    Returns:
        Search results with scene metadata
    """
    if start_date is None:
        start_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    # Build bounding box from center + radius
    deg_offset = radius_km / 111.32
    bbox = f"{lon - deg_offset},{lat - deg_offset},{lon + deg_offset},{lat + deg_offset}"

    # OData filter
    filter_parts = [
        f"Collection/Name eq 'SENTINEL-1'",
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}')",
        f"ContentDate/Start gt {start_date}T00:00:00.000Z",
        f"ContentDate/Start lt {end_date}T23:59:59.999Z",
        f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})')",
    ]

    filter_str = " and ".join(filter_parts)
    url = (f"{COPERNICUS_ODATA_BASE}/Products?"
           f"$filter={urllib.request.quote(filter_str)}"
           f"&$top={max_results}"
           f"&$orderby=ContentDate/Start desc")

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        scenes = []
        for product in data.get("value", []):
            scenes.append({
                "id": product.get("Id", ""),
                "name": product.get("Name", ""),
                "acquisition_date": product.get("ContentDate", {}).get("Start", ""),
                "platform": product.get("Name", "")[:3],  # S1A or S1B
                "product_type": product_type,
                "size_mb": round(product.get("ContentLength", 0) / (1024 * 1024), 1),
                "online": product.get("Online", False),
            })

        return {
            "status": "SUCCESS",
            "query": {
                "center": {"lat": lat, "lon": lon},
                "radius_km": radius_km,
                "date_range": f"{start_date} to {end_date}",
                "product_type": product_type,
            },
            "total_results": len(scenes),
            "scenes": scenes,
            "source": "Copernicus Data Space Ecosystem (CDSE)",
        }

    except Exception as e:
        logger.warning(f"Copernicus scene search failed: {e}")
        return {
            "status": "ERROR",
            "error": str(e),
            "scenes": [],
            "source": "Copernicus Data Space Ecosystem (CDSE)",
        }


def search_sentinel2_scenes(
    lat: float,
    lon: float,
    *,
    radius_km: float = 50.0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_cloud_cover: float = 30.0,
    max_results: int = 10,
) -> Dict[str, Any]:
    """Search for Sentinel-2 optical scenes near a coordinate.

    Adapted from ImageToDEM's Google Earth Engine approach,
    but uses the free Copernicus CDSE API instead.
    """
    if start_date is None:
        start_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    filter_parts = [
        f"Collection/Name eq 'SENTINEL-2'",
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {max_cloud_cover})",
        f"ContentDate/Start gt {start_date}T00:00:00.000Z",
        f"ContentDate/Start lt {end_date}T23:59:59.999Z",
        f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})')",
    ]

    filter_str = " and ".join(filter_parts)
    url = (f"{COPERNICUS_ODATA_BASE}/Products?"
           f"$filter={urllib.request.quote(filter_str)}"
           f"&$top={max_results}"
           f"&$orderby=ContentDate/Start desc")

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        scenes = []
        for product in data.get("value", []):
            scenes.append({
                "id": product.get("Id", ""),
                "name": product.get("Name", ""),
                "acquisition_date": product.get("ContentDate", {}).get("Start", ""),
                "platform": product.get("Name", "")[:3],
                "size_mb": round(product.get("ContentLength", 0) / (1024 * 1024), 1),
                "online": product.get("Online", False),
            })

        return {
            "status": "SUCCESS",
            "query": {
                "center": {"lat": lat, "lon": lon},
                "radius_km": radius_km,
                "date_range": f"{start_date} to {end_date}",
                "max_cloud_cover": max_cloud_cover,
            },
            "total_results": len(scenes),
            "scenes": scenes,
            "source": "Copernicus Data Space Ecosystem (CDSE)",
        }

    except Exception as e:
        logger.warning(f"Copernicus S2 search failed: {e}")
        return {
            "status": "ERROR",
            "error": str(e),
            "scenes": [],
        }


def get_ocean_color_data(
    lat: float,
    lon: float,
    *,
    date: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch ocean color / chlorophyll concentration from NASA OceanColor.

    Useful for lookalike discrimination (algae bloom vs oil spill).
    """
    if date is None:
        date = datetime.utcnow().strftime("%Y-%m-%d")

    # NASA OceanColor WMS
    url = (f"https://oceancolor.gsfc.nasa.gov/cgi/l3?"
           f"per=DAY&res=9km&prod=chlor_a&date={date}"
           f"&lat={lat}&lon={lon}")

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "OCEAN-SHIELD/1.0 (research screening)"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type:
                data = json.loads(resp.read().decode())
                return {"status": "SUCCESS", "data": data}
            else:
                return {
                    "status": "SUCCESS",
                    "note": "Ocean color data available but requires image processing",
                    "url": url,
                }
    except Exception as e:
        return {
            "status": "UNAVAILABLE",
            "error": str(e),
            "note": "NASA OceanColor API may be temporarily unavailable",
        }
