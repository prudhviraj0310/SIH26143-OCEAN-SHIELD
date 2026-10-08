"""
KDE Source-Region Extraction — Probabilistic Origin Area Estimation

Adapted from AlgoRise (Team_AlgoRise_OilSpill_detection/backend/app/services/hindcast.py)
Implements Gaussian KDE over backward-advected Lagrangian particles to extract
highest-density regions (HDR) as candidate source polygons.

Differences from AlgoRise:
- No OpenDrift dependency (uses OCEAN-SHIELD's native drift engine)
- Works with our existing particle trajectory format
- Returns GeoJSON-compatible polygons for frontend rendering
- Conservative labeling: KDE mass fraction, NOT a calibrated probability

Reference: AlgoRise hindcast.py + Shapely contour extraction
"""

from __future__ import annotations

import math
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0



def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert to a JSON-safe float (no NaN/Inf)."""
    try:
        f = float(val)
        if math.isfinite(f):
            return f
    except (TypeError, ValueError, OverflowError):
        pass
    return default


def lonlat_to_xy(lons: np.ndarray, lats: np.ndarray,
                 lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert WGS-84 lon/lat to local tangent-plane km coordinates.
    Adapted from AlgoRise's hindcast._lonlat_to_xy."""
    lat0_rad = np.radians(lat0)
    x = EARTH_RADIUS_KM * np.cos(lat0_rad) * np.radians(lons - lon0)
    y = EARTH_RADIUS_KM * np.radians(lats - lat0)
    return x, y


def xy_to_lonlat(x: np.ndarray, y: np.ndarray,
                 lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert local tangent-plane km coordinates back to WGS-84.
    Adapted from AlgoRise's hindcast._xy_to_lonlat."""
    lat0_rad = np.radians(lat0)
    lons = lon0 + np.degrees(x / (EARTH_RADIUS_KM * np.cos(lat0_rad)))
    lats = lat0 + np.degrees(y / EARTH_RADIUS_KM)
    return lons, lats


def extract_source_region(
    particle_lons: np.ndarray,
    particle_lats: np.ndarray,
    *,
    grid_resolution: int = 100,
    margin_km: float = 5.0,
    hdr_mass_fraction: float = 0.95,
) -> Dict[str, Any]:
    """Extract source-region probability contour from particle cloud.

    Uses 2D Gaussian KDE to estimate the probability density of the
    particle distribution and extracts the highest-density region
    containing hdr_mass_fraction of the total mass.

    Args:
        particle_lons: 1D array of particle longitudes
        particle_lats: 1D array of particle latitudes
        grid_resolution: Number of grid cells per axis for KDE evaluation
        margin_km: Margin beyond particle extent (km)
        hdr_mass_fraction: Target density mass fraction (0-1)

    Returns:
        Dict with source region polygons, centroid, and metadata
    """
    from scipy.stats import gaussian_kde

    # Remove NaN particles
    valid = ~np.isnan(particle_lons) & ~np.isnan(particle_lats)
    lons = particle_lons[valid]
    lats = particle_lats[valid]

    if len(lons) < 3:
        return {
            "status": "INSUFFICIENT_PARTICLES",
            "error": f"Only {len(lons)} valid particles — need at least 3 for KDE",
            "candidates": [],
        }

    # Metric projection centered on particle mean
    lon0 = float(np.mean(lons))
    lat0 = float(np.mean(lats))
    x, y = lonlat_to_xy(lons, lats, lon0, lat0)

    # 2D Gaussian KDE
    positions_xy = np.vstack([x, y])
    try:
        kde = gaussian_kde(positions_xy)
    except np.linalg.LinAlgError:
        return {
            "status": "KDE_FAILED",
            "error": "Particles are collinear — KDE singular matrix",
            "candidates": [],
        }

    # Evaluate on grid
    x_min, x_max = x.min() - margin_km, x.max() + margin_km
    y_min, y_max = y.min() - margin_km, y.max() + margin_km

    x_grid, y_grid = np.mgrid[
        x_min:x_max:complex(grid_resolution),
        y_min:y_max:complex(grid_resolution),
    ]
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    kde_values = kde(grid_coords)
    kde_surface = np.reshape(kde_values, x_grid.shape)

    # MAP (maximum a-posteriori density) estimate
    max_idx = np.argmax(kde_surface)
    max_idx_2d = np.unravel_index(max_idx, kde_surface.shape)
    map_x = float(x_grid[max_idx_2d])
    map_y = float(y_grid[max_idx_2d])
    map_lons, map_lats = xy_to_lonlat(np.array([map_x]), np.array([map_y]), lon0, lat0)

    # HDR threshold
    kde_normalized = kde_surface / np.sum(kde_surface)
    sorted_probs = np.sort(kde_normalized.flatten())[::-1]
    sorted_densities = np.sort(kde_surface.flatten())[::-1]
    cumulative_probs = np.cumsum(sorted_probs)

    idx_threshold = int(np.argmax(cumulative_probs >= hdr_mass_fraction))
    hdr_density_threshold = float(sorted_densities[idx_threshold])
    actual_mass = float(cumulative_probs[idx_threshold])

    # Contour extraction
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        cs = ax.contour(x_grid, y_grid, kde_surface, levels=[hdr_density_threshold])
        plt.close(fig)

        polygons = []
        for path in cs.get_paths():
            for poly_pts in path.to_polygons():
                poly_lons, poly_lats = xy_to_lonlat(
                    poly_pts[:, 0], poly_pts[:, 1], lon0, lat0
                )
                # Ensure closed ring
                if poly_lons[0] != poly_lons[-1] or poly_lats[0] != poly_lats[-1]:
                    poly_lons = np.append(poly_lons, poly_lons[0])
                    poly_lats = np.append(poly_lats, poly_lats[0])

                coords = [[float(lo), float(la)] for lo, la in zip(poly_lons, poly_lats)]
                if len(coords) >= 4:
                    polygons.append(coords)

    except Exception as e:
        logger.warning(f"Contour extraction failed: {e}")
        polygons = []

    # Build GeoJSON source regions
    candidates = []
    for i, coords in enumerate(polygons):
        # Sanitize coordinates
        safe_coords = [[_safe_float(c[0]), _safe_float(c[1])] for c in coords]
        candidates.append({
            "id": f"cr-{uuid.uuid4().hex[:12]}",
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [safe_coords],
            },
            "properties": {
                "kde_mass_fraction": _safe_float(round(actual_mass, 4)),
                "label": f"Source Candidate Region {i + 1}",
                "note": "KDE density mass fraction — NOT a calibrated probability",
            },
        })

    # Spread metrics
    spread_km = _safe_float(float(np.sqrt(np.var(x) + np.var(y))))

    return {
        "status": "SUCCESS",
        "centroid": {
            "lat": _safe_float(float(map_lats[0])),
            "lon": _safe_float(float(map_lons[0])),
        },
        "particle_count": int(len(lons)),
        "particle_spread_km": _safe_float(round(spread_km, 2)),
        "hdr_mass_fraction": _safe_float(round(actual_mass, 4)),
        "hdr_density_threshold": _safe_float(hdr_density_threshold),
        "candidates": candidates,
        "methodology": "2D Gaussian KDE with highest-density region extraction "
                        "(adapted from AlgoRise hindcast methodology)",
    }


def compute_spill_trajectory_overlap(
    predicted_coords: List[List[float]],
    observed_polygon: List[List[float]],
) -> Dict[str, float]:
    """Compute the overlap between a predicted trajectory and observed spill.

    Adapted from AlgoRise's forward counterfactual verification.
    Used to validate whether a suspect vessel's predicted release
    plausibly matches the observed oil slick.
    """
    try:
        from shapely.geometry import Polygon, LineString, MultiPoint

        # Build observed polygon
        if len(observed_polygon) < 3:
            return {"overlap_score": 0.0, "status": "INSUFFICIENT_POINTS"}

        obs_poly = Polygon(observed_polygon)
        if not obs_poly.is_valid:
            obs_poly = obs_poly.buffer(0)

        # Build predicted footprint (convex hull of trajectory)
        if len(predicted_coords) < 2:
            return {"overlap_score": 0.0, "status": "INSUFFICIENT_TRAJECTORY"}

        pred_points = MultiPoint([(c[0], c[1]) for c in predicted_coords])
        pred_hull = pred_points.convex_hull

        if pred_hull.geom_type == "Point" or pred_hull.geom_type == "LineString":
            # Trajectory is degenerate — buffer it
            pred_hull = pred_hull.buffer(0.01)

        # Compute overlap
        intersection = obs_poly.intersection(pred_hull)
        intersection_area = intersection.area
        union_area = obs_poly.union(pred_hull).area

        iou = intersection_area / max(union_area, 1e-10)
        containment = intersection_area / max(obs_poly.area, 1e-10)

        # Centroid distance
        obs_centroid = obs_poly.centroid
        pred_centroid = pred_hull.centroid
        dist_deg = math.sqrt(
            (obs_centroid.x - pred_centroid.x) ** 2 +
            (obs_centroid.y - pred_centroid.y) ** 2
        )
        dist_km = dist_deg * 111.32

        return {
            "overlap_score": round(iou, 4),
            "containment_fraction": round(containment, 4),
            "centroid_distance_km": round(dist_km, 2),
            "predicted_area_deg2": round(pred_hull.area, 6),
            "observed_area_deg2": round(obs_poly.area, 6),
            "status": "SUCCESS",
        }

    except ImportError:
        return {"overlap_score": 0.0, "status": "SHAPELY_NOT_AVAILABLE"}
    except Exception as e:
        return {"overlap_score": 0.0, "status": f"ERROR: {e}"}
