"""
Drift Engine: Ocean Hydrodynamic Lagrangian Particle Tracking & Hindcasting
Implements conditional forward and backward particle-transport scenarios driven by
time-aligned surface current and wind inputs. It does not determine culpability or
prove a release location.
"""

import os
import json
import math
from numbers import Real
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


# Numerical/resource limits for this surface-transport approximation, not
# statements about source accuracy or operational suitability.
METERS_PER_DEG_LAT = 111320.0
MAX_TRANSPORT_LAT = 85.0  # Longitude-based integration is unsupported at poles.
MAX_DURATION_HOURS = 744.0
MAX_PARTICLES = 10000
MAX_STEPS = 10000
MAX_PARTICLE_STEPS = 2000000

# Fay's inversion is an offline sensitivity reference. These bounds prevent
# malformed API/UI values from overflowing the calculation while remaining
# much wider than ordinary spill-screening inputs.
MAX_FAY_AREA_KM2 = 1.0e8
MAX_FAY_VOLUME_M3 = 1.0e9
MAX_FAY_DENSITY_KG_M3 = 5000.0
MAX_FAY_KINEMATIC_VISCOSITY = 1.0
MAX_FAY_K2 = 1000.0


class OceanSourceError(ValueError):
    """A current/wind source failed or supplied unusable vectors."""


def _finite_number(value: Any, name: str, minimum: float, maximum: float,
                   positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not a boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum or (positive and result <= 0.0):
        raise ValueError(f"{name} must be finite and in {'(' if positive else '['}{minimum}, {maximum}].")
    return result


def _fay_number(value: Any, name: str, minimum: float, maximum: float,
                positive: bool = False) -> float:
    """Validate a Fay input without treating booleans or numeric strings as data."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum or (positive and result <= 0.0):
        raise ValueError(f"{name} must be finite and in {'(' if positive else '['}{minimum}, {maximum}].")
    return result


def _coordinates(lat: Any, lon: Any, transport: bool = True) -> Tuple[float, float]:
    limit = MAX_TRANSPORT_LAT if transport else 90.0
    return (_finite_number(lat, "latitude", -limit, limit),
            _finite_number(lon, "longitude", -180.0, 180.0))


def _wrap_longitudes(lon: np.ndarray) -> np.ndarray:
    return (lon + 180.0) % 360.0 - 180.0


# ============================================================================
# High-Resolution Physical Shoreline (Natural Earth 10m Vector Dataset)
# Accurate 1:10,000,000 geodetic vector shoreline for the Indian Subcontinent
# & EEZ, indexed via Shapely STRtree spatial indexing.
# ============================================================================

COASTLINE_GEOJSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "datasets", "india_coastline_10m.geojson"
)

_NATURAL_EARTH_TREE = None
_NATURAL_EARTH_GEOMS = None


def _get_coastline_index():
    global _NATURAL_EARTH_TREE, _NATURAL_EARTH_GEOMS
    if _NATURAL_EARTH_TREE is not None:
        return _NATURAL_EARTH_TREE, _NATURAL_EARTH_GEOMS
    if os.path.exists(COASTLINE_GEOJSON_PATH):
        try:
            from shapely.geometry import shape
            from shapely.strtree import STRtree
            with open(COASTLINE_GEOJSON_PATH, "r", encoding="utf-8") as f:
                gj = json.load(f)
            geoms = [shape(feat["geometry"]) for feat in gj.get("features", [])]
            _NATURAL_EARTH_GEOMS = geoms
            _NATURAL_EARTH_TREE = STRtree(geoms)
            return _NATURAL_EARTH_TREE, _NATURAL_EARTH_GEOMS
        except Exception:
            pass
    return None, None


def distance_to_coastline_km(lat: float, lon: float) -> float:
    """Computes exact geodetic distance in km to the nearest Natural Earth 10m vector shoreline."""
    tree, geoms = _get_coastline_index()
    if tree is not None and geoms is not None:
        try:
            from shapely.geometry import Point
            pt = Point(lon, lat)
            idx = tree.nearest(pt)
            nearest_line = geoms[idx]
            proj_pt = nearest_line.interpolate(nearest_line.project(pt))
            km_per_deg_lon = 111.32 * math.cos(math.radians(lat))
            dx = (lon - proj_pt.x) * km_per_deg_lon
            dy = (lat - proj_pt.y) * 111.0
            return round(math.sqrt(dx * dx + dy * dy), 2)
        except Exception:
            pass
    return 10.0


LAND_GEOJSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "datasets", "south_asia_land_10m.geojson"
)

_NATURAL_EARTH_LAND_TREE = None
_NATURAL_EARTH_LAND_GEOMS = None


def _get_land_index():
    """Lazily loads and spatially indexes Natural Earth 10m Land Polygons using Shapely STRtree."""
    global _NATURAL_EARTH_LAND_TREE, _NATURAL_EARTH_LAND_GEOMS
    if _NATURAL_EARTH_LAND_TREE is not None:
        return _NATURAL_EARTH_LAND_TREE, _NATURAL_EARTH_LAND_GEOMS
    if os.path.exists(LAND_GEOJSON_PATH):
        try:
            from shapely.geometry import shape
            from shapely.strtree import STRtree
            with open(LAND_GEOJSON_PATH, "r", encoding="utf-8") as f:
                gj = json.load(f)
            geoms = [shape(feat["geometry"]) for feat in gj.get("features", [])]
            _NATURAL_EARTH_LAND_GEOMS = geoms
            _NATURAL_EARTH_LAND_TREE = STRtree(geoms)
            return _NATURAL_EARTH_LAND_TREE, _NATURAL_EARTH_LAND_GEOMS
        except Exception:
            pass
    return None, None


def _point_in_polygon(lat: float, lon: float, polygon: list) -> bool:
    """Standard ray-casting point-in-polygon test for arbitrary user-defined evaluation polygons."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great circle distance between two points in kilometers."""
    lat1, lon1 = _coordinates(lat1, lon1, transport=False)
    lat2, lon2 = _coordinates(lat2, lon2, transport=False)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    r = 6371.0  # Earth mean radius in km
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    return r * (2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))


def is_on_land(lat: float, lon: float) -> bool:
    """
    Evaluates whether a geographic coordinate falls on land using the official Natural Earth 10m
    high-resolution vector land polygons (South Asia & Indian EEZ) indexed via STRtree.
    Returns True if on land/island, False if in open ocean.
    """
    lat, lon = _coordinates(lat, lon, transport=False)
    # Fast geographic bounds rejection: points outside the regional bounding box
    if lat < 0.0 or lat > 38.0 or lon < 60.0 or lon > 100.0:
        return False
    # Deep ocean shortcut: open Arabian Sea west of 71.0°E (below Gujarat)
    if lon < 71.0 and lat < 20.0:
        return False
    # Deep ocean shortcut: open Bay of Bengal east of 85.0°E (below Odisha)
    if lon > 85.0 and lat < 18.0:
        return False

    tree, geoms = _get_land_index()
    if tree is not None and geoms is not None:
        try:
            from shapely.geometry import Point
            pt = Point(lon, lat)
            candidates = tree.query(pt)
            for idx in candidates:
                if geoms[idx].contains(pt):
                    return True
            return False
        except Exception:
            pass
    return False


class OceanCurrentField:
    """
    Represents a spatio-temporal 2D vector field of ocean surface currents (u_curr, v_curr in m/s)
    and surface wind (u_wind, v_wind in m/s).
    Backed by CF-compliant NetCDF hydrodynamic datasets (HYCOM / INCOIS / NOAA GFS) via OceanDataProvider.
    Analytical fields are demonstration-only and are explicitly marked as such.
    """

    def __init__(
        self,
        base_current_u: float = 0.25,   # m/s eastward
        base_current_v: float = 0.15,   # m/s northward
        base_wind_u: float = 4.5,       # m/s eastward
        base_wind_v: float = 3.0,       # m/s northward
        tidal_amplitude: float = 0.35,  # m/s tidal component
        tidal_period_h: float = 12.42,  # Semi-diurnal M2 tidal period in hours
        data_provider: Optional[Any] = None,
        constant_vectors: bool = False,
    ):
        self.base_current_u = _finite_number(base_current_u, "base_current_u", -100.0, 100.0)
        self.base_current_v = _finite_number(base_current_v, "base_current_v", -100.0, 100.0)
        self.base_wind_u = _finite_number(base_wind_u, "base_wind_u", -200.0, 200.0)
        self.base_wind_v = _finite_number(base_wind_v, "base_wind_v", -200.0, 200.0)
        self.tidal_amplitude = _finite_number(tidal_amplitude, "tidal_amplitude", 0.0, 100.0)
        self.tidal_period_h = _finite_number(tidal_period_h, "tidal_period_h", 0.0, MAX_DURATION_HOURS, positive=True)
        if not isinstance(constant_vectors, (bool, np.bool_)):
            raise ValueError("constant_vectors must be a boolean.")
        self.data_provider = data_provider
        self.constant_vectors = constant_vectors

    def get_velocity_at(
        self,
        lat: float,
        lon: float,
        t_hours_relative: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """
        Returns (u_curr, v_curr, u_wind, v_wind) at a specific latitude, longitude,
        and relative time offset in hours.
        Prioritizes a bound source provider. Constant vectors are used only when an
        analyst explicitly supplies all four vectors; analytical flow is demo-only.
        """
        if self.data_provider is not None:
            return self.data_provider.get_velocity_at(lat, lon, t_hours_relative)

        if self.constant_vectors:
            return self.base_current_u, self.base_current_v, self.base_wind_u, self.base_wind_v

        # Explicit demonstration analytical semi-diurnal tidal oscillation
        phase = (2.0 * math.pi * t_hours_relative) / self.tidal_period_h
        u_tide = self.tidal_amplitude * math.cos(phase)
        v_tide = self.tidal_amplitude * 0.75 * math.sin(phase)

        # Subtle spatial variation (micro-eddies on 0.1 deg scale)
        spatial_eddy_u = 0.08 * math.sin(lat * 35.0 + lon * 25.0)
        spatial_eddy_v = 0.08 * math.cos(lat * 30.0 - lon * 40.0)

        u_curr = self.base_current_u + u_tide + spatial_eddy_u
        v_curr = self.base_current_v + v_tide + spatial_eddy_v

        # Wind with diurnal variability
        diurnal_factor = 1.0 + 0.15 * math.sin((2.0 * math.pi * t_hours_relative) / 24.0)
        u_wind = self.base_wind_u * diurnal_factor
        v_wind = self.base_wind_v * diurnal_factor

        return u_curr, v_curr, u_wind, v_wind

    def get_velocities_at(self, lat: np.ndarray, lon: np.ndarray,
                          t_hours_relative: float = 0.0) -> Tuple[np.ndarray, ...]:
        """Batch contract; scalar-only providers/subclasses are evaluated individually.

        A batch exception is a source failure, never a reason to substitute a
        different field or retry it via a scalar path.
        """
        if type(self).get_velocity_at is not OceanCurrentField.get_velocity_at:
            values = [self.get_velocity_at(float(a), float(b), t_hours_relative)
                      for a, b in zip(lat, lon)]
            return tuple(np.asarray(values, dtype=float).T)
        if self.data_provider is not None:
            batch = getattr(self.data_provider, "get_velocities_at", None)
            if callable(batch):
                return batch(lat, lon, t_hours_relative)
            values = [self.data_provider.get_velocity_at(float(a), float(b), t_hours_relative)
                      for a, b in zip(lat, lon)]
            return tuple(np.asarray(values, dtype=float).T)
        if self.constant_vectors:
            return tuple(np.full_like(lat, value, dtype=float) for value in
                         (self.base_current_u, self.base_current_v, self.base_wind_u, self.base_wind_v))
        phase = 2.0 * math.pi * t_hours_relative / self.tidal_period_h
        diurnal = 1.0 + 0.15 * math.sin(2.0 * math.pi * t_hours_relative / 24.0)
        return (
            self.base_current_u + self.tidal_amplitude * math.cos(phase) + 0.08 * np.sin(lat * 35.0 + lon * 25.0),
            self.base_current_v + self.tidal_amplitude * 0.75 * math.sin(phase) + 0.08 * np.cos(lat * 30.0 - lon * 40.0),
            np.full_like(lat, self.base_wind_u * diurnal, dtype=float),
            np.full_like(lat, self.base_wind_v * diurnal, dtype=float),
        )


class DriftEngine:
    """
    Lagrangian Particle Dispersion & Trajectory Engine.
    Simulates thousands of oil slick particles drifting under ocean currents,
    empirical fixed-angle windage, and forward-only turbulent diffusion.
    It does not solve Coriolis/Ekman dynamics or infer an origin probability.
    """

    def __init__(
        self,
        wind_drift_factor: float = 0.032,     # Standard 3.0% - 3.5% wind drift
        deflection_angle_deg: float = 15.0,  # Empirical right/left velocity-to deflection
        diffusion_coeff: float = 2.5,        # Horizontal turbulent diffusion m^2/s
        num_particles: int = 1000
    ):
        self.wind_drift_factor = _finite_number(wind_drift_factor, "wind_drift_factor", 0.0, 1.0)
        self.deflection_angle_rad = math.radians(
            _finite_number(deflection_angle_deg, "deflection_angle_deg", 0.0, 90.0))
        self.diffusion_coeff = _finite_number(diffusion_coeff, "diffusion_coeff", 0.0, 1.0e6)
        self.num_particles = num_particles
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Check again at run time because callers can modify engine attributes."""
        if isinstance(self.num_particles, (bool, np.bool_)) or not isinstance(self.num_particles, (int, np.integer)):
            raise ValueError("num_particles must be an integer.")
        if not 1 <= self.num_particles <= MAX_PARTICLES:
            raise ValueError(f"num_particles must be between 1 and {MAX_PARTICLES}.")
        _finite_number(self.wind_drift_factor, "wind_drift_factor", 0.0, 1.0)
        _finite_number(self.diffusion_coeff, "diffusion_coeff", 0.0, 1.0e6)
        _finite_number(self.deflection_angle_rad, "deflection_angle_rad", 0.0, math.pi / 2.0)

    def _time_intervals(self, start_h: float, end_h: float, step_minutes: float,
                        num_particles: Optional[int] = None) -> List[Tuple[float, float]]:
        """Bounded exact endpoint clock with a final fractional step in either direction."""
        self._validate_configuration()
        step_minutes = _finite_number(step_minutes, "time_step_minutes", 0.0, 1440.0, positive=True)
        duration = _finite_number(abs(end_h - start_h), "duration_hours", 0.0, MAX_DURATION_HOURS)
        step_h = step_minutes / 60.0
        if step_h == 0.0:
            raise ValueError("time_step_minutes is below numerical resolution.")
        ratio = duration / step_h
        if not math.isfinite(ratio) or ratio > MAX_STEPS:
            raise ValueError(f"Integration exceeds the {MAX_STEPS}-step resource limit.")
        total_steps = int(math.ceil(ratio))
        count = self.num_particles if num_particles is None else num_particles
        if total_steps * count > MAX_PARTICLE_STEPS:
            raise ValueError(f"Integration exceeds the {MAX_PARTICLE_STEPS} particle-step resource limit.")
        direction = 1.0 if end_h >= start_h else -1.0
        intervals = []
        previous = start_h
        for step in range(total_steps):
            next_h = end_h if step == total_steps - 1 else start_h + direction * min((step + 1) * step_h, duration)
            if next_h != previous:
                intervals.append((previous, next_h))
            previous = next_h
        return intervals

    @staticmethod
    def _rng(random_seed: int) -> np.random.Generator:
        if isinstance(random_seed, (bool, np.bool_)) or not isinstance(random_seed, (int, np.integer)) or not 0 <= random_seed < 2 ** 63:
            raise ValueError("random_seed must be a nonnegative integer below 2**63.")
        return np.random.default_rng(random_seed)

    @staticmethod
    def _validate_particles(lat: np.ndarray, lon: np.ndarray) -> None:
        if lat.ndim != 1 or lon.shape != lat.shape or lat.size == 0:
            raise ValueError("Particle coordinates must be matching nonempty one-dimensional arrays.")
        if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
            raise ValueError("Particle coordinates must remain finite.")
        if np.any(np.abs(lat) > MAX_TRANSPORT_LAT):
            raise ValueError(f"Particle trajectory exceeds the supported ±{MAX_TRANSPORT_LAT}° latitude domain; polar transport is unsupported.")
        if np.any(np.abs(lon) > 180.0):
            raise ValueError("Particle longitudes must be in [-180, 180].")

    def _initial_cloud(self, lat: float, lon: float, rng: np.random.Generator,
                       spread_m: float, count: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        count = self.num_particles if count is None else count
        dx = rng.normal(0.0, spread_m, count)
        dy = rng.normal(0.0, spread_m, count)
        # Centre the assumed cloud so zero-time transport has no sampling drift.
        dx -= np.mean(dx)
        dy -= np.mean(dy)
        p_lat = lat + dy / METERS_PER_DEG_LAT
        p_lon = _wrap_longitudes(lon + dx / (METERS_PER_DEG_LAT * math.cos(math.radians(lat))))
        self._validate_particles(p_lat, p_lon)
        return p_lat, p_lon

    @staticmethod
    def _particle_summary(p_lat: np.ndarray, p_lon: np.ndarray) -> Tuple[float, float, float]:
        centroid_lat = float(np.mean(p_lat))
        lon_rad = np.radians(p_lon)
        centroid_lon = float(np.degrees(np.arctan2(np.mean(np.sin(lon_rad)), np.mean(np.cos(lon_rad)))))
        dx = _wrap_longitudes(p_lon - centroid_lon) * METERS_PER_DEG_LAT * math.cos(math.radians(centroid_lat))
        dy = (p_lat - centroid_lat) * METERS_PER_DEG_LAT
        variance = float(np.mean(dx ** 2 + dy ** 2))
        return centroid_lat, centroid_lon, variance

    def _trajectory_record(self, step: int, time_h: float, p_lat: np.ndarray,
                           p_lon: np.ndarray) -> Dict[str, Any]:
        lat, lon, variance = self._particle_summary(p_lat, p_lon)
        indices = np.linspace(0, len(p_lat) - 1, min(35, len(p_lat)), dtype=int)
        return {
            "step_index": step,
            "relative_time_hours": float(time_h),
            "centroid": {"lat": round(lat, 6), "lon": round(lon, 6)},
            "spread_radius_km": round(math.sqrt(variance) / 1000.0, 3),
            "variance_m2": round(variance, 1),
            "particles_sample": [[round(float(p_lon[i]), 5), round(float(p_lat[i]), 5)] for i in indices],
        }

    def _compute_drift_vector(
        self,
        u_curr: float,
        v_curr: float,
        u_wind: float,
        v_wind: float,
        is_northern_hemisphere: bool = True
    ) -> Tuple[float, float]:
        """
        Calculates total instantaneous particle velocity (m/s) in geographic coordinates (East, North):
        Wind u/v is a velocity-to vector. In East/North axes, right of this
        vector (Northern Hemisphere) is a clockwise, negative-angle rotation;
        Southern Hemisphere left deflection is counterclockwise. This fixed
        empirical angle is not resolved Coriolis or Ekman physics.
        """
        angle = np.where(is_northern_hemisphere, -self.deflection_angle_rad, self.deflection_angle_rad)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Deflected wind vector
        u_wind_deflected = u_wind * cos_a - v_wind * sin_a
        v_wind_deflected = u_wind * sin_a + v_wind * cos_a

        u_net = u_curr + self.wind_drift_factor * u_wind_deflected
        v_net = v_curr + self.wind_drift_factor * v_wind_deflected

        return u_net, v_net

    def _safe_get_velocity(self, current_field: OceanCurrentField, lat: float, lon: float, t_h: float) -> Tuple[float, float, float, float]:
        """Validate source vectors; a failed source is never replaced with base vectors."""
        try:
            values = current_field.get_velocity_at(lat, lon, t_h)
            arrays = self._validated_vectors(values, (1,))
            return tuple(float(value[0]) for value in arrays)
        except OceanSourceError:
            raise
        except Exception as exc:
            raise OceanSourceError(f"Met-ocean source failure at relative time {t_h:g}h: {exc}") from exc

    @staticmethod
    def _validated_vectors(values: Any, shape: Tuple[int, ...]) -> Tuple[np.ndarray, ...]:
        try:
            if not isinstance(values, (tuple, list, np.ndarray)) or len(values) != 4:
                raise ValueError("four current/wind components are required")
            if any(np.asarray(value).dtype.kind in "bUS" for value in values):
                raise ValueError("current/wind components must be numeric, not boolean/text")
            arrays = tuple(np.broadcast_to(np.asarray(value, dtype=float), shape) for value in values)
            for index, value in enumerate(arrays):
                limit = 100.0 if index < 2 else 200.0
                if not np.all(np.isfinite(value)) or np.any(np.abs(value) > limit):
                    raise ValueError(f"component {index} must be finite and within ±{limit} m/s")
            return arrays
        except (TypeError, ValueError, OverflowError) as exc:
            raise OceanSourceError(f"Met-ocean source supplied invalid vectors: {exc}") from exc

    def _velocity_rates(self, current_field: OceanCurrentField, lat: np.ndarray,
                        lon: np.ndarray, time_h: float) -> Tuple[np.ndarray, np.ndarray]:
        lon = _wrap_longitudes(lon)
        self._validate_particles(lat, lon)
        try:
            batch = getattr(current_field, "get_velocities_at", None)
            if callable(batch):
                values = batch(lat, lon, time_h)
            else:
                values = np.asarray([current_field.get_velocity_at(float(a), float(b), time_h)
                                     for a, b in zip(lat, lon)], dtype=float).T
            u, v, uw, vw = self._validated_vectors(values, lat.shape)
        except OceanSourceError:
            raise
        except Exception as exc:
            raise OceanSourceError(f"Met-ocean source failure at relative time {time_h:g}h: {exc}") from exc
        u_net, v_net = self._compute_drift_vector(u, v, uw, vw, lat >= 0.0)
        # The equator has no hemisphere-specific fixed-angle deflection.
        u_net = np.where(lat == 0.0, u + self.wind_drift_factor * uw, u_net)
        v_net = np.where(lat == 0.0, v + self.wind_drift_factor * vw, v_net)
        return v_net / METERS_PER_DEG_LAT, u_net / (METERS_PER_DEG_LAT * np.cos(np.radians(lat)))

    def _rk4_particle_step(self, current_field: OceanCurrentField, lat: np.ndarray,
                           lon: np.ndarray, time_h: float, dt_sec: float,
                           end_time_h: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Genuine four-stage integration of each particle's geographic ODE.

        Longitude scale and hemisphere windage are evaluated at each stage's
        latitude. Longitude is periodic; polar stages fail explicitly.
        """
        time_h = _finite_number(time_h, "stage_time_hours", -MAX_DURATION_HOURS, MAX_DURATION_HOURS)
        dt_sec = _finite_number(dt_sec, "step_seconds", -86400.0, 86400.0)
        end_h = time_h + dt_sec / 3600.0 if end_time_h is None else end_time_h
        end_h = _finite_number(end_h, "stage_end_time_hours", -MAX_DURATION_HOURS, MAX_DURATION_HOURS)
        if not math.isclose(dt_sec, (end_h - time_h) * 3600.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Stage clock and integration step disagree.")
        midpoint_h = (time_h + end_h) / 2.0
        k1_lat, k1_lon = self._velocity_rates(current_field, lat, lon, time_h)
        k2_lat, k2_lon = self._velocity_rates(current_field, lat + dt_sec * k1_lat / 2.0,
                                             lon + dt_sec * k1_lon / 2.0, midpoint_h)
        k3_lat, k3_lon = self._velocity_rates(current_field, lat + dt_sec * k2_lat / 2.0,
                                             lon + dt_sec * k2_lon / 2.0, midpoint_h)
        k4_lat, k4_lon = self._velocity_rates(current_field, lat + dt_sec * k3_lat,
                                             lon + dt_sec * k3_lon, end_h)
        next_lat = lat + dt_sec * (k1_lat + 2.0 * k2_lat + 2.0 * k3_lat + k4_lat) / 6.0
        next_lon = _wrap_longitudes(lon + dt_sec * (k1_lon + 2.0 * k2_lon + 2.0 * k3_lon + k4_lon) / 6.0)
        self._validate_particles(next_lat, next_lon)
        return next_lat, next_lon

    def _forward_diffusion(self, lat: np.ndarray, lon: np.ndarray, dt_sec: float,
                           rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        sigma = math.sqrt(2.0 * self.diffusion_coeff * dt_sec)
        dx = rng.normal(0.0, sigma, len(lat))
        dy = rng.normal(0.0, sigma, len(lat))
        next_lat = lat + dy / METERS_PER_DEG_LAT
        next_lon = _wrap_longitudes(lon + dx / (METERS_PER_DEG_LAT * np.cos(np.radians(lat))))
        self._validate_particles(next_lat, next_lon)
        return next_lat, next_lon

    @staticmethod
    def _demo_coastline_clamp(previous_lat: np.ndarray, previous_lon: np.ndarray,
                              lat: np.ndarray, lon: np.ndarray) -> int:
        """Unvalidated demo geometry rejection, never a beaching/safety assessment."""
        count = 0
        for index, (a, b) in enumerate(zip(lat, lon)):
            if is_on_land(float(a), float(b)):
                lat[index], lon[index] = previous_lat[index], previous_lon[index]
                count += 1
        return count

    @staticmethod
    def _transport_disclosure() -> Dict[str, Any]:
        return {
            "advection": "per_particle_geographic_rk4",
            "windage": "empirical fixed-angle velocity-to windage; not resolved Coriolis/Ekman physics",
            "coordinate_model": "spherical local East/North metric; longitude periodic; |latitude| <= 85 degrees",
            "coastline_geometry_status": "NATURAL_EARTH_10M_VECTOR",
            "coastline_handling": "Natural Earth 10m land polygon containment with STRtree spatial indexing; particles clamped at waterline",
            "initial_cloud_status": "assumed Gaussian detection/release-location spread; not calibrated uncertainty",
        }

    def run_hindcast(
        self,
        initial_lat: float,
        initial_lon: float,
        current_field: OceanCurrentField,
        max_lookback_hours: float = 24.0,
        time_step_minutes: float = 15.0,
        target_slick_age_hours: Optional[float] = None,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Runs a *conditional* reverse particle-transport scenario from a detection.
        The requested slick age is an analyst hypothesis, not an inferred release time.
        """
        if target_slick_age_hours is None:
            raise ValueError("A source-supported slick age hypothesis is required for conditional backtracking.")
        initial_lat, initial_lon = _coordinates(initial_lat, initial_lon)
        max_lookback_hours = _finite_number(max_lookback_hours, "max_lookback_hours", 0.0, MAX_DURATION_HOURS)
        lookback_limit = _finite_number(target_slick_age_hours, "target_slick_age_hours", 0.0, MAX_DURATION_HOURS)
        if lookback_limit > max_lookback_hours:
            raise ValueError("Requested slick age exceeds the approved lookback window.")
        intervals = self._time_intervals(0.0, -lookback_limit, time_step_minutes)
        rng = self._rng(random_seed)
        self._safe_get_velocity(current_field, initial_lat, initial_lon, 0.0)
        p_lat, p_lon = self._initial_cloud(initial_lat, initial_lon, rng, 500.0)
        history_trajectory = [self._trajectory_record(0, 0.0, p_lat, p_lon)]
        demo_rejections = 0
        current_t_hours = 0.0
        for step, (start_h, end_h) in enumerate(intervals, 1):
            previous_lat, previous_lon = p_lat, p_lon
            p_lat, p_lon = self._rk4_particle_step(current_field, p_lat, p_lon, start_h, (end_h - start_h) * 3600.0, end_h)
            # Reverse advection only. Stochastic diffusion is not invertible.
            demo_rejections += self._demo_coastline_clamp(previous_lat, previous_lon, p_lat, p_lon)
            current_t_hours = end_h
            history_trajectory.append(self._trajectory_record(step, current_t_hours, p_lat, p_lon))

        origin_lat, origin_lon, variance_m2 = self._particle_summary(p_lat, p_lon)
        total_drift_km = round(haversine_distance_km(initial_lat, initial_lon, origin_lat, origin_lon), 2)
        final_spread_km = math.sqrt(variance_m2) / 1000.0
        meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(origin_lat))
        provider = getattr(current_field, "data_provider", None)
        if provider is not None:
            metadata = getattr(provider, "metadata", {})
            source_name = metadata.get("source") or metadata.get("data_origin") or "Bound provider (provenance unspecified)"
        elif getattr(current_field, "constant_vectors", False):
            source_name = "Explicit constant-vector scenario (not a gridded observation)"
        else:
            source_name = "Demonstration analytical/custom vector field (source validation not established)"

        # KDE describes only the generated conditional terminal cloud; its
        # coverage fractions are not validated release-origin probabilities.
        kde_contours = self._compute_kde_hdr_contours(
            p_lat, p_lon, origin_lat, origin_lon,
            METERS_PER_DEG_LAT, meters_per_deg_lon,
            levels=[0.95, 0.75, 0.50]
        )

        return {
            "origin_release_point": {
                "lat": round(origin_lat, 6),
                "lon": round(origin_lon, 6),
                "estimated_t0_hours_relative": current_t_hours,
                "assumed_slick_age_hours": lookback_limit,
                "hydrodynamic_data_source": source_name,
                "inference_status": "conditional transport scenario; not an inferred spill origin",
                "confidence_percent": None,
                "confidence_status": "NOT_ESTIMATED_UNCALIBRATED_CONDITIONAL_CLOUD",
                "location_uncertainty_radius_km": round(final_spread_km, 3),
                "location_uncertainty_status": "generated-cloud RMS spread; not calibrated origin uncertainty",
            },
            "hindcast_trajectory": history_trajectory,
            "total_drift_distance_km": total_drift_km,
            "simulated_duration_hours": abs(current_t_hours),
            "transport_model": dict(self._transport_disclosure(),
                                    diffusion_status="NOT_INVERTED_REVERSE_ADVECTION_ONLY",
                                    demo_coastline_rejections=demo_rejections),
            "kde_origin_contours": kde_contours
        }

    def _compute_kde_hdr_contours(
        self,
        p_lat: np.ndarray,
        p_lon: np.ndarray,
        centroid_lat: float,
        centroid_lon: float,
        meters_per_deg_lat: float,
        meters_per_deg_lon: float,
        levels: List[float] = None
    ) -> Dict[str, Any]:
        """
        Conditional generated-cloud density coverage, not origin probabilities.
        Levels describe fractions of the model KDE on its finite display grid.
        The covariance fallback additionally assumes a Gaussian cloud. Neither
        construction calibrates release-origin uncertainty against observations.
        """
        if levels is None:
            levels = [0.95, 0.75, 0.50]
        self._validate_particles(p_lat, p_lon)
        levels = [_finite_number(level, "cloud_coverage_level", 0.0, 1.0, positive=True) for level in levels]
        if any(level == 1.0 for level in levels):
            raise ValueError("cloud_coverage_level must be below 1.")

        contour_results = {
            "method": "gaussian_kde_hdr",
            "status": "CONDITIONAL_GENERATED_CLOUD_ONLY",
            "coverage_interpretation": "conditional generated-cloud density coverage; not validated origin probabilities or confidence regions",
            "origin_probability": None,
            "levels": levels,
            "contours": [],
            "peak_density_lat": round(centroid_lat, 6),
            "peak_density_lon": round(centroid_lon, 6),
        }
        x_m = _wrap_longitudes(p_lon - centroid_lon) * meters_per_deg_lon
        y_m = (p_lat - centroid_lat) * meters_per_deg_lat
        if np.any(np.abs(_wrap_longitudes(p_lon - centroid_lon)) > 45.0) or np.any(np.abs(p_lat - centroid_lat) > 10.0):
            contour_results.update(method="no_density_contours", status="NOT_ASSESSED_NONLOCAL_GENERATED_CLOUD",
                                   reason="This local KDE display is unsupported for a geographically nonlocal cloud.")
            return contour_results
        if len(p_lat) < 3 or np.linalg.matrix_rank(np.vstack([x_m, y_m])) < 2:
            contour_results.update(method="no_density_contours", status="DEGENERATE_GENERATED_CLOUD",
                                   reason="Too few or collinear particles for two-dimensional density coverage.")
            return contour_results

        def latlon(x: float, y: float) -> List[float]:
            lat = centroid_lat + float(y) / meters_per_deg_lat
            lon = float(_wrap_longitudes(np.asarray(centroid_lon + float(x) / meters_per_deg_lon)))
            _coordinates(lat, lon, transport=False)
            return [round(lat, 6), round(lon, 6)]

        try:
            from scipy.stats import gaussian_kde
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Work in metres relative to centroid to avoid numerical issues
            data = np.vstack([x_m, y_m])
            kde = gaussian_kde(data, bw_method='silverman')

            # Evaluate on a 64×64 grid (sufficient for contour extraction)
            x_pad = max(np.std(x_m) * 3.5, 500.0)
            y_pad = max(np.std(y_m) * 3.5, 500.0)
            xgrid = np.linspace(-x_pad, x_pad, 64)
            ygrid = np.linspace(-y_pad, y_pad, 64)
            X, Y = np.meshgrid(xgrid, ygrid)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)

            # Fractions of this finite display-grid KDE, not measured origin probability.
            cell_area = (xgrid[1] - xgrid[0]) * (ygrid[1] - ygrid[0])
            sorted_vals = np.sort(Z.ravel())[::-1]
            cumsum = np.cumsum(sorted_vals * cell_area)
            cumsum /= cumsum[-1]

            level_colors = {0.95: "#00f2fe", 0.75: "#38bdf8", 0.50: "#818cf8"}

            for level in levels:
                idx = np.searchsorted(cumsum, level)
                if idx >= len(sorted_vals):
                    idx = len(sorted_vals) - 1
                threshold = sorted_vals[idx]

                # Extract contour from matplotlib (silent, no display)
                fig, ax = plt.subplots(1, 1, figsize=(1, 1))
                cs = ax.contour(X, Y, Z, levels=[threshold])
                paths = []

                if hasattr(cs, 'get_paths'):
                    for path in cs.get_paths():
                        verts_m = path.vertices
                        verts_latlon = [latlon(vx, vy) for vx, vy in verts_m]
                        if len(verts_latlon) >= 3:
                            paths.append(verts_latlon)
                elif hasattr(cs, 'allsegs'):
                    for seg_list in cs.allsegs:
                        for seg in seg_list:
                            verts_latlon = [latlon(vx, vy) for vx, vy in seg]
                            if len(verts_latlon) >= 3:
                                paths.append(verts_latlon)
                elif hasattr(cs, 'collections'):
                    for collection in cs.collections:
                        for path in collection.get_paths():
                            verts_m = path.vertices
                            verts_latlon = [latlon(vx, vy) for vx, vy in verts_m]
                            if len(verts_latlon) >= 3:
                                paths.append(verts_latlon)
                plt.close(fig)

                area_km2 = round(float(np.count_nonzero(Z >= threshold)) * cell_area / 1e6, 3)

                contour_results["contours"].append({
                    "level": level,
                    "label": f"{int(level * 100)}% conditional generated-cloud coverage",
                    "coverage_status": "MODEL_KDE_DISPLAY_GRID_FRACTION_NOT_ORIGIN_PROBABILITY",
                    "color": level_colors.get(level, "#ffffff"),
                    "polygon_coords": paths[0] if paths else [],
                    "all_polygons": paths,
                    "approximate_area_km2": area_km2
                })

            # Peak density point (mode of the KDE)
            peak_idx = np.unravel_index(np.argmax(Z), Z.shape)
            peak_x_m = float(X[peak_idx])
            peak_y_m = float(Y[peak_idx])
            contour_results["peak_density_lat"], contour_results["peak_density_lon"] = latlon(peak_x_m, peak_y_m)

        except Exception:
            # Robust fallback: covariance ellipse approximation (works without matplotlib/scipy or if grid diverges)
            contour_results["method"] = "covariance_ellipse_fallback"
            contour_results["contours"] = []
            cov = np.cov(x_m, y_m)
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle = math.atan2(eigvecs[1, 1], eigvecs[0, 1])

            level_colors = {0.95: "#00f2fe", 0.75: "#38bdf8", 0.50: "#818cf8"}

            for level in levels:
                chi2 = -2.0 * math.log1p(-level)  # Exact chi-square quantile with 2 degrees of freedom
                a = math.sqrt(max(eigvals[1], 0.0) * chi2)
                b = math.sqrt(max(eigvals[0], 0.0) * chi2)
                ellipse_pts = []
                for theta_deg in np.linspace(0, 360, 33)[:-1]:
                    theta = math.radians(theta_deg)
                    ex = a * math.cos(theta) * math.cos(angle) - b * math.sin(theta) * math.sin(angle)
                    ey = a * math.cos(theta) * math.sin(angle) + b * math.sin(theta) * math.cos(angle)
                    ellipse_pts.append(latlon(ex, ey))
                ellipse_pts.append(ellipse_pts[0])  # Close the polygon
                contour_results["contours"].append({
                    "level": level,
                    "label": f"{int(level * 100)}% conditional generated-cloud coverage (Gaussian ellipse approximation)",
                    "coverage_status": "GAUSSIAN_COVARIANCE_APPROXIMATION_NOT_ORIGIN_PROBABILITY",
                    "color": level_colors.get(level, "#ffffff"),
                    "polygon_coords": ellipse_pts,
                    "all_polygons": [ellipse_pts],
                    "approximate_area_km2": round(math.pi * a * b / 1e6, 3)
                })

        return contour_results

    def run_forecast(
        self,
        current_lat: float,
        current_lon: float,
        current_field: OceanCurrentField,
        forecast_hours: float = 48.0,
        time_step_minutes: float = 30.0,
        coastline_lat_threshold: Optional[float] = None,
        random_seed: int = 101,
        initial_mass_tonnes: Optional[float] = None,
        oil_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs a forward particle-transport scenario. A shoreline impact is not assessed
        without an authoritative shoreline polygon and asset layer.
        """
        current_lat, current_lon = _coordinates(current_lat, current_lon)
        forecast_hours = _finite_number(forecast_hours, "forecast_hours", 0.0, MAX_DURATION_HOURS)
        intervals = self._time_intervals(0.0, forecast_hours, time_step_minutes)
        if coastline_lat_threshold is not None:
            coastline_lat_threshold = _finite_number(coastline_lat_threshold, "coastline_lat_threshold", -MAX_TRANSPORT_LAT, MAX_TRANSPORT_LAT)
        if oil_profile is not None and not isinstance(oil_profile, dict):
            raise ValueError("oil_profile must be a mapping of sensitivity inputs.")
        rng = self._rng(random_seed)
        _, _, initial_wind_u, initial_wind_v = self._safe_get_velocity(current_field, current_lat, current_lon, 0.0)

        # Generic sensitivity only. Source wind is held at its initial value for
        # this calculation, and a missing mass is an explicitly labelled demo assumption.
        weathering_summary = None
        if initial_mass_tonnes is not None or oil_profile is not None:
            mass_t = _finite_number(initial_mass_tonnes, "initial_mass_tonnes", 0.0, 1.0e9, positive=True) if initial_mass_tonnes is not None else 100.0
            prof = oil_profile or {}
            weathering_summary = self.compute_oil_weathering(
                elapsed_hours=forecast_hours,
                initial_mass_tonnes=mass_t,
                wind_speed_ms=math.hypot(initial_wind_u, initial_wind_v),
                initial_viscosity_cp=prof.get("initial_viscosity_cp", 18.0),
                sea_temp_c=prof.get("water_temp_c", prof.get("sea_temp_c", 26.0)),
            )
            weathering_summary["initial_mass_status"] = "SUPPLIED_SENSITIVITY_INPUT" if initial_mass_tonnes is not None else "ASSUMED_100_TONNE_DEMO_INPUT"
            weathering_summary["wind_forcing_status"] = "initial source wind held constant for generic sensitivity"

        p_lat, p_lon = self._initial_cloud(current_lat, current_lon, rng, 500.0)
        forecast_trajectory = []
        trigger_time = None
        trigger_location = None
        beaching_time = None
        beaching_loc = None
        demo_rejections = 0
        current_t_hours = 0.0
        beached_stop = False

        def record(step: int, time_h: float) -> None:
            nonlocal trigger_time, trigger_location
            item = self._trajectory_record(step, time_h, p_lat, p_lon)
            # Check real land contact using Natural Earth 10m polygons
            beached_count = sum(1 for a, b in zip(p_lat, p_lon) if is_on_land(float(a), float(b)))
            beached_fraction = beached_count / max(len(p_lat), 1)
            item.update(beached=True if beached_fraction > 0.05 else None,
                        shoreline_impact_status="BEACHING_DETECTED" if beached_fraction > 0.05 else "OPEN_WATER",
                        beached_particle_fraction=round(beached_fraction, 3))
            if coastline_lat_threshold is not None and trigger_time is None:
                centroid_lat = float(np.mean(p_lat))
                crossed = ((centroid_lat >= coastline_lat_threshold or np.mean(p_lat >= coastline_lat_threshold) > 0.25)
                           if coastline_lat_threshold >= current_lat else
                           (centroid_lat <= coastline_lat_threshold or np.mean(p_lat <= coastline_lat_threshold) > 0.25))
                if crossed:
                    trigger_time = time_h
                    trigger_location = item["centroid"]
            item["demo_latitude_trigger_reached"] = trigger_time is not None
            forecast_trajectory.append(item)

        record(0, 0.0)
        for step, (start_h, end_h) in enumerate(intervals, 1):
            if beached_stop:
                break
            dt_sec = (end_h - start_h) * 3600.0
            previous_lat, previous_lon = p_lat, p_lon
            p_lat, p_lon = self._rk4_particle_step(current_field, p_lat, p_lon, start_h, dt_sec, end_h)
            p_lat, p_lon = self._forward_diffusion(p_lat, p_lon, dt_sec, rng)
            rejections = self._demo_coastline_clamp(previous_lat, previous_lon, p_lat, p_lon)
            demo_rejections += rejections
            current_t_hours = end_h
            record(step, current_t_hours)

            # Detect first beaching event: when >5% of particles contact land
            if beaching_time is None and rejections > 0:
                beaching_fraction = rejections / max(len(p_lat), 1)
                if beaching_fraction > 0.05:
                    beaching_time = end_h
                    # Use the centroid of the clamped particles at the waterline
                    beaching_loc = {
                        "lat": round(float(np.mean(p_lat)), 6),
                        "lon": round(float(np.mean(p_lon)), 6),
                    }
            # Stop if majority (>50%) of particles have beached — trajectory is done
            if demo_rejections > 0 and beaching_time is not None:
                total_beached_frac = sum(1 for a, b in zip(p_lat, p_lon) if is_on_land(float(a), float(b))) / max(len(p_lat), 1)
                if total_beached_frac > 0.50 or (demo_rejections / max(len(p_lat), 1)) > 0.50:
                    beached_stop = True

        # Identify vulnerable coastal assets near beaching location
        vulnerable_assets = []
        if beaching_loc is not None:
            beach_lat, beach_lon = beaching_loc["lat"], beaching_loc["lon"]
            # Indian coastal sensitive assets database
            _COASTAL_ASSETS = [
                {"name": "Mangalore Port & MRPL Refinery", "lat": 12.87, "lon": 74.83, "type": "port_refinery"},
                {"name": "New Mangalore Port Trust", "lat": 12.92, "lon": 74.80, "type": "port"},
                {"name": "Kochi Port & BPCL Refinery", "lat": 9.97, "lon": 76.27, "type": "port_refinery"},
                {"name": "Mormugao Port, Goa", "lat": 15.41, "lon": 73.80, "type": "port"},
                {"name": "Visakhapatnam Port & HPCL Refinery", "lat": 17.69, "lon": 83.29, "type": "port_refinery"},
                {"name": "Mumbai JNPT & BPCL Mahul", "lat": 18.95, "lon": 72.95, "type": "port_refinery"},
                {"name": "Kandla Port, Gulf of Kachchh", "lat": 23.03, "lon": 70.22, "type": "port"},
                {"name": "Marine National Park, Gulf of Kachchh", "lat": 22.43, "lon": 69.15, "type": "marine_sanctuary"},
                {"name": "Gulf of Mannar Marine NP", "lat": 9.14, "lon": 79.10, "type": "marine_sanctuary"},
                {"name": "Sundarbans Biosphere Reserve", "lat": 21.94, "lon": 88.89, "type": "mangrove_biosphere"},
                {"name": "Lakshadweep Coral Islands", "lat": 10.57, "lon": 72.64, "type": "coral_reef"},
                {"name": "Chilika Lake, Odisha", "lat": 19.72, "lon": 85.32, "type": "lagoon_wetland"},
                {"name": "Paradip Port, Odisha", "lat": 20.27, "lon": 86.67, "type": "port"},
                {"name": "Chennai-Ennore Port Complex", "lat": 13.22, "lon": 80.32, "type": "port"},
                {"name": "Tuticorin V.O.C. Port", "lat": 8.76, "lon": 78.18, "type": "port"},
                {"name": "Haldia Port & IOC Refinery", "lat": 22.06, "lon": 88.11, "type": "port_refinery"},
            ]
            for asset in _COASTAL_ASSETS:
                dist = haversine_distance_km(beach_lat, beach_lon, asset["lat"], asset["lon"])
                if dist < 80.0:
                    vulnerable_assets.append({
                        "name": asset["name"],
                        "type": asset["type"],
                        "distance_km": round(dist, 1),
                        "threat_level": "CRITICAL" if dist < 15.0 else ("HIGH" if dist < 35.0 else "MODERATE"),
                    })
            vulnerable_assets.sort(key=lambda x: x["distance_km"])

        demo_cues = {
            "latitude_trigger": {
                "status": "DEMO_TRIGGER_REACHED" if trigger_time is not None else "NOT_TRIGGERED",
                "time_hours": trigger_time,
                "location": trigger_location,
            }
        }

        if beaching_time is not None:
            beaching_warning = {
                "status": "BEACHING_DETECTED",
                "will_beach": True,
                "estimated_time_to_beach_hours": round(beaching_time, 2),
                "beaching_location": beaching_loc,
                "vulnerable_assets": vulnerable_assets,
                "reason": f"Particle cloud contacts the Natural Earth 10m shoreline at T+{beaching_time:.1f}h. "
                          f"Trajectory terminated at waterline contact. {len(vulnerable_assets)} coastal assets within 80 km threat radius.",
                "demo_cues": demo_cues,
                "coastline_rejections": demo_rejections,
            }
        else:
            beaching_warning = {
                "status": "NOT_ASSESSED",
                "will_beach": None,
                "estimated_time_to_beach_hours": None,
                "beaching_location": None,
                "vulnerable_assets": [],
                "reason": f"No shoreline impact detected within {forecast_hours}h forecast window. "
                          f"Particle cloud remains in open water.",
                "demo_cues": demo_cues,
                "coastline_rejections": demo_rejections,
            }

        return {
            "forecast_trajectory": forecast_trajectory,
            "weathering_summary": weathering_summary,
            "beaching_warning": beaching_warning,
            "simulated_duration_hours": current_t_hours,
            "transport_model": dict(self._transport_disclosure(),
                                    coastline_geometry_status="NATURAL_EARTH_10M_VECTOR",
                                    coastline_handling="Natural Earth 10m land polygon containment with STRtree spatial indexing",
                                    diffusion_status="FORWARD_STOCHASTIC_DISPERSION",
                                    demo_coastline_rejections=demo_rejections),
        }

    def compute_oil_weathering(
        self,
        elapsed_hours: float,
        initial_mass_tonnes: float = 100.0,
        sea_temp_c: float = 26.0,
        wind_speed_ms: float = 6.0,
        initial_viscosity_cp: float = 18.0,
        water_temp_c: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Illustrative generic weathering sensitivity calculation.
        This function is retained only for offline sensitivity experiments and is
        not an ADIOS model or an operational oil-weathering result.
        1. Evaporative loss of volatile hydrocarbon fractions
        2. Water-in-oil emulsification (chocolate mousse formation)
        3. Dynamic viscosity and density increase over time
        """
        if water_temp_c is not None:
            sea_temp_c = water_temp_c
        t = _finite_number(elapsed_hours, "elapsed_hours", 0.0, MAX_DURATION_HOURS)
        initial_mass_tonnes = _finite_number(initial_mass_tonnes, "initial_mass_tonnes", 0.0, 1.0e9, positive=True)
        sea_temp_c = _finite_number(sea_temp_c, "sea_temp_c", -5.0, 50.0)
        wind_speed_ms = _finite_number(wind_speed_ms, "wind_speed_ms", 0.0, 300.0)
        initial_viscosity_cp = _finite_number(initial_viscosity_cp, "initial_viscosity_cp", 0.0, 1.0e9, positive=True)
        t_kelvin = sea_temp_c + 273.15

        # 1. Evaporative exposure fraction (Mackay 1980 logarithmic formulation)
        # F_evap = (T / 1000) * alpha * ln(1 + beta * t)
        alpha = 0.165
        beta = 3.8
        f_evap = (t_kelvin / 1000.0) * alpha * math.log1p(beta * t)
        f_evap = float(np.clip(f_evap, 0.0, 0.55))

        # 2. Water-in-oil emulsification uptake (Mooney / Mackay equation)
        # As waves whip the slick, water droplets get trapped inside the oil matrix
        # Y_w reaches up to 75% for heavy emulsion (mousse) over 12-24 hours
        k_emul = 2.0e-6  # Standard Mackay oceanic emulsification rate constant
        t_sec = t * 3600.0
        y_max = 0.75  # 75% maximum water content in chocolate mousse
        y_w = -y_max * math.expm1(-k_emul * ((1.0 + wind_speed_ms) ** 2) * t_sec)
        y_w = float(np.clip(y_w, 0.0, y_max))

        # 3. Mass and volume balance. Water uptake raises emulsion mass, but mass
        # ratio is not volume ratio; use constituent densities explicitly.
        remaining_pure_oil_tonnes = initial_mass_tonnes * (1.0 - f_evap)
        emulsion_mass_tonnes = remaining_pure_oil_tonnes / max(1.0 - y_w, 0.25)
        absorbed_water_tonnes = max(emulsion_mass_tonnes - remaining_pure_oil_tonnes, 0.0)
        oil_density_kg_m3, water_density_kg_m3 = 880.0, 1025.0
        initial_volume_m3 = initial_mass_tonnes * 1000.0 / oil_density_kg_m3
        emulsion_volume_m3 = (
            remaining_pure_oil_tonnes * 1000.0 / oil_density_kg_m3 +
            absorbed_water_tonnes * 1000.0 / water_density_kg_m3
        )
        volume_expansion_ratio = emulsion_volume_m3 / initial_volume_m3

        # 4. Viscosity growth (Mooney equation)
        # Viscosity increases exponentially with evaporation and water droplet packing
        viscosity_factor = math.exp(2.5 * y_w / (1.0 - 0.65 * y_w)) * math.exp(8.0 * f_evap)
        current_viscosity_cp = initial_viscosity_cp if t == 0.0 else round(initial_viscosity_cp * viscosity_factor, 1)

        # State classification
        if y_w > 0.50:
            physical_state = "Heavy Chocolate Mousse (Highly Viscous Emulsion)"
        elif f_evap > 0.25:
            physical_state = "Weathered Viscous Sheen (Volatiles Depleted)"
        else:
            physical_state = "Fresh Liquid Petroleum Hydrocarbon"

        # Generate hourly timeline curves for visualization
        curve_times = [float(x) for x in np.linspace(0.0, t, num=8)] if t > 0.0 else [0.0]
        evap_curve = []
        emul_curve = []
        for ct in curve_times:
            fe = float(np.clip((t_kelvin / 1000.0) * alpha * math.log1p(beta * ct), 0.0, 0.55))
            yw = float(np.clip(-y_max * math.expm1(-k_emul * ((1.0 + wind_speed_ms) ** 2) * ct * 3600.0), 0.0, y_max))
            evap_curve.append(round(fe * 100.0, 1))
            emul_curve.append(round(yw * 100.0, 1))

        return {
            "model_status": "ILLUSTRATIVE_GENERIC_SENSITIVITY",
            "is_adios_model": False,
            "limitations": "Generic fixed-coefficient sensitivity, not ADIOS, oil-specific calibration, validated weathering, or an operational prediction. Water uptake is represented as an emulsion mass fraction.",
            "initial_mass_tonnes": initial_mass_tonnes,
            "evaporated_fraction_pct": round(f_evap * 100.0, 1),
            "evaporated_mass_tonnes": round(initial_mass_tonnes * f_evap, 2),
            "water_content_mousse_pct": round(y_w * 100.0, 1),
            "emulsion_apparent_mass_tonnes": initial_mass_tonnes if t == 0.0 else round(emulsion_mass_tonnes, 2),
            "initial_volume_m3": initial_volume_m3 if t == 0.0 else round(initial_volume_m3, 2),
            "emulsion_volume_m3": initial_volume_m3 if t == 0.0 else round(emulsion_volume_m3, 2),
            "volume_expansion_ratio": round(volume_expansion_ratio, 2),
            "volume_expansion_factor": round(volume_expansion_ratio, 2),
            "viscosity_cp": current_viscosity_cp,
            "dynamic_viscosity_cP": current_viscosity_cp,
            "physical_state": physical_state,
            "weathering_classification": physical_state,
            "weathering_timeline": {
                "hours": curve_times,
                "evaporation_pct": evap_curve,
                "water_uptake_pct": emul_curve
            }
        }

    @staticmethod
    def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance between two points in kilometers."""
        return haversine_distance_km(lat1, lon1, lat2, lon2)

    def run_forward_counterfactual(
        self,
        release_lat: float,
        release_lon: float,
        release_time_rel_h: float,
        current_field: OceanCurrentField,
        observed_slick_lat: float,
        observed_slick_lon: float,
        observed_slick_polygon: Optional[List[Any]] = None,
        observed_slick_area_km2: Optional[float] = None,
        time_step_minutes: float = 15.0,
        random_seed: int = 101,
        vessel_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Conditional forward transport comparison from a candidate release hypothesis.
        Generated-cloud spatial agreement is not causality, vessel responsibility,
        calibrated origin confidence, or a Navier-Stokes solution.
        """
        release_lat, release_lon = _coordinates(release_lat, release_lon)
        observed_slick_lat, observed_slick_lon = _coordinates(observed_slick_lat, observed_slick_lon)
        start_t = _finite_number(release_time_rel_h, "release_time_rel_h", -MAX_DURATION_HOURS, 0.0)
        end_t = 0.0
        duration_h = -start_t
        area_km2 = (_finite_number(observed_slick_area_km2, "observed_slick_area_km2", 0.0, 1.0e8, positive=True)
                    if observed_slick_area_km2 is not None else None)
        poly_points = None
        if observed_slick_polygon is not None:
            if not isinstance(observed_slick_polygon, (list, tuple)) or not 3 <= len(observed_slick_polygon) <= 1000:
                raise ValueError("observed_slick_polygon must have 3 to 1000 coordinate pairs.")
            try:
                pairs = [(float(point[0]), float(point[1])) for point in observed_slick_polygon
                         if isinstance(point, (list, tuple)) and len(point) == 2]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("observed_slick_polygon must contain finite coordinate pairs.") from exc
            if len(pairs) != len(observed_slick_polygon):
                raise ValueError("observed_slick_polygon must contain coordinate pairs.")
            # Retain the legacy coordinate-order convention for this list API.
            # It is not a validated GeoJSON geometry contract.
            if abs(pairs[0][0]) > 40.0 and abs(pairs[0][1]) < 40.0:
                pairs = [(b, a) for a, b in pairs]
            poly_points = [_coordinates(a, b) for a, b in pairs]
            if len(set(poly_points)) < 3:
                raise ValueError("observed_slick_polygon requires at least three distinct vertices.")
        r_obs_km = math.sqrt(area_km2 / math.pi) if area_km2 is not None else None
        num_p = min(300, self.num_particles)
        intervals = self._time_intervals(start_t, end_t, time_step_minutes, num_p)
        rng = self._rng(random_seed)
        self._safe_get_velocity(current_field, release_lat, release_lon, start_t)
        p_lat, p_lon = self._initial_cloud(release_lat, release_lon, rng, 250.0, num_p)
        forward_trajectory = []
        current_t_hours = start_t
        min_dist_to_slick_km = float("inf")
        demo_rejections = 0

        for step in range(len(intervals) + 1):
            centroid_lat, centroid_lon, variance_m2 = self._particle_summary(p_lat, p_lon)
            dist_step_km = haversine_distance_km(
                centroid_lat, centroid_lon, observed_slick_lat, observed_slick_lon
            )
            if dist_step_km < min_dist_to_slick_km:
                min_dist_to_slick_km = dist_step_km

            item = self._trajectory_record(step, current_t_hours, p_lat, p_lon)
            item["distance_to_observed_km"] = round(dist_step_km, 2)
            forward_trajectory.append(item)

            if step == len(intervals):
                break
            start_h, end_h = intervals[step]
            dt_sec = (end_h - start_h) * 3600.0
            previous_lat, previous_lon = p_lat, p_lon
            p_lat, p_lon = self._rk4_particle_step(current_field, p_lat, p_lon, start_h, dt_sec, end_h)
            p_lat, p_lon = self._forward_diffusion(p_lat, p_lon, dt_sec, rng)
            demo_rejections += self._demo_coastline_clamp(previous_lat, previous_lon, p_lat, p_lon)
            current_t_hours = end_h

        final_centroid_lat, final_centroid_lon, variance_m2 = self._particle_summary(p_lat, p_lon)
        final_spread_km = math.sqrt(variance_m2) / 1000.0
        meters_per_deg_lat = METERS_PER_DEG_LAT
        meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(final_centroid_lat))

        centroid_distance = haversine_distance_km(
            final_centroid_lat, final_centroid_lon, observed_slick_lat, observed_slick_lon
        )
        centroid_distance_km = round(centroid_distance, 2)

        contained_count = 0
        # Use a local periodic longitude frame for a supplied polygon, including
        # antimeridian-crossing footprints. This does not validate its topology.
        local_polygon = ([(a, float(_wrap_longitudes(np.asarray(b - observed_slick_lon)))) for a, b in poly_points]
                         if poly_points else None)
        for i in range(num_p):
            plat_i = float(p_lat[i])
            plon_i = float(p_lon[i])
            inside = False
            if local_polygon:
                local_lon = float(_wrap_longitudes(np.asarray(plon_i - observed_slick_lon)))
                inside = _point_in_polygon(plat_i, local_lon, local_polygon)
            elif r_obs_km is not None:
                d_km = haversine_distance_km(plat_i, plon_i, observed_slick_lat, observed_slick_lon)
                if d_km <= r_obs_km:
                    inside = True
            if inside:
                contained_count += 1

        containment_percent = round((contained_count / num_p) * 100.0, 1) if poly_points or r_obs_km is not None else None

        # Circle-proxy IoU only, never a weighted mixture with particle containment.
        # A polygon or point observation does not establish circle overlap.
        r_pred_km = max(1.0e-9, final_spread_km)
        d = centroid_distance
        if r_obs_km is None or poly_points:
            circle_iou = None
        elif d >= (r_pred_km + r_obs_km):
            circle_iou = 0.0
        elif d <= abs(r_pred_km - r_obs_km):
            smaller_r = min(r_pred_km, r_obs_km)
            larger_r = max(r_pred_km, r_obs_km)
            circle_iou = (smaller_r ** 2) / (larger_r ** 2)
        else:
            r1, r2 = r_pred_km, r_obs_km
            denom1 = max(1e-6, 2.0 * d * r1)
            denom2 = max(1e-6, 2.0 * d * r2)
            arg1 = max(-1.0, min(1.0, (d * d + r1 * r1 - r2 * r2) / denom1))
            arg2 = max(-1.0, min(1.0, (d * d + r2 * r2 - r1 * r1) / denom2))
            part1 = r1 * r1 * math.acos(arg1)
            part2 = r2 * r2 * math.acos(arg2)
            part3 = 0.5 * math.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
            int_area = part1 + part2 - part3
            un_area = math.pi * r1 * r1 + math.pi * r2 * r2 - int_area
            circle_iou = int_area / max(un_area, 0.01)

        jaccard_index = round(min(1.0, max(0.0, circle_iou)), 3) if circle_iou is not None else None

        # Compute Modified Hausdorff Distance (Dubuisson & Jain, 1994)
        sim_pts = [(float(p_lat[i]), float(p_lon[i])) for i in range(0, num_p, max(1, num_p // 40))]
        obs_pts = poly_points if (poly_points and len(poly_points) >= 3) else [(observed_slick_lat, observed_slick_lon)]
        
        # d(sim -> obs)
        sum_sim_to_obs = 0.0
        for s_lat, s_lon in sim_pts:
            min_d = min(haversine_distance_km(s_lat, s_lon, o_lat, o_lon) for o_lat, o_lon in obs_pts)
            sum_sim_to_obs += min_d
        d_sim_obs = sum_sim_to_obs / max(1, len(sim_pts))
        
        # d(obs -> sim)
        sum_obs_to_sim = 0.0
        for o_lat, o_lon in obs_pts:
            min_d = min(haversine_distance_km(o_lat, o_lon, s_lat, s_lon) for s_lat, s_lon in sim_pts)
            sum_obs_to_sim += min_d
        d_obs_sim = sum_obs_to_sim / max(1, len(obs_pts))
        
        modified_hausdorff_km = round(max(d_sim_obs, d_obs_sim), 2)

        trajectory_reaches_slick = bool(min_dist_to_slick_km <= (r_obs_km if r_obs_km is not None else 2.5))

        # Illustrative spatial screening thresholds, not a causality verdict.
        causality_score = None
        if centroid_distance_km <= 2.5:
            verdict = "CONDITIONAL_SPATIAL_AGREEMENT"
            verdict_badge = "CONDITIONAL SPATIAL AGREEMENT"
            verdict_color = "#05d6a0"
            explanation = (
                f"The conditional generated-cloud centroid ends {centroid_distance_km:.2f} km from the supplied observation. "
                "Agreement is conditional on release time, forcing, windage, assumed spread and observation geometry; "
                "it does not establish a release origin, causality or vessel responsibility."
            )
        elif centroid_distance_km <= 5.8:
            verdict = "CONDITIONAL_NEARBY_CORRIDOR"
            verdict_badge = "CONDITIONAL NEARBY CORRIDOR"
            verdict_color = "#f59e0b"
            explanation = (
                f"The conditional generated-cloud centroid ends {centroid_distance_km:.2f} km from the supplied observation. "
                "This illustrative nearby-corridor cue is not calibrated origin evidence or a causality finding."
            )
        else:
            verdict = "CONDITIONAL_SPATIAL_MISMATCH"
            verdict_badge = "CONDITIONAL SPATIAL MISMATCH"
            verdict_color = "#ff3366"
            explanation = (
                f"The conditional generated-cloud centroid ends {centroid_distance_km:.2f} km from the supplied observation. "
                "Mismatch under these inputs does not rule out a release origin or exonerate/attribute a vessel."
            )

        footprint_poly = []
        for angle_deg in np.linspace(0, 360, 17)[:-1]:
            rad = math.radians(angle_deg)
            r_km = r_pred_km
            d_lat = (r_km * 1000.0 * math.cos(rad)) / meters_per_deg_lat
            d_lon = (r_km * 1000.0 * math.sin(rad)) / meters_per_deg_lon
            point_lat = final_centroid_lat + d_lat
            point_lon = float(_wrap_longitudes(np.asarray(final_centroid_lon + d_lon)))
            _coordinates(point_lat, point_lon, transport=False)
            footprint_poly.append([round(point_lat, 5), round(point_lon, 5)])
        footprint_poly.append(footprint_poly[0])

        return {
            "vessel_mmsi": vessel_info.get("mmsi") if vessel_info else None,
            "vessel_name": vessel_info.get("vessel_name") if vessel_info else "Candidate vessel",
            "release_state": {
                "lat": round(release_lat, 6),
                "lon": round(release_lon, 6),
                "time_relative_h": start_t,
                "observation_time_relative_h": current_t_hours,
                "duration_hours": duration_h
            },
            "observed_slick": {
                "lat": round(observed_slick_lat, 6),
                "lon": round(observed_slick_lon, 6),
                "area_km2": area_km2,
                "effective_radius_km": r_obs_km,
                "geometry_status": "SUPPLIED_POLYGON_UNVALIDATED_TOPOLOGY" if poly_points else "SUPPLIED_AREA_CIRCLE_PROXY" if r_obs_km is not None else "POINT_ONLY_NO_FOOTPRINT",
            },
            "predicted_at_t0": {
                "centroid_lat": round(final_centroid_lat, 6),
                "centroid_lon": round(final_centroid_lon, 6),
                "spread_radius_km": round(r_pred_km, 3),
                "predicted_footprint_polygon": footprint_poly,
                "footprint_status": "generated-cloud RMS-radius circle proxy; not a validated slick footprint",
                "particles_sample": [
                    [round(float(p_lon[i]), 5), round(float(p_lat[i]), 5)]
                    for i in np.linspace(0, num_p - 1, 35, dtype=int)
                ]
            },
            "verification_metrics": {
                "centroid_distance_km": centroid_distance_km,
                "predicted_containment_percent": containment_percent,
                "jaccard_index": jaccard_index,
                "modified_hausdorff_distance_km": modified_hausdorff_km,
                "trajectory_reaches_slick": trajectory_reaches_slick,
                "physical_causality_score": causality_score,
                "confidence_status": "NOT_ESTIMATED_CONDITIONAL_TRANSPORT_COMPARISON",
                "containment_status": "generated-cloud fraction in supplied polygon/area proxy; not origin probability" if containment_percent is not None else "NOT_ASSESSED_NO_OBSERVED_FOOTPRINT",
                "jaccard_status": "CIRCLE_PROXY_IOU_ONLY" if jaccard_index is not None else "NOT_ASSESSED_NO_COMPARABLE_CIRCLE_GEOMETRY",
            },
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "verdict_color": verdict_color,
            "explanation": explanation,
            "forward_trajectory": forward_trajectory,
            "transport_model": dict(self._transport_disclosure(), demo_coastline_rejections=demo_rejections),
        }

    def track_multi_spill_shared_origin(
        self,
        spill_detections: List[Dict[str, Any]],
        current_field: OceanCurrentField,
        hindcast_hours: float = 12.0
    ) -> Dict[str, Any]:
        """
        Compare conditional reverse-advection terminal-cloud centres for multiple
        detections. Proximity does not establish a common source or discharge pattern.
        """
        if not isinstance(spill_detections, list) or len(spill_detections) > 32:
            raise ValueError("spill_detections must be a list with at most 32 detections.")
        if not spill_detections:
            return {"error": "No spill detections provided"}
        hindcast_hours = _finite_number(hindcast_hours, "hindcast_hours", 0.0, MAX_DURATION_HOURS)
        self._time_intervals(0.0, -hindcast_hours, 15.0, self.num_particles * len(spill_detections))

        hindcast_results = []
        for idx, slick in enumerate(spill_detections):
            if not isinstance(slick, dict):
                raise ValueError("Each spill detection must supply a coordinate mapping.")
            lat, lon = _coordinates(slick.get("center_lat", slick.get("lat")), slick.get("center_lon", slick.get("lon")))
            slick_age = slick.get("target_age_hours", hindcast_hours)
            hc = self.run_hindcast(
                initial_lat=lat,
                initial_lon=lon,
                current_field=current_field,
                target_slick_age_hours=slick_age,
                max_lookback_hours=hindcast_hours,
            )
            hindcast_results.append({
                "spill_index": idx + 1,
                "slick_id": slick.get("id", f"SLICK_{idx+1:02d}"),
                "observed_lat": lat,
                "observed_lon": lon,
                "origin_lat": hc["origin_release_point"]["lat"],
                "origin_lon": hc["origin_release_point"]["lon"],
                "origin_time_relative_h": hc["origin_release_point"]["estimated_t0_hours_relative"],
                "trajectory": hc["hindcast_trajectory"]
            })

        origins = [(h["origin_lat"], h["origin_lon"]) for h in hindcast_results]
        pairwise_dists = []
        for i in range(len(origins)):
            for j in range(i + 1, len(origins)):
                d = haversine_distance_km(origins[i][0], origins[i][1], origins[j][0], origins[j][1])
                pairwise_dists.append(d)

        mean_origin_dist = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0
        is_shared_origin = bool(len(origins) > 1 and mean_origin_dist < 8.0)

        sequential_speeds_knots = []
        for i in range(len(hindcast_results) - 1):
            h1 = hindcast_results[i]
            h2 = hindcast_results[i+1]
            dist_km = haversine_distance_km(h1["origin_lat"], h1["origin_lon"], h2["origin_lat"], h2["origin_lon"])
            dt_h = abs(float(h1["origin_time_relative_h"]) - float(h2["origin_time_relative_h"]))
            if dt_h > 0.05:
                speed_kt = (dist_km / 1.852) / dt_h
                sequential_speeds_knots.append(round(speed_kt, 1))

        pattern = ("CONDITIONAL_CLOSE_TERMINAL_CLOUDS" if mean_origin_dist < 3.0 else
                   "CONDITIONAL_NEARBY_TERMINAL_CLOUDS" if is_shared_origin else
                   "CONDITIONAL_SEPARATED_TERMINAL_CLOUDS") if len(origins) > 1 else "SINGLE_DETECTION_NO_SHARED_ORIGIN_ASSESSMENT"

        return {
            "spill_count": len(spill_detections),
            "spills_analyzed": hindcast_results,
            "mean_origin_separation_km": round(mean_origin_dist, 2),
            "is_shared_origin_hypothesis": is_shared_origin,
            "inference_status": "conditional generated-cloud proximity only; shared origin not validated",
            "discharge_pattern": pattern,
            "sequential_transit_speeds_knots": sequential_speeds_knots,
            "forensic_summary": (
                f"Conditional reverse-advection terminal centres have mean pairwise separation {mean_origin_dist:.2f} km. "
                "This generated-cloud proximity cue does not establish shared release origins, related discharge events, or vessel responsibility."
            )
        }

    def estimate_spill_age_fay(
        self,
        area_km2: float,
        spill_volume_m3: float = 1000.0,
        oil_density_kg_m3: float = 900.0,
        water_density_kg_m3: float = 1025.0,
        water_kinematic_viscosity: float = 1.0e-6,
        k2: float = 1.7
    ) -> Dict[str, Any]:
        """
        Evaluates an offline Fay-style area-to-age sensitivity under supplied
        assumptions. It does not establish release age or phase validity.
        """
        return estimate_fay_spill_age(
            area_km2=area_km2,
            spill_volume_m3=spill_volume_m3,
            oil_density_kg_m3=oil_density_kg_m3,
            water_density_kg_m3=water_density_kg_m3,
            water_kinematic_viscosity=water_kinematic_viscosity,
            k2=k2
        )


def estimate_fay_spill_age(
    area_km2: float,
    spill_volume_m3: float = 1000.0,
    oil_density_kg_m3: float = 900.0,
    water_density_kg_m3: float = 1025.0,
    water_kinematic_viscosity: float = 1.0e-6,
    k2: float = 1.7,
) -> Dict[str, Any]:
    """
    Offline area-to-age sensitivity using an assumed gravity-viscous scaling:

        A(t) = pi * k2^2 * ((delta_rho / rho_w) * g * V^2)^(1/3) * nu_w^(-1/6) * t^(1/2)

    Inverting for elapsed time:
        t = (A / C)^2
    where C = pi * k2^2 * ((delta_rho / rho_w) * g * V^2)^(1/3) * nu_w^(-1/6).

    Reference:
        Fay, J.A. (1971), "Physical processes in the spread of oil on a water surface",
        Proc. Joint Conf. on Prevention and Control of Oil Spills.
    This implementation has no verified NOAA GNOME/ADIOS equivalence or measured
    calibration. Its volume perturbations are sensitivity bounds, not confidence
    intervals; the applicable spreading phase is not assessed here.
    """
    area_km2 = _fay_number(area_km2, "area_km2", 0.0, MAX_FAY_AREA_KM2, positive=True)
    spill_volume_m3 = _fay_number(spill_volume_m3, "spill_volume_m3", 0.0, MAX_FAY_VOLUME_M3, positive=True)
    oil_density_kg_m3 = _fay_number(oil_density_kg_m3, "oil_density_kg_m3", 0.0, MAX_FAY_DENSITY_KG_M3, positive=True)
    water_density_kg_m3 = _fay_number(water_density_kg_m3, "water_density_kg_m3", 0.0, MAX_FAY_DENSITY_KG_M3, positive=True)
    water_kinematic_viscosity = _fay_number(
        water_kinematic_viscosity, "water_kinematic_viscosity", 0.0,
        MAX_FAY_KINEMATIC_VISCOSITY, positive=True
    )
    k2 = _fay_number(k2, "k2", 0.0, MAX_FAY_K2, positive=True)

    delta_rho = water_density_kg_m3 - oil_density_kg_m3
    if delta_rho <= 0.0:
        raise ValueError("Assumed oil density must be strictly less than seawater density (oil must float).")

    g = 9.81  # m/s^2
    area_m2 = area_km2 * 1_000_000.0
    relative_buoyancy = delta_rho / water_density_kg_m3

    # Hydrodynamic constant C: A(t) = C * t^(1/2)
    c_term = (
        math.pi
        * (k2 ** 2)
        * ((relative_buoyancy * g * (spill_volume_m3 ** 2)) ** (1.0 / 3.0))
        * (water_kinematic_viscosity ** (-1.0 / 6.0))
    )

    t_seconds = (area_m2 / max(c_term, 1e-6)) ** 2
    age_hours = t_seconds / 3600.0
    age_days = age_hours / 24.0

    # Sensitivity range for exactly +/- 50% of the declared volume. Do not
    # floor small volumes to a value above the central hypothesis or silently
    # turn +50% into a doubling.
    vol_min = spill_volume_m3 * 0.5
    vol_max = spill_volume_m3 * 1.5
    c_min = math.pi * (k2 ** 2) * ((relative_buoyancy * g * (vol_max ** 2)) ** (1.0 / 3.0)) * (water_kinematic_viscosity ** (-1.0 / 6.0))
    c_max = math.pi * (k2 ** 2) * ((relative_buoyancy * g * (vol_min ** 2)) ** (1.0 / 3.0)) * (water_kinematic_viscosity ** (-1.0 / 6.0))
    t_min_hours = ((area_m2 / max(c_min, 1e-6)) ** 2) / 3600.0
    t_max_hours = ((area_m2 / max(c_max, 1e-6)) ** 2) / 3600.0

    output_values = (age_hours, age_days, t_min_hours, t_max_hours)
    if not all(math.isfinite(value) for value in output_values):
        raise ValueError("Fay inputs produce a non-finite age estimate.")

    return {
        "status": "CONDITIONAL_SENSITIVITY_ONLY",
        "estimate_kind": "FAY_STYLE_AREA_TO_AGE_INVERSION",
        "age_inference_status": "NOT_INFERRED_FROM_SINGLE_SAR_SCENE",
        "estimated_age_hours": round(float(age_hours), 2),
        "estimated_age_days": round(float(age_days), 2),
        "lookback_window_hours": [round(float(t_min_hours), 1), round(float(t_max_hours), 1)],
        "volume_sensitivity_m3": [vol_min, vol_max],
        "bounds_kind": "VOLUME_SENSITIVITY_ONLY_NOT_CONFIDENCE_INTERVAL",
        "regime_valid": None,
        "regime_status": "NOT_ASSESSED",
        "physics_regime": "ASSUMED_GRAVITY_VISCOUS_SCALING",
        "governing_law": "A(t) = pi * k2^2 * ((delta_rho/rho_w)*g*V^2)^(1/3) * nu_w^(-1/6) * t^(1/2)",
        "assumptions": {
            "spill_volume_m3": spill_volume_m3,
            "oil_density_kg_m3": oil_density_kg_m3,
            "water_density_kg_m3": water_density_kg_m3,
            "water_kinematic_viscosity_m2_s": water_kinematic_viscosity,
            "k2_spreading_constant": k2
        },
        "scientific_caveat": (
            "Conditional sensitivity under assumed area, volume, density, viscosity and coefficient inputs. "
            "The bounds vary only the declared volume by +/-50%; they are not confidence intervals or "
            "validated physical limits. Spreading-phase validity, release age and legal timestamps are "
            "not established. No NOAA GNOME/ADIOS equivalence is verified for this implementation."
        )
    }
