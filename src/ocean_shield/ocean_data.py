"""
Oceanographic & Meteorological Data Ingestion and Spatiotemporal Interpolation Engine.
Smart India Hackathon 2026 (SIH26143 / NTRO)

Provides physical hydrodynamic ocean currents (HYCOM, INCOIS, CMEMS) and atmospheric
wind vectors (NOAA GFS, ECMWF) in CF-compliant NetCDF format or via live Open-Meteo APIs.
Supplies continuous 4D vector interpolation for Lagrangian oil spill trajectory simulation.
"""

import os
import math
import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import netCDF4 as nc


class OceanDataProvider:
    """
    Spatiotemporal Oceanographic and Meteorological Vector Field Provider.
    Ingests CF-compliant NetCDF (.nc) grids containing:
      - water_u: Sea surface eastward velocity (m/s)
      - water_v: Sea surface northward velocity (m/s)
      - wind_u: 10-meter atmospheric eastward wind (m/s)
      - wind_v: 10-meter atmospheric northward wind (m/s)
    Performs multidimensional interpolation across (time, latitude, longitude).
    """

    def __init__(
        self,
        netcdf_path: Optional[str] = None,
        provider_name: str = "HYCOM GOFS 3.1 / NOAA GFS",
        center_lat: float = 22.38,
        center_lon: float = 69.12
    ):
        self.provider_name = provider_name
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.interpolators: Dict[str, RegularGridInterpolator] = {}
        self.metadata: Dict[str, Any] = {}
        self.is_loaded = False

        if netcdf_path:
            if os.path.exists(netcdf_path):
                self.load_netcdf(netcdf_path)
            else:
                self.create_standard_ocean_grid(netcdf_path, center_lat, center_lon)
                self.load_netcdf(netcdf_path)
        else:
            # Prefer real downloaded HYCOM data if available
            real_nc = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "datasets", "ocean_met", "hycom_real_gulf_kachchh.nc"
                )
            )
            default_nc = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "datasets", "ocean_met", "hycom_currents_sample.nc"
                )
            )
            if os.path.exists(real_nc):
                self.load_netcdf(real_nc)
            elif os.path.exists(default_nc):
                self.load_netcdf(default_nc)
            else:
                self.create_standard_ocean_grid(default_nc, center_lat, center_lon)
                self.load_netcdf(default_nc)

        # Bind real observed Open-Meteo atmospheric wind data if available
        self.load_openmeteo_wind()

    def create_standard_ocean_grid(
        self,
        output_path: str,
        center_lat: float = 22.38,
        center_lon: float = 69.12,
        time_span_hours: float = 48.0,
        grid_points: int = 25
    ) -> str:
        """
        Synthesizes a CF-1.8 compliant NetCDF-4 oceanographic and meteorological file
        using physical tidal models (M2 semidiurnal, 12.42h period), Coriolis acceleration,
        and geostrophic boundary flow. Structure mirrors HYCOM GOFS 3.1 schema.
        NOTE: This is SYNTHETIC data generated from analytical models, not downloaded
        from HYCOM/INCOIS/CMEMS. File is clearly labeled as such in global attributes.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Spatial bounds (~1.0 degree box centered at spill location, ~100km x 100km)
        lat_range = np.linspace(center_lat - 0.6, center_lat + 0.6, grid_points, dtype=np.float32)
        lon_range = np.linspace(center_lon - 0.6, center_lon + 0.6, grid_points, dtype=np.float32)

        # Time range: -36h to +12h relative to T0 (hourly resolution)
        time_steps = int(time_span_hours) + 1
        time_range = np.linspace(-36.0, 12.0, time_steps, dtype=np.float32)

        # Meshgrid (time, lat, lon)
        T, LAT, LON = np.meshgrid(time_range, lat_range, lon_range, indexing="ij")

        # Physical M2 tidal frequency (period = 12.42h)
        omega_m2 = 2.0 * math.pi / 12.42
        tide_phase = omega_m2 * T

        # Coriolis frequency: f = 2 * Omega * sin(lat)
        f_coriolis = 2.0 * 7.2921e-5 * np.sin(np.radians(LAT))
        coriolis_scale = np.clip(f_coriolis / (2.0 * 7.2921e-5 * np.sin(np.radians(22.0))), 0.8, 1.2)

        # Realistic HYCOM hydrodynamic current field (m/s):
        # 1. Base geostrophic coastal jet (0.22 m/s eastward, 0.14 m/s northward)
        # 2. Semi-diurnal tidal ellipse (0.32 m/s semi-major, 0.20 m/s semi-minor)
        # 3. Mesoscale eddy circulation
        eddy_u = -0.06 * np.sin((LAT - center_lat) * 12.0) * np.cos((LON - center_lon) * 12.0)
        eddy_v = 0.06 * np.cos((LAT - center_lat) * 12.0) * np.sin((LON - center_lon) * 12.0)

        u_curr = 0.22 + 0.32 * np.cos(tide_phase) * coriolis_scale + eddy_u
        v_curr = 0.14 + 0.22 * np.sin(tide_phase) * coriolis_scale + eddy_v

        # Realistic NOAA GFS 10m atmospheric wind field (m/s):
        # Prevailing southwest/west monsoon: 4.8 m/s eastward, 3.2 m/s northward
        diurnal_factor = 1.0 + 0.18 * np.sin((2.0 * math.pi * T) / 24.0)
        u_wind = (4.8 + 0.6 * np.sin(LAT * 5.0)) * diurnal_factor
        v_wind = (3.2 + 0.4 * np.cos(LON * 5.0)) * diurnal_factor

        # Write CF-1.8 Compliant NetCDF-4
        with nc.Dataset(output_path, "w", format="NETCDF4") as ds:
            ds.title = "Physically-Modeled Oceanographic Surface Current & Wind Grid (Synthetic)"
            ds.institution = "OCEAN-SHIELD / NTRO (SIH26143)"
            ds.source = "Analytical M2 tidal model + geostrophic flow + mesoscale eddies (SYNTHETIC — not downloaded from HYCOM/INCOIS)"
            ds.Conventions = "CF-1.8"
            ds.data_origin = "SYNTHETIC: Generated by OCEAN-SHIELD ocean_data.py using physical tidal equations. Structure mirrors HYCOM GOFS 3.1 schema but data is NOT from real observations."
            ds.history = f"Generated {datetime.datetime.utcnow().isoformat()}Z"

            # Create dimensions
            ds.createDimension("time", len(time_range))
            ds.createDimension("lat", len(lat_range))
            ds.createDimension("lon", len(lon_range))

            # Coordinate variables
            v_time = ds.createVariable("time", "f4", ("time",))
            v_time.units = "hours relative to satellite acquisition (T0)"
            v_time.standard_name = "time"
            v_time[:] = time_range

            v_lat = ds.createVariable("lat", "f4", ("lat",))
            v_lat.units = "degrees_north"
            v_lat.standard_name = "latitude"
            v_lat[:] = lat_range

            v_lon = ds.createVariable("lon", "f4", ("lon",))
            v_lon.units = "degrees_east"
            v_lon.standard_name = "longitude"
            v_lon[:] = lon_range

            # Data variables
            var_u = ds.createVariable("water_u", "f4", ("time", "lat", "lon"), zlib=True)
            var_u.units = "m s-1"
            var_u.standard_name = "eastward_sea_water_velocity"
            var_u.long_name = "Sea Surface Eastward Current Velocity"
            var_u[:] = u_curr.astype(np.float32)

            var_v = ds.createVariable("water_v", "f4", ("time", "lat", "lon"), zlib=True)
            var_v.units = "m s-1"
            var_v.standard_name = "northward_sea_water_velocity"
            var_v.long_name = "Sea Surface Northward Current Velocity"
            var_v[:] = v_curr.astype(np.float32)

            var_wu = ds.createVariable("wind_u", "f4", ("time", "lat", "lon"), zlib=True)
            var_wu.units = "m s-1"
            var_wu.standard_name = "eastward_wind"
            var_wu.long_name = "10m Atmospheric Eastward Wind"
            var_wu[:] = u_wind.astype(np.float32)

            var_wv = ds.createVariable("wind_v", "f4", ("time", "lat", "lon"), zlib=True)
            var_wv.units = "m s-1"
            var_wv.standard_name = "northward_wind"
            var_wv.long_name = "10m Atmospheric Northward Wind"
            var_wv[:] = v_wind.astype(np.float32)

        return output_path

    def load_netcdf(self, file_path: str):
        """
        Loads and binds RegularGridInterpolators from a CF-compliant NetCDF dataset.
        Handles both synthetic (3D: time,lat,lon) and real HYCOM (4D: time,depth,lat,lon) files.
        Handles absolute time coordinates (converts to relative hours centered on the middle timestep).
        Handles missing wind variables (uses Open-Meteo or analytical fallback).
        """
        with nc.Dataset(file_path, "r") as ds:
            time_var = ds.variables["time"]
            time_arr = np.array(time_var[:], dtype=np.float64)
            lat_arr = np.array(ds.variables["lat"][:], dtype=np.float64)
            lon_arr = np.array(ds.variables["lon"][:], dtype=np.float64)

            # Detect absolute time (e.g. "hours since 2000-01-01") vs relative time
            time_units = getattr(time_var, "units", "")
            is_absolute_time = "since" in time_units.lower()
            if is_absolute_time:
                # Convert to relative hours centered on the middle timestep (T0 = detection)
                t_center = time_arr[len(time_arr) // 2]
                time_arr = time_arr - t_center  # Now ranges around 0

            # Load ocean current variables — netCDF4 returns MaskedArrays for int16 with _FillValue.
            # Must call .filled() BEFORE dtype conversion to replace fill values (e.g. -30000) with 0.0.
            u_raw = ds.variables["water_u"][:]
            v_raw = ds.variables["water_v"][:]
            u_curr_raw = np.ma.filled(u_raw, 0.0).astype(np.float32)
            v_curr_raw = np.ma.filled(v_raw, 0.0).astype(np.float32)

            # Squeeze depth dimension if present: (time, depth, lat, lon) → (time, lat, lon)
            if u_curr_raw.ndim == 4:
                u_curr_raw = u_curr_raw[:, 0, :, :]  # Surface (depth index 0)
                v_curr_raw = v_curr_raw[:, 0, :, :]

            # Replace any remaining NaN/inf with 0
            u_curr_raw = np.nan_to_num(u_curr_raw, nan=0.0, posinf=0.0, neginf=0.0)
            v_curr_raw = np.nan_to_num(v_curr_raw, nan=0.0, posinf=0.0, neginf=0.0)

            # Load wind variables if available; otherwise use analytical fallback
            has_wind = "wind_u" in ds.variables and "wind_v" in ds.variables
            if has_wind:
                u_wind = np.array(ds.variables["wind_u"][:], dtype=np.float32)
                v_wind = np.array(ds.variables["wind_v"][:], dtype=np.float32)
                if u_wind.ndim == 4:
                    u_wind = u_wind[:, 0, :, :]
                    v_wind = v_wind[:, 0, :, :]
                u_wind = np.nan_to_num(u_wind, nan=0.0, posinf=0.0, neginf=0.0)
                v_wind = np.nan_to_num(v_wind, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # Create synthetic wind grid from southwest monsoon model (will be overridden by Open-Meteo if available)
                T, LAT, _ = np.meshgrid(time_arr, lat_arr, lon_arr, indexing="ij")
                diurnal = 1.0 + 0.18 * np.sin((2.0 * math.pi * T) / 24.0)
                u_wind = (4.8 + 0.6 * np.sin(LAT * 5.0)) * diurnal
                v_wind = np.full_like(u_wind, 3.2) * diurnal
                u_wind = u_wind.astype(np.float32)
                v_wind = v_wind.astype(np.float32)

            is_real_data = getattr(ds, "institution", "").startswith("Fleet Numerical") or \
                           "HYCOM archive" in getattr(ds, "source", "") or \
                           "SYNTHETIC" not in getattr(ds, "data_origin", "").upper()
            data_origin = getattr(ds, "data_origin", getattr(ds, "source", "Unknown"))

            # Store bounds and provenance
            self.metadata = {
                "title": getattr(ds, "title", "Ocean Hydrodynamic Dataset"),
                "institution": getattr(ds, "institution", "HYCOM/NOAA/INCOIS"),
                "source": getattr(ds, "source", "HYCOM GOFS 3.1"),
                "data_origin": data_origin,
                "is_real_observed": is_real_data,
                "has_real_wind": has_wind,
                "time_units_original": time_units,
                "time_min": float(time_arr.min()),
                "time_max": float(time_arr.max()),
                "lat_min": float(lat_arr.min()),
                "lat_max": float(lat_arr.max()),
                "lon_min": float(lon_arr.min()),
                "lon_max": float(lon_arr.max()),
                "file_path": file_path
            }

        # Build 3D RegularGridInterpolators (time, lat, lon)
        grid_points = (time_arr, lat_arr, lon_arr)
        self.interpolators["water_u"] = RegularGridInterpolator(grid_points, u_curr_raw, bounds_error=False, fill_value=None)
        self.interpolators["water_v"] = RegularGridInterpolator(grid_points, v_curr_raw, bounds_error=False, fill_value=None)
        self.interpolators["wind_u"] = RegularGridInterpolator(grid_points, u_wind, bounds_error=False, fill_value=None)
        self.interpolators["wind_v"] = RegularGridInterpolator(grid_points, v_wind, bounds_error=False, fill_value=None)
        self.is_loaded = True

    def get_velocity_at(
        self,
        lat: float,
        lon: float,
        t_hours_relative: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """
        Interpolates (u_curr, v_curr, u_wind, v_wind) at the given space-time coordinates.
        Clamps time to valid grid range. Returns velocities in meters per second.
        """
        if not self.is_loaded:
            # Fallback to analytical constants if dataset uninitialized
            return 0.25, 0.15, 4.5, 3.0

        t_eval = float(np.clip(t_hours_relative, self.metadata["time_min"], self.metadata["time_max"]))
        lat_eval = float(np.clip(lat, self.metadata["lat_min"], self.metadata["lat_max"]))
        lon_eval = float(np.clip(lon, self.metadata["lon_min"], self.metadata["lon_max"]))
        pt = np.array([[t_eval, lat_eval, lon_eval]])

        u_c = float(self.interpolators["water_u"](pt)[0])
        v_c = float(self.interpolators["water_v"](pt)[0])

        real_wind = self.get_real_wind_at(t_hours_relative)
        if real_wind is not None:
            u_w, v_w = real_wind
        else:
            u_w = float(self.interpolators["wind_u"](pt)[0])
            v_w = float(self.interpolators["wind_v"](pt)[0])

        return u_c, v_c, u_w, v_w

    def get_telemetry_summary(self, lat: float, lon: float, t_h: float = 0.0) -> Dict[str, Any]:
        """Provides human-readable ocean and atmospheric diagnostic metrics."""
        u_c, v_c, u_w, v_w = self.get_velocity_at(lat, lon, t_h)
        curr_speed = math.hypot(u_c, v_c)
        curr_dir_deg = (math.degrees(math.atan2(u_c, v_c)) + 360.0) % 360.0
        wind_speed = math.hypot(u_w, v_w)
        wind_dir_deg = (math.degrees(math.atan2(u_w, v_w)) + 360.0) % 360.0

        wind_src = getattr(self, "_openmeteo_source", self.metadata.get("source", self.provider_name))

        return {
            "source": self.metadata.get("source", self.provider_name),
            "institution": self.metadata.get("institution", "Unknown"),
            "data_origin": self.metadata.get("data_origin", "Unknown"),
            "is_real_observed_currents": self.metadata.get("is_real_observed", False),
            "wind_source": wind_src,
            "format": "CF-1.8 NetCDF-4" if not hasattr(self, "_openmeteo_wind") else "CF-1.8 NetCDF-4 + Real ECMWF/Open-Meteo Wind",
            "evaluated_time_relative_h": round(t_h, 2),
            "surface_current": {
                "speed_ms": round(curr_speed, 3),
                "speed_knots": round(curr_speed * 1.94384, 2),
                "heading_degrees": round(curr_dir_deg, 1),
                "u_east_ms": round(u_c, 3),
                "v_north_ms": round(v_c, 3)
            },
            "surface_wind_10m": {
                "speed_ms": round(wind_speed, 2),
                "speed_knots": round(wind_speed * 1.94384, 1),
                "direction_degrees": round(wind_dir_deg, 1),
                "u_east_ms": round(u_w, 2),
                "v_north_ms": round(v_w, 2),
                "is_real_observed": hasattr(self, "_openmeteo_wind") and bool(self._openmeteo_wind)
            }
        }

    def load_openmeteo_wind(self, json_path: str = None) -> bool:
        """
        Loads real historical wind data from an Open-Meteo JSON archive file.
        Converts wind speed (km/h to m/s) and direction (degrees) to u/v components
        and stores them for interpolation by hour.
        Returns True if loaded successfully.
        """
        import json as json_mod

        if json_path is None:
            json_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "datasets", "ocean_met", "openmeteo_wind_kachchh.json"
                )
            )

        if not os.path.exists(json_path):
            return False

        try:
            with open(json_path, "r") as f:
                data = json_mod.load(f)

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            speeds = hourly.get("wind_speed_10m", [])
            directions = hourly.get("wind_direction_10m", [])

            if not times or not speeds:
                return False

            self._openmeteo_wind = []
            for i in range(len(times)):
                spd_kmh = speeds[i] if i < len(speeds) and speeds[i] is not None else 0.0
                spd_ms = float(spd_kmh) / 3.6  # Convert km/h to physical m/s
                d = float(directions[i]) if i < len(directions) and directions[i] is not None else 0.0
                # Oceanographic wind vector pointing towards the advection direction:
                d_rad = math.radians(d)
                u_w = -spd_ms * math.sin(d_rad)
                v_w = -spd_ms * math.cos(d_rad)
                self._openmeteo_wind.append((u_w, v_w, spd_ms))

            self._openmeteo_source = "Open-Meteo Real Historical Archive (ECMWF/ERA5 Reanalysis)"
            return True
        except Exception:
            return False

    def get_real_wind_at(self, t_hours_relative: float = 0.0) -> Optional[Tuple[float, float]]:
        """
        Returns (u_wind, v_wind) from real Open-Meteo data if loaded.
        Maps relative time to the nearest hourly record, anchored to T0 reference index.
        Returns None if no real wind data is available.
        """
        if not hasattr(self, "_openmeteo_wind") or not self._openmeteo_wind:
            return None
        ref_offset = 36  # Center of 72h window corresponds to T0 (detection)
        idx = int(np.clip(round(ref_offset + t_hours_relative), 0, len(self._openmeteo_wind) - 1))
        return self._openmeteo_wind[idx][0], self._openmeteo_wind[idx][1]

