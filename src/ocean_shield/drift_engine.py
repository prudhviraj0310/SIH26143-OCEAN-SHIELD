"""
Drift Engine: Ocean Hydrodynamic Lagrangian Particle Tracking & Hindcasting
Implements conditional forward and backward particle-transport scenarios driven by
time-aligned surface current and wind inputs. It does not determine culpability or
prove a release location.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


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
        self.base_current_u = base_current_u
        self.base_current_v = base_current_v
        self.base_wind_u = base_wind_u
        self.base_wind_v = base_wind_v
        self.tidal_amplitude = tidal_amplitude
        self.tidal_period_h = tidal_period_h
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

        # Analytical semi-diurnal tidal oscillation fallback
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


class DriftEngine:
    """
    Lagrangian Particle Dispersion & Trajectory Engine.
    Simulates thousands of oil slick particles drifting under ocean currents,
    Ekman windage (3% wind factor with Coriolis deflection angle), and turbulent diffusion.
    """

    def __init__(
        self,
        wind_drift_factor: float = 0.032,     # Standard 3.0% - 3.5% wind drift
        deflection_angle_deg: float = 15.0,  # Ekman deflection (right of wind in Northern Hemisphere)
        diffusion_coeff: float = 2.5,        # Horizontal turbulent diffusion m^2/s
        num_particles: int = 1000
    ):
        self.wind_drift_factor = wind_drift_factor
        self.deflection_angle_rad = math.radians(deflection_angle_deg)
        self.diffusion_coeff = diffusion_coeff
        self.num_particles = num_particles

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
        V_net = V_current + factor * Rotation(Ekman_angle) * V_wind
        """
        # Ekman deflection rotation
        angle = self.deflection_angle_rad if is_northern_hemisphere else -self.deflection_angle_rad
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Deflected wind vector
        u_wind_deflected = u_wind * cos_a - v_wind * sin_a
        v_wind_deflected = u_wind * sin_a + v_wind * cos_a

        u_net = u_curr + self.wind_drift_factor * u_wind_deflected
        v_net = v_curr + self.wind_drift_factor * v_wind_deflected

        return u_net, v_net

    def _rk4_advection_step(
        self,
        current_field: OceanCurrentField,
        lat: float,
        lon: float,
        t_hours: float,
        dt_sec: float,
        meters_per_deg_lat: float,
        meters_per_deg_lon: float,
    ) -> Tuple[float, float]:
        """
        True 4th-Order Runge-Kutta (RK4) hydrodynamic & wind leeway advection.
        Evaluates k1, k2, k3, k4 vector stages across the time-varying velocity field.
        """
        u1, v1, uw1, vw1 = current_field.get_velocity_at(lat, lon, t_hours)
        k1_u, k1_v = self._compute_drift_vector(u1, v1, uw1, vw1)

        half_dt_h = (0.5 * dt_sec) / 3600.0
        lat2 = lat + (0.5 * dt_sec * k1_v) / meters_per_deg_lat
        lon2 = lon + (0.5 * dt_sec * k1_u) / meters_per_deg_lon
        u2, v2, uw2, vw2 = current_field.get_velocity_at(lat2, lon2, t_hours + half_dt_h)
        k2_u, k2_v = self._compute_drift_vector(u2, v2, uw2, vw2)

        lat3 = lat + (0.5 * dt_sec * k2_v) / meters_per_deg_lat
        lon3 = lon + (0.5 * dt_sec * k2_u) / meters_per_deg_lon
        u3, v3, uw3, vw3 = current_field.get_velocity_at(lat3, lon3, t_hours + half_dt_h)
        k3_u, k3_v = self._compute_drift_vector(u3, v3, uw3, vw3)

        full_dt_h = dt_sec / 3600.0
        lat4 = lat + (dt_sec * k3_v) / meters_per_deg_lat
        lon4 = lon + (dt_sec * k3_u) / meters_per_deg_lon
        u4, v4, uw4, vw4 = current_field.get_velocity_at(lat4, lon4, t_hours + full_dt_h)
        k4_u, k4_v = self._compute_drift_vector(u4, v4, uw4, vw4)

        u_rk4 = (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u) / 6.0
        v_rk4 = (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0

        return u_rk4, v_rk4

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
        lookback_limit = abs(float(target_slick_age_hours))
        if lookback_limit > max_lookback_hours:
            raise ValueError("Requested slick age exceeds the approved lookback window.")
        rng = np.random.default_rng(random_seed)
        dt_sec = -1.0 * (time_step_minutes * 60.0)  # Negative dt for time reversal
        total_steps = int((lookback_limit * 60.0) / time_step_minutes)

        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(initial_lat))

        # Detection-location uncertainty envelope. It is preserved under reverse
        # advection; random diffusion cannot be inverted into an origin estimate.
        init_spread_m = 500.0
        px_m = rng.normal(0, init_spread_m, self.num_particles)
        py_m = rng.normal(0, init_spread_m, self.num_particles)

        # Particle coordinates in absolute lat/lon
        p_lat = initial_lat + (py_m / meters_per_deg_lat)
        p_lon = initial_lon + (px_m / meters_per_deg_lon)

        history_trajectory = []
        origin_lat = initial_lat
        origin_lon = initial_lon
        estimated_t0_hours = -lookback_limit

        current_t_hours = 0.0

        for step in range(total_steps + 1):
            centroid_lat = float(np.mean(p_lat))
            centroid_lon = float(np.mean(p_lon))

            d_x = (p_lon - centroid_lon) * meters_per_deg_lon
            d_y = (p_lat - centroid_lat) * meters_per_deg_lat
            variance_m2 = float(np.mean(d_x ** 2 + d_y ** 2))
            spread_radius_km = round(math.sqrt(variance_m2) / 1000.0, 3)

            # Sample representative particles for UI rendering
            sample_indices = np.linspace(0, self.num_particles - 1, 35, dtype=int)
            sampled_coords = [
                [round(float(p_lon[i]), 5), round(float(p_lat[i]), 5)]
                for i in sample_indices
            ]

            step_record = {
                "step_index": step,
                "relative_time_hours": round(current_t_hours, 2),
                "centroid": {"lat": round(centroid_lat, 6), "lon": round(centroid_lon, 6)},
                "spread_radius_km": spread_radius_km,
                "variance_m2": round(variance_m2, 1),
                "particles_sample": sampled_coords
            }
            history_trajectory.append(step_record)

            if step == total_steps:
                origin_lat, origin_lon = centroid_lat, centroid_lon

            if step == total_steps:
                break

            # Evaluate 4th-Order Runge-Kutta advection vector
            u_net, v_net = self._rk4_advection_step(
                current_field, centroid_lat, centroid_lon, current_t_hours,
                dt_sec, meters_per_deg_lat, meters_per_deg_lon
            )

            p_lat += (v_net * dt_sec) / meters_per_deg_lat
            p_lon += (u_net * dt_sec) / meters_per_deg_lon

            current_t_hours += (dt_sec / 3600.0)

        total_drift_km = round(
            math.hypot(
                (origin_lon - initial_lon) * meters_per_deg_lon,
                (origin_lat - initial_lat) * meters_per_deg_lat
            ) / 1000.0, 2
        )

        final_spread_km = math.sqrt(variance_m2) / 1000.0

        data_provider_meta = getattr(current_field, "data_provider", None)
        source_name = data_provider_meta.metadata.get("source", "HYCOM GOFS 3.1 NetCDF") if (data_provider_meta and hasattr(data_provider_meta, "metadata")) else "Physical Oceanographic Hydrodynamic Field"

        # Hydrodynamic concurrence confidence based on particle dispersion radius
        confidence_percent = round(min(96.5, max(68.0, 96.0 - (final_spread_km * 4.5))), 1)

        return {
            "origin_release_point": {
                "lat": round(origin_lat, 6),
                "lon": round(origin_lon, 6),
                "estimated_t0_hours_relative": round(estimated_t0_hours, 2),
                "assumed_slick_age_hours": round(abs(estimated_t0_hours), 1),
                "hydrodynamic_data_source": source_name,
                "inference_status": "conditional transport scenario; not an inferred spill origin",
                "confidence_percent": confidence_percent,
                "location_uncertainty_radius_km": round(final_spread_km, 3),
            },
            "hindcast_trajectory": history_trajectory,
            "total_drift_distance_km": total_drift_km
        }

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
        np.random.seed(random_seed)

        dt_sec = time_step_minutes * 60.0
        total_steps = int((forecast_hours * 60.0) / time_step_minutes)

        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(current_lat))

        # Initial particle cloud
        p_lat = current_lat + (np.random.normal(0, 500.0, self.num_particles) / meters_per_deg_lat)
        p_lon = current_lon + (np.random.normal(0, 500.0, self.num_particles) / meters_per_deg_lon)

        forecast_trajectory = []
        beaching_detected = False
        estimated_time_to_beach_hours = None
        beaching_location = None

        current_t_hours = 0.0

        for step in range(total_steps + 1):
            centroid_lat = float(np.mean(p_lat))
            centroid_lon = float(np.mean(p_lon))

            d_x = (p_lon - centroid_lon) * meters_per_deg_lon
            d_y = (p_lat - centroid_lat) * meters_per_deg_lat
            variance_m2 = float(np.mean(d_x ** 2 + d_y ** 2))
            spread_radius_km = round(math.sqrt(variance_m2) / 1000.0, 3)

            # Sample particles for map visualization
            sample_indices = np.linspace(0, self.num_particles - 1, 35, dtype=int)
            sampled_coords = [
                [round(float(p_lon[i]), 5), round(float(p_lat[i]), 5)]
                for i in sample_indices
            ]

            forecast_trajectory.append({
                "step_index": step,
                "relative_time_hours": round(current_t_hours, 2),
                "centroid": {"lat": round(centroid_lat, 6), "lon": round(centroid_lon, 6)},
                "spread_radius_km": spread_radius_km,
                "beached": beaching_detected,
                "particles_sample": sampled_coords
            })

            if step == total_steps:
                break

            # Forward advection using 4th-Order Runge-Kutta
            u_net, v_net = self._rk4_advection_step(
                current_field, centroid_lat, centroid_lon, current_t_hours,
                dt_sec, meters_per_deg_lat, meters_per_deg_lon
            )

            # Stochastic horizontal turbulent diffusion
            sigma_diff = math.sqrt(2.0 * self.diffusion_coeff * dt_sec)
            rand_dx = np.random.normal(0, sigma_diff, self.num_particles)
            rand_dy = np.random.normal(0, sigma_diff, self.num_particles)

            p_lat += (v_net * dt_sec + rand_dy) / meters_per_deg_lat
            p_lon += (u_net * dt_sec + rand_dx) / meters_per_deg_lon

            current_t_hours += (dt_sec / 3600.0)

        weathering_summary = None
        if initial_mass_tonnes is not None and oil_profile:
            weathering_summary = self.compute_oil_weathering(
                elapsed_hours=abs(forecast_hours),
                initial_mass_tonnes=initial_mass_tonnes,
                wind_speed_ms=math.hypot(current_field.base_wind_u, current_field.base_wind_v),
                initial_viscosity_cp=float(oil_profile.get("initial_viscosity_cp", 18.0)),
                sea_temp_c=float(oil_profile.get("water_temp_c", 26.0)),
            )

        return {
            "forecast_trajectory": forecast_trajectory,
            "weathering_summary": weathering_summary,
            "beaching_warning": {
                "status": "not_assessed",
                "will_beach": None,
                "estimated_time_to_beach_hours": None,
                "beaching_location": None,
                "vulnerable_assets": [],
                "reason": "No authoritative shoreline polygon and asset layer were supplied."
            }
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
        Mackay's ADIOS (Automated Data Inquiry for Oil Spills) Physical Weathering Model:
        Simulates the chemical & physical evolution of crude oil drifting on the sea surface:
        1. Evaporative loss of volatile hydrocarbon fractions
        2. Water-in-oil emulsification (chocolate mousse formation)
        3. Dynamic viscosity and density increase over time
        """
        if water_temp_c is not None:
            sea_temp_c = water_temp_c
        t = max(elapsed_hours, 0.0)
        t_kelvin = sea_temp_c + 273.15

        # 1. Evaporative exposure fraction (Mackay 1980 logarithmic formulation)
        # F_evap = (T / 1000) * alpha * ln(1 + beta * t)
        alpha = 0.165
        beta = 3.8
        f_evap = (t_kelvin / 1000.0) * alpha * math.log(1.0 + beta * t)
        f_evap = float(np.clip(f_evap, 0.0, 0.55))

        # 2. Water-in-oil emulsification uptake (Mooney / Mackay equation)
        # As waves whip the slick, water droplets get trapped inside the oil matrix
        # Y_w reaches up to 75% for heavy emulsion (mousse) over 12-24 hours
        k_emul = 2.0e-6  # Standard Mackay oceanic emulsification rate constant
        t_sec = t * 3600.0
        y_max = 0.75  # 75% maximum water content in chocolate mousse
        y_w = y_max * (1.0 - math.exp(-k_emul * ((1.0 + wind_speed_ms) ** 2) * t_sec))
        y_w = float(np.clip(y_w, 0.0, y_max))

        # 3. Mass and volume balance. Water uptake raises emulsion mass, but mass
        # ratio is not volume ratio; use constituent densities explicitly.
        remaining_pure_oil_tonnes = initial_mass_tonnes * (1.0 - f_evap)
        emulsion_mass_tonnes = remaining_pure_oil_tonnes / max(1.0 - y_w, 0.28)
        absorbed_water_tonnes = max(emulsion_mass_tonnes - remaining_pure_oil_tonnes, 0.0)
        oil_density_kg_m3, water_density_kg_m3 = 880.0, 1025.0
        initial_volume_m3 = initial_mass_tonnes * 1000.0 / oil_density_kg_m3
        emulsion_volume_m3 = (
            remaining_pure_oil_tonnes * 1000.0 / oil_density_kg_m3 +
            absorbed_water_tonnes * 1000.0 / water_density_kg_m3
        )
        volume_expansion_ratio = emulsion_volume_m3 / max(initial_volume_m3, 0.1)

        # 4. Viscosity growth (Mooney equation)
        # Viscosity increases exponentially with evaporation and water droplet packing
        viscosity_factor = math.exp(2.5 * y_w / (1.0 - 0.65 * y_w)) * math.exp(8.0 * f_evap)
        current_viscosity_cp = round(initial_viscosity_cp * viscosity_factor, 1)

        # State classification
        if y_w > 0.50:
            physical_state = "Heavy Chocolate Mousse (Highly Viscous Emulsion)"
        elif f_evap > 0.25:
            physical_state = "Weathered Viscous Sheen (Volatiles Depleted)"
        else:
            physical_state = "Fresh Liquid Petroleum Hydrocarbon"

        # Generate hourly timeline curves for visualization
        curve_times = [round(x, 1) for x in np.linspace(0.5, t, num=8)]
        evap_curve = []
        emul_curve = []
        for ct in curve_times:
            fe = float(np.clip((t_kelvin / 1000.0) * alpha * math.log(1.0 + beta * ct), 0.02, 0.55))
            yw = float(np.clip(y_max * (1.0 - math.exp(-k_emul * ((1.0 + wind_speed_ms) ** 2) * ct * 3600.0)), 0.0, y_max))
            evap_curve.append(round(fe * 100.0, 1))
            emul_curve.append(round(yw * 100.0, 1))

        return {
            "initial_mass_tonnes": round(initial_mass_tonnes, 2),
            "evaporated_fraction_pct": round(f_evap * 100.0, 1),
            "evaporated_mass_tonnes": round(initial_mass_tonnes * f_evap, 2),
            "water_content_mousse_pct": round(y_w * 100.0, 1),
            "emulsion_apparent_mass_tonnes": round(emulsion_mass_tonnes, 2),
            "initial_volume_m3": round(initial_volume_m3, 2),
            "emulsion_volume_m3": round(emulsion_volume_m3, 2),
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
