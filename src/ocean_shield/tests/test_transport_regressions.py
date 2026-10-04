"""Small deterministic transport/source regressions; no model downloads or field trials."""

import datetime
import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.ocean_shield.drift_engine import (
    DriftEngine, OceanCurrentField, OceanSourceError, METERS_PER_DEG_LAT,
)
from src.ocean_shield.ocean_data import OceanDataProvider, DataCoverageError


class PositiveJet(OceanCurrentField):
    def get_velocity_at(self, lat, lon, t_hours_relative=0.0):
        return 0.5 * (1.0 - math.exp(-(lat / 0.004) ** 2)), 0.0, 0.0, 0.0


class ScalarProvider:
    """A provider deliberately rejecting array calls."""
    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after
        self.metadata = {"source": "test scalar forcing"}

    def get_velocity_at(self, lat, lon, t_hours_relative=0.0):
        if not isinstance(lat, float) or not isinstance(lon, float):
            raise TypeError("scalar coordinates required")
        self.calls.append((lat, lon, t_hours_relative))
        if self.fail_after is not None and t_hours_relative > self.fail_after:
            raise DataCoverageError("requested stage time unavailable")
        return 0.25, 0.0, 0.0, 0.0


def constant_field(u=0.25, v=0.0):
    return OceanCurrentField(base_current_u=u, base_current_v=v,
                             base_wind_u=0.0, base_wind_v=0.0,
                             constant_vectors=True)


def memory_grid(time_bounds=(-1.0, 1.0), missing_wind=False):
    """Real interpolation contracts on a tiny in-memory grid, without source files."""
    provider = OceanDataProvider.__new__(OceanDataProvider)
    provider.is_loaded = True
    provider._grid_center_time_utc = None
    provider._detection_time_utc = None
    provider.metadata = {
        "time_min": time_bounds[0], "time_max": time_bounds[1],
        "lat_min": -0.05, "lat_max": 0.05,
        "lon_min": -0.05, "lon_max": 0.05, "has_real_wind": not missing_wind,
        "source": "small in-memory test grid",
    }
    axes = (np.array(time_bounds), np.array([-0.05, 0.05]), np.array([-0.05, 0.05]))
    provider.interpolators = {}
    for key, value in (("water_u", 0.25), ("water_v", 0.0),
                       ("wind_u", np.nan if missing_wind else 0.0), ("wind_v", 0.0)):
        provider.interpolators[key] = RegularGridInterpolator(axes, np.full((2, 2, 2), value), bounds_error=True)
    return provider


class TestPerParticleTransport(unittest.TestCase):
    def setUp(self):
        self.engine = DriftEngine(num_particles=80, wind_drift_factor=0.0, diffusion_coeff=0.0)

    def test_positive_bounded_jet_never_moves_particles_west(self):
        result = self.engine.run_forecast(0.0, 0.0, PositiveJet(), forecast_hours=0.1, time_step_minutes=6.0)
        initial, final = result["forecast_trajectory"]
        for before, after in zip(initial["particles_sample"], final["particles_sample"]):
            east_m = (after[0] - before[0]) * METERS_PER_DEG_LAT
            self.assertGreaterEqual(east_m, 0.0)
            self.assertLessEqual(east_m, 182.0)  # 0.5 m/s * 360 s, plus display rounding
            self.assertEqual(after[1], before[1])

        lat = np.array([-0.006, -0.002, 0.0, 0.001, 0.004, 0.007])
        lon = np.zeros_like(lat)
        next_lat, next_lon = self.engine._rk4_particle_step(PositiveJet(), lat, lon, 0.0, 360.0)
        expected_u = 0.5 * (1.0 - np.exp(-(lat / 0.004) ** 2))
        expected_lon = 360.0 * expected_u / (METERS_PER_DEG_LAT * np.cos(np.radians(lat)))
        np.testing.assert_allclose(next_lon, expected_lon, atol=2e-14)
        np.testing.assert_array_equal(next_lat, lat)
        _, previous_lon = self.engine._rk4_particle_step(PositiveJet(), lat, lon, 0.0, -360.0)
        np.testing.assert_allclose(previous_lon, -expected_lon, atol=2e-14)

    def test_constant_flow_uses_stage_latitude_geographic_metric(self):
        lat = np.array([25.0, 40.0, 60.0])
        lon = np.array([-10.0, -20.0, -30.0])
        u, v, dt = 0.4, 0.2, 7200.0
        next_lat, next_lon = self.engine._rk4_particle_step(constant_field(u, v), lat, lon, 0.0, dt)
        expected_lat = lat + v * dt / METERS_PER_DEG_LAT
        phi0, phi1 = np.radians(lat), np.radians(expected_lat)
        expected_lon = lon + u / v * 180.0 / math.pi * np.log(
            np.tan(math.pi / 4.0 + phi1 / 2.0) / np.tan(math.pi / 4.0 + phi0 / 2.0))
        np.testing.assert_allclose(next_lat, expected_lat, atol=1e-12)
        np.testing.assert_allclose(next_lon, expected_lon, atol=1e-10, rtol=0.0)

    def test_sheared_flow_matches_analytic_each_particle_solution(self):
        class Shear(OceanCurrentField):
            def get_velocity_at(self, lat, lon, t_hours_relative=0.0):
                return (0.3 + 4.0 * lat) * math.cos(math.radians(lat)), 0.2, 0.0, 0.0

        lat = np.array([-0.01, 0.0, 0.02])
        lon = np.array([-0.1, 0.0, 0.1])
        dt = 7200.0
        next_lat, next_lon = self.engine._rk4_particle_step(Shear(), lat, lon, 0.0, dt)
        expected_lat = lat + 0.2 * dt / METERS_PER_DEG_LAT
        expected_lon = lon + ((0.3 + 4.0 * lat) * dt + 2.0 * 0.2 * dt ** 2 / METERS_PER_DEG_LAT) / METERS_PER_DEG_LAT
        np.testing.assert_allclose(next_lat, expected_lat, atol=1e-14)
        np.testing.assert_allclose(next_lon, expected_lon, atol=2e-14)

    def test_time_dependent_forcing_and_fractional_final_step(self):
        class Timed(OceanCurrentField):
            def __init__(self):
                super().__init__()
                self.times = []

            def get_velocity_at(self, lat, lon, t_hours_relative=0.0):
                self.times.append(t_hours_relative)
                return (0.2 + 0.05 * t_hours_relative) * math.cos(math.radians(lat)), 0.0, 0.0, 0.0

        field = Timed()
        result = self.engine.run_forecast(0.0, 0.0, field, forecast_hours=0.7, time_step_minutes=15.0)
        trajectory = result["forecast_trajectory"]
        self.assertEqual([item["relative_time_hours"] for item in trajectory], [0.0, 0.25, 0.5, 0.7])
        expected_m = 3600.0 * (0.2 * 0.7 + 0.025 * 0.7 ** 2)
        actual_m = trajectory[-1]["centroid"]["lon"] * METERS_PER_DEG_LAT
        self.assertAlmostEqual(actual_m, expected_m, delta=0.06)
        self.assertAlmostEqual(max(field.times), 0.7)
        self.assertLessEqual(max(field.times), 0.7 + 1e-15)

    def test_short_forward_and_hindcast_reach_actual_endpoint(self):
        for duration in (0.0, 0.001, 0.1, 0.7):
            with self.subTest(duration=duration):
                forecast = self.engine.run_forecast(0.0, 0.0, constant_field(), forecast_hours=duration, time_step_minutes=15.0)
                with patch.object(self.engine, "_compute_kde_hdr_contours", return_value={}):
                    hindcast = self.engine.run_hindcast(0.0, 0.0, constant_field(),
                        target_slick_age_hours=duration, max_lookback_hours=1.0, time_step_minutes=15.0)
                self.assertEqual(forecast["forecast_trajectory"][-1]["relative_time_hours"], duration)
                self.assertEqual(forecast["simulated_duration_hours"], duration)
                self.assertEqual(hindcast["hindcast_trajectory"][-1]["relative_time_hours"], -duration)
                origin = hindcast["origin_release_point"]
                self.assertEqual(origin["estimated_t0_hours_relative"], -duration)
                self.assertEqual(origin["assumed_slick_age_hours"], duration)
                self.assertIsNone(origin["confidence_percent"])
                expected_lon = 0.25 * duration * 3600.0 / METERS_PER_DEG_LAT
                self.assertAlmostEqual(forecast["forecast_trajectory"][-1]["centroid"]["lon"], expected_lon, delta=5.1e-7)
                self.assertAlmostEqual(origin["lon"], -expected_lon, delta=5.1e-7)

    def test_counterfactual_fractional_duration_has_no_minimum_or_overshoot(self):
        for duration in (0.0, 0.001, 0.1, 0.7):
            with self.subTest(duration=duration):
                result = self.engine.run_forward_counterfactual(0.0, 0.0, -duration, constant_field(), 0.0, 0.0,
                                                                time_step_minutes=15.0)
                self.assertEqual(result["release_state"]["duration_hours"], duration)
                self.assertEqual(result["forward_trajectory"][-1]["relative_time_hours"], 0.0)
                expected_lon = 0.25 * duration * 3600.0 / METERS_PER_DEG_LAT
                self.assertAlmostEqual(result["predicted_at_t0"]["centroid_lon"], expected_lon, delta=5.1e-7)
                metrics = result["verification_metrics"]
                self.assertIsNone(metrics["physical_causality_score"])
                self.assertIsNone(metrics["predicted_containment_percent"])
                self.assertIsNone(metrics["jaccard_index"])
                self.assertTrue(result["verdict"].startswith("CONDITIONAL_"))

    def test_reverse_transport_does_not_invert_diffusion(self):
        other = DriftEngine(num_particles=80, wind_drift_factor=0.0, diffusion_coeff=100.0)
        with patch.object(self.engine, "_compute_kde_hdr_contours", return_value={}), patch.object(other, "_compute_kde_hdr_contours", return_value={}):
            a = self.engine.run_hindcast(0.0, 0.0, constant_field(), target_slick_age_hours=0.1)
            b = other.run_hindcast(0.0, 0.0, constant_field(), target_slick_age_hours=0.1)
        self.assertEqual(a["hindcast_trajectory"], b["hindcast_trajectory"])
        self.assertEqual(b["transport_model"]["diffusion_status"], "NOT_INVERTED_REVERSE_ADVECTION_ONLY")

    def test_kde_is_conditional_cloud_coverage_not_origin_confidence(self):
        result = self.engine.run_hindcast(0.0, 0.0, constant_field(), target_slick_age_hours=0.1)
        contours = result["kde_origin_contours"]
        self.assertIsNone(contours["origin_probability"])
        self.assertIn("not validated origin probabilities", contours["coverage_interpretation"])
        self.assertEqual(len(contours["contours"]), 3)
        for contour in contours["contours"]:
            self.assertIn("conditional generated-cloud coverage", contour["label"])
            self.assertIn("NOT_ORIGIN_PROBABILITY", contour["coverage_status"])
        one = DriftEngine(num_particles=1, diffusion_coeff=0.0)
        degenerate = one.run_hindcast(0.0, 0.0, constant_field(), target_slick_age_hours=0.0)
        self.assertEqual(degenerate["kde_origin_contours"]["status"], "DEGENERATE_GENERATED_CLOUD")
        self.assertEqual(degenerate["kde_origin_contours"]["contours"], [])

    def test_velocity_to_windage_turns_right_north_left_south(self):
        engine = DriftEngine(wind_drift_factor=0.1, deflection_angle_deg=15.0, diffusion_coeff=0.0)
        u_n, v_n = engine._compute_drift_vector(0.0, 0.0, 10.0, 0.0, True)
        u_s, v_s = engine._compute_drift_vector(0.0, 0.0, 10.0, 0.0, False)
        self.assertAlmostEqual(u_n, math.cos(math.radians(15.0)))
        self.assertAlmostEqual(u_s, u_n)
        self.assertLess(v_n, 0.0)
        self.assertGreater(v_s, 0.0)
        wind = OceanCurrentField(base_current_u=0.0, base_current_v=0.0,
                                 base_wind_u=10.0, base_wind_v=0.0, constant_vectors=True)
        lat_rate, _ = engine._velocity_rates(wind, np.array([-10.0, 0.0, 10.0]), np.zeros(3), 0.0)
        self.assertGreater(lat_rate[0], 0.0)
        self.assertEqual(lat_rate[1], 0.0)
        self.assertLess(lat_rate[2], 0.0)

    def test_antimeridian_centroid_and_samples_remain_geographic(self):
        result = self.engine.run_forecast(0.0, 179.999, constant_field(1.0), forecast_hours=0.1, time_step_minutes=15.0)
        self.assertGreater(abs(result["forecast_trajectory"][0]["centroid"]["lon"]), 179.99)
        expected = (179.999 + 360.0 / METERS_PER_DEG_LAT + 180.0) % 360.0 - 180.0
        self.assertAlmostEqual(result["forecast_trajectory"][-1]["centroid"]["lon"], expected, delta=5.1e-7)
        for item in result["forecast_trajectory"]:
            for lon, lat in item["particles_sample"]:
                self.assertTrue(-180.0 <= lon <= 180.0 and -85.0 <= lat <= 85.0)

    def test_demo_coastline_trigger_never_changes_not_assessed_state(self):
        for threshold in (0.001, 1.0, None):
            result = self.engine.run_forecast(0.0, 0.0, constant_field(0.0, 1.0),
                forecast_hours=0.1, time_step_minutes=6.0, coastline_lat_threshold=threshold)
            warning = result["beaching_warning"]
            self.assertEqual(warning["status"], "NOT_ASSESSED")
            for key in ("will_beach", "estimated_time_to_beach_hours", "beaching_location"):
                self.assertIsNone(warning[key])
            self.assertEqual(warning["vulnerable_assets"], [])
            self.assertTrue(all(item["beached"] is None for item in result["forecast_trajectory"]))
            if threshold == 0.001:
                self.assertEqual(warning["demo_cues"]["latitude_trigger"]["status"], "DEMO_TRIGGER_REACHED")


class TestTransportSources(unittest.TestCase):
    def setUp(self):
        self.engine = DriftEngine(num_particles=8, wind_drift_factor=0.0, diffusion_coeff=0.0)

    def test_scalar_provider_is_evaluated_per_particle_per_stage(self):
        provider = ScalarProvider()
        self.engine.run_forecast(0.0, 0.0, OceanCurrentField(data_provider=provider), forecast_hours=0.1, time_step_minutes=6.0)
        self.assertEqual(len(provider.calls), 1 + 4 * 8)
        self.assertEqual(len({round(call[0], 10) for call in provider.calls[1:9]}), 8)
        self.assertEqual({call[2] for call in provider.calls}, {0.0, 0.05, 0.1})

    def test_provider_stage_failure_is_explicit_in_all_directions(self):
        for mode in ("forecast", "hindcast", "counterfactual"):
            class Broken(ScalarProvider):
                def get_velocity_at(self, lat, lon, t_hours_relative=0.0):
                    if t_hours_relative != (0.0 if mode != "counterfactual" else -0.1):
                        raise RuntimeError("source time/grid unavailable")
                    return super().get_velocity_at(lat, lon, t_hours_relative)
            with self.subTest(mode=mode):
                field = OceanCurrentField(data_provider=Broken())
                with self.assertRaisesRegex(OceanSourceError, "source failure.*source time/grid unavailable"):
                    if mode == "forecast":
                        self.engine.run_forecast(0.0, 0.0, field, forecast_hours=0.1)
                    elif mode == "hindcast":
                        self.engine.run_hindcast(0.0, 0.0, field, target_slick_age_hours=0.1)
                    else:
                        self.engine.run_forward_counterfactual(0.0, 0.0, -0.1, field, 0.0, 0.0)

    def test_batch_failure_is_not_retried_with_scalar_base_vectors(self):
        class BrokenBatch(ScalarProvider):
            def __init__(self):
                super().__init__()
                self.batch_calls = 0

            def get_velocities_at(self, lat, lon, time):
                self.batch_calls += 1
                raise DataCoverageError("batch grid unavailable")

        provider = BrokenBatch()
        with self.assertRaisesRegex(OceanSourceError, "batch grid unavailable"):
            self.engine.run_forecast(0.0, 0.0, OceanCurrentField(data_provider=provider), forecast_hours=0.1)
        self.assertEqual(provider.batch_calls, 1)
        self.assertEqual(len(provider.calls), 1)

    def test_batch_and_scalar_grid_interpolation_agree_and_fail_closed(self):
        provider = memory_grid((-0.1, 0.1))
        lat, lon = np.array([-0.01, 0.0, 0.01]), np.array([0.01, 0.0, -0.01])
        batch = provider.get_velocities_at(lat, lon, 0.05)
        scalar = np.asarray([provider.get_velocity_at(float(a), float(b), 0.05) for a, b in zip(lat, lon)]).T
        np.testing.assert_array_equal(np.asarray(batch), scalar)
        with self.assertRaises(DataCoverageError):
            provider.get_velocity_at(0.0, 0.0, 0.1001)
        with self.assertRaises(DataCoverageError):
            provider.get_velocities_at(np.array([0.0, 0.051]), np.zeros(2), 0.0)
        field = OceanCurrentField(data_provider=provider)
        result = self.engine.run_forecast(0.0, 0.0, field, forecast_hours=0.1)
        self.assertEqual(result["simulated_duration_hours"], 0.1)
        with self.assertRaises(OceanSourceError):
            self.engine.run_forecast(0.0, 0.0, field, forecast_hours=0.2)
        with self.assertRaises(OceanSourceError):
            self.engine.run_forecast(0.0499, 0.0, field, forecast_hours=0.1)

    def test_missing_wind_and_invalid_vectors_are_source_failures(self):
        with self.assertRaisesRegex(OceanSourceError, "missing current or wind"):
            self.engine.run_forecast(0.0, 0.0, OceanCurrentField(data_provider=memory_grid(missing_wind=True)), forecast_hours=0.1)
        for vectors in ((np.nan, 0.0, 0.0, 0.0), (0.0, np.inf, 0.0, 0.0),
                        (0.0, 0.0, 201.0, 0.0), (0.0, 0.0, 0.0), (True, 0.0, 0.0, 0.0), "0000"):
            class Invalid:
                def get_velocity_at(self, lat, lon, time):
                    return vectors
            with self.subTest(vectors=vectors), self.assertRaises(OceanSourceError):
                self.engine.run_forecast(0.0, 0.0, OceanCurrentField(data_provider=Invalid()), forecast_hours=0.1)

    def test_wind_is_time_interpolated_and_unavailable_times_raise(self):
        provider = memory_grid()
        t0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        provider._detection_time_utc = t0
        provider._openmeteo_wind = [(t0, 0.0, -2.0, 2.0), (t0 + datetime.timedelta(hours=1), 4.0, 2.0, 4.0)]
        self.assertEqual(provider.get_real_wind_at(0.5), (2.0, 0.0))
        with self.assertRaisesRegex(DataCoverageError, "wind archive coverage"):
            provider.get_velocity_at(0.0, 0.0, -0.5)

    def test_missing_grid_and_missing_netcdf_library_do_not_generate_replacements(self):
        import src.ocean_shield.ocean_data as module
        with patch.object(module, "HAS_NETCDF4", False), self.assertRaises(DataCoverageError):
            OceanDataProvider("unavailable-test-source.nc")
        with patch.object(module, "HAS_NETCDF4", True), patch.object(module.os.path, "exists", return_value=False), \
                patch.object(OceanDataProvider, "create_standard_ocean_grid") as generate:
            with self.assertRaisesRegex(DataCoverageError, "no synthetic replacement"):
                OceanDataProvider("unavailable-test-source.nc")
            generate.assert_not_called()

    def test_absolute_seconds_grid_converts_to_hours_and_preserves_masked_wind(self):
        import src.ocean_shield.ocean_data as module

        class Variable:
            def __init__(self, data, units=""):
                self.data, self.units = data, units

            def __getitem__(self, key):
                return self.data[key]

        class Dataset:
            source = "in-memory test model forcing"
            data_origin = "test grid; source authenticity not asserted"
            institution = "test fixture"

            def __init__(self, masked):
                shape = (3, 2, 2)
                wind = np.ma.array(np.zeros(shape, dtype=np.int16), mask=np.full(shape, masked))
                self.variables = {
                    "time": Variable(np.array([0.0, 3600.0, 7200.0]), "seconds since 2026-01-01 00:00:00"),
                    "lat": Variable(np.array([-0.05, 0.05])), "lon": Variable(np.array([-0.05, 0.05])),
                    "water_u": Variable(np.full(shape, 0.25), "m s-1"), "water_v": Variable(np.zeros(shape), "m s-1"),
                    "wind_u": Variable(wind, "m s-1"), "wind_v": Variable(np.zeros(shape), "m s-1"),
                }

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        t0 = datetime.datetime(2026, 1, 1)
        dates = [t0 + datetime.timedelta(hours=i) for i in range(3)]
        for masked in (False, True):
            fake_nc = SimpleNamespace(Dataset=lambda *args: Dataset(masked), num2date=lambda *args, **kwargs: dates)
            provider = memory_grid()
            with patch.object(module, "HAS_NETCDF4", True), patch.object(module, "nc", fake_nc):
                provider.load_netcdf("in-memory.nc")
            self.assertEqual((provider.metadata["time_min"], provider.metadata["time_max"]), (-1.0, 1.0))
            provider.bind_detection_time("2026-01-01T01:00:00Z", 1.0, 1.0)
            if masked:
                with self.assertRaisesRegex(DataCoverageError, "missing current or wind"):
                    provider.get_velocity_at(0.0, 0.0, 0.0)
            else:
                self.assertEqual(provider.get_velocity_at(0.0, 0.0, 0.0), (0.25, 0.0, 0.0, 0.0))

        dataset = Dataset(False)
        dataset.variables["water_u"].units = "cm/s"
        fake_nc = SimpleNamespace(Dataset=lambda *args: dataset, num2date=lambda *args, **kwargs: dates)
        with patch.object(module, "HAS_NETCDF4", True), patch.object(module, "nc", fake_nc):
            with self.assertRaisesRegex(DataCoverageError, "m/s velocity units"):
                provider.load_netcdf("in-memory.nc")
        self.assertFalse(provider.is_loaded)

    def test_failed_source_rebinding_invalidates_previous_alignment(self):
        provider = memory_grid()
        t0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        provider._grid_center_time_utc = t0
        provider.metadata.update(source_start_utc=(t0 - datetime.timedelta(hours=1)).isoformat(),
                                 source_end_utc=(t0 + datetime.timedelta(hours=1)).isoformat())
        provider.bind_detection_time(t0.isoformat(), 0.5, 0.5)
        self.assertTrue(provider.has_coverage(0.5, 0.5))
        provider._openmeteo_wind = [(t0, 0.0, 0.0, 0.0), (t0 + datetime.timedelta(hours=0.25), 0.0, 0.0, 0.0)]
        self.assertFalse(provider.has_coverage(0.5, 0.5))
        with self.assertRaises(DataCoverageError):
            provider.bind_detection_time(t0.isoformat(), 0.5, 0.5)
        self.assertIsNone(provider._detection_time_utc)
        with self.assertRaisesRegex(DataCoverageError, "detection timestamp"):
            provider.get_velocity_at(0.0, 0.0, 0.0)

    def test_scalar_only_provider_subclass_retains_its_custom_field(self):
        class Custom(OceanDataProvider):
            def __init__(self):
                self.calls = 0

            def get_velocity_at(self, lat, lon, t_hours_relative=0.0):
                self.calls += 1
                return 0.25, 0.0, 0.0, 0.0

        provider = Custom()
        result = self.engine.run_forecast(0.0, 0.0, OceanCurrentField(data_provider=provider), forecast_hours=0.1)
        self.assertEqual(provider.calls, 1 + 4 * 8)
        self.assertEqual(result["simulated_duration_hours"], 0.1)

    def test_exact_fractional_clock_does_not_round_past_source_endpoint(self):
        duration = 0.7173299113638919
        step_h = 0.2772154519411929
        provider = memory_grid((-1.0, duration))
        result = self.engine.run_forecast(0.0, 0.0, OceanCurrentField(data_provider=provider),
                                         forecast_hours=duration, time_step_minutes=step_h * 60.0)
        self.assertEqual(result["forecast_trajectory"][-1]["relative_time_hours"], duration)

    def test_coverage_metadata_does_not_round_up_unknown_time(self):
        provider = memory_grid((-1.0, 1.0006))
        t0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        provider._grid_center_time_utc = t0
        provider.metadata.update(source_start_utc=(t0 - datetime.timedelta(hours=1)).isoformat(),
                                 source_end_utc=(t0 + datetime.timedelta(hours=1.0006)).isoformat())
        provider.bind_detection_time(t0.isoformat(), 0.0, 1.0006)
        self.assertTrue(provider.has_coverage(0.0, 1.0006))
        self.assertFalse(provider.has_coverage(0.0, 1.0008))
        with self.assertRaises(DataCoverageError):
            provider.bind_detection_time("2026-01-01T00:00:00", 0.0, 1.0)

    def test_empty_wind_archive_does_not_claim_observed_archive_forcing(self):
        provider = memory_grid()
        provider._openmeteo_source = "stale previous archive"
        provider._openmeteo_wind = []
        summary = provider.get_telemetry_summary(0.0, 0.0)
        self.assertEqual(summary["format"], "CF-1.8 NetCDF-4")
        self.assertEqual(summary["wind_source"], provider.metadata["source"])
        self.assertFalse(summary["surface_wind_10m"]["is_real_observed"])
        self.assertFalse(summary["surface_wind_10m"]["archive_time_aligned"])


class TestTransportBoundariesAndWeathering(unittest.TestCase):
    def setUp(self):
        self.engine = DriftEngine(num_particles=8, wind_drift_factor=0.0, diffusion_coeff=0.0)

    def test_engine_and_field_configuration_reject_invalid_domains(self):
        for count in (0, -1, 1.5, True, 10001, np.nan):
            with self.subTest(count=count), self.assertRaises(ValueError):
                DriftEngine(num_particles=count)
        for name, values in (("wind_drift_factor", (-0.1, np.nan, np.inf, 1.01)),
                             ("diffusion_coeff", (-1.0, np.nan, np.inf, 1e7)),
                             ("deflection_angle_deg", (-1.0, 91.0, np.nan, np.inf))):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    DriftEngine(**{name: value})
        for params in ({"base_current_u": np.nan}, {"base_wind_v": np.inf},
                       {"tidal_period_h": 0.0}, {"tidal_period_h": -1.0},
                       {"tidal_amplitude": -1.0}, {"constant_vectors": "yes"}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                OceanCurrentField(**params)

    def test_invalid_coordinates_durations_and_steps_fail_before_arrays(self):
        for lat, lon in ((np.nan, 0.0), (np.inf, 0.0), (90.0, 0.0), (-90.0, 0.0),
                         (85.0001, 0.0), (0.0, np.nan), (0.0, 180.01), (0.0, -180.01), (True, 0.0)):
            with self.subTest(lat=lat, lon=lon), self.assertRaises(ValueError):
                self.engine.run_forecast(lat, lon, constant_field(), forecast_hours=0.1)
        for duration in (-0.1, np.nan, np.inf, 745.0, True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.engine.run_forecast(0.0, 0.0, constant_field(), forecast_hours=duration)
        for step in (0.0, -1.0, np.nan, np.inf, 1441.0, 1e-300, 0.00001):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.engine.run_forecast(0.0, 0.0, constant_field(), forecast_hours=0.1, time_step_minutes=step)
        for age in (-0.1, np.nan, np.inf, 1.1):
            with self.subTest(age=age), self.assertRaises(ValueError):
                self.engine.run_hindcast(0.0, 0.0, constant_field(), target_slick_age_hours=age, max_lookback_hours=1.0)
        with self.assertRaises(ValueError):
            self.engine.run_hindcast(0.0, 0.0, constant_field(), target_slick_age_hours=0.1, max_lookback_hours=-1.0)
        with self.assertRaises(ValueError):
            self.engine.run_forecast(0.0, 0.0, constant_field(), forecast_hours=0.1, coastline_lat_threshold=np.nan)
        with self.assertRaisesRegex(ValueError, "particle-step resource limit"):
            DriftEngine(num_particles=10000).run_forecast(0.0, 0.0, constant_field(), forecast_hours=744.0, time_step_minutes=144.0)
        self.engine.num_particles = 0
        with self.assertRaises(ValueError):
            self.engine.run_forecast(0.0, 0.0, constant_field(), forecast_hours=0.0)

    def test_advected_polar_stage_is_rejected(self):
        lat, lon = np.array([84.999]), np.array([0.0])
        with self.assertRaisesRegex(ValueError, "polar transport is unsupported"):
            self.engine._rk4_particle_step(constant_field(0.0, 1.0), lat, lon, 0.0, 3600.0)

    def test_counterfactual_invalid_release_and_geometry_do_not_default(self):
        for params in ({"release_time_rel_h": 0.1}, {"release_time_rel_h": np.nan},
                       {"observed_slick_area_km2": 0.0}, {"observed_slick_area_km2": -1.0},
                       {"observed_slick_area_km2": np.inf}, {"observed_slick_polygon": [[0, 0], [1, 1]]},
                       {"observed_slick_polygon": [[0, 0], [1, 1], [np.nan, 0]]}):
            args = dict(release_lat=0.0, release_lon=0.0, release_time_rel_h=-0.1,
                        current_field=constant_field(), observed_slick_lat=0.0, observed_slick_lon=0.0)
            args.update(params)
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.engine.run_forward_counterfactual(**args)

    def test_counterfactual_circle_iou_is_not_blended_with_containment(self):
        result = self.engine.run_forward_counterfactual(0.0, 0.0, 0.0, constant_field(), 0.0, 0.0,
                                                        observed_slick_area_km2=math.pi * 2.0 ** 2)
        metrics = result["verification_metrics"]
        self.assertEqual(metrics["predicted_containment_percent"], 100.0)
        predicted_radius = result["predicted_at_t0"]["spread_radius_km"]
        self.assertAlmostEqual(metrics["jaccard_index"], predicted_radius ** 2 / 2.0 ** 2, delta=0.001)
        polygon = [[0.05, 0.05], [0.06, 0.05], [0.06, 0.06], [0.05, 0.06]]
        result = self.engine.run_forward_counterfactual(0.0, 0.0, 0.0, constant_field(), 0.0, 0.0,
            observed_slick_area_km2=math.pi * 2.0 ** 2, observed_slick_polygon=polygon)
        self.assertEqual(result["verification_metrics"]["predicted_containment_percent"], 0.0)
        self.assertIsNone(result["verification_metrics"]["jaccard_index"])

    def test_counterfactual_polygon_containment_handles_antimeridian(self):
        polygon = [[-0.02, 179.98], [0.02, 179.98], [0.02, -179.98], [-0.02, -179.98]]
        result = self.engine.run_forward_counterfactual(0.0, 179.999, 0.0, constant_field(), 0.0, 179.999,
                                                        observed_slick_polygon=polygon)
        self.assertEqual(result["verification_metrics"]["predicted_containment_percent"], 100.0)

    def test_zero_time_weathering_preserves_mass_volume_viscosity_and_curves(self):
        result = self.engine.compute_oil_weathering(0.0, initial_mass_tonnes=0.001, initial_viscosity_cp=0.125)
        self.assertEqual(result["initial_mass_tonnes"], 0.001)
        self.assertEqual(result["emulsion_apparent_mass_tonnes"], 0.001)
        self.assertEqual(result["evaporated_mass_tonnes"], 0.0)
        self.assertEqual(result["evaporated_fraction_pct"], 0.0)
        self.assertEqual(result["water_content_mousse_pct"], 0.0)
        self.assertEqual(result["initial_volume_m3"], result["emulsion_volume_m3"])
        self.assertEqual(result["volume_expansion_ratio"], 1.0)
        self.assertEqual(result["viscosity_cp"], 0.125)
        self.assertEqual(result["weathering_timeline"], {"hours": [0.0], "evaporation_pct": [0.0], "water_uptake_pct": [0.0]})
        self.assertFalse(result["is_adios_model"])
        self.assertEqual(result["model_status"], "ILLUSTRATIVE_GENERIC_SENSITIVITY")

    def test_weathering_mass_fraction_fix_keeps_density_based_volume(self):
        result = self.engine.compute_oil_weathering(18.0, initial_mass_tonnes=100.0,
                                                   wind_speed_ms=5.0, sea_temp_c=28.0)
        remaining_oil = result["initial_mass_tonnes"] - result["evaporated_mass_tonnes"]
        water = result["emulsion_apparent_mass_tonnes"] - remaining_oil
        mass_fraction = water / result["emulsion_apparent_mass_tonnes"]
        self.assertAlmostEqual(mass_fraction * 100.0, result["water_content_mousse_pct"], delta=0.051)
        expected_volume = 1000.0 * (remaining_oil / 880.0 + water / 1025.0)
        self.assertAlmostEqual(result["emulsion_volume_m3"], expected_volume, delta=0.02)
        self.assertLess(result["volume_expansion_ratio"], result["emulsion_apparent_mass_tonnes"] / 100.0)
        self.assertGreater(result["volume_expansion_ratio"], 1.0)
        self.assertEqual(result["weathering_timeline"]["hours"][0], 0.0)
        self.assertEqual(result["weathering_timeline"]["evaporation_pct"][0], 0.0)

    def test_weathering_invalid_inputs_raise_without_silent_fallback(self):
        for params in ({"elapsed_hours": -1.0}, {"elapsed_hours": np.nan}, {"elapsed_hours": np.inf},
                       {"initial_mass_tonnes": 0.0}, {"initial_mass_tonnes": -1.0}, {"initial_mass_tonnes": np.inf},
                       {"wind_speed_ms": -1.0}, {"wind_speed_ms": np.nan}, {"sea_temp_c": -273.15},
                       {"water_temp_c": 51.0}, {"initial_viscosity_cp": 0.0}, {"initial_viscosity_cp": np.nan}):
            args = {"elapsed_hours": 1.0}
            args.update(params)
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.engine.compute_oil_weathering(**args)
        for params in ({"initial_mass_tonnes": 0.0}, {"initial_mass_tonnes": np.nan},
                       {"oil_profile": {"water_temp_c": 100.0}}, {"oil_profile": {"initial_viscosity_cp": -1.0}},
                       {"oil_profile": []}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.engine.run_forecast(0.0, 0.0, constant_field(), forecast_hours=0.1, **params)


if __name__ == "__main__":
    unittest.main()
