"""
Unit tests for Fay's (1971) Gravity-Viscous Hydrodynamic Spreading Model.
Validates:
  - Inversion of Fay's law: A(t) = C * t^(1/2) => t = (A/C)^2
  - Bounded sensitivity calculations (+/- 50% spill volume range)
  - Strict input domain validation (rejecting non-finite, zero, or negative areas, and unphysical densities)
  - Seamless SAR engine payload integration
"""

import math
import unittest
import cv2
import numpy as np

from src.ocean_shield.drift_engine import estimate_fay_spill_age
from src.ocean_shield.sar_engine import SAREngine


class TestFaySpreadingModel(unittest.TestCase):

    def test_standard_fay_calculation(self):
        """Declared inputs produce a conditional inversion and volume sensitivity."""
        res = estimate_fay_spill_age(area_km2=5.0, spill_volume_m3=1000.0)

        self.assertIn("estimated_age_hours", res)
        self.assertIn("estimated_age_days", res)
        self.assertIn("lookback_window_hours", res)
        self.assertIn("physics_regime", res)

        age_h = res["estimated_age_hours"]
        self.assertGreater(age_h, 0.0)
        self.assertTrue(math.isfinite(age_h))

        # Lookback window: lower bound should be less than upper bound
        t_min, t_max = res["lookback_window_hours"]
        self.assertLess(t_min, t_max)
        self.assertLessEqual(t_min, age_h)
        self.assertGreaterEqual(t_max, age_h)
        self.assertEqual(res["status"], "CONDITIONAL_SENSITIVITY_ONLY")
        self.assertEqual(res["age_inference_status"], "NOT_INFERRED_FROM_SINGLE_SAR_SCENE")

    def test_fay_domain_validation(self):
        """Must strictly reject invalid inputs: non-finite values, non-positive areas, non-floating oil."""
        # Non-finite values
        with self.assertRaises(ValueError):
            estimate_fay_spill_age(float("nan"))
        with self.assertRaises(ValueError):
            estimate_fay_spill_age(float("inf"))

        # Non-positive area
        with self.assertRaises(ValueError):
            estimate_fay_spill_age(0.0)
        with self.assertRaises(ValueError):
            estimate_fay_spill_age(-2.5)

        # Non-positive volume
        with self.assertRaises(ValueError):
            estimate_fay_spill_age(5.0, spill_volume_m3=0.0)
        with self.assertRaises(ValueError):
            estimate_fay_spill_age(5.0, spill_volume_m3=-500.0)

        # Oil denser than or equal to water (non-floating)
        with self.assertRaises(ValueError):
            estimate_fay_spill_age(5.0, oil_density_kg_m3=1025.0, water_density_kg_m3=1025.0)
        with self.assertRaises(ValueError):
            estimate_fay_spill_age(5.0, oil_density_kg_m3=1050.0, water_density_kg_m3=1025.0)

        # Booleans, strings, non-positive densities, and overflow-prone bounds
        # must not pass Python's bool-is-int or implicit string conversion rules.
        for kwargs in (
            {"area_km2": True},
            {"area_km2": "5"},
            {"area_km2": np.float64("nan")},
            {"area_km2": 5.0, "oil_density_kg_m3": 0.0},
            {"area_km2": 5.0, "water_density_kg_m3": 0.0},
            {"area_km2": 5.0, "water_kinematic_viscosity": True},
            {"area_km2": 1.0e20},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    estimate_fay_spill_age(**kwargs)

    def test_sar_engine_integration(self):
        """Scene processing must not infer a Fay age from an unobserved volume."""
        sar = SAREngine()
        image = np.full((128, 128), 128, np.uint8)
        cv2.ellipse(image, (64, 64), (20, 8), 0, 0, 360, 30, -1)
        res = sar.process_sar_scene(image, 0, 0, model_type="cfar_edge")["primary_slick"]
        self.assertIsNotNone(res)
        self.assertIsNone(res["estimated_age_hours"])
        self.assertEqual(res["fay_spreading_age"]["status"], "NOT_ASSESSED")

    def test_fay_regime_and_bounds_do_not_claim_phase_validity_or_confidence(self):
        for area, volume in ((0.1, 1000), (5, 1000), (5, 1)):
            with self.subTest(area=area, volume=volume):
                res = estimate_fay_spill_age(area, spill_volume_m3=volume)
                self.assertIsNone(res["regime_valid"])
                self.assertEqual(res["regime_status"], "NOT_ASSESSED")
                self.assertEqual(res["physics_regime"], "ASSUMED_GRAVITY_VISCOUS_SCALING")
                self.assertEqual(res["bounds_kind"], "VOLUME_SENSITIVITY_ONLY_NOT_CONFIDENCE_INTERVAL")
                self.assertIn("not confidence intervals", res["scientific_caveat"])
                self.assertIn("No NOAA GNOME/ADIOS equivalence is verified", res["scientific_caveat"])

    def test_volume_sensitivity_contains_small_volume_hypotheses(self):
        """The documented +/-50% range must contain the declared volume and central age."""
        for volume in (1.0, 10.0, 1000.0):
            with self.subTest(volume=volume):
                result = estimate_fay_spill_age(5.0, spill_volume_m3=volume)
                self.assertEqual(result["volume_sensitivity_m3"], [volume * 0.5, volume * 1.5])
                low, high = result["lookback_window_hours"]
                self.assertLess(low, result["estimated_age_hours"])
                self.assertGreater(high, result["estimated_age_hours"])


if __name__ == "__main__":
    unittest.main()
