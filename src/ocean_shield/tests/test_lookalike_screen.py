"""
Unit tests for Scale-Free Dark-Patch Look-Alike Screening Module.
Validates:
  - Robustness to degenerate, empty, and non-finite (NaN / Inf) inputs
  - Scale-free ratio calculations on synthetic test patterns
  - Uncalibrated triage classification and disclaimers contract
"""

import math
import unittest
import numpy as np
import cv2

from src.ocean_shield.lookalike_screen import (
    compute_scale_free_ratios,
    evaluate_lookalike_screening,
    LookalikeScreenResult,
)


class TestLookalikeScreen(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 200x200 8-bit image with ambient sea background (mean ~ 128, std ~ 15)
        np.random.seed(42)
        self.sea_bg = np.random.normal(128, 15, (200, 200)).clip(0, 255).astype(np.uint8)

    def test_empty_or_corrupt_inputs(self):
        """Must handle empty, None, or sub-pixel contours gracefully without raising exceptions."""
        empty_cnt = np.array([], dtype=np.int32)
        res_empty = compute_scale_free_ratios(self.sea_bg, empty_cnt)
        self.assertIn("darkness_z", res_empty)
        self.assertIsNone(res_empty["darkness_z"])
        self.assert_unavailable(res_empty, "INVALID_INPUT")

        # None inputs
        res_none = compute_scale_free_ratios(None, None)
        self.assertIsNone(res_none["darkness_z"])
        self.assert_unavailable(res_none, "INVALID_INPUT")

        # Image smaller than 5x5
        tiny_img = np.zeros((3, 3), dtype=np.uint8)
        cnt = np.array([[[1, 1]], [[1, 2]], [[2, 1]]], dtype=np.int32)
        res_tiny = compute_scale_free_ratios(tiny_img, cnt)
        self.assertIsNone(res_tiny["darkness_z"])
        self.assert_unavailable(res_tiny, "INVALID_INPUT")

    def test_nan_inf_image_hardening(self):
        """Missing pixels outside the local support must not change valid triage."""
        float_img = self.sea_bg.astype(np.float32)
        float_img[10:20, 10:20] = np.nan
        float_img[30:40, 30:40] = np.inf

        # Draw a synthetic dark patch
        cv2.circle(float_img, (100, 100), 20, 40.0, -1)
        cnt = np.array([
            [[80, 80]], [[120, 80]], [[120, 120]], [[80, 120]]
        ], dtype=np.int32)

        ratios = compute_scale_free_ratios(float_img, cnt)
        self.assertEqual(ratios["status"], "ASSESSED")
        for key, val in evaluate_lookalike_screening(ratios).features.items():
            self.assertTrue(math.isfinite(val), f"Ratio {key} was non-finite: {val}")

    def test_elongated_dark_slick_triage(self):
        """An elongated, sharply damped dark patch should receive high contrast candidate triage."""
        img = self.sea_bg.copy()
        # Draw dark elongated ellipse (major axis 60, minor axis 15, value ~30)
        cv2.ellipse(img, (100, 100), (60, 15), 30, 0, 360, 30, -1)

        # Extract contour of the ellipse
        mask = (img < 50).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertTrue(len(contours) > 0)
        slick_cnt = max(contours, key=cv2.contourArea)

        ratios = compute_scale_free_ratios(img, slick_cnt)
        self.assertGreater(ratios["darkness_z"], 1.5, "Expected strong darkness contrast")
        self.assertGreater(ratios["elongation"], 2.0, "Expected significant elongation")

        result = evaluate_lookalike_screening(ratios)
        self.assertIsInstance(result, LookalikeScreenResult)
        self.assertIn(result.verdict, ["CANDIDATE_HIGH_CONTRAST", "AMBIGUOUS_INTERMEDIATE"])
        self.assertGreaterEqual(result.screening_index, 0.50)
        self.assertFalse(hasattr(result, "oil_probability"), "A triage index must not be exposed as probability")
        self.assertEqual(result.calibration_status, "UNCALIBRATED_EXPERIMENTAL_HEURISTIC")
        self.assertEqual(result.status, "ASSESSED")
        self.assertIn("uncalibrated morphological heuristic", result.disclaimer)

    def test_shallow_diffuse_lookalike_triage(self):
        """A shallow, round patch with diffuse boundary should receive low-priority lookalike triage."""
        ratios = {
            "darkness_z": 0.40,
            "darkness_p10_z": 0.55,
            "texture_ratio": 1.40,
            "edge_sharpness": 0.85,
            "compactness": 0.85,
            "solidity": 0.95,
            "elongation": 1.10,
        }
        result = evaluate_lookalike_screening(ratios)
        self.assertEqual(result.verdict, "LOOKALIKE_PRIORITY_LOW")
        self.assertLessEqual(result.screening_index, 0.35)
        self.assertTrue(any("Low backscatter contrast" in r for r in result.rejection_reasons))

    def test_malformed_feature_values_and_inputs_fail_closed(self):
        """Malformed inputs cannot invent usable feature defaults or a triage tier."""
        default = compute_scale_free_ratios(np.zeros((32, 32), dtype=np.uint8), np.array([1, 2, 3]))
        self.assert_unavailable(default, "INVALID_INPUT")
        self.assert_unavailable(compute_scale_free_ratios(np.zeros(32, dtype=np.uint8), np.zeros((3, 1, 2), dtype=np.int32)), "INVALID_INPUT")
        self.assert_unavailable(compute_scale_free_ratios(self.sea_bg, np.array([[[1, 1]], [[2, 2]], [[3, 1]]]), annulus_px=-1), "INVALID_INPUT")
        self.assert_unavailable({"darkness_z": "not-a-number", "texture_ratio": [1.0], "elongation": True}, "INVALID_INPUT")

    def assert_unavailable(self, features, status="UNAVAILABLE"):
        result = evaluate_lookalike_screening(features)
        self.assertEqual(result.status, status)
        self.assertEqual(result.verdict, "NOT_ASSESSED")
        self.assertIsNone(result.screening_index)
        self.assertTrue(result.reason)
        self.assertEqual(result.rejection_reasons, [])
        self.assertEqual(result.supporting_reasons, [])
        self.assertTrue(all(value is None for value in result.features.values()))

    def dark_fixture(self):
        image = self.sea_bg.copy()
        contour = np.array([[[70, 90]], [[130, 90]], [[130, 110]], [[70, 110]]], np.int32)
        cv2.drawContours(image, [contour], -1, 30, -1)
        return image, contour

    def test_all_masked_pixels_never_use_hidden_backing_data(self):
        image, contour = self.dark_fixture()
        self.assertGreater(evaluate_lookalike_screening(compute_scale_free_ratios(image, contour)).screening_index, 0.65)
        for backing in (image, np.zeros_like(image), np.full_like(image, 255)):
            with self.subTest(backing_mean=float(backing.mean())):
                self.assert_unavailable(compute_scale_free_ratios(np.ma.array(backing, mask=True), contour))

    def test_partial_mask_or_nonfinite_in_patch_annulus_and_gradient_support_is_unavailable(self):
        image, contour = self.dark_fixture()
        for y, x in ((100, 100), (100, 145), (100, 69)):
            for missing in ("masked", float("nan"), float("inf"), -float("inf")):
                with self.subTest(pixel=(y, x), missing=missing):
                    if missing == "masked":
                        sample = np.ma.array(image, mask=np.zeros_like(image, dtype=bool))
                        sample.mask[y, x] = True
                    else:
                        sample = image.astype(float)
                        sample[y, x] = missing
                    self.assert_unavailable(compute_scale_free_ratios(sample, contour))

    def test_missing_pixels_outside_required_support_preserve_score(self):
        image, contour = self.dark_fixture()
        expected = evaluate_lookalike_screening(compute_scale_free_ratios(image, contour))
        sample = np.ma.array(image.astype(float), mask=np.zeros_like(image, dtype=bool))
        sample.mask[0:10, 0:10] = True
        sample.data[0:10, 0:10] = np.nan
        result = evaluate_lookalike_screening(compute_scale_free_ratios(sample, contour))
        self.assertEqual(result.status, "ASSESSED")
        self.assertEqual(result.screening_index, expected.screening_index)

    def test_insufficient_background_and_degenerate_contours_are_not_low_risk(self):
        image, contour = self.dark_fixture()
        self.assert_unavailable(compute_scale_free_ratios(np.full_like(image, 128), contour))
        boundary_contour = contour - np.array([[[70, 90]]], np.int32)
        self.assert_unavailable(compute_scale_free_ratios(image, boundary_contour))
        for invalid in (np.full_like(image, np.nan, dtype=float), np.full_like(image, np.inf, dtype=float)):
            self.assert_unavailable(compute_scale_free_ratios(invalid, contour))
        for invalid_contour in (np.array([[[1, 1]], [[2, 2]], [[3, 3]]], np.int32),
                                contour.astype(float) + np.nan, contour + 500):
            self.assert_unavailable(compute_scale_free_ratios(image, invalid_contour), "INVALID_INPUT")

    def test_missing_features_remain_unavailable(self):
        for missing in ({}, None, [], {"darkness_z": 1.0}):
            with self.subTest(features=missing):
                self.assert_unavailable(missing)

    def test_each_feature_requires_a_finite_real_in_domain(self):
        image, contour = self.dark_fixture()
        valid = evaluate_lookalike_screening(compute_scale_free_ratios(image, contour)).features
        invalid_values = (None, True, np.bool_(False), "1", [], np.nan, np.inf, 1 + 0j, np.ma.masked)
        for key in valid:
            for value in invalid_values:
                with self.subTest(feature=key, value=value):
                    self.assert_unavailable({**valid, key: value}, "INVALID_INPUT")
        for key, value in (("texture_ratio", -0.01), ("edge_sharpness", -1),
                           ("compactness", 1.01), ("compactness", -0.1), ("solidity", 1.01),
                           ("solidity", 0), ("elongation", 0.99)):
            with self.subTest(feature=key, value=value):
                self.assert_unavailable({**valid, key: value}, "INVALID_INPUT")

    def test_extraction_status_and_support_metadata_cannot_override_missing_evidence(self):
        image, contour = self.dark_fixture()
        valid = compute_scale_free_ratios(image, contour)
        for status in ("NOT_ASSESSED", None, [], np.array(["ASSESSED", "UNAVAILABLE"])):
            with self.subTest(status=status):
                self.assert_unavailable({**valid, "status": status}, "INVALID_INPUT")
        for support in (None, {}, {**valid["support"], "required_pixels": True},
                        {**valid["support"], "required_pixels": 1}):
            with self.subTest(support=support):
                self.assert_unavailable({**valid, "support": support}, "INVALID_INPUT")
        self.assert_unavailable({**valid, "support": {**valid["support"], "missing_required_pixels": 1}})

    def test_nonreal_images_and_numeric_overflow_do_not_fabricate_ratios(self):
        image, contour = self.dark_fixture()
        for sample in (image.astype(bool), image.astype(complex), image.astype(object)):
            self.assert_unavailable(compute_scale_free_ratios(sample, contour), "INVALID_INPUT")
        sample = np.random.default_rng(4).choice((-1.0e308, 1.0e308), size=image.shape)
        self.assert_unavailable(compute_scale_free_ratios(sample, contour))
        valid = evaluate_lookalike_screening(compute_scale_free_ratios(image, contour)).features
        self.assert_unavailable({**valid, "darkness_z": 1.7e308, "darkness_p10_z": 1.7e308}, "INVALID_INPUT")


if __name__ == "__main__":
    unittest.main()
