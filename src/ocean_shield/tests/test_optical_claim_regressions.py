"""Adjacent optical path must not fabricate confidence or spectral authenticity."""
import json
import unittest
import numpy as np
from src.ocean_shield.eo_engine import EOEngine


class OpticalClaimRegressions(unittest.TestCase):
    def test_supplied_bands_do_not_establish_sensor_calibration_or_oil_code(self):
        _, diagnostic = EOEngine().process_multispectral_scene({"red": np.ones((40, 40)), "nir": np.ones((40, 40)) * 1.3})
        self.assertIsNone(diagnostic["confidence"])
        self.assertIsNone(diagnostic["lookalike_algae_rejected"])
        self.assertFalse(diagnostic["has_calibrated_nir"])
        self.assertEqual(diagnostic["calibration_status"], "NOT_ASSESSED")
        self.assertNotIn("Bonn", diagnostic["classification"])
        json.dumps(diagnostic, allow_nan=False)

    def test_invalid_or_misaligned_spectral_support_is_rejected(self):
        mask = np.ma.array(np.ones((8, 8)), mask=False)
        mask.mask[2, 2] = True
        for band in (np.ones((7, 8)), np.full((8, 8), np.nan), mask):
            with self.subTest(band_type=str(type(band))), self.assertRaises(ValueError):
                EOEngine().process_multispectral_scene({"red": np.ones((8, 8)), "nir": band})

    def test_pseudo_nir_is_not_observed_or_calibrated_confidence(self):
        _, diagnostic = EOEngine().segment_optical_slick(np.ones((40, 40, 3), dtype=np.uint8))
        self.assertIsNone(diagnostic["confidence"])
        self.assertEqual(diagnostic["sensor_mode"], "RGB_ESTIMATED_PSEUDO_NIR")
        self.assertFalse(diagnostic["has_calibrated_nir"])
        self.assertEqual(diagnostic["assessment_scope"], "RGB_APPEARANCE_SCREEN_ONLY")


if __name__ == "__main__":
    unittest.main()
