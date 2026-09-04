"""
Unit and Integration Test Suite for OCEAN-SHIELD
Smart India Hackathon 2026 - Problem Statement SIH26143
"""

import os
import unittest
import numpy as np

from src.ocean_shield.sar_engine import SAREngine
from src.ocean_shield.drift_engine import DriftEngine, OceanCurrentField
from src.ocean_shield.ais_engine import AISEngine
from src.ocean_shield.scenarios import get_scenario_sar_and_currents, get_all_scenarios
from src.ocean_shield.report_generator import DossierReportGenerator


class TestOceanShieldPipeline(unittest.TestCase):

    def setUp(self):
        self.sar_engine = SAREngine()
        self.drift_engine = DriftEngine()
        self.ais_engine = AISEngine()
        self.report_gen = DossierReportGenerator(output_dir="reports")

    def test_sar_engine_segmentation_and_super_resolution(self):
        """Validates radar speckle filtering, CFAR segmentation, and Super-Resolution."""
        sar_img, _, scenario_data = get_scenario_sar_and_currents("gulf_of_kachchh")
        self.assertEqual(sar_img.shape, (512, 512))

        # Test Super-Resolution
        sr_img = self.sar_engine.enhance_sar_super_resolution(sar_img, scale_factor=2)
        self.assertEqual(sr_img.shape, (1024, 1024))

        # Test Lee Filter
        filtered = self.sar_engine.enhanced_lee_filter(sar_img)
        self.assertEqual(filtered.shape, sar_img.shape)

        # Test Segmentation
        mask, contours = self.sar_engine.segment_oil_slick(sar_img)
        self.assertGreater(len(contours), 0, "Should detect at least one slick contour")

        # Test Full Scene Processing
        results = self.sar_engine.process_sar_scene(
            sar_img, scenario_data["center"]["lat"], scenario_data["center"]["lon"]
        )
        self.assertGreater(results["total_slicks_detected"], 0)
        slick = results["primary_slick"]
        self.assertIsNotNone(slick)
        self.assertGreater(slick["area_km2"], 0.0)
        self.assertGreater(slick["estimated_mass_tonnes"], 0.0)
        self.assertIn("polygon_geojson", slick)
        self.assertGreater(slick["confidence_score"], 60.0, "Mineral oil should have high confidence")

    def test_drift_engine_hindcast_and_forecast(self):
        """Validates reverse Lagrangian particle hindcasting and forward forecasting."""
        sar_img, current_field, scenario_data = get_scenario_sar_and_currents("gulf_of_kachchh")
        center_lat = scenario_data["center"]["lat"]
        center_lon = scenario_data["center"]["lon"]

        # Run Backward Hindcast
        hindcast = self.drift_engine.run_hindcast(
            center_lat, center_lon, current_field,
            max_lookback_hours=18.0, target_slick_age_hours=10.5
        )
        origin = hindcast["origin_release_point"]
        self.assertAlmostEqual(origin["slick_age_hours"], 10.5, delta=1.5)
        self.assertGreater(hindcast["total_drift_distance_km"], 0.0)
        self.assertGreater(len(hindcast["hindcast_trajectory"]), 10)

        # Run Forward Forecast
        forecast = self.drift_engine.run_forecast(
            center_lat, center_lon, current_field,
            forecast_hours=24.0,
            coastline_lat_threshold=scenario_data["coastline_hazard"]["coastline_lat_threshold"]
        )
        self.assertIn("forecast_trajectory", forecast)
        self.assertIn("beaching_warning", forecast)
        self.assertTrue(forecast["beaching_warning"]["will_beach"])

    def test_ais_correlation_and_culprit_attribution(self):
        """Validates spatio-temporal corridor filtering, kinematic anomalies, and rogue ship attribution."""
        _, _, scenario_data = get_scenario_sar_and_currents("gulf_of_kachchh")
        ground_truth = scenario_data["ground_truth_culprit"]

        results = self.ais_engine.attribute_oil_spill(
            scenario_data["ais_vessels"],
            ground_truth["discharge_lat"],
            ground_truth["discharge_lon"],
            ground_truth["discharge_time_rel_h"]
        )

        culprit = results["primary_culprit"]
        self.assertIsNotNone(culprit)
        self.assertEqual(culprit["vessel_name"], ground_truth["vessel_name"])
        self.assertEqual(culprit["imo"], ground_truth["imo"])
        self.assertGreaterEqual(culprit["composite_suspect_score"], 80.0, "Culprit should have high suspect score")
        self.assertLess(culprit["closest_approach"]["distance_nm"], 1.0, "Culprit should be directly over origin")
        self.assertGreaterEqual(culprit["kinematics"]["speed_drop_knots"], 6.0, "Culprit should exhibit speed drop")

    def test_all_scenarios_and_dossier_pdf(self):
        """Validates all 3 Indian maritime sectors and generates an official PDF dossier."""
        scenarios = get_all_scenarios()
        self.assertEqual(len(scenarios), 3)

        for sid in scenarios:
            sar_img, current_field, data = get_scenario_sar_and_currents(sid)
            self.assertEqual(sar_img.shape, (512, 512))

        # Test Dossier PDF generation
        sar_img, current_field, scenario_data = get_scenario_sar_and_currents("gulf_of_kachchh")
        sar_res = self.sar_engine.process_sar_scene(
            sar_img, scenario_data["center"]["lat"], scenario_data["center"]["lon"]
        )
        origin = {"lat": 22.385, "lon": 69.115, "slick_age_hours": 10.5, "estimated_t0_hours_relative": -10.5}
        ais_res = self.ais_engine.attribute_oil_spill(
            scenario_data["ais_vessels"], origin["lat"], origin["lon"], -10.5
        )

        pdf_path = self.report_gen.generate_pdf_dossier(
            scenario_data, sar_res,
            {
                "origin_release_point": origin,
                "total_drift_distance_km": 14.8,
                "forecast_warning": {"will_beach": True, "estimated_time_to_beach_hours": 11.2, "beaching_location": {"lat": 22.58, "lon": 69.45}}
            },
            ais_res,
            filename="Test_Verification_Dossier.pdf"
        )
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000, "PDF should be non-empty and well-structured")


if __name__ == "__main__":
    unittest.main()
