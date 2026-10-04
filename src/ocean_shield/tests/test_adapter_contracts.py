"""Regression checks for disconnected source adapters and public screening labels."""

import unittest

from src.ocean_shield.ais_ingestion import ICG_VTMS_Adapter
from src.ocean_shield.ocean_data import Oceansat3_OCM_Adapter


class AdapterContractRegressions(unittest.TestCase):
    def test_ocm_adapter_does_not_fabricate_optical_confirmation(self):
        result = Oceansat3_OCM_Adapter().verify_slick_optical_signature(22.5, 69.2)
        self.assertIsNone(result["sun_glint_detected"])
        self.assertIsNone(result["slick_reflectance_anomaly"])
        self.assertEqual(result["optical_confirmation"], "NOT_ASSESSED_NO_OCM3_SCENE")
        self.assertEqual(result["status"], "ADAPTER_NOT_CONNECTED")

    def test_vtms_adapter_reports_metadata_only(self):
        adapter = ICG_VTMS_Adapter()
        telemetry = adapter.get_station_telemetry()
        self.assertIsNone(telemetry["chain_operational"])
        self.assertIsNone(telemetry["active_radar_stations"])
        self.assertEqual(telemetry["status"], "ADAPTER_NOT_CONNECTED")
        self.assertIn("not assessed", telemetry["notice"].lower())


if __name__ == "__main__":
    unittest.main()
