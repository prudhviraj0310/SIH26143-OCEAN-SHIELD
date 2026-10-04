"""Bounded local fixtures: one segmentation must drive metrics and overlays."""

import base64
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.ocean_shield import server
from src.ocean_shield.sar_engine import SAREngine


def threshold_fixture():
    image = np.full((128, 128), 128, dtype=np.uint8)
    cv2.ellipse(image, (64, 64), (20, 8), 0, 0, 360, 90, -1)
    return image


def decode_overlay(value):
    payload = value.split(",", 1)[-1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(payload), dtype=np.uint8), cv2.IMREAD_COLOR)


def expected_overlay(image, mask):
    return cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.65,
                           cv2.applyColorMap(mask, cv2.COLORMAP_JET), 0.35, 0)


class SARSegmentationRegressions(unittest.TestCase):
    def setUp(self):
        self.engine = SAREngine()
        self.engine.sr_model = None
        self.image = threshold_fixture()
        self.scenario = {"id": "segmentation_fixture", "center": {"lat": 0, "lon": 0},
                         "satellite_metadata": {"data_origin": "synthetic regression fixture", "pixel_spacing_m": 10},
                         "ocean_conditions": {"base_wind_u": 5, "base_wind_v": 0}}
        self.addCleanup(patch.stopall)
        patch.object(server, "sar_engine", self.engine).start()
        patch.object(server, "resolve_scenario_sar_and_currents",
                     side_effect=lambda _: (self.image, None, self.scenario)).start()
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)

    def test_requested_threshold_changes_metrics_and_the_returned_mask_together(self):
        for threshold, count in ((5, 1), (80, 0)):
            with self.subTest(threshold=threshold):
                mask, _ = self.engine.segment_oil_slick(self.image, threshold_offset=threshold)
                result, accepted_mask = self.engine.process_sar_scene(
                    self.image, 0, 0, model_type="cfar_edge", threshold_offset=threshold, return_mask=True)
                np.testing.assert_array_equal(accepted_mask, mask)
                self.assertEqual(result["total_slicks_detected"], count)
                self.assertEqual(result["segmentation"]["threshold_offset"], threshold)
                self.assertEqual(result["segmentation"]["threshold_offset_status"], "APPLIED")
                if count:
                    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                    self.assertEqual(result["primary_slick"]["area_km2"], round(cv2.contourArea(contour) * 100 / 1e6, 3))
                else:
                    self.assertIsNone(result["primary_slick"])

    def test_unet_unavailable_fallback_applies_requested_threshold(self):
        with patch.object(self.engine, "_ensure_unet_loaded"):
            result, mask = self.engine.process_sar_scene(self.image, 0, 0, threshold_offset=80, return_mask=True)
        self.assertIsNone(result["primary_slick"])
        self.assertFalse(np.any(mask))
        self.assertEqual(result["segmentation"]["threshold_offset"], 80)

    def test_invalid_direct_threshold_inputs_are_rejected(self):
        for value in (True, "20", None, np.nan, np.inf, 0, 101):
            with self.subTest(threshold=value), self.assertRaises(ValueError):
                self.engine.process_sar_scene(self.image, 0, 0, model_type="cfar_edge", threshold_offset=value)

    def test_scenario_api_metrics_overlay_and_threshold_share_one_segmentation(self):
        for threshold, count in ((5, 1), (80, 0)):
            with self.subTest(threshold=threshold):
                mask = self.engine.segment_oil_slick(self.image, threshold_offset=threshold)[0]
                with patch.object(self.engine, "segment_oil_slick", wraps=self.engine.segment_oil_slick) as segment:
                    response = self.client.post("/api/analyze-sar", json={
                        "scenario_id": "segmentation_fixture", "model_type": "cfar_edge", "demo_mode": True,
                        "threshold_offset": threshold, "use_super_resolution": False})
                self.assertEqual(response.status_code, 200)
                segment.assert_called_once_with(self.image, threshold_offset=threshold)
                body = response.json()
                self.assertEqual(body["sar_results"]["total_slicks_detected"], count)
                np.testing.assert_array_equal(decode_overlay(body["segmentation_overlay_base64"]), expected_overlay(self.image, mask))

    def test_uploaded_api_metrics_overlay_and_threshold_share_one_segmentation(self):
        content = cv2.imencode(".png", self.image)[1].tobytes()
        for threshold, count in ((5, 1), (80, 0)):
            with self.subTest(threshold=threshold):
                mask = self.engine.segment_oil_slick(self.image, threshold_offset=threshold)[0]
                with patch.object(self.engine, "segment_oil_slick", wraps=self.engine.segment_oil_slick) as segment:
                    response = self.client.post("/api/analyze-sar-upload", content=content, headers={
                        "X-Center-Lat": "0", "X-Center-Lon": "0", "X-Model-Type": "cfar_edge",
                        "X-Threshold-Offset": str(threshold)})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(segment.call_count, 1)
                self.assertEqual(segment.call_args.kwargs["threshold_offset"], threshold)
                body = response.json()
                self.assertEqual(body["sar_results"]["total_slicks_detected"], count)
                self.assertEqual(body["sar_results"]["segmentation"]["threshold_offset"], threshold)
                self.assertEqual(body["provenance"]["freshness_status"], "UNVERIFIED")
                self.assertIn("freshness is unverified", body["screening_notice"])
                np.testing.assert_array_equal(decode_overlay(body["segmentation_overlay_base64"]), expected_overlay(self.image, mask))

    def test_tiled_and_single_api_overlays_reuse_the_selected_inference_once(self):
        self.engine.model_loaded = True
        for size, mode in ((128, "SINGLE"), (512, "TILED")):
            self.image = np.full((size, size), 128, np.uint8)
            mask = np.zeros_like(self.image)
            cv2.rectangle(mask, (size // 2 - 20, size // 2 - 8), (size // 2 + 20, size // 2 + 8), 255, -1)
            for uploaded in (False, True):
                with self.subTest(size=size, uploaded=uploaded), \
                        patch.object(self.engine, "predict_unet", return_value=(mask, mask / 255)) as single, \
                        patch.object(self.engine, "predict_unet_tiled", return_value=(mask, mask / 255)) as tiled:
                    if uploaded:
                        response = self.client.post("/api/analyze-sar-upload", content=cv2.imencode(".png", self.image)[1].tobytes(),
                                                    headers={"X-Center-Lat": "0", "X-Center-Lon": "0", "X-Model-Type": "unet"})
                    else:
                        response = self.client.post("/api/analyze-sar", json={"scenario_id": "segmentation_fixture",
                                                                            "model_type": "unet", "use_super_resolution": False})
                    self.assertEqual(response.status_code, 200)
                    selected, unused = (tiled, single) if mode == "TILED" else (single, tiled)
                    selected.assert_called_once()
                    unused.assert_not_called()
                    body = response.json()
                    self.assertEqual(body["sar_results"]["segmentation"]["inference_mode"], mode)
                    self.assertIsNone(body["sar_results"]["segmentation"]["threshold_offset"])
                    np.testing.assert_array_equal(decode_overlay(body["segmentation_overlay_base64"]), expected_overlay(self.image, mask))

    def test_frame_and_tiny_rejections_are_absent_from_the_returned_overlay_mask(self):
        self.engine.model_loaded = True
        for mask in (np.full_like(self.image, 255), np.zeros_like(self.image)):
            mask[60:62, 60:62] = 255
            with patch.object(self.engine, "predict_unet", return_value=(mask, mask / 255)):
                result, accepted = self.engine.process_sar_scene(self.image, 0, 0, return_mask=True)
            self.assertEqual(result["total_slicks_detected"], 0)
            self.assertFalse(np.any(accepted))

    def test_sar_screen_payload_propagates_missing_support_without_index(self):
        contour = np.array([[[40, 50]], [[80, 50]], [[80, 70]], [[40, 70]]], np.int32)
        for image in (None, np.ma.array(self.image, mask=True),
                      np.ma.array(np.dstack([self.image] * 3), mask=True)):
            with self.subTest(image_kind=type(image).__name__):
                result = self.engine.extract_geometric_metrics(contour, 0, 0, img_shape=self.image.shape, image=image)
                screen = result["lookalike_screening"]
                self.assertEqual(screen["status"], "UNAVAILABLE")
                self.assertEqual(screen["verdict"], "NOT_ASSESSED")
                self.assertIsNone(screen["screening_index"])
                self.assertTrue(screen["reason"])


if __name__ == "__main__":
    unittest.main()
