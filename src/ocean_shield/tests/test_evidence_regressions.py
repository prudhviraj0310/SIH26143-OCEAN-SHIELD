"""Evidence-state regressions using only bounded synthetic inputs and local models."""

import copy
import base64
import json
import math
import os
import re
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.ocean_shield.ais_engine import AISEngine
from src.ocean_shield.falsification import FalsificationAndAbstentionEngine as Gate
from src.ocean_shield.models.super_resolution import enhance_sar_deep_learning
from src.ocean_shield.sar_engine import SAREngine
from src.ocean_shield.report_generator import DossierReportGenerator
from src.ocean_shield.ocean_data import DataCoverageError
from src.ocean_shield.drift_engine import OceanSourceError


def continuous_track():
    return [
        {"relative_time_hours": t, "lat": 0.0, "lon": t * 0.001,
         "sog_knots": 6.0, "cog_degrees": 90.0}
        for t in (-0.4, -0.2, 0.0, 0.2)
    ]


def candidate(score=90.0, mmsi=419001234):
    return {"mmsi": mmsi, "composite_score": score, "trajectory": continuous_track(),
            "closest_approach": {"distance_nm": 0.1, "time_diff_h": 0.0,
                                 "time_relative_h": 0.0}}


def assessed_candidate(score=90.0, mmsi=419001234):
    vessel = candidate(score, mmsi)
    vessel["spoofing_audit"] = AISEngine.detect_ais_spoofing_and_gaps(
        vessel["trajectory"], mmsi=mmsi)
    vessel["adversarial_stress_test"] = Gate.run_adversarial_stress_test(
        vessel, 0.0, 0.0, 1.0, 10.0)
    return vessel


class EvidenceDistributionRegressions(unittest.TestCase):
    def test_invalid_scores_abstain_without_fabricating_zero(self):
        for value in (None, float("nan"), float("inf"), -1, 101, True, np.bool_(True), "bad", [90]):
            with self.subTest(score=value):
                vessel = assessed_candidate(value)
                result = Gate.evaluate_decision_theoretic_abstention(
                    [vessel], coverage_validated=True)
                self.assertTrue(result["is_abstention"])
                self.assertEqual(result["status"], "INVALID_INPUT")
                self.assertIsNone(vessel["composite_score"])
                self.assertIsNone(vessel["lead_priority_weight"])
                self.assertIsNone(result["confidence_score"])

    def test_unknown_distribution_present_at_every_count(self):
        for count in (0, 1, 2, 5):
            with self.subTest(count=count):
                vessels = [assessed_candidate(90 - i * 10, 419001234 + i)
                           for i in range(count)]
                result = Gate.evaluate_decision_theoretic_abstention(vessels)
                distribution = result["hypothesis_distribution"]
                self.assertEqual(len(distribution), count + 1)
                self.assertEqual(distribution[-1]["hypothesis"], "UNKNOWN_SOURCE")
                self.assertGreater(distribution[-1]["lead_priority_weight"], 0)
                self.assertAlmostEqual(sum(row["lead_priority_weight"]
                                           for row in distribution), 1.0, places=14)

    def test_binary_entropy_includes_unknown_at_full_precision(self):
        vessels = [assessed_candidate(55.0)]
        result = Gate.evaluate_decision_theoretic_abstention(vessels)
        p = 1.0 / (1.0 + math.exp(-10.0 / Gate.TEMPERATURE))
        expected = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        self.assertAlmostEqual(result["entropy_metrics"]["normalized_entropy"], expected, places=14)
        self.assertGreater(expected, 0.82)
        self.assertTrue(result["is_abstention"])

    def test_unknown_cannot_be_disabled_by_legacy_keyword(self):
        result = Gate.evaluate_decision_theoretic_abstention(
            [assessed_candidate(90)], unknown_source_hypothesis=False)
        self.assertGreater(result["unknown_source_weight"], 0)

    def test_missing_coverage_is_an_evidentiary_hold(self):
        result = Gate.evaluate_decision_theoretic_abstention([assessed_candidate(90)])
        self.assertTrue(result["is_abstention"])
        self.assertEqual(result["coverage_status"], "NOT_ASSESSED")
        self.assertIsNone(result["confidence_score"])

    def test_valid_coverage_integrity_and_stress_allow_only_screening_lead(self):
        result = Gate.evaluate_decision_theoretic_abstention(
            [assessed_candidate(90)], coverage_validated=True)
        self.assertFalse(result["is_abstention"])
        self.assertEqual(result["decision"], "SCREENING_LEAD")
        self.assertEqual(result["weight_kind"], "uncalibrated_lead_priority")
        self.assertIsNone(result["confidence_score"])

    def test_missing_or_malformed_stress_cannot_promote_a_candidate(self):
        vessel = assessed_candidate()
        for stress in (None, {}, {"status": "ASSESSED", "stress_passed": True,
                                 "challenges": [{"survived": True}] * 4}):
            with self.subTest(stress=stress):
                vessel["adversarial_stress_test"] = stress
                result = Gate.evaluate_decision_theoretic_abstention([vessel], coverage_validated=True)
                self.assertTrue(result["is_abstention"])

    def test_unknown_winner_and_unsorted_competitors_are_compared(self):
        result = Gate.evaluate_decision_theoretic_abstention(
            [assessed_candidate(0)], coverage_validated=True)
        self.assertTrue(result["is_abstention"])
        self.assertLess(result["separation_margin"], 0)
        vessels = [assessed_candidate(20, 419001235), assessed_candidate(90)]
        result = Gate.evaluate_decision_theoretic_abstention(vessels, coverage_validated=True)
        self.assertEqual(result["leading_candidate_mmsi"], 419001234)
        self.assertGreater(result["separation_margin"], 0.9)

    def test_entropy_threshold_uses_unrounded_value_and_inclusive_eligibility(self):
        vessel = assessed_candidate(90)
        original = Gate.calculate_shannon_entropy
        for entropy, abstains in ((0.82, False), (0.820001, True), (0.819999, False)):
            with self.subTest(entropy=entropy):
                with patch.object(Gate, "calculate_shannon_entropy", return_value={
                    **original([0.9, 0.1]), "normalized_entropy": entropy}):
                    result = Gate.evaluate_decision_theoretic_abstention(
                        [copy.deepcopy(vessel)], coverage_validated=True)
                self.assertEqual(result["is_abstention"], abstains)
        metrics = original([0.6970592839654074, 0.3029407160345926])
        self.assertNotEqual(metrics["normalized_entropy"], round(metrics["normalized_entropy"], 4))


class AISIntegrityRegressions(unittest.TestCase):
    def test_missing_malformed_or_sparse_telemetry_is_not_assessed(self):
        for track in ([], [None], [{}], continuous_track()[:1],
                      [{"relative_time_hours": 0, "sog_knots": 6},
                       {"relative_time_hours": 0.2, "sog_knots": 6}],
                      [{**p, "lat": float("nan")} for p in continuous_track()]):
            with self.subTest(track=track):
                audit = AISEngine.detect_ais_spoofing_and_gaps(track, mmsi=419001234)
                self.assertEqual(audit["status"], "NOT_ASSESSED")
                self.assertNotEqual(audit["integrity_rating"], "VERIFIED_CONTINUOUS")
                self.assertFalse(Gate.assess_ais_integrity(audit)["passed"])

    def test_minimum_position_and_distinct_time_coverage(self):
        audit = AISEngine.detect_ais_spoofing_and_gaps(continuous_track(), mmsi=419001234)
        self.assertTrue(Gate.assess_ais_integrity(audit)["passed"])
        duplicate_times = [{**p, "relative_time_hours": 0} for p in continuous_track()]
        self.assertFalse(Gate.assess_ais_integrity(AISEngine.detect_ais_spoofing_and_gaps(
            duplicate_times, mmsi=419001234))["passed"])
        self.assertFalse(Gate.assess_ais_integrity(AISEngine.detect_ais_spoofing_and_gaps(
            continuous_track(), mmsi=419001234, cpa_time_relative_h=2))["passed"])

    def test_nonempty_but_incomplete_audit_schema_cannot_pass(self):
        for audit in ({"irrelevant": True}, {"has_anomalies": False, "mmsi_valid": True},
                      {**AISEngine.detect_ais_spoofing_and_gaps(continuous_track(), 419001234),
                       "corridor_blackout": "false"}):
            with self.subTest(audit=audit):
                vessel = candidate()
                vessel["spoofing_audit"] = audit
                result = Gate.run_adversarial_stress_test(vessel, 0, 0, 1, 10)
                self.assertFalse(result["challenges"][3]["survived"])
                self.assertEqual(result["integrity_status"], "NOT_ASSESSED")

    def test_one_speed_jump_blocks_independently_of_anomaly_count(self):
        track = continuous_track()
        track[1] = {**track[1], "relative_time_hours": -0.38, "sog_knots": 18.0}
        audit = AISEngine.detect_ais_spoofing_and_gaps(track, 419001234)
        self.assertTrue(audit["impossible_speed_jump"])
        self.assertFalse(Gate.assess_ais_integrity(audit)["passed"])
        vessel = assessed_candidate()
        vessel["spoofing_audit"] = audit
        vessel["adversarial_stress_test"] = Gate.run_adversarial_stress_test(vessel, 0, 0, 1, 10)
        result = Gate.evaluate_decision_theoretic_abstention([vessel], coverage_validated=True)
        self.assertTrue(result["is_abstention"])
        self.assertEqual(result["integrity_status"], "COMPROMISED")

    def test_blackout_or_identity_flag_blocks_without_has_anomalies_conjunction(self):
        for flag in ("corridor_blackout", "identity_conflict"):
            with self.subTest(flag=flag):
                audit = AISEngine.detect_ais_spoofing_and_gaps(continuous_track(), 419001234)
                audit[flag] = True
                self.assertFalse(Gate.assess_ais_integrity(audit)["passed"])
        audit = AISEngine.detect_ais_spoofing_and_gaps(continuous_track(), 999001234)
        self.assertFalse(Gate.assess_ais_integrity(audit)["passed"])

    def test_final_decision_after_stress_failure_and_no_verdict_aliases(self):
        vessels = [candidate(mmsi=419001234), candidate(mmsi=419001235)]
        vessels[1]["closest_approach"]["distance_nm"] = 5
        ranked = AISEngine().score_and_rank_suspects(vessels, 0, 0, 0, coverage_validated=True)
        self.assertIsNot(ranked[0]["abstention_verdict"], ranked[1]["abstention_verdict"])
        self.assertTrue(ranked[1]["abstention_verdict"]["is_abstention"])
        ranked[0]["abstention_verdict"]["reason"] = "mutation witness"
        self.assertNotEqual(ranked[1]["abstention_verdict"]["reason"], "mutation witness")
        with patch.object(Gate, "run_adversarial_stress_test", side_effect=RuntimeError("test gate failure")):
            lead = AISEngine().score_and_rank_suspects([candidate()], 0, 0, 0, coverage_validated=True)[0]
        self.assertTrue(lead["abstention_verdict"]["is_abstention"])
        self.assertEqual(lead["abstention_verdict"]["status"], "UNAVAILABLE")

    def test_evaluator_exception_is_explicit_unavailable(self):
        with patch.object(Gate, "evaluate_decision_theoretic_abstention", side_effect=RuntimeError("test gate failure")):
            lead = AISEngine().score_and_rank_suspects([candidate()], 0, 0, 0)[0]
        self.assertEqual(lead["abstention_verdict"]["status"], "UNAVAILABLE")
        self.assertTrue(lead["abstention_verdict"]["is_abstention"])
        self.assertIsNone(lead["abstention_verdict"]["confidence_score"])
        self.assertEqual(len(lead["abstention_verdict"]["hypothesis_distribution"]), 2)
        self.assertEqual(lead["abstention_verdict"]["hypothesis_distribution"][-1]["hypothesis"], "UNKNOWN_SOURCE")

    def test_invalid_cpa_and_track_are_json_safe_holds(self):
        vessel = candidate()
        vessel["closest_approach"]["distance_nm"] = float("nan")
        vessel["trajectory"][0]["lat"] = float("nan")
        result = AISEngine().score_and_rank_suspects([vessel], 0, 0, 0)
        json.dumps(result, allow_nan=False)
        self.assertTrue(result[0]["abstention_verdict"]["is_abstention"])
        self.assertIsNone(result[0]["composite_score"])

    def test_failed_audit_implementation_is_unavailable(self):
        with patch.object(AISEngine, "detect_ais_spoofing_and_gaps", side_effect=RuntimeError("fixture audit unavailable")):
            lead = AISEngine().score_and_rank_suspects([candidate()], 0, 0, 0)[0]
        self.assertTrue(lead["abstention_verdict"]["is_abstention"])
        self.assertEqual(lead["abstention_verdict"]["status"], "UNAVAILABLE")

    def test_empty_corridor_gate_exception_still_has_unavailable_verdict(self):
        with patch.object(Gate, "evaluate_decision_theoretic_abstention", side_effect=RuntimeError("fixture gate unavailable")):
            result = AISEngine().attribute_oil_spill([], 0, 0, 0)
        self.assertTrue(result["is_abstention"])
        self.assertEqual(result["screening_gate"]["status"], "UNAVAILABLE")
        self.assertEqual(result["screening_gate"]["hypothesis_distribution"][-1]["hypothesis"], "UNKNOWN_SOURCE")


class SARPreviewRegressions(unittest.TestCase):
    def test_direct_untrained_fallback_is_repeatable_and_disclosed(self):
        class Untrained:
            weights_loaded = False

            def __call__(self, *args):
                raise AssertionError("random weights must never run")

        image = np.arange(64, dtype=np.uint8).reshape(8, 8)
        out, metadata = enhance_sar_deep_learning(image, Untrained(), return_metadata=True)
        np.testing.assert_array_equal(out, enhance_sar_deep_learning(image, Untrained()))
        np.testing.assert_array_equal(out, cv2.resize(image, (16, 16), interpolation=cv2.INTER_CUBIC))
        self.assertEqual(metadata["interpolation"], "bicubic")
        self.assertFalse(metadata["native_radiometry_preserved"])
        self.assertFalse(metadata["trained_weights_used"])

    def test_direct_and_outer_failure_use_same_display_fallback(self):
        image = np.arange(64, dtype=np.float32).reshape(8, 8) - 20
        class Untrained:
            weights_loaded = False
        direct, direct_metadata = enhance_sar_deep_learning(image, Untrained(), return_metadata=True)
        engine = SAREngine()
        with patch.object(engine, "_ensure_sr_loaded", side_effect=RuntimeError("test checkpoint failure")):
            outer, metadata = engine.enhance_sar_super_resolution(image, return_metadata=True)
        np.testing.assert_array_equal(direct, outer)
        self.assertEqual(metadata["interpolation"], direct_metadata["interpolation"])
        self.assertEqual(metadata["normalization"], "min_max_display_scaling")
        self.assertEqual(metadata["fallback_reason"], "model_or_inference_unavailable")

    def test_missing_checkpoint_and_scale_factor_are_deterministic(self):
        image = np.arange(64, dtype=np.uint8).reshape(8, 8)
        with patch("src.ocean_shield.models.super_resolution.os.path.exists", return_value=False):
            first, metadata = enhance_sar_deep_learning(image, return_metadata=True)
            second = enhance_sar_deep_learning(image)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(metadata["trained_weights_used"])
        engine = SAREngine()
        with patch.object(engine, "_ensure_sr_loaded", return_value=None):
            self.assertEqual(engine.enhance_sar_super_resolution(image, scale_factor=3).shape, (24, 24))

    def test_morphology_score_does_not_fabricate_confidence(self):
        contour = np.array([[[10, 10]], [[30, 10]], [[30, 40]], [[10, 40]]], dtype=np.int32)
        metrics = SAREngine().extract_geometric_metrics(contour, 0, 0)
        self.assertIsNone(metrics["confidence_score"])
        self.assertEqual(metrics["confidence_status"], "UNCALIBRATED_SCREENING_SCORE")
        self.assertGreater(metrics["screening_score"], 0)


def pdf_page_streams(path):
    """Decode ReportLab's actual page streams without adding a PDF dependency."""
    data = Path(path).read_bytes()
    decoded = []
    for stream in re.findall(rb"stream\r?\n(.*?)endstream", data, re.S):
        try:
            decoded.append(zlib.decompress(base64.a85decode(stream.strip(), adobe=True)).decode("latin1"))
        except (ValueError, zlib.error):
            continue
    return "\n".join(decoded)


class PublicEvidenceStateRegressions(unittest.TestCase):
    def setUp(self):
        from src.ocean_shield import server
        from src.ocean_shield.drift_engine import OceanCurrentField
        self.server = server
        self.image = np.full((64, 64), 120, dtype=np.uint8)
        self.field = OceanCurrentField(constant_vectors=True)
        self.scenario = {
            "id": "evidence_fixture", "region": "Synthetic unit fixture", "center": {"lat": 0, "lon": 0},
            "satellite_metadata": {"data_origin": "synthetic regression fixture", "pixel_spacing_m": 10},
            "ocean_conditions": {"base_current_u": 0.1, "base_current_v": 0.1,
                                 "base_wind_u": 3, "base_wind_v": 3},
            "ais_vessels": [candidate()],
        }
        engine = SAREngine()
        engine.sr_model = SimpleNamespace(weights_loaded=False)
        self.addCleanup(patch.stopall)
        patch.object(server, "sar_engine", engine).start()
        patch.object(server, "resolve_scenario_sar_and_currents",
                     return_value=(self.image, self.field, self.scenario)).start()
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)

    def ais_request(self):
        return {"scenario_id": "evidence_fixture", "origin_lat": 0, "origin_lon": -0.0002,
                "origin_time_rel_h": -0.2, "vessels": [candidate()],
                "ais_provenance": {"source_kind": "synthetic_unit_fixture", "coverage_validated": True},
                "hindcast_trajectory": [{"relative_time_hours": -0.2,
                                         "centroid": {"lat": 0, "lon": -0.0002}}]}

    def test_ais_api_never_promotes_unvalidated_coverage_declaration(self):
        response = self.client.post("/api/correlate-ais", json=self.ais_request())
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["is_abstention"])
        self.assertIsNone(result["evidentiary_lead"])
        self.assertEqual(result["screening_gate"]["coverage_status"], "NOT_ASSESSED")
        self.assertTrue(result["primary_review_lead"]["abstention_verdict"]["is_abstention"])

    def test_optional_counterfactual_failure_is_explicit_in_all_gate_locations(self):
        self.field.data_provider = SimpleNamespace(is_loaded=True)
        with patch.object(self.server.drift_engine, "run_forward_counterfactual", side_effect=DataCoverageError("fixture forcing unavailable")):
            response = self.client.post("/api/correlate-ais", json=self.ais_request())
        result = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(result["counterfactual_status"], "UNAVAILABLE")
        self.assertFalse(result["counterfactual_ready"])
        for gate in (result["screening_gate"], result["bayesian_legal_gate"],
                     result["primary_review_lead"]["abstention_verdict"]):
            self.assertEqual(gate["status"], "UNAVAILABLE")
            self.assertTrue(gate["is_abstention"])
            self.assertIsNone(gate["confidence_score"])

    def test_provider_failure_during_integration_is_503_unavailable(self):
        with patch.object(self.server.drift_engine, "run_hindcast", side_effect=DataCoverageError("fixture forcing unavailable")):
            response = self.client.post("/api/simulate-drift", json={
                "scenario_id": "evidence_fixture", "demo_mode": True,
                "slick_lat": 0, "slick_lon": 0, "slick_age_hours": 1, "forecast_hours": 1})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "UNAVAILABLE")
        self.assertTrue(response.json()["is_abstention"])
        self.assertNotIn("forecast_trajectory", response.json())

    def test_counterfactual_provider_failure_is_503_unavailable(self):
        with patch.object(self.server.drift_engine, "run_forward_counterfactual", side_effect=DataCoverageError("fixture forcing unavailable")):
            response = self.client.post("/api/verify-counterfactual", json={
                "scenario_id": "evidence_fixture", "demo_mode": True,
                "release_lat": 0, "release_lon": 0, "release_time_rel_h": -1,
                "observed_slick_lat": 0, "observed_slick_lon": 0})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["transport_status"], "UNAVAILABLE")

    def test_engine_source_error_contract_is_503_not_invalid_input(self):
        with patch.object(self.server.drift_engine, "run_hindcast", side_effect=OceanSourceError("fixture source unavailable")):
            response = self.client.post("/api/simulate-drift", json={
                "scenario_id": "evidence_fixture", "demo_mode": True,
                "slick_lat": 0, "slick_lon": 0, "slick_age_hours": 1, "forecast_hours": 1})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "UNAVAILABLE")

    def test_real_field_provider_exception_has_no_fallback_or_forecast(self):
        class BrokenProvider:
            is_loaded = True
            metadata = {"source": "synthetic failure fixture"}
            _has_real_wind = True

            def bind_detection_time(self, *args):
                pass

            def get_velocity_at(self, *args):
                raise RuntimeError("synthetic fixture source unavailable")

        self.field.data_provider = BrokenProvider()
        response = self.client.post("/api/simulate-drift", json={
            "scenario_id": "evidence_fixture", "demo_mode": False,
            "scene_acquired_at_utc": "2026-10-04T00:00:00Z",
            "slick_lat": 0, "slick_lon": 0, "slick_age_hours": 1, "forecast_hours": 1})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["transport_status"], "UNAVAILABLE")
        self.assertNotIn("forecast_trajectory", response.json())

    def test_sr_png_headers_and_json_disclose_interpolation(self):
        response = self.client.get("/api/scenario/evidence_fixture/sr-preview.png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-image-interpolation"], "bicubic")
        self.assertEqual(response.headers["x-native-radiometry-preserved"], "false")
        response = self.client.post("/api/analyze-sar", json={"scenario_id": "evidence_fixture", "model_type": "cfar_edge"})
        self.assertEqual(response.status_code, 200)
        metadata = response.json()["super_resolution_metadata"]
        self.assertEqual(metadata["status"], "INTERPOLATED_DISPLAY_PREVIEW")
        self.assertFalse(metadata["native_radiometry_preserved"])
        self.assertFalse(metadata["trained_weights_used"])

    def test_super_resolution_request_flag_is_honored(self):
        response = self.client.post("/api/analyze-sar", json={
            "scenario_id": "evidence_fixture", "model_type": "cfar_edge", "use_super_resolution": False})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["super_resolution_base64"])
        self.assertEqual(response.json()["super_resolution_metadata"]["status"], "NOT_REQUESTED")

    def test_quality_gate_preserves_unverified_area_without_mutating_input(self):
        feature = {"area_km2": 3.5, "centroid": {"lat": 0, "lon": 0}, "confidence_score": 95}
        raw = {"primary_slick": feature, "all_slicks": [feature]}
        result = self.server._apply_sar_quality_gate(raw, {"operational_eligible": False})
        self.assertEqual(result["primary_slick"]["area_km2_unverified"], 3.5)
        self.assertIsNone(result["primary_slick"]["confidence_score"])
        self.assertEqual(raw["primary_slick"]["area_km2"], 3.5)

    def test_explicit_benchmark_demo_keeps_simulated_geometry_labelled(self):
        feature = {"area_km2": 3.5, "centroid": {"lat": 0, "lon": 0}, "confidence_score": 95}
        raw = {"primary_slick": feature, "all_slicks": [feature], "radar_detected_ships": [{"lat": 0, "lon": 0}]}
        result = self.server._apply_sar_quality_gate(raw, {"operational_eligible": False}, demo_mode=True)
        self.assertEqual(result["primary_slick"]["area_km2"], 3.5)
        self.assertEqual(result["primary_slick"]["geometry_status"], "SIMULATED_BENCHMARK_GEOMETRY")
        self.assertIsNone(result["primary_slick"]["confidence_score"])
        self.assertTrue(result["demo_mode"])
        self.assertEqual(len(result["radar_detected_ships"]), 1)

    def test_sar_demo_api_contract_allows_playback_but_not_operational_eligibility(self):
        feature = {"area_km2": 3.5, "centroid": {"lat": 0, "lon": 0},
                   "polygon_geojson": None, "confidence_score": 95, "screening_score": 60,
                   "elongation": 1.2, "slick_id": "FIXTURE"}
        raw = {"primary_slick": feature, "all_slicks": [feature], "radar_detected_ships": [],
               "active_engine": "fixture"}
        with patch.object(self.server.sar_engine, "process_sar_scene",
                          return_value=(raw, np.zeros_like(self.image))):
            response = self.client.post("/api/analyze-sar", json={
                "scenario_id": "evidence_fixture", "demo_mode": True,
                "model_type": "cfar_edge", "use_super_resolution": False})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["sar_results"]["primary_slick"]["geometry_status"], "SIMULATED_BENCHMARK_GEOMETRY")
        self.assertFalse(result["sar_results"]["observability"]["operational_eligible"])
        self.assertIn("synthetic geometry", result["screening_notice"])

    def test_unknown_sar_model_is_rejected(self):
        response = self.client.post("/api/analyze-sar", json={
            "scenario_id": "evidence_fixture", "model_type": "not-a-model"})
        self.assertEqual(response.status_code, 422)

    def test_eo_metadata_keeps_lookalike_screen_as_unresolved(self):
        rgb = np.full((64, 64, 3), 80, dtype=np.uint8)
        nir = np.full((64, 64), 90, dtype=np.uint8)
        swir = np.full((64, 64), 85, dtype=np.uint8)
        with patch.object(self.server, "get_scenario_eo_data", return_value=(rgb, nir, swir, self.scenario)):
            response = self.client.post("/api/analyze-eo", json={"scenario_id": "evidence_fixture"})
        self.assertEqual(response.status_code, 200)
        metadata = response.json()["metadata"]
        self.assertIn("not determined", metadata["lookalike_discrimination"].lower())
        self.assertNotIn("discarded", metadata["lookalike_discrimination"].lower())

    def test_export_route_preserves_unavailable_pdf_state(self):
        with tempfile.TemporaryDirectory(prefix="ocean-evidence-http-", dir=os.environ.get("OCEAN_SHIELD_TEST_TMP")) as directory:
            with patch.object(self.server, "report_gen", DossierReportGenerator(directory)), \
                    patch.object(self.server, "get_all_scenarios", return_value={"evidence_fixture": self.scenario}):
                response = self.client.post("/api/export-case-summary", json={
                    "scenario_id": "evidence_fixture", "sar_results": {"primary_slick": {
                        "centroid": None, "area_km2": None, "screening_score": 60}},
                    "drift_results": {"status": "UNAVAILABLE"},
                    "ais_results": {"screening_gate": Gate.unavailable_verdict("Fixture unavailable")}})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-document-kind"], "analyst-review-screening-summary")
            files = list(Path(directory).glob("*.pdf"))
            self.assertEqual(len(files), 1)
            content = pdf_page_streams(files[0])
            self.assertIn("EVIDENTIARY LEAD WITHHELD", content)
            self.assertIn("UNAVAILABLE", content)

    def test_export_rendering_exception_is_unavailable_without_download(self):
        with patch.object(self.server, "get_all_scenarios", return_value={"evidence_fixture": self.scenario}), \
                patch.object(self.server.report_gen, "generate_pdf_dossier", side_effect=RuntimeError("fixture rendering unavailable")):
            response = self.client.post("/api/export-case-summary", json={
                "scenario_id": "evidence_fixture", "sar_results": {"primary_slick": {"screening_score": 60}},
                "drift_results": {}, "ais_results": {}})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "UNAVAILABLE")
        self.assertTrue(response.json()["is_abstention"])


class ScreeningPDFRegressions(unittest.TestCase):
    def render(self, ais):
        directory = tempfile.TemporaryDirectory(prefix="ocean-evidence-", dir=os.environ.get("OCEAN_SHIELD_TEST_TMP"))
        self.addCleanup(directory.cleanup)
        generator = DossierReportGenerator(directory.name)
        path = generator.generate_pdf_dossier(
            {"id": "evidence_fixture", "region": "Synthetic unit fixture", "satellite_metadata": {}},
            {"primary_slick": {"centroid": None, "area_km2": None, "screening_score": 60},
             "observability": {"status": "NOT_OPERATIONALLY_INTERPRETABLE"}},
            {"status": "UNAVAILABLE", "counterfactual_verification": {
                "status": "UNAVAILABLE", "verdict": "NOT_ASSESSED", "verification_metrics": None}},
            ais,
        )
        self.assertGreater(Path(path).stat().st_size, 5000)
        content = pdf_page_streams(path)
        self.assertTrue(content, "Actual PDF page streams must be decodable")
        return content

    def test_pdf_propagates_unavailable_and_null_geometry_without_legal_labels(self):
        lead = AISEngine().score_and_rank_suspects([candidate()], 0, 0, 0)[0]
        gate = Gate.unavailable_verdict("Synthetic failure witness")
        lead["abstention_verdict"] = gate
        content = self.render({"primary_review_lead": lead, "screening_gate": gate})
        self.assertIn("EVIDENTIARY LEAD WITHHELD", content)
        self.assertIn("UNAVAILABLE", content)
        self.assertIn("NOT_SUPPLIED", content)
        for unsupported in ("Bayesian Legal Gate", "Prevents false accusation", "CRITICAL VIOLATION",
                            "Admissible Hash Ledger", "CHAIN-OF-CUSTODY DIGITAL SEAL", "Conf)"):
            self.assertNotIn(unsupported, content)

    def test_pdf_does_not_treat_incomplete_clean_audit_as_verified(self):
        gate = {"status": "ASSESSED", "decision": "SCREENING_LEAD", "is_abstention": False,
                "coverage_status": "VALIDATED", "confidence_score": 99}
        content = self.render({"screening_gate": gate, "primary_review_lead": {
            "spoofing_audit": {"has_anomalies": False, "mmsi_valid": True},
            "abstention_verdict": gate}})
        self.assertIn("EVIDENTIARY LEAD WITHHELD", content)
        self.assertIn("NOT_ASSESSED", content)
        self.assertNotIn("Verified Continuous", content)

    def test_pdf_malformed_audit_and_stress_containers_stay_not_assessed(self):
        content = self.render({"screening_gate": {
            "status": "ASSESSED", "is_abstention": False, "decision": "SCREENING_LEAD",
            "coverage_status": "VALIDATED", "entropy_metrics": ["malformed"]},
            "primary_review_lead": {"spoofing_audit": ["malformed"],
                                    "adversarial_stress_test": ["malformed"]}})
        self.assertIn("EVIDENTIARY LEAD WITHHELD", content)
        self.assertIn("NOT_ASSESSED", content)

    def test_pdf_uses_current_conditional_verdicts_without_causality_score(self):
        directory = tempfile.TemporaryDirectory(prefix="ocean-conditional-pdf-", dir=os.environ.get("OCEAN_SHIELD_TEST_TMP"))
        self.addCleanup(directory.cleanup)
        path = DossierReportGenerator(directory.name).generate_pdf_dossier(
            {"id": "evidence_fixture", "region": "Synthetic unit fixture", "satellite_metadata": {}},
            {"primary_slick": {"centroid": None, "area_km2": None, "screening_score": 60}},
            {"status": "CONDITIONAL_DEMO_SCENARIO", "counterfactual_verification": {
                "verdict": "CONDITIONAL_SPATIAL_AGREEMENT", "verdict_badge": "CONDITIONAL SPATIAL AGREEMENT",
                "verification_metrics": {"centroid_distance_km": 0.0, "predicted_containment_percent": 0.0,
                                          "jaccard_index": 0.0, "physical_causality_score": None},
            }},
            {},
        )
        content = pdf_page_streams(path)
        searchable_content = re.sub(r"\)\s*Tj\s*T\*\s*\(", " ", content)
        self.assertIn("CONDITIONAL SPATIAL AGREEMENT", searchable_content)
        self.assertIn("NOT ESTIMATED", content)
        self.assertNotIn("None/100", content)


if __name__ == "__main__":
    unittest.main()
