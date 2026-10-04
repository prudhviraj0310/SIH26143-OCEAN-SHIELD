"""Bounded adversarial regressions against both historical and repaired contracts.

Run using the existing Ocean venv with --depth-root PATH --json PATH.
Depth functions are extracted unchanged via AST to avoid model/network startup.
Only input/provider boundaries are faked; algorithms under test are real code.
"""
import argparse
import ast
import asyncio
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import typing
import unittest
from unittest.mock import patch
import numpy as np
from scipy import ndimage

WITNESSES = {}
DEPTH = None
BENCHMARK = None
SERVER_FUNCTION = None
CLI_FUNCTION = None


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_depth(root):
    path = root / "src/depth_wizard/elevation_engine.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ElevationEngine")
    # Keep methods plus their actual support helpers; algorithms are not copied,
    # replaced or mocked. __new__ below avoids only the neural-loading constructor.
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
    fit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "robust_affine_calibration_irls")
    namespace = {"np": np, "ndimage": ndimage, "HAS_SRTM_PROVIDER": False,
                 "HAS_RASTERIO": False, "HAS_TORCH_TRANSFORMERS": False, "Path": Path,
                 **{n: getattr(typing, n) for n in ("Dict", "Any", "Optional", "Tuple")}}
    module = ast.fix_missing_locations(ast.Module(body=[fit, cls], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def extract_function(path, name):
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    fn.decorator_list = []
    return compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec")


class WitnessTest(unittest.TestCase):
    def note(self, value):
        WITNESSES[self.id()] = value


class OceanAdversarial(WitnessTest):
    def setUp(self):
        from src.ocean_shield.falsification import FalsificationAndAbstentionEngine
        from src.ocean_shield.ais_engine import AISEngine
        self.gate = FalsificationAndAbstentionEngine
        self.ais = AISEngine()

    def candidate(self, score=90):
        return {"mmsi": 419001234, "composite_score": score,
                "closest_approach": {"distance_nm": .1, "time_diff_h": 1}}

    def stress(self, audit):
        c = self.candidate()
        if audit is not None:
            c["spoofing_audit"] = audit
        return self.gate.run_adversarial_stress_test(c, 0, 0, 1, 10)

    def test_missing_sr_checkpoint_has_working_fallback(self):
        from src.ocean_shield.models.super_resolution import enhance_sar_deep_learning
        class Untrained:
            weights_loaded = False
        out = enhance_sar_deep_learning(np.arange(64, dtype=np.uint8).reshape(8, 8), Untrained())
        self.assertEqual(out.shape, (16, 16))

    def test_exception_handler_preserves_gate_failure_not_nameerror(self):
        with patch.object(self.gate, "evaluate_decision_theoretic_abstention", side_effect=RuntimeError("AUDIT_ONLY gate unavailable")):
            try:
                self.ais.score_and_rank_suspects([], 0, 0, 0)
            except Exception as exc:
                self.note({"exception": type(exc).__name__, "message": str(exc)})
                self.assertNotIsInstance(exc, NameError)

    def test_none_score_is_rejected_without_typeerror(self):
        try:
            r = self.gate.evaluate_decision_theoretic_abstention([self.candidate(None)])
        except (TypeError, NameError) as exc:
            self.fail(f"Unexpected crash after score sanitization: {type(exc).__name__}: {exc}")
        self.assertTrue(r["is_abstention"])

    def test_nan_score_cannot_produce_nonabstained_lead(self):
        r = self.gate.evaluate_decision_theoretic_abstention([self.candidate(float("nan"))])
        self.note(r)
        self.assertTrue(r["is_abstention"])

    def test_unknown_source_is_in_single_candidate_entropy(self):
        r = self.gate.evaluate_decision_theoretic_abstention([self.candidate(55)])
        self.note(r)
        self.assertGreater(r["entropy_metrics"]["normalized_entropy"], .82)

    def test_unknown_source_not_dropped_with_two_candidates(self):
        candidates = [self.candidate(90), self.candidate(20)]
        r = self.gate.evaluate_decision_theoretic_abstention(candidates)
        self.note({"weights": [c["lead_priority_weight"] for c in candidates], "decision": r["decision"]})
        self.assertLess(sum(c["lead_priority_weight"] for c in candidates), .9999)

    def test_missing_ais_audit_now_blocks_stress_verdict(self):
        r = self.stress(None)
        self.note(r)
        self.assertFalse(r["challenges"][3]["survived"])
        self.assertIn("VULNERABLE", r["verdict"])

    def test_malformed_nonempty_ais_audit_cannot_pass(self):
        r = self.stress({"irrelevant": True})
        self.note(r)
        self.assertFalse(r["challenges"][3]["survived"])

    def test_one_impossible_speed_jump_blocks_integrity(self):
        audit = self.ais.detect_ais_spoofing_and_gaps([
            {"relative_time_hours": -1, "sog_knots": 6},
            {"relative_time_hours": -.98, "sog_knots": 18}], mmsi=419001234)
        r = self.stress(audit)
        self.note({"audit": audit, "stress": r})
        self.assertFalse(r["challenges"][3]["survived"])

    def test_empty_track_is_not_verified_continuous(self):
        audit = self.ais.detect_ais_spoofing_and_gaps([], mmsi=419001234)
        r = self.stress(audit)
        self.note({"audit": audit, "stress": r})
        self.assertFalse(r["challenges"][3]["survived"])

    def test_ais_failure_reaches_final_abstention(self):
        audit = {"has_anomalies": True, "corridor_blackout": True, "mmsi_valid": False,
                 "anomaly_count": 2, "anomalies_detected": ["AUDIT_ONLY integrity failure"]}
        vessel = self.candidate()
        vessel["trajectory"] = []
        with patch.object(self.ais, "detect_ais_spoofing_and_gaps", return_value=audit):
            lead = self.ais.score_and_rank_suspects([vessel], 0, 0, 0)[0]
        self.note(lead)
        self.assertTrue(lead["abstention_verdict"]["is_abstention"])

    def test_fractional_hindcast_clock_matches_assumed_age(self):
        from src.ocean_shield.drift_engine import DriftEngine, OceanCurrentField
        e = DriftEngine(num_particles=40)
        e._compute_kde_hdr_contours = lambda *a, **k: {"audit_only": "clock check"}
        f = OceanCurrentField(constant_vectors=True)
        r = e.run_hindcast(0, 0, f, target_slick_age_hours=.1, time_step_minutes=15)
        self.note({"last_time": r["hindcast_trajectory"][-1]["relative_time_hours"], "reported_t0": r["origin_release_point"]["estimated_t0_hours_relative"]})
        self.assertAlmostEqual(r["hindcast_trajectory"][-1]["relative_time_hours"], -.1)

    def test_nonnegative_current_never_moves_particles_west(self):
        from src.ocean_shield.drift_engine import DriftEngine, OceanCurrentField
        class PositiveJet(OceanCurrentField):
            def get_velocity_at(self, lat, lon, t_hours_relative=0):
                # Smooth eastward-only jet, bounded everywhere by 0.5 m/s.
                return float(.5 * (1 - np.exp(-(lat / .004) ** 2))), 0, 0, 0
        e = DriftEngine(num_particles=80, diffusion_coeff=0, wind_drift_factor=0)
        r = e.run_forecast(0, 0, PositiveJet(), forecast_hours=.1, time_step_minutes=6)
        a, b = r["forecast_trajectory"][0]["particles_sample"], r["forecast_trajectory"][-1]["particles_sample"]
        dx = [q[0] - p[0] for p, q in zip(a, b)]
        self.note({"westbound_sample_count": sum(v < -1e-5 for v in dx), "minimum_delta_lon_deg": min(dx)})
        self.assertGreaterEqual(min(dx), -1e-5)

    def test_weathering_water_mass_fraction_now_consistent(self):
        from src.ocean_shield.drift_engine import DriftEngine
        r = DriftEngine().compute_oil_weathering(24, wind_speed_ms=6)
        remaining = r["initial_mass_tonnes"] - r["evaporated_mass_tonnes"]
        implied = 100 * (1 - remaining / r["emulsion_apparent_mass_tonnes"])
        self.note({"reported_water_pct": r["water_content_mousse_pct"], "implied_pct": implied})
        self.assertAlmostEqual(implied, r["water_content_mousse_pct"], delta=.02)


class FakeBenchmarkInputs:
    ids = ["isro_sac_ahmedabad", "gamus_dc_11_33", "gamus_hilly_ridge", "gamus_dc_02_26"]

    def __init__(self, errors=None, missing=(), synthetic=False):
        self.errors = errors or {}
        self.missing = set(missing)
        self.synthetic = synthetic
        self.gt = np.arange(16, dtype=np.float32).reshape(4, 4) + 10

    def load_gamus_scene(self, sid, **kwargs):
        if sid in self.missing:
            raise RuntimeError("AUDIT_ONLY missing scene")
        self.sid = sid
        return {"rgb_image": np.zeros((4, 4, 3), dtype=np.uint8), "base_elevation_m": 10,
                "max_structural_height_m": 25, "ground_truth_dsm": self.gt,
                "geo_metadata": {}, "name": "AUDIT_ONLY synthetic fixture",
                "is_synthetic": self.synthetic}

    def extract_relative_depth(self, rgb):
        return np.zeros((4, 4), dtype=np.float32)

    def calibrate_to_absolute_dsm(self, **kwargs):
        # Controlled synthetic inputs exercise aggregation even though bundled
        # production scenes cannot establish metric controls. Not accuracy evidence.
        return {"dsm": self.gt + self.errors.get(self.sid, 0),
                "stats": {"is_metric": True, "status": "SUCCESS", "control_status": "AUDIT_ONLY_CONTROL"}}


class DepthAdversarial(WitnessTest):
    def setUp(self):
        self.engine = DEPTH["ElevationEngine"].__new__(DEPTH["ElevationEngine"])

    def run_benchmark(self, inputs):
        ns = {"engine": inputs, "DepthWizardBenchmark": BENCHMARK}
        exec(SERVER_FUNCTION, ns)
        with contextlib.redirect_stdout(io.StringIO()):
            return asyncio.run(ns["run_full_benchmark"]())

    def test_all_missing_scenes_now_fail(self):
        r = self.run_benchmark(FakeBenchmarkInputs(missing=FakeBenchmarkInputs.ids))
        self.note(r)
        self.assertEqual(r["status"], "FAILED")
        self.assertNotIn("APPROVED", json.dumps(r))
        self.assertNotIn("overall_isro_compliance", r["benchmark_summary"])

    def test_partial_landscape_suite_not_success(self):
        r = self.run_benchmark(FakeBenchmarkInputs(missing=FakeBenchmarkInputs.ids[:3]))
        self.note(r)
        self.assertNotEqual(r["status"], "SUCCESS")

    def test_catastrophic_mountain_not_hidden_by_macro_average(self):
        r = self.run_benchmark(FakeBenchmarkInputs(errors={"gamus_hilly_ridge": 200}))
        self.note(r)
        self.assertNotIn("APPROVED", json.dumps(r))
        self.assertEqual(r["certification"], "NOT CERTIFIED")

    def test_cli_does_not_approve_1000m_error(self):
        inputs = FakeBenchmarkInputs(errors={sid: 1000 for sid in FakeBenchmarkInputs.ids})
        ns = {"ElevationEngine": lambda: inputs, "DepthWizardBenchmark": BENCHMARK,
              "print_banner": lambda: None}
        exec(CLI_FUNCTION, ns)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            ns["run_benchmark"]()
        self.note({"stdout_tail": stream.getvalue()[-850:]})
        self.assertNotIn("VERDICT: APPROVED", stream.getvalue())

    def test_synthetic_provenance_survives_aggregate(self):
        r = self.run_benchmark(FakeBenchmarkInputs(synthetic=True))
        self.note(r)
        self.assertTrue(all(row["provenance"].get("is_synthetic") is True for row in r["scene_evaluations"]))
        self.assertEqual(r["provenance_groups"]["synthetic"]["evaluated_scene_count"], 4)
        self.assertEqual(r["provenance_groups"]["real"]["evaluated_scene_count"], 0)

    def test_irls_recovers_independent_affine_anchors(self):
        x = np.linspace(0, 1, 20)
        r = DEPTH["robust_affine_calibration_irls"](x, 3 * x + 20)
        self.note(r)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["scale"], 3, places=5)
        self.assertAlmostEqual(r["offset"], 20, places=5)

    def test_irls_rank_deficient_controls_not_converged(self):
        r = DEPTH["robust_affine_calibration_irls"](np.ones(10), np.arange(10))
        self.note(r)
        self.assertFalse(r["converged"])

    def test_irls_iteration_exhaustion_not_converged(self):
        r = DEPTH["robust_affine_calibration_irls"](np.arange(10), np.arange(10), max_iter=0)
        self.note(r)
        self.assertFalse(r["converged"])

    def test_calibrated_dsm_dtm_and_agl_are_consistent(self):
        yy, xx = np.mgrid[:24, :24]
        anchors = 100 + 20 * xx / 23 + 50 * np.sin(2 * np.pi * yy / 23)
        class Provider:
            @staticmethod
            def get_elevation_grid(**kwargs):
                return anchors.astype(np.float32)
        with patch.dict(DEPTH, {"HAS_SRTM_PROVIDER": True, "SRTMElevationProvider": Provider}):
            r = self.engine.calibrate_to_absolute_dsm((xx / 23).astype(np.float32), 100, geo_bounds=[0, 0, 1, 1])
        mismatch = float(np.max(np.abs(r["dsm"] - r["dtm"] - r["structural_heights"])))
        self.note({"maximum_DSM_minus_DTM_minus_AGL_m": mismatch, "irls": r["stats"]["irls_huber_fit"]})
        self.assertLess(mismatch, .001)

    def test_nodata_anchors_cannot_establish_metric_dsm(self):
        class Provider:
            @staticmethod
            def get_elevation_grid(**kwargs):
                return np.full((8, 8), np.nan, dtype=np.float32)
        with patch.dict(DEPTH, {"HAS_SRTM_PROVIDER": True, "SRTMElevationProvider": Provider}):
            r = self.engine.calibrate_to_absolute_dsm(np.linspace(0, 1, 64).reshape(8, 8), 50, geo_bounds=[0, 0, 1, 1])
        self.note({"is_metric": r["stats"]["is_metric"], "finite_pixels": int(np.isfinite(r["dsm"]).sum())})
        self.assertFalse(r["stats"]["is_metric"])

    def test_annular_hlz_no_longer_selects_obstacle(self):
        dtm = np.zeros((101, 101), dtype=np.float32)
        heights = dtm.copy()
        yy, xx = np.ogrid[:101, :101]
        heights[(yy - 50)**2 + (xx - 50)**2 < 100] = 10
        r = self.engine.detect_landing_zones(dtm + heights, dtm, heights, ground_res_m=1, pad_radius_m=4)
        self.note(r)
        self.assertGreater(len(r["candidate_zones"]), 0)
        self.assertTrue(all(heights[z["pixel_y"], z["pixel_x"]] <= .5 for z in r["candidate_zones"]))

    def test_hlz_cannot_certify_nan_elevation(self):
        dtm = np.zeros((40, 40), dtype=np.float32)
        r = self.engine.detect_landing_zones(np.full_like(dtm, np.nan), dtm, dtm, ground_res_m=1, pad_radius_m=4)
        self.note(r)
        self.assertEqual(r["candidate_zones"], [])

    def test_missing_geology_does_not_claim_bis_tehd(self):
        dtm = np.tile(np.linspace(0, 1000, 64), (64, 1))
        r = self.engine.screen_landslide_risk(dtm)
        self.note(r)
        self.assertNotIn("bis_is14496_hazard_class", r)

    def test_nodata_landslide_assessment_not_success(self):
        r = self.engine.screen_landslide_risk(np.full((8, 8), np.nan))
        self.note(r)
        self.assertNotEqual(r["status"], "SUCCESS")


class RecordedResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)
        self.receipts = getattr(self, "receipts", [])

    def addSuccess(self, test):
        super().addSuccess(test)
        self.receipts.append({"test": test.id(), "status": "PASS"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.receipts.append({"test": test.id(), "status": "FAIL", "detail": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.receipts.append({"test": test.id(), "status": "ERROR", "detail": self._exc_info_to_string(err, test)})


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main():
    global DEPTH, BENCHMARK, SERVER_FUNCTION, CLI_FUNCTION
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocean-root", type=Path, default=Path.cwd())
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    ocean_root, depth_root = args.ocean_root.resolve(), args.depth_root.resolve()
    sys.path.insert(0, str(ocean_root))
    DEPTH = extract_depth(depth_root)
    BENCHMARK = load_file("audit_depth_benchmark", depth_root / "src/depth_wizard/benchmark.py").DepthWizardBenchmark
    SERVER_FUNCTION = extract_function(depth_root / "src/depth_wizard/server.py", "run_full_benchmark")
    CLI_FUNCTION = extract_function(depth_root / "src/depth_wizard/cli.py", "run_benchmark")
    hashes = {}
    for label, root, files in [
        ("ocean", ocean_root, ["src/ocean_shield/falsification.py", "src/ocean_shield/ais_engine.py", "src/ocean_shield/drift_engine.py", "src/ocean_shield/models/super_resolution.py"]),
        ("depth", depth_root, ["src/depth_wizard/elevation_engine.py", "src/depth_wizard/server.py", "src/depth_wizard/cli.py", "src/depth_wizard/benchmark.py"]),
    ]:
        for path in files:
            hashes[f"{label}/{path}"] = hashlib.sha256((root / path).read_bytes()).hexdigest()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(OceanAdversarial), unittest.defaultTestLoader.loadTestsFromTestCase(DepthAdversarial)])
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    summary = {"run": result.testsRun, "passed": result.testsRun - len(result.failures) - len(result.errors), "failures": len(result.failures), "errors": len(result.errors)}
    payload = {"summary": summary, "source_sha256": hashes, "tests": result.receipts, "witnesses": WITNESSES,
               "scope": "Ocean actual modules; unchanged Depth functions via AST. Fake input/provider boundaries. No real model/field performance claim."}
    args.json.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    print("ADVERSARIAL_RECEIPT", summary)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
