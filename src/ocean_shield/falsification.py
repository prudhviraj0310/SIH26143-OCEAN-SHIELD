"""Heuristic screening weights, sensitivity checks, and evidence-state holds.

Softmax weights, entropy, and project thresholds are uncalibrated triage signals.
They do not establish source identity, responsibility, or a false-accusation rate.
Eligibility requires entropy <= 0.82 and margin >= 0.15, plus assessed inputs.
"""

import math
from numbers import Real
from typing import Dict, List, Any, Optional, Tuple


class FalsificationAndAbstentionEngine:
    """Keeps incomplete or unstable evidence on an analyst-review hold."""

    MAX_NORMALIZED_ENTROPY_THRESHOLD = 0.82  # Uncertainty too high if entropy > 82% of uniform
    MIN_SEPARATION_MARGIN = 0.15            # Top candidate must lead second candidate by >= 15%
    CURRENT_VARIATION_STRESS = 0.20         # +/- 20% hydrodynamic current sensitivity challenge
    GPS_JITTER_STRESS_NM = 1.0              # 1.0 nautical mile GPS corridor uncertainty challenge
    TEMPERATURE = 12.0                      # Heuristic scaling, not calibration
    UNKNOWN_SOURCE_SCORE = 45.0             # Project baseline, not a measured prior

    @staticmethod
    def finite_number(value: Any, minimum: Optional[float] = None,
                      maximum: Optional[float] = None) -> Optional[float]:
        """Canonicalize a scalar once; absent/invalid evidence stays absent."""
        if isinstance(value, bool) or not isinstance(value, (Real, str)):
            return None
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError):
            return None
        if not math.isfinite(number):
            return None
        if minimum is not None and number < minimum:
            return None
        if maximum is not None and number > maximum:
            return None
        return number

    @classmethod
    def unavailable_verdict(cls, reason: str, candidates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        distribution = [
            {"hypothesis": "VESSEL_CANDIDATE", "candidate_index": index,
             "mmsi": candidate.get("mmsi"), "score": candidate.get("composite_score"),
             "lead_priority_weight": None}
            for index, candidate in enumerate(candidates or [])
        ]
        distribution.append({"hypothesis": "UNKNOWN_SOURCE", "score": cls.UNKNOWN_SOURCE_SCORE,
                             "lead_priority_weight": None})
        return {
            "status": "UNAVAILABLE", "decision": "EVIDENCE_GATE_UNAVAILABLE",
            "is_abstention": True, "reason": reason,
            "actionable_recommendation": "Restore and rerun the evidence checks before reviewing a lead.",
            "confidence_score": None, "confidence_status": "UNCALIBRATED",
            "weight_kind": "uncalibrated_lead_priority",
            "hypothesis_distribution": distribution, "unknown_source_weight": None,
            "coverage_status": "NOT_ASSESSED", "integrity_status": "NOT_ASSESSED",
            "stress_status": "UNAVAILABLE", "entropy_metrics": {
                "status": "NOT_ASSESSED", "entropy_bits": None,
                "max_possible_entropy": None, "normalized_entropy": None,
                "uncertainty_level": "NOT_ASSESSED",
            },
        }

    @classmethod
    def assess_ais_integrity(cls, audit: Any) -> Dict[str, Any]:
        """Validate the complete v1 audit; no permissive telemetry defaults."""
        def hold(reason: str, status: str = "NOT_ASSESSED") -> Dict[str, Any]:
            return {"status": status, "passed": False, "reason": reason}

        if isinstance(audit, dict) and audit.get("status") == "UNAVAILABLE":
            return hold("AIS integrity audit is unavailable.", "UNAVAILABLE")
        if not isinstance(audit, dict) or type(audit.get("schema_version")) is not int or audit["schema_version"] != 1:
            return hold("AIS audit missing or unsupported schema.")
        bool_fields = ("has_anomalies", "corridor_blackout", "impossible_speed_jump",
                       "identity_conflict", "position_jump", "telemetry_validated", "cpa_time_covered")
        count_fields = ("anomaly_count", "total_fixes", "valid_time_fixes", "valid_position_fixes",
                        "valid_speed_fixes", "rejected_fixes")
        numeric_fields = ("max_gap_minutes", "max_acceleration_kts_min", "time_span_hours")
        if any(type(audit.get(key)) is not bool for key in bool_fields):
            return hold("AIS audit boolean fields are incomplete or malformed.")
        if any(type(audit.get(key)) is not int or audit[key] < 0 for key in count_fields):
            return hold("AIS audit telemetry counts are incomplete or malformed.")
        if any(cls.finite_number(audit.get(key), 0) is None for key in numeric_fields):
            return hold("AIS audit metrics are incomplete or nonfinite.")
        anomalies = audit.get("anomalies_detected")
        reasons = audit.get("assessment_reasons")
        if (not isinstance(anomalies, list) or any(not isinstance(a, str) for a in anomalies)
                or audit["anomaly_count"] != len(anomalies)
                or not isinstance(reasons, list) or any(not isinstance(r, str) for r in reasons)
                or audit.get("status") not in ("ASSESSED", "NOT_ASSESSED")
                or audit.get("integrity_rating") not in ("VERIFIED_CONTINUOUS", "COMPROMISED / ANOMALOUS", "NOT_ASSESSED")
                or (audit.get("mmsi_valid") is not None and type(audit.get("mmsi_valid")) is not bool)):
            return hold("AIS audit state or anomaly schema is malformed.")
        if any(audit[key] > audit["total_fixes"] for key in count_fields[2:]):
            return hold("AIS audit counts contradict the number of fixes.")
        if audit["valid_position_fixes"] > audit["valid_time_fixes"] or audit["valid_speed_fixes"] > audit["valid_time_fixes"]:
            return hold("AIS audit position/speed counts contradict valid timestamp counts.")

        # These flags independently block integrity, even if has_anomalies is false.
        compromised = (audit["has_anomalies"] or audit["anomaly_count"] > 0
                       or audit["corridor_blackout"] or audit["impossible_speed_jump"]
                       or audit["identity_conflict"] or audit["position_jump"]
                       or audit.get("mmsi_valid") is False
                       or audit["max_acceleration_kts_min"] > 3.0)
        if compromised:
            return hold("; ".join(anomalies) or "Severe AIS identity, blackout, or kinematic integrity flag.", "COMPROMISED")
        if (audit["status"] != "ASSESSED" or not audit["telemetry_validated"]
                or audit["valid_time_fixes"] < 2 or audit["valid_position_fixes"] < 2
                or audit["time_span_hours"] <= 0 or not audit["cpa_time_covered"]
                or audit["rejected_fixes"] > 0 or audit.get("mmsi_valid") is not True
                or audit["integrity_rating"] != "VERIFIED_CONTINUOUS"):
            return hold("; ".join(reasons) or "Insufficient valid time/position coverage or identity support.")
        return {"status": "ASSESSED", "passed": True,
                "reason": "Supplied telemetry passed continuity checks; receiver coverage is separate."}

    @classmethod
    def calculate_shannon_entropy(cls, probabilities: List[float]) -> Dict[str, Any]:
        """
        Computes Shannon Information Entropy H(p) and Normalized Entropy H_norm.
        """
        values = [cls.finite_number(p, 0, 1) for p in probabilities]
        k = len(probabilities)
        if not values or any(p is None for p in values) or sum(values) <= 0:
            return {"status": "NOT_ASSESSED", "entropy_bits": None,
                    "max_possible_entropy": None, "normalized_entropy": None,
                    "uncertainty_level": "NOT_ASSESSED"}
        if k <= 1:
            return {
                "status": "ASSESSED",
                "entropy_bits": 0.0,
                "max_possible_entropy": 0.0,
                "normalized_entropy": 0.0,
                "uncertainty_level": "SINGLE HYPOTHESIS (NO COMPARISON)"
            }
        total = math.fsum(values)
        p_norm = [p / total for p in values]
        h_bits = -math.fsum(p * math.log2(p) for p in p_norm if p > 0)
        h_max = math.log2(k)
        h_normalized = h_bits / h_max
        if h_normalized > cls.MAX_NORMALIZED_ENTROPY_THRESHOLD:
            level = "CRITICAL (UNIFORM / HIGH CONFUSION)"
        elif h_normalized >= 0.50:
            level = "MODERATE (AMBIGUOUS SCREENING)"
        else:
            level = "LOW (HIGH CONVERGENCE)"

        return {
            "status": "ASSESSED",
            "entropy_bits": h_bits,
            "max_possible_entropy": h_max,
            "normalized_entropy": h_normalized,
            "uncertainty_level": level
        }

    @classmethod
    def evaluate_decision_theoretic_abstention(
        cls,
        ranked_candidates: List[Dict[str, Any]],
        *,
        coverage_validated: bool = False,
        unknown_source_hypothesis: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate the leading screening candidate after its integrity/stress checks.

        The legacy unknown_source_hypothesis argument cannot remove the mandatory
        unknown-source baseline. A coverage declaration is accepted only as strict
        True; callers must perform the independent receiver-coverage assessment.
        """
        if not isinstance(ranked_candidates, list) or any(not isinstance(c, dict) for c in ranked_candidates):
            result = cls.unavailable_verdict("Candidate list is malformed.")
            result["status"] = "INVALID_INPUT"
            return result

        scores = []
        distribution = []
        for index, candidate in enumerate(ranked_candidates):
            score = cls.finite_number(candidate.get("composite_score"), 0.0, 100.0)
            candidate["composite_score"] = score
            candidate["score_status"] = "VALID" if score is not None else "INVALID_INPUT"
            candidate["lead_priority_weight"] = None
            scores.append(score)
            distribution.append({"hypothesis": "VESSEL_CANDIDATE", "candidate_index": index,
                                 "mmsi": candidate.get("mmsi"), "score": score,
                                 "lead_priority_weight": None})
        distribution.append({"hypothesis": "UNKNOWN_SOURCE", "score": cls.UNKNOWN_SOURCE_SCORE,
                             "lead_priority_weight": None})
        result = {
            "status": "ASSESSED", "decision": "INSUFFICIENT_EVIDENCE", "is_abstention": True,
            "confidence_score": None, "confidence_status": "UNCALIBRATED",
            "weight_kind": "uncalibrated_lead_priority", "leading_candidate_mmsi": None,
            "hypothesis_distribution": distribution, "unknown_source_weight": None,
            "unknown_source_baseline": "Heuristic score 45/100; not a measured prior or likelihood.",
            "coverage_status": "VALIDATED" if coverage_validated is True else "NOT_ASSESSED",
            "integrity_status": "NOT_ASSESSED", "stress_status": "NOT_ASSESSED",
            "separation_margin": None,
            "entropy_metrics": cls.calculate_shannon_entropy([]),
            "thresholds": {"maximum_normalized_entropy_inclusive": cls.MAX_NORMALIZED_ENTROPY_THRESHOLD,
                           "minimum_separation_margin_inclusive": cls.MIN_SEPARATION_MARGIN,
                           "minimum_score_inclusive": 55.0},
            "actionable_recommendation": "Validate source coverage and telemetry; review independent corroboration.",
        }
        if any(score is None for score in scores):
            result.update(status="INVALID_INPUT", reason="Candidate scores must be finite scalars in [0, 100]; ranking withheld.")
            return result

        all_scores = scores + [cls.UNKNOWN_SOURCE_SCORE]
        maximum = max(all_scores)
        exp_scores = [math.exp((score - maximum) / cls.TEMPERATURE) for score in all_scores]
        total = math.fsum(exp_scores)
        weights = [value / total for value in exp_scores]
        for row, weight in zip(distribution, weights):
            row["lead_priority_weight"] = weight
        for candidate, weight in zip(ranked_candidates, weights):
            candidate["lead_priority_weight"] = weight
        result["unknown_source_weight"] = weights[-1]
        result["entropy_metrics"] = cls.calculate_shannon_entropy(weights)
        if not scores:
            result.update(status="NOT_ASSESSED", reason="No vessel candidates; only the unknown-source hypothesis remains.")
            return result

        top_index = max(range(len(scores)), key=lambda index: scores[index])
        top = ranked_candidates[top_index]
        top_weight = weights[top_index]
        runner_weight = max(weight for index, weight in enumerate(weights) if index != top_index)
        margin = top_weight - runner_weight
        result.update(leading_candidate_mmsi=top.get("mmsi"), leading_candidate_index=top_index,
                      lead_priority_weight=top_weight, separation_margin=margin)
        integrity = cls.assess_ais_integrity(top.get("spoofing_audit"))
        stress = top.get("adversarial_stress_test")
        challenges = stress.get("challenges") if isinstance(stress, dict) else None
        stress_passed = (isinstance(stress, dict) and stress.get("status") == "ASSESSED"
                         and stress.get("stress_passed") is True and isinstance(challenges, list)
                         and len(challenges) == 4
                         and all(isinstance(c, dict) and c.get("status") == "ASSESSED"
                                 and c.get("survived") is True for c in challenges))
        result["integrity_status"] = integrity["status"]
        result["stress_status"] = stress.get("status", "NOT_ASSESSED") if isinstance(stress, dict) else "NOT_ASSESSED"

        blockers = []
        if weights[-1] >= top_weight:
            blockers.append("Unknown-source weight equals or exceeds the leading vessel weight.")
        entropy = result["entropy_metrics"]["normalized_entropy"]
        if entropy is None or entropy > cls.MAX_NORMALIZED_ENTROPY_THRESHOLD:
            blockers.append("Full-hypothesis entropy exceeds the project <=0.82 screening rule or is unassessed.")
        if margin < cls.MIN_SEPARATION_MARGIN:
            blockers.append("Separation from all other hypotheses, including unknown, is below 0.15.")
        if scores[top_index] < 55.0:
            blockers.append("Leading score is below the project 55/100 screening rule.")
        if coverage_validated is not True:
            blockers.append("AIS satellite/receiver coverage has not been independently validated.")
            result["status"] = "NOT_ASSESSED"
        if not integrity["passed"]:
            blockers.append(integrity["reason"])
            if integrity["status"] in ("NOT_ASSESSED", "UNAVAILABLE"):
                result["status"] = integrity["status"]
        if not stress_passed:
            blockers.append("Candidate sensitivity checks are missing, unavailable, or failed.")
            if result["stress_status"] in ("NOT_ASSESSED", "UNAVAILABLE"):
                result["status"] = result["stress_status"]
        if blockers:
            result["reason"] = " ".join(blockers)
            # Retain an advisory label for UI compatibility, but evidentiary hold is true.
            if result["status"] == "NOT_ASSESSED" and coverage_validated is not True:
                result["decision"] = "PROVISIONAL_SCREENING_LEAD"
            if result["status"] == "UNAVAILABLE":
                result["decision"] = "EVIDENCE_GATE_UNAVAILABLE"
            return result

        result.update(decision="SCREENING_LEAD", is_abstention=False,
                      reason="Leading candidate passes the project screening checks; responsibility is not determined.")
        return result

    @classmethod
    def run_adversarial_stress_test(
        cls,
        candidate: Dict[str, Any],
        origin_lat: float,
        origin_lon: float,
        current_speed_knots: float,
        wind_speed_knots: float
    ) -> Dict[str, Any]:
        """
        Executes 4 adversarial falsification attacks against a candidate attribution:
        1. Hydrodynamic Current Perturbation (+/- 20% ocean velocity)
        2. Wind Drift Variation (+/- 1.0% leeway factor)
        3. GPS Position Jitter (+/- 1.0 nm AIS uncertainty)
        4. AIS Integrity / Spoofing Challenge
        """
        cpa = candidate.get("closest_approach")
        cpa = cpa if isinstance(cpa, dict) else {}
        dist_nm = cls.finite_number(cpa.get("distance_nm"), 0.0, 10810.0)
        time_diff_h = cls.finite_number(cpa.get("time_diff_h"), 0.0, 744.0)
        current_speed = cls.finite_number(current_speed_knots, 0.0, 200.0)
        wind_speed = cls.finite_number(wind_speed_knots, 0.0, 400.0)
        physics_valid = (dist_nm is not None and time_diff_h is not None
                         and current_speed is not None and wind_speed is not None
                         and cls.finite_number(origin_lat, -90, 90) is not None
                         and cls.finite_number(origin_lon, -180, 180) is not None)

        challenges = []
        passed_challenges = 0

        # Attack 1: Hydrodynamic Current Perturbation (+/- 20%)
        current_displacement_nm = current_speed * cls.CURRENT_VARIATION_STRESS * max(1.0, time_diff_h) if physics_valid else None
        c1_survived = physics_valid and (dist_nm + current_displacement_nm) <= 4.0
        challenges.append({
            "challenge_name": "Hydrodynamic Current Perturbation (+/- 20%)",
            "stress_parameter": "+/-20% supplied surface-current speed",
            "status": "ASSESSED" if physics_valid else "NOT_ASSESSED",
            "estimated_origin_shift_nm": round(current_displacement_nm, 2) if physics_valid else None,
            "survived": c1_survived,
            "rationale": "Candidate remains within the project corridor under this scalar perturbation." if c1_survived else "Current-sensitivity check failed or its inputs are unavailable."
        })
        if c1_survived:
            passed_challenges += 1

        # Attack 2: Wind Leeway Deflection (+/- 1% leeway)
        wind_displacement_nm = wind_speed * 0.01 * max(1.0, time_diff_h) if physics_valid else None
        c2_survived = physics_valid and (dist_nm + wind_displacement_nm) <= 4.5
        challenges.append({
            "challenge_name": "Atmospheric Leeway Perturbation (+/- 1% wind drag)",
            "stress_parameter": "+/-1.0% supplied wind-speed factor",
            "status": "ASSESSED" if physics_valid else "NOT_ASSESSED",
            "estimated_origin_shift_nm": round(wind_displacement_nm, 2) if physics_valid else None,
            "survived": c2_survived,
            "rationale": "Candidate track robust to atmospheric wind drag variability." if c2_survived else "Wind variation exceeds spatial correlation window."
        })
        if c2_survived:
            passed_challenges += 1

        # Attack 3: GPS Sensor Jitter & Spatial Sensor Uncertainty (+/- 1 nm)
        jittered_dist = dist_nm + cls.GPS_JITTER_STRESS_NM if physics_valid else None
        c3_survived = physics_valid and jittered_dist < 4.0
        challenges.append({
            "challenge_name": "GPS Sensor Jitter & Interpolation Error (+/- 1.0 nm)",
            "stress_parameter": f"+/-{cls.GPS_JITTER_STRESS_NM:.1f} nm transponder noise",
            "status": "ASSESSED" if physics_valid else "NOT_ASSESSED",
            "effective_cpa_nm": round(jittered_dist, 2) if physics_valid else None,
            "survived": c3_survived,
            "rationale": "Candidate retains proximity under the project 1 nm position perturbation." if c3_survived else "Position-sensitivity check failed or its inputs are unavailable."
        })
        if c3_survived:
            passed_challenges += 1

        # Attack 4: AIS Spoofing & Continuity Audit (Hard gate: Telemetry must be present and verified)
        integrity = cls.assess_ais_integrity(candidate.get("spoofing_audit"))
        c4_survived = integrity["passed"]

        challenges.append({
            "challenge_name": "AIS Continuity & Anti-Spoofing Challenge",
            "stress_parameter": "Speed anomalies, teleportation jumps, MMSI duplication",
            "status": integrity["status"],
            "survived": c4_survived,
            "rationale": integrity["reason"]
        })
        if c4_survived:
            passed_challenges += 1

        robustness_score = round((passed_challenges / 4.0) * 100.0, 1)
        v_name = candidate.get("vessel_name") or candidate.get("name") or "Suspect Vessel"

        # All checks must pass. A percentage of checks is not calibrated robustness.
        if not c4_survived:
            final_verdict = f"FALSIFICATION_VULNERABLE (AIS_INTEGRITY_{integrity['status']})"
        elif passed_challenges == 4:
            final_verdict = "ADVERSARIAL_PHYSICS_ROBUST (SCREENING_GRADE)"
        else:
            final_verdict = "FALSIFICATION_VULNERABLE (CIRCUMSTANTIAL)"

        return {
            "status": ("UNAVAILABLE" if integrity["status"] == "UNAVAILABLE" else
                       "NOT_ASSESSED" if not physics_valid or integrity["status"] == "NOT_ASSESSED" else "ASSESSED"),
            "stress_passed": passed_challenges == 4,
            "integrity_status": integrity["status"],
            "metric_kind": "fraction_of_project_checks_passed_not_calibrated_robustness",
            "forcing_status": "illustrative_scalar_perturbations_not_provider_error_bounds",
            "mmsi": candidate.get("mmsi"),
            "name": v_name,
            "challenges_tested": 4,
            "challenges_passed": passed_challenges,
            "adversarial_robustness_score": robustness_score,
            "verdict": final_verdict,
            "challenges": challenges
        }
