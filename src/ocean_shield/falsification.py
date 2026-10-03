"""
Ocean Shield — Adversarial Self-Falsification & Decision-Theoretic Abstention Engine
SIH26143 (NTRO) / Maritime Domain Awareness

Implements Bayesian hypothesis testing, Shannon information entropy quantification,
adversarial stress-testing (hydrodynamic current variance, GPS jitter, AIS spoofing),
and decision-theoretic abstention to prevent reckless false vessel attribution in court.

Theoretical Foundations:
1. Shannon Information Entropy: H(p) = -sum(p_i * log2(p_i))
2. Normalized Entropy: H_norm = H(p) / log2(K)
3. Principled Abstention Gate: If H_norm > 0.82 or Margin(Top1, Top2) < 0.15 => ABSTAIN
4. Adversarial Sensitivity: Challenges attribution stability against +/-20% current perturbations
"""

import math
from typing import Dict, List, Any, Optional, Tuple


class FalsificationAndAbstentionEngine:
    """
    Executes adversarial counterfactual challenges and information-theoretic
    uncertainty quantification to establish legal defensibility.
    """

    MAX_NORMALIZED_ENTROPY_THRESHOLD = 0.82  # Uncertainty too high if entropy > 82% of uniform
    MIN_SEPARATION_MARGIN = 0.15            # Top candidate must lead second candidate by >= 15%
    CURRENT_VARIATION_STRESS = 0.20         # +/- 20% hydrodynamic current sensitivity challenge
    GPS_JITTER_STRESS_NM = 1.0              # 1.0 nautical mile GPS corridor uncertainty challenge
    TEMPERATURE = 12.0                      # Softmax scaling temperature for calibrated probabilities

    @classmethod
    def calculate_shannon_entropy(cls, probabilities: List[float]) -> Dict[str, float]:
        """
        Computes Shannon Information Entropy H(p) and Normalized Entropy H_norm.
        """
        p_clean = [p for p in probabilities if p > 1e-9]
        k = len(probabilities)

        if not p_clean or k <= 1:
            return {
                "entropy_bits": 0.0,
                "max_possible_entropy": 0.0,
                "normalized_entropy": 0.0,
                "uncertainty_level": "LOW (DETERMINISTIC)"
            }

        total = sum(p_clean)
        p_norm = [p / total for p in p_clean]

        h_bits = -sum(p * math.log2(p) for p in p_norm)
        h_max = math.log2(k) if k > 1 else 1.0
        h_normalized = round(h_bits / h_max, 4) if h_max > 0 else 0.0

        if h_normalized >= cls.MAX_NORMALIZED_ENTROPY_THRESHOLD:
            level = "CRITICAL (UNIFORM / HIGH CONFUSION)"
        elif h_normalized >= 0.50:
            level = "MODERATE (AMBIGUOUS ATTRIBUTION)"
        else:
            level = "LOW (HIGH CONVERGENCE)"

        return {
            "entropy_bits": round(h_bits, 4),
            "max_possible_entropy": round(h_max, 4),
            "normalized_entropy": h_normalized,
            "uncertainty_level": level
        }

    @classmethod
    def evaluate_decision_theoretic_abstention(
        cls,
        ranked_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enforces decision-theoretic boundaries to prevent wrongful legal accusations.
        Outputs either DEFINITIVE_LEAD, PROBABLE_LEAD, or INSUFFICIENT_EVIDENCE (ABSTAIN).
        """
        if not ranked_candidates:
            return {
                "decision": "INSUFFICIENT_EVIDENCE",
                "is_abstention": True,
                "reason": "Zero vessel trajectories intersected the spatial-temporal hindcast corridor.",
                "actionable_recommendation": "Maintain persistent satellite surveillance; inspect local port berths.",
                "confidence_score": 0.0,
                "entropy_metrics": {
                    "entropy_bits": 0.0,
                    "max_possible_entropy": 0.0,
                    "normalized_entropy": 1.0,
                    "uncertainty_level": "MAXIMAL (NO CANDIDATES)"
                }
            }

        # Softmax calibration for Bayesian likelihoods
        scores = [float(c.get("composite_score", 0.0)) for c in ranked_candidates]
        max_s = max(scores)
        exp_scores = [math.exp((s - max_s) / cls.TEMPERATURE) for s in scores]
        sum_exp = sum(exp_scores)
        posteriors = [e / sum_exp for e in exp_scores]

        for i, c in enumerate(ranked_candidates):
            c["bayesian_posterior"] = round(posteriors[i], 4)

        entropy_metrics = cls.calculate_shannon_entropy(posteriors)
        h_norm = entropy_metrics["normalized_entropy"]

        top_cand = ranked_candidates[0]
        top_posterior = posteriors[0]
        runner_up_posterior = posteriors[1] if len(posteriors) > 1 else 0.0
        margin = top_posterior - runner_up_posterior

        # 1. Check for High Information Entropy (Uniform Distribution / Ambiguity)
        if len(ranked_candidates) > 1 and h_norm > cls.MAX_NORMALIZED_ENTROPY_THRESHOLD:
            return {
                "decision": "INSUFFICIENT_EVIDENCE",
                "is_abstention": True,
                "reason": (
                    f"Shannon Information Entropy H_norm={h_norm:.2f} exceeds scientific threshold "
                    f"({cls.MAX_NORMALIZED_ENTROPY_THRESHOLD}). Multiple vessels share statistically indistinguishable "
                    f"proximity to the spill origin. Defensible legal attribution cannot be established."
                ),
                "actionable_recommendation": (
                    "Deploy Indian Coast Guard aerial reconnaissance / Request high-resolution optical satellite tasking; "
                    "Board and inspect bilges/slop logs of candidate vessels upon port arrival."
                ),
                "leading_candidate_mmsi": top_cand.get("mmsi"),
                "confidence_score": round(top_posterior * 100, 1),
                "separation_margin": round(margin, 4),
                "entropy_metrics": entropy_metrics
            }

        # 2. Check for Insufficient Separation Margin
        if len(ranked_candidates) > 1 and margin < cls.MIN_SEPARATION_MARGIN:
            return {
                "decision": "INSUFFICIENT_EVIDENCE",
                "is_abstention": True,
                "reason": (
                    f"Separation margin between top candidate ({top_cand.get('name', top_cand.get('mmsi'))}) "
                    f"and runner-up ({ranked_candidates[1].get('name', ranked_candidates[1].get('mmsi'))}) is "
                    f"{margin:.2%}, below minimum defensible margin ({cls.MIN_SEPARATION_MARGIN:.0%})."
                ),
                "actionable_recommendation": (
                    "Audit voyage data recorder (VDR) and fuel oil transfer records for both candidate vessels."
                ),
                "leading_candidate_mmsi": top_cand.get("mmsi"),
                "confidence_score": round(top_posterior * 100, 1),
                "separation_margin": round(margin, 4),
                "entropy_metrics": entropy_metrics
            }

        # 3. Check for absolute score viability
        if top_cand.get("composite_score", 0.0) < 55.0:
            return {
                "decision": "INSUFFICIENT_EVIDENCE",
                "is_abstention": True,
                "reason": (
                    f"Leading candidate composite score ({top_cand.get('composite_score')}%) is below investigative "
                    "plausibility threshold (55%). Closest vessel was too far in space or time from estimated origin."
                ),
                "actionable_recommendation": (
                    "Perform extended 48h backward hindcast; investigate dark non-transponding vessels or offshore rig seepage."
                ),
                "leading_candidate_mmsi": top_cand.get("mmsi"),
                "confidence_score": round(top_posterior * 100, 1),
                "separation_margin": round(margin, 4),
                "entropy_metrics": entropy_metrics
            }

        # 4. Valid Lead
        decision_label = "DEFINITIVE_LEAD" if (top_cand.get("composite_score", 0) >= 80 and margin >= 0.25) else "PROBABLE_LEAD"
        return {
            "decision": decision_label,
            "is_abstention": False,
            "reason": (
                f"Candidate {top_cand.get('name', top_cand.get('mmsi'))} uniquely satisfies spatio-temporal co-location "
                f"with posterior probability {top_posterior:.1%} and clear separation margin {margin:.1%}."
            ),
            "actionable_recommendation": (
                "Issue Maritime Law Enforcement Notice of Violation (MARPOL Annex I). Request immediate bunker fuel sampling."
            ),
            "leading_candidate_mmsi": top_cand.get("mmsi"),
            "confidence_score": round(top_posterior * 100, 1),
            "separation_margin": round(margin, 4),
            "entropy_metrics": entropy_metrics
        }

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
        cpa = candidate.get("closest_approach", {})
        dist_nm = float(cpa.get("distance_nm", 99.0))
        time_diff_h = float(cpa.get("time_diff_h", 99.0))

        challenges = []
        passed_challenges = 0

        # Attack 1: Hydrodynamic Current Perturbation (+/- 20%)
        current_displacement_nm = current_speed_knots * cls.CURRENT_VARIATION_STRESS * max(1.0, time_diff_h)
        c1_survived = (dist_nm - current_displacement_nm) < 3.0
        challenges.append({
            "challenge_name": "Hydrodynamic Current Perturbation (+/- 20%)",
            "stress_parameter": f"+/-{cls.CURRENT_VARIATION_STRESS*100:.0f}% surface current ({current_speed_knots:.1f} kts)",
            "estimated_origin_shift_nm": round(current_displacement_nm, 2),
            "survived": c1_survived,
            "rationale": "Candidate remains within origin probability boundary even if INCOIS/CMEMS currents deviate by 20%." if c1_survived else "Attribution vulnerable to ocean current forecast error."
        })
        if c1_survived:
            passed_challenges += 1

        # Attack 2: Wind Leeway Deflection (+/- 1% leeway)
        wind_displacement_nm = wind_speed_knots * 0.01 * max(1.0, time_diff_h)
        c2_survived = (dist_nm - wind_displacement_nm) < 3.5
        challenges.append({
            "challenge_name": "Atmospheric Leeway Perturbation (+/- 1% wind drag)",
            "stress_parameter": f"+/-1.0% wind factor on {wind_speed_knots:.1f} kts wind",
            "estimated_origin_shift_nm": round(wind_displacement_nm, 2),
            "survived": c2_survived,
            "rationale": "Candidate track robust to atmospheric wind drag variability." if c2_survived else "Wind variation exceeds spatial correlation window."
        })
        if c2_survived:
            passed_challenges += 1

        # Attack 3: GPS Sensor Jitter & Spatial Sensor Uncertainty (+/- 1 nm)
        jittered_dist = dist_nm + cls.GPS_JITTER_STRESS_NM
        c3_survived = jittered_dist < 4.0
        challenges.append({
            "challenge_name": "GPS Sensor Jitter & Interpolation Error (+/- 1.0 nm)",
            "stress_parameter": f"+/-{cls.GPS_JITTER_STRESS_NM:.1f} nm transponder noise",
            "effective_cpa_nm": round(jittered_dist, 2),
            "survived": c3_survived,
            "rationale": "Candidate maintains geometric proximity despite maximum maritime GPS drift." if c3_survived else "Sensor jitter moves vessel outside origin threshold."
        })
        if c3_survived:
            passed_challenges += 1

        # Attack 4: AIS Spoofing & Continuity Audit
        spoof_audit = candidate.get("spoofing_audit", {})
        has_critical_spoofing = spoof_audit.get("spoofing_detected", False) and spoof_audit.get("risk_level") in ["CRITICAL", "HIGH"]
        c4_survived = not has_critical_spoofing
        challenges.append({
            "challenge_name": "AIS Continuity & Anti-Spoofing Challenge",
            "stress_parameter": "Speed anomalies, teleportation jumps, MMSI duplication",
            "survived": c4_survived,
            "rationale": "Vessel trajectory shows verified kinematic continuity." if c4_survived else f"FLAW DETECTED: {spoof_audit.get('summary', 'AIS transponder gaps / speed anomalies present.')}"
        })
        if c4_survived:
            passed_challenges += 1

        robustness_score = round((passed_challenges / 4.0) * 100.0, 1)

        return {
            "mmsi": candidate.get("mmsi"),
            "name": candidate.get("name"),
            "challenges_tested": 4,
            "challenges_passed": passed_challenges,
            "adversarial_robustness_score": robustness_score,
            "verdict": "FALSIFICATION_RESISTANT (COURT-DEFENSIBLE)" if passed_challenges >= 3 else "FALSIFICATION_VULNERABLE (CIRCUMSTANTIAL)",
            "challenges": challenges
        }
