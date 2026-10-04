"""
AIS Engine: Automatic Identification System Maritime Traffic Correlation
Ingests historical vessel transponder trajectories, filters irrelevant maritime traffic,
reconstructs spatio-temporal vessel positions around the spill origin window (x0, y0, t0),
and produces analyst-review traffic leads. It does not determine responsibility.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class AISEngine:
    """
    Spatio-Temporal AIS Trajectory Ingestion, Spatial Indexing,
    Kinematic Screening, and Analyst-Review Lead Ranking.
    """

    # Baseline capacity priors: modest 10% weight so attribution is evidence-driven (CPA, time, kinematics)
    # rather than predetermined by ship type.
    VESSEL_TYPE_WEIGHTS = {
        "Crude / Product Tanker": 65.0,
        "Hazardous Category A Tanker": 70.0,
        "Container / Bulk Cargo Carrier": 55.0,
        "Tug / Towing / Offshore Support": 45.0,
        "Commercial Fishing Vessel": 40.0,
        "Passenger Ferry": 35.0,
        "Crude Oil Tanker": 65.0,
        "Product Tanker": 65.0,
        "General Cargo": 55.0,
        "Bulk Carrier": 55.0,
        "Container Ship": 55.0,
        "Fishing Vessel": 40.0,
        "Other / Unclassified Vessel": 40.0,
        "Other / Unknown": 40.0
    }

    def __init__(
        self,
        cpa_distance_sigma_nm: float = 1.8,   # Closest Point of Approach scale in Nautical Miles
        temporal_sigma_hours: float = 1.2,     # Time difference decay parameter
        weight_proximity: float = 0.40,        # 40% spatial CPA
        weight_temporal: float = 0.30,         # 30% temporal coincidence
        weight_speed_anomaly: float = 0.15,    # 15% speed drop / discharge profile
        weight_course_anomaly: float = 0.05,   # 5% heading variance
        weight_vessel_type: float = 0.10       # 10% baseline capacity prior
    ):
        self.cpa_sigma_nm = cpa_distance_sigma_nm
        self.temporal_sigma_hours = temporal_sigma_hours
        self.w_prox = weight_proximity
        self.w_time = weight_temporal
        self.w_speed = weight_speed_anomaly
        self.w_course = weight_course_anomaly
        self.w_type = weight_vessel_type

    @staticmethod
    def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Computes great circle distance between two points in kilometers."""
        try:
            phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
            dphi = math.radians(float(lat2) - float(lat1))
            dlambda = math.radians(float(lon2) - float(lon1))
        except (ValueError, TypeError):
            return 99999.0

        r = 6371.0  # Earth mean radius in km
        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        a = min(1.0, max(0.0, a))
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return r * c

    def haversine_distance_nm(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Computes distance in Nautical Miles (1 NM = 1.852 km)."""
        return self.haversine_distance_km(lat1, lon1, lat2, lon2) / 1.852

    def filter_vessel_traffic(
        self,
        vessels: List[Dict[str, Any]],
        origin_lat: float,
        origin_lon: float,
        origin_time_relative_h: float,
        spatial_radius_nm: float = 25.0,
        temporal_window_h: float = 5.0,
        hindcast_trajectory: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Reconstructs and filters maritime traffic around the origin window in space and time.
        If hindcast_trajectory is provided, cross-correlates vessel positions against the
        entire drifting plume track to find exact space-time coincidence.
        """
        filtered_candidates = []

        for v in vessels:
            track = sorted(v.get("trajectory", []), key=lambda point: point.get("relative_time_hours", 0.0))
            if not track:
                continue

            min_dist_nm = float("inf")
            best_point = None

            # 1. Primary correlation against the estimated release origin (x0, y0)
            for pt in track:
                d_nm = self.haversine_distance_nm(pt["lat"], pt["lon"], origin_lat, origin_lon)
                if d_nm < min_dist_nm:
                    min_dist_nm = d_nm
                    best_point = pt

            # 2. Match the trajectory in both space and time.  A spatially close
            # ping at the wrong time is not a corridor match.
            if hindcast_trajectory:
                for pt in track:
                    pt_t = pt.get("relative_time_hours", 0.0)
                    for step in hindcast_trajectory:
                        step_t = step.get("relative_time_hours", 0.0)
                        if abs(pt_t - step_t) <= 1.0:
                            d_nm = self.haversine_distance_nm(
                                pt["lat"], pt["lon"],
                                step["centroid"]["lat"], step["centroid"]["lon"]
                            )
                            if d_nm < min_dist_nm:
                                min_dist_nm = d_nm
                                best_point = pt

            # Check if vessel traversed within corridor around the release event
            if min_dist_nm <= spatial_radius_nm and best_point is not None:
                pt_rel_t = best_point.get("relative_time_hours", 0.0)
                time_diff_h = abs(pt_rel_t - origin_time_relative_h)

                if time_diff_h <= temporal_window_h:
                    v_copy = dict(v)
                    v_copy["closest_approach"] = {
                        "distance_nm": round(min_dist_nm, 2),
                        "distance_km": round(min_dist_nm * 1.852, 2),
                        "time_relative_h": round(pt_rel_t, 2),
                        "time_diff_h": round(time_diff_h, 2),
                        "point": best_point
                    }
                    filtered_candidates.append(v_copy)

        return filtered_candidates

    def analyze_vessel_kinematics(
        self,
        vessel: Dict[str, Any],
        origin_time_relative_h: float
    ) -> Dict[str, Any]:
        """
        Computes navigation context only. Speed changes, course changes, and AIS
        reporting gaps have many benign causes and are not evidence of discharge.
        """
        track = sorted(vessel.get("trajectory", []), key=lambda point: point.get("relative_time_hours", 0.0))
        if len(track) < 3:
            return {
                "speed_drop_knots": 0.0,
                "min_speed_near_origin": 0.0,
                "cruise_speed": 14.0,
                "speed_anomaly_score": 10.0,
                "course_variance_deg": 0.0,
                "course_anomaly_score": 10.0,
                "has_ais_gap": False,
                "behavior_summary": "Insufficient track points for kinematic analysis"
            }

        speeds = [pt.get("sog_knots", 14.0) for pt in track]
        courses = [pt.get("cog_degrees", 0.0) for pt in track]
        times = [pt.get("relative_time_hours", 0.0) for pt in track]

        # Cruise speed baseline (median of speeds outside origin window)
        far_speeds = [
            s for s, t in zip(speeds, times)
            if abs(t - origin_time_relative_h) > 2.0
        ]
        cruise_speed = float(np.median(far_speeds)) if far_speeds else float(np.median(speeds))

        # Speed near origin release window (|t - t0| <= 1.5h)
        near_speeds = [
            s for s, t in zip(speeds, times)
            if abs(t - origin_time_relative_h) <= 1.5
        ]
        min_near_speed = float(np.min(near_speeds)) if near_speeds else cruise_speed
        speed_drop = max(0.0, cruise_speed - min_near_speed)

        # Navigation-context score; never describe it as a discharge profile.
        if speed_drop >= 7.0 and 3.5 <= min_near_speed <= 8.5:
            speed_score = 70.0
            speed_comment = f"Large speed change ({cruise_speed:.1f} -> {min_near_speed:.1f} kts); requires navigation-context review"
        elif speed_drop >= 4.0:
            speed_score = 75.0
            speed_comment = f"Moderate deceleration ({speed_drop:.1f} kts drop) detected near origin window"
        elif speed_drop >= 2.0:
            speed_score = 45.0
            speed_comment = "Minor speed fluctuation observed"
        else:
            speed_score = 15.0
            speed_comment = "Steady cruising speed maintained"

        # Course anomaly (standard deviation of heading change)
        course_diffs = []
        for i in range(1, len(courses)):
            diff = abs(courses[i] - courses[i - 1])
            if diff > 180:
                diff = 360 - diff
            course_diffs.append(diff)

        course_std = float(np.std(course_diffs)) if course_diffs else 0.0
        if course_std > 25.0:
            course_score = 88.0
            course_comment = "Erratic course zigzag maneuvers detected"
        elif course_std > 12.0:
            course_score = 55.0
            course_comment = "Noticeable course heading adjustments"
        else:
            course_score = 15.0
            course_comment = "Straight-line transit following navigational channel"

        # Report coverage gaps without treating them as transponder-disable evidence.
        has_ais_gap = False
        max_time_gap_min = 0.0
        for i in range(1, len(times)):
            gap_min = abs(times[i] - times[i - 1]) * 60.0
            if gap_min > max_time_gap_min:
                max_time_gap_min = gap_min
            if gap_min > 45.0:
                has_ais_gap = True

        return {
            "speed_drop_knots": round(speed_drop, 1),
            "min_speed_near_origin": round(min_near_speed, 1),
            "cruise_speed": round(cruise_speed, 1),
            "speed_anomaly_score": round(speed_score, 1),
            "course_variance_deg": round(course_std, 1),
            "course_anomaly_score": round(course_score, 1),
            "has_ais_gap": has_ais_gap,
            "max_ais_gap_minutes": round(max_time_gap_min, 1),
            "speed_comment": speed_comment,
            "course_comment": course_comment
        }

    @staticmethod
    def detect_ais_spoofing_and_gaps(
        trajectory: List[Dict[str, Any]],
        mmsi: Optional[int] = None,
        origin_time_relative_h: float = 0.0,
        cpa_time_relative_h: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Forensic AIS Integrity & Spoofing Detector:
        1. Transponder Blackout Gaps: flags blackout > 30 min in proximity to origin.
        2. Kinematic Speed Jumps: flags physically impossible acceleration > 3.0 kts/min.
        3. Draught Drop / De-ballasting Signature: checks for sudden draught reduction.
        4. MMSI / MID Validity: verifies standard 9-digit ITU MID (201-775).
        """
        track = sorted(trajectory, key=lambda p: p.get("relative_time_hours", 0.0))
        anomalies = []
        is_suspicious = False
        max_gap_min = 0.0
        max_accel_kts_min = 0.0

        # 1. MMSI Validity check
        mmsi_valid = True
        if mmsi is not None:
            mmsi_str = str(mmsi)
            if len(mmsi_str) != 9:
                anomalies.append("INVALID_MMSI_LENGTH: Transponder ID does not match 9-digit ITU standard")
                mmsi_valid = False
            else:
                try:
                    mid = int(mmsi_str[:3])
                    if not (201 <= mid <= 775):
                        anomalies.append(f"UNALLOCATED_MID: Maritime Identification Digit {mid} is outside allocated ITU range")
                        mmsi_valid = False
                except ValueError:
                    mmsi_valid = False

        # 2. Transponder Gap & Acceleration checks
        cpa_t = cpa_time_relative_h if cpa_time_relative_h is not None else origin_time_relative_h
        corridor_gap = False

        for i in range(1, len(track)):
            p_prev = track[i-1]
            p_curr = track[i]
            t_prev_raw = p_prev.get("relative_time_hours")
            t_curr_raw = p_curr.get("relative_time_hours")
            try:
                t_prev = float(t_prev_raw) if t_prev_raw is not None else 0.0
            except (ValueError, TypeError):
                t_prev = 0.0

            try:
                t_curr = float(t_curr_raw) if t_curr_raw is not None else 0.0
            except (ValueError, TypeError):
                t_curr = 0.0

            dt_hours = abs(t_curr - t_prev)
            dt_min = dt_hours * 60.0

            if dt_min > max_gap_min:
                max_gap_min = dt_min

            # Check if gap occurs near CPA / origin window (within 2.5 hours)
            if dt_min > 30.0 and abs(t_prev - cpa_t) <= 2.5:
                corridor_gap = True
                anomalies.append(f"CORRIDOR_TRANSPONDER_BLACKOUT: {dt_min:.1f} min gap during closest approach window")

            # Speed jump check (acceleration > 3.0 kts / min)
            sog_prev_raw = p_prev.get("sog_knots")
            sog_curr_raw = p_curr.get("sog_knots")
            try:
                sog_prev = float(sog_prev_raw) if sog_prev_raw is not None else 0.0
            except (ValueError, TypeError):
                sog_prev = 0.0

            try:
                sog_curr = float(sog_curr_raw) if sog_curr_raw is not None else 0.0
            except (ValueError, TypeError):
                sog_curr = 0.0

            if dt_min > 0.05:
                accel = abs(sog_curr - sog_prev) / dt_min
                if accel > max_accel_kts_min:
                    max_accel_kts_min = accel
                if accel > 3.0:
                    anomalies.append(f"KINEMATIC_SPEED_JUMP: Impossible acceleration of {accel:.1f} kts/min (GPS spoofing / replay artifact)")

        # 3. Draught change check (if draught field provided)
        draughts = []
        for p in track:
            d_raw = p.get("draught_m")
            if d_raw is not None:
                try:
                    d_val = float(d_raw)
                    if not math.isnan(d_val) and d_val > 0:
                        draughts.append(d_val)
                except (ValueError, TypeError):
                    continue

        if len(draughts) >= 2 and draughts[0] > 0:
            draught_drop = draughts[0] - draughts[-1]
            if draught_drop >= 0.4:
                anomalies.append(f"DRAUGHT_REDUCTION_DETECTED: {draught_drop:.2f}m draught reduction across corridor (cargo / ballast discharge indicator)")

        if anomalies:
            is_suspicious = True

        return {
            "has_anomalies": is_suspicious,
            "anomalies_detected": anomalies,
            "anomaly_count": len(anomalies),
            "max_gap_minutes": round(max_gap_min, 1),
            "max_acceleration_kts_min": round(max_accel_kts_min, 2),
            "corridor_blackout": corridor_gap,
            "mmsi_valid": mmsi_valid,
            "integrity_rating": "COMPROMISED / ANOMALOUS" if is_suspicious else "VERIFIED_CONTINUOUS"
        }

    @staticmethod
    def compute_topsis_rankings(
        candidate_vessels: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
        Multi-Criteria Decision Analysis (MCDA) + Borda Count Rank Aggregation.

        Evaluates 5 orthogonal attribution criteria:
          C1 [Cost]: CPA Distance (NM) — lower is closer to discharge origin
          C2 [Cost]: Delta-T Time Coincidence (h) — lower is better time alignment
          C3 [Benefit]: Speed Deceleration Near Origin (knots) — higher drop indicates discharge
          C4 [Benefit]: Ship Hazard Prior Score — higher tanker/hazardous capacity
          C5 [Benefit]: Course Zigzag Variance (deg) — higher maneuvering
        """
        if not candidate_vessels:
            return []
        if len(candidate_vessels) == 1:
            v = candidate_vessels[0]
            v["topsis_closeness_score"] = 100.0
            v["topsis_rank"] = 1
            v["borda_points"] = 1
            return [v]

        n = len(candidate_vessels)
        weights = np.array([0.35, 0.25, 0.20, 0.10, 0.10], dtype=np.float64)
        is_benefit = np.array([False, False, True, True, True], dtype=bool)

        X = np.zeros((n, 5), dtype=np.float64)
        for i, v in enumerate(candidate_vessels):
            cpa = v.get("closest_approach") or {}
            kin = v.get("kinematics") or {}

            d_raw = cpa.get("distance_nm")
            try:
                d_val = float(d_raw) if d_raw is not None and not math.isnan(float(d_raw)) else 15.0
            except (ValueError, TypeError):
                d_val = 15.0

            t_raw = cpa.get("time_diff_h")
            try:
                t_val = float(t_raw) if t_raw is not None and not math.isnan(float(t_raw)) else 5.0
            except (ValueError, TypeError):
                t_val = 5.0

            s_drop_raw = kin.get("speed_drop_knots")
            try:
                s_drop_val = float(s_drop_raw) if s_drop_raw is not None and not math.isnan(float(s_drop_raw)) else 0.0
            except (ValueError, TypeError):
                s_drop_val = 0.0

            c_var_raw = kin.get("course_variance_deg")
            try:
                c_var_val = float(c_var_raw) if c_var_raw is not None and not math.isnan(float(c_var_raw)) else 0.0
            except (ValueError, TypeError):
                c_var_val = 0.0

            X[i, 0] = max(0.05, d_val)
            X[i, 1] = max(0.05, t_val)
            X[i, 2] = max(0.0, s_drop_val)
            X[i, 3] = float(AISEngine.VESSEL_TYPE_WEIGHTS.get(v.get("vessel_type"), 40.0))
            X[i, 4] = max(0.0, c_var_val)

        # 1. Vector normalization
        norms = np.sqrt(np.sum(X ** 2, axis=0))
        norms[norms == 0] = 1.0
        R = X / norms

        # 2. Weighted normalized matrix
        V = R * weights

        # 3. Positive-Ideal (A+) and Negative-Ideal (A-)
        A_plus = np.zeros(5)
        A_minus = np.zeros(5)
        for j in range(5):
            if is_benefit[j]:
                A_plus[j] = np.max(V[:, j])
                A_minus[j] = np.min(V[:, j])
            else:
                A_plus[j] = np.min(V[:, j])
                A_minus[j] = np.max(V[:, j])

        # 4. Euclidean separation measures
        S_plus = np.sqrt(np.sum((V - A_plus) ** 2, axis=1))
        S_minus = np.sqrt(np.sum((V - A_minus) ** 2, axis=1))

        # 5. Relative Closeness to Ideal Solution C_i in [0, 1]
        denom = S_plus + S_minus
        denom[denom == 0] = 1e-6
        C = S_minus / denom

        topsis_scores = [round(float(c * 100.0), 1) for c in C]
        rank_indices = np.argsort(topsis_scores)[::-1]

        for rank, idx in enumerate(rank_indices, 1):
            candidate_vessels[idx]["topsis_closeness_score"] = topsis_scores[idx]
            candidate_vessels[idx]["topsis_rank"] = rank
            candidate_vessels[idx]["borda_points"] = n - rank + 1

        return candidate_vessels

    def score_and_rank_suspects(
        self,
        candidate_vessels: List[Dict[str, Any]],
        origin_lat: float,
        origin_lon: float,
        origin_time_relative_h: float
    ) -> List[Dict[str, Any]]:
        """
        Computes an uncalibrated lead-priority score. Only spatio-temporal
        co-location affects ranking; vessel class and navigation context are shown
        for review but never treated as responsibility evidence.
        """
        ranked_vessels = []

        for v in candidate_vessels:
            cpa = v.get("closest_approach") or {}
            raw_dist = cpa.get("distance_nm", 999.0)
            raw_time = cpa.get("time_diff_h", 999.0)
            try:
                dist_nm = float(raw_dist) if not math.isnan(float(raw_dist)) and float(raw_dist) >= 0 else 999.0
            except (ValueError, TypeError):
                dist_nm = 999.0

            try:
                time_diff_h = float(raw_time) if not math.isnan(float(raw_time)) else 999.0
            except (ValueError, TypeError):
                time_diff_h = 999.0

            # 1. Proximity score (Gaussian spatial decay)
            s_prox = 100.0 * math.exp(-(dist_nm ** 2) / (2.0 * (self.cpa_sigma_nm ** 2)))

            # 2. Temporal coincidence score (Gaussian temporal decay)
            s_time = 100.0 * math.exp(-(time_diff_h ** 2) / (2.0 * (self.temporal_sigma_hours ** 2)))

            # Navigation context is descriptive, not an attribution factor.
            kinematics = self.analyze_vessel_kinematics(v, origin_time_relative_h)
            s_speed = kinematics["speed_anomaly_score"]
            s_course = kinematics["course_anomaly_score"]
            v_type = v.get("vessel_type", "Other / Unknown")
            s_type = None
            composite_score = round(0.6 * s_prox + 0.4 * s_time, 1)

            # Lead-priority tier. Scores are heuristic ranking signals, not a
            # calibrated probability or a finding of responsibility.
            if composite_score >= 80.0:
                attribution_tier = "HIGH-PRIORITY LEAD (REVIEW REQUIRED)"
                flag_color = "#ff3366"  # Red
            elif composite_score >= 60.0:
                attribution_tier = "MEDIUM-PRIORITY LEAD (REVIEW REQUIRED)"
                flag_color = "#ffaa00"  # Amber
            else:
                attribution_tier = "LOW-PRIORITY CORRIDOR TRAFFIC"
                flag_color = "#05d6a0"  # Green

            pt = cpa.get("point") or {}
            candidate_release_point = {
                "lat": pt.get("lat", origin_lat),
                "lon": pt.get("lon", origin_lon),
                "relative_time_hours": pt.get("relative_time_hours", origin_time_relative_h)
            }

            # AIS Spoofing & Integrity Audit
            spoofing_audit = self.detect_ais_spoofing_and_gaps(
                v.get("trajectory", []),
                v.get("mmsi"),
                origin_time_relative_h,
                cpa.get("time_diff_h")
            )

            v_result = {
                "mmsi": v.get("mmsi"),
                "imo": v.get("imo"),
                "vessel_name": v.get("vessel_name", "UNKNOWN VESSEL"),
                "call_sign": v.get("call_sign", "N/A"),
                "flag_state": v.get("flag_state", "Unknown"),
                "vessel_type": v_type,
                "length_m": v.get("length_m"),
                "width_m": v.get("width_m"),
                "dwt_tonnes": v.get("dwt_tonnes"),
                "lead_priority_score": composite_score,
                "composite_suspect_score": composite_score,
                "mfa_attribution_index": composite_score,
                "total_score": composite_score,
                "attribution_tier": attribution_tier,
                "flag_color": flag_color,
                "candidate_release_point": candidate_release_point,
                "score_breakdown": {
                    "proximity_score": round(s_prox, 1),
                    "temporal_score": round(s_time, 1),
                    "speed_anomaly_score": round(s_speed, 1),
                    "course_anomaly_score": round(s_course, 1),
                    "vessel_type_score": None
                },
                "closest_approach": cpa,
                "kinematics": kinematics,
                "spoofing_audit": spoofing_audit,
                "full_trajectory": v.get("trajectory", []),
                "data_origin": v.get("data_origin", "Scenario Physics Simulation (Synthetic Trajectory)"),
                "is_real_ais": v.get("is_real_ais", False)
            }
            ranked_vessels.append(v_result)

        # Multi-Criteria Decision Analysis (TOPSIS + Borda Count)
        ranked_vessels = self.compute_topsis_rankings(ranked_vessels)

        # Sort by composite suspect score descending
        ranked_vessels.sort(key=lambda x: x["lead_priority_score"], reverse=True)

        # Bayesian Posterior Calibration, Shannon Entropy & Adversarial Falsification Stress Tests
        try:
            from .falsification import FalsificationAndAbstentionEngine
            for v in ranked_vessels:
                v["composite_score"] = v.get("lead_priority_score", 0.0)
            
            abstention_verdict = FalsificationAndAbstentionEngine.evaluate_decision_theoretic_abstention(ranked_vessels)
            for v in ranked_vessels:
                v["abstention_verdict"] = abstention_verdict
                # Run adversarial stress testing on candidate leads
                v["adversarial_stress_test"] = FalsificationAndAbstentionEngine.run_adversarial_stress_test(
                    v, origin_lat, origin_lon, current_speed_knots=1.5, wind_speed_knots=12.0
                )
        except Exception as e:
            pass

        return ranked_vessels

    def correlate_radar_targets_with_ais(
        self,
        radar_targets: List[Dict[str, Any]],
        ais_vessels: List[Dict[str, Any]],
        origin_lat: float,
        origin_lon: float,
        match_threshold_nm: float = 3.0,
        max_time_difference_h: float = 1.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Cross-correlates SAR radar metallic ship targets against AIS broadcast transponder pings.
        Produces radar/AIS correlation review cues. A no-match is never evidence
        that a vessel disabled its transponder; AIS coverage and time alignment
        must be verified independently.
        """
        dark_vessels = []
        matched_targets = []

        for tgt in radar_targets:
            tgt_copy = dict(tgt)
            try:
                target_time = float(tgt_copy.get("relative_time_hours", 0.0))
            except (ValueError, TypeError):
                target_time = 0.0

            try:
                tgt_lat = float(tgt_copy.get("lat", 0.0))
                tgt_lon = float(tgt_copy.get("lon", 0.0))
            except (ValueError, TypeError):
                continue

            min_dist_nm = float("inf")
            matched_vessel = None
            time_aligned_positions = []
            for vessel in ais_vessels:
                for point in vessel.get("trajectory", []):
                    try:
                        time_delta = abs(float(point.get("relative_time_hours", 0.0)) - target_time)
                        p_lat = float(point["lat"])
                        p_lon = float(point["lon"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if time_delta <= max_time_difference_h:
                        time_aligned_positions.append({
                            "mmsi": vessel.get("mmsi"),
                            "name": vessel.get("vessel_name", "UNKNOWN"),
                            "lat": p_lat,
                            "lon": p_lon,
                            "time_difference_h": time_delta,
                        })

            for ap in time_aligned_positions:
                d_nm = self.haversine_distance_nm(tgt_lat, tgt_lon, ap["lat"], ap["lon"])
                if d_nm < min_dist_nm:
                    min_dist_nm = d_nm
                    matched_vessel = ap

            if min_dist_nm <= match_threshold_nm and matched_vessel is not None:
                tgt_copy["has_matched_ais"] = True
                tgt_copy["matched_mmsi"] = matched_vessel["mmsi"]
                tgt_copy["matched_name"] = matched_vessel["name"]
                tgt_copy["status"] = "COOPERATIVE_AIS_VESSEL"
                matched_targets.append(tgt_copy)
            else:
                dist_to_origin_nm = self.haversine_distance_nm(tgt_lat, tgt_lon, origin_lat, origin_lon)
                tgt_copy["has_matched_ais"] = False
                tgt_copy["matched_mmsi"] = "NONE (NO TIME-ALIGNED AIS POSITION)"
                tgt_copy["distance_to_spill_origin_nm"] = round(dist_to_origin_nm, 2)
                if time_aligned_positions:
                    tgt_copy["status"] = "RADAR_AIS_SPATIAL_MISMATCH_REVIEW"
                    tgt_copy["threat_classification"] = "RADAR/AIS MISMATCH — VERIFY COVERAGE" 
                else:
                    tgt_copy["status"] = "AIS_TEMPORAL_COVERAGE_GAP"
                    tgt_copy["threat_classification"] = "AIS TIME COVERAGE INSUFFICIENT"
                dark_vessels.append(tgt_copy)

        return matched_targets, dark_vessels

    def attribute_oil_spill(
        self,
        raw_vessels: List[Dict[str, Any]],
        origin_lat: float,
        origin_lon: float,
        origin_time_relative_h: float,
        spatial_radius_nm: float = 30.0,
        temporal_window_h: float = 5.0,
        hindcast_trajectory: Optional[List[Dict[str, Any]]] = None,
        radar_targets: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Complete end-to-end attribution workflow:
        1. Filters out irrelevant traffic (using full trajectory matching if provided)
        2. Reconstructs space-time corridor
        3. Analyzes kinematic anomalies
        4. Identifies and ranks investigative leads
        5. Cross-correlates radar targets with time-aligned AIS positions
        """
        filtered = self.filter_vessel_traffic(
            raw_vessels, origin_lat, origin_lon, origin_time_relative_h,
            spatial_radius_nm, temporal_window_h,
            hindcast_trajectory=hindcast_trajectory
        )

        ranked = self.score_and_rank_suspects(
            filtered, origin_lat, origin_lon, origin_time_relative_h
        )

        primary_suspect = ranked[0] if ranked else None

        # Dark ship detection
        dark_vessels = []
        matched_radar = []
        if radar_targets:
            matched_radar, dark_vessels = self.correlate_radar_targets_with_ais(
                radar_targets, raw_vessels, origin_lat, origin_lon
            )

        return {
            "total_vessels_in_region": len(raw_vessels),
            "vessels_evaluated_in_corridor": len(ranked),
            "primary_review_lead": primary_suspect,
            "primary_culprit": None,
            "ranked_suspects": ranked,
            "dark_vessels_detected": dark_vessels,
            "dark_vessels_count": len(dark_vessels),
            "bayesian_legal_gate": primary_suspect.get("abstention_verdict") if primary_suspect else None,
            "adversarial_stress_test": primary_suspect.get("adversarial_stress_test") if primary_suspect else None,
            "origin_query": {
                "origin_lat": origin_lat,
                "origin_lon": origin_lon,
                "origin_time_relative_h": origin_time_relative_h
            },
            "screening_notice": "Lead-priority ranking is not a probability, allegation, or finding of responsibility."
        }


def categorize_marpol_violation(
    vessel_type: str,
    speed_knots: float,
    distance_to_coast_nm: float = 28.0,
    estimated_volume_m3: float = 2.5,
    instantaneous_discharge_l_per_nm: Optional[float] = None,
    is_special_area: bool = False
) -> Dict[str, Any]:
    """Return a legal-review checklist, never an automated violation finding.

    A SAR-derived area cannot establish discharge volume, 15 ppm concentration,
    equipment bypass, vessel status, or jurisdiction.  The competent authority
    must apply the current MARPOL text and domestic law to original records.
    """
    return {
        "is_marpol_violation": None,
        "status": "LEGAL_REVIEW_REQUIRED",
        "violation_count": None,
        "overall_severity": "NOT_DETERMINED",
        "primary_breach": None,
        "regulations_breached": [],
        "violation_details": [],
        "review_inputs_required": [
            "verified vessel identity, flag, voyage, ship type, and jurisdiction",
            "certified Oil Record Book, VDR, discharge-monitoring and equipment records",
            "validated spill sample/source fingerprinting and contemporaneous observations",
            "competent-authority interpretation of the current MARPOL Annex I and Indian law",
        ],
        "screening_context": {
            "reported_vessel_type": vessel_type,
            "reported_speed_knots": speed_knots,
            "reported_distance_to_coast_nm": distance_to_coast_nm,
            "unverified_estimated_volume_m3": estimated_volume_m3,
            "unverified_rate_l_per_nm": instantaneous_discharge_l_per_nm,
            "reported_special_area": is_special_area,
        },
        "statutory_nexus": "Analyst aid only; no legal conclusion or enforcement instruction."
    }
