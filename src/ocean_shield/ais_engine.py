"""
AIS Engine: Automatic Identification System Maritime Traffic Correlation
Ingests historical vessel transponder trajectories, filters irrelevant maritime traffic,
reconstructs spatio-temporal vessel positions around the spill origin window (x0, y0, t0),
and produces analyst-review traffic leads. It does not determine responsibility.
"""

import logging
import math
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from .falsification import FalsificationAndAbstentionEngine

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Unavailable numeric evidence is null, never a nonstandard JSON NaN."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


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

    @staticmethod
    def _valid_position_track(trajectory: Any) -> List[Dict[str, Any]]:
        """Use only finite actual fixes for proximity; the audit retains rejections."""
        number = FalsificationAndAbstentionEngine.finite_number
        track = []
        for point in trajectory if isinstance(trajectory, (list, tuple)) else []:
            if not isinstance(point, dict):
                continue
            time = number(point.get("relative_time_hours"))
            lat, lon = number(point.get("lat"), -90, 90), number(point.get("lon"), -180, 180)
            if time is None or lat is None or lon is None:
                continue
            track.append({**point, "relative_time_hours": time, "lat": lat, "lon": lon,
                          "sog_knots": number(point.get("sog_knots"), 0, 102.2),
                          "cog_degrees": number(point.get("cog_degrees"), 0, 360)})
        return sorted(track, key=lambda point: point["relative_time_hours"])

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
            if not isinstance(v, dict):
                continue
            track = self._valid_position_track(v.get("trajectory"))
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
                        if not isinstance(step, dict) or not isinstance(step.get("centroid"), dict):
                            continue
                        step_t = FalsificationAndAbstentionEngine.finite_number(step.get("relative_time_hours"))
                        step_lat = FalsificationAndAbstentionEngine.finite_number(step["centroid"].get("lat"), -90, 90)
                        step_lon = FalsificationAndAbstentionEngine.finite_number(step["centroid"].get("lon"), -180, 180)
                        if step_t is None or step_lat is None or step_lon is None:
                            continue
                        if abs(pt_t - step_t) <= 1.0:
                            d_nm = self.haversine_distance_nm(
                                pt["lat"], pt["lon"],
                                step_lat, step_lon
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
        track = self._valid_position_track(vessel.get("trajectory"))
        if len(track) < 3 or any(pt["sog_knots"] is None or pt["cog_degrees"] is None for pt in track):
            return {
                "status": "NOT_ASSESSED", "speed_drop_knots": None,
                "min_speed_near_origin": None, "cruise_speed": None,
                "speed_anomaly_score": None, "course_variance_deg": None,
                "course_anomaly_score": None, "has_ais_gap": None,
                "behavior_summary": "Insufficient valid speed/course telemetry for navigation context"
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
            "status": "ASSESSED",
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
        """Audit supplied telemetry, never infer receiver coverage from silence.

        Continuity needs >=2 finite position/time fixes at distinct times spanning
        the CPA time, with no rejected fixes. Severe flags block independently.
        MMSI validation is a format/range check, not registry authentication.
        """
        number = FalsificationAndAbstentionEngine.finite_number
        input_valid = isinstance(trajectory, (list, tuple))
        raw_track = trajectory if input_valid else []
        anomalies, reasons = [], []
        timed_track, position_track = [], []
        rejected = 0
        identity_conflict = False
        mmsi_valid = None
        if mmsi is None:
            reasons.append("Vessel identity is missing.")
        else:
            mmsi_str = str(mmsi)
            mmsi_valid = len(mmsi_str) == 9 and mmsi_str.isdigit() and 201 <= int(mmsi_str[:3]) <= 775
            if not mmsi_valid:
                anomalies.append("INVALID_MMSI_OR_UNALLOCATED_MID: ID fails the 9-digit maritime MID range check.")
        if not input_valid:
            reasons.append("Trajectory must be a list of telemetry records.")
        for raw in raw_track:
            if not isinstance(raw, dict):
                rejected += 1
                continue
            time = number(raw.get("relative_time_hours"))
            lat, lon = number(raw.get("lat"), -90, 90), number(raw.get("lon"), -180, 180)
            speed = number(raw.get("sog_knots"), 0, 102.2)
            if raw.get("mmsi") is not None and str(raw["mmsi"]) != str(mmsi):
                identity_conflict = True
            if time is None:
                rejected += 1
                continue
            point = {**raw, "relative_time_hours": time, "lat": lat, "lon": lon, "sog_knots": speed}
            timed_track.append(point)
            if lat is not None and lon is not None:
                position_track.append(point)
            if lat is None or lon is None or (raw.get("sog_knots") is not None and speed is None):
                rejected += 1
        timed_track.sort(key=lambda p: p["relative_time_hours"])
        position_track.sort(key=lambda p: p["relative_time_hours"])
        if identity_conflict:
            anomalies.append("IDENTITY_CONFLICT: A position record reports a different MMSI.")

        cpa_t = number(cpa_time_relative_h if cpa_time_relative_h is not None else origin_time_relative_h)
        max_gap_min = max_accel_kts_min = 0.0
        corridor_gap = impossible_speed_jump = position_jump = False
        for previous, current in zip(timed_track, timed_track[1:]):
            t_previous, t_current = previous["relative_time_hours"], current["relative_time_hours"]
            dt_min = (t_current - t_previous) * 60.0
            max_gap_min = max(max_gap_min, dt_min)
            if (dt_min > 30 and cpa_t is not None
                    and t_previous <= cpa_t + 2.5 and t_current >= cpa_t - 2.5):
                corridor_gap = True
                anomalies.append(f"CORRIDOR_TRANSPONDER_BLACKOUT: {dt_min:.1f} min reporting gap near CPA; verify receiver coverage.")
            if previous["sog_knots"] is not None and current["sog_knots"] is not None:
                speed_difference = abs(current["sog_knots"] - previous["sog_knots"])
                acceleration = speed_difference / dt_min if dt_min > 0 else 0.0
                max_accel_kts_min = max(max_accel_kts_min, acceleration)
                if acceleration > 3 or (dt_min == 0 and speed_difference > 0):
                    impossible_speed_jump = True
                    anomalies.append("KINEMATIC_SPEED_JUMP: Inconsistent simultaneous speeds or acceleration above 3 kts/min.")
        for previous, current in zip(position_track, position_track[1:]):
            dt_hours = current["relative_time_hours"] - previous["relative_time_hours"]
            distance_nm = AISEngine.haversine_distance_km(
                previous["lat"], previous["lon"], current["lat"], current["lon"]) / 1.852
            if (dt_hours == 0 and distance_nm > 0.1) or (dt_hours > 0 and distance_nm / dt_hours > 100):
                position_jump = True
                anomalies.append("POSITION_JUMP: Position displacement requires more than 100 knots or contradicts simultaneous fixes.")

        position_times = [p["relative_time_hours"] for p in position_track]
        span = position_times[-1] - position_times[0] if position_times else 0.0
        cpa_covered = bool(position_times and cpa_t is not None and position_times[0] <= cpa_t <= position_times[-1])
        if len(position_track) < 2:
            reasons.append("At least two valid geographic position/time fixes are required.")
        if len(set(position_times)) < 2:
            reasons.append("At least two distinct finite telemetry times are required.")
        if rejected:
            reasons.append(f"{rejected} telemetry records have invalid/missing required fields.")
        if not cpa_covered:
            reasons.append("Valid position telemetry does not span the CPA time.")
        telemetry_validated = input_valid and len(position_track) >= 2 and span > 0 and rejected == 0 and cpa_covered
        status = "ASSESSED" if telemetry_validated and mmsi_valid is not None else "NOT_ASSESSED"
        compromised = bool(anomalies)
        return {
            "schema_version": 1, "status": status,
            "has_anomalies": compromised, "anomalies_detected": anomalies, "anomaly_count": len(anomalies),
            "assessment_reasons": reasons, "total_fixes": len(raw_track), "rejected_fixes": rejected,
            "valid_time_fixes": len(timed_track), "valid_position_fixes": len(position_track),
            "valid_speed_fixes": sum(p["sog_knots"] is not None for p in timed_track),
            "time_span_hours": span, "cpa_time_covered": cpa_covered,
            "telemetry_validated": telemetry_validated,
            "max_gap_minutes": max_gap_min, "max_acceleration_kts_min": max_accel_kts_min,
            "corridor_blackout": corridor_gap, "impossible_speed_jump": impossible_speed_jump,
            "identity_conflict": identity_conflict, "position_jump": position_jump, "mmsi_valid": mmsi_valid,
            "integrity_rating": ("COMPROMISED / ANOMALOUS" if compromised else
                                 "VERIFIED_CONTINUOUS" if status == "ASSESSED" else "NOT_ASSESSED"),
            "identity_check": "format_and_mid_range_only_not_registry_authentication",
        }

    @staticmethod
    def compute_topsis_rankings(
        candidate_vessels: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
        Multi-Criteria Decision Analysis (MCDA) + Borda Count Rank Aggregation.

        Descriptive comparison of five criteria; not the evidence-gate ranking:
          C1 [Cost]: CPA Distance (NM) — lower is closer to discharge origin
          C2 [Cost]: Delta-T Time Coincidence (h) — lower is better time alignment
          C3 [Benefit]: Speed Deceleration Near Origin (knots) — navigation context
          C4 [Benefit]: Vessel Class Screen — descriptive metadata only
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
        origin_time_relative_h: float,
        *,
        coverage_validated: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Computes an uncalibrated lead-priority score. Only spatio-temporal
        co-location affects ranking; vessel class and navigation context are shown
        for review but never treated as responsibility evidence.
        """
        ranked_vessels = []

        for v in candidate_vessels:
            cpa = v.get("closest_approach")
            cpa = cpa if isinstance(cpa, dict) else {}
            dist_nm = FalsificationAndAbstentionEngine.finite_number(cpa.get("distance_nm"), 0)
            time_diff_h = FalsificationAndAbstentionEngine.finite_number(cpa.get("time_diff_h"), 0)

            # 1. Proximity score (Gaussian spatial decay)
            s_prox = 100.0 * math.exp(-0.5 * min(40.0, dist_nm / self.cpa_sigma_nm) ** 2) if dist_nm is not None else None

            # 2. Temporal coincidence score (Gaussian temporal decay)
            s_time = 100.0 * math.exp(-0.5 * min(40.0, time_diff_h / self.temporal_sigma_hours) ** 2) if time_diff_h is not None else None

            # Navigation context is descriptive, not an attribution factor.
            kinematics = self.analyze_vessel_kinematics(v, origin_time_relative_h)
            s_speed = kinematics["speed_anomaly_score"]
            s_course = kinematics["course_anomaly_score"]
            v_type = v.get("vessel_type", "Other / Unknown")
            s_type = None
            composite_score = 0.6 * s_prox + 0.4 * s_time if s_prox is not None and s_time is not None else None

            # Lead-priority tier. Scores are heuristic ranking signals, not a
            # calibrated probability or a finding of responsibility.
            if composite_score is None:
                attribution_tier = "NOT_ASSESSED (INVALID CPA INPUT)"
                flag_color = "#64748b"
            elif composite_score >= 80.0:
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
                "lat": FalsificationAndAbstentionEngine.finite_number(pt.get("lat"), -90, 90),
                "lon": FalsificationAndAbstentionEngine.finite_number(pt.get("lon"), -180, 180),
                "relative_time_hours": FalsificationAndAbstentionEngine.finite_number(pt.get("relative_time_hours"))
            }
            if any(value is None for value in candidate_release_point.values()):
                candidate_release_point = None

            # AIS Spoofing & Integrity Audit
            try:
                spoofing_audit = self.detect_ais_spoofing_and_gaps(
                    v.get("trajectory", []), v.get("mmsi"), origin_time_relative_h,
                    cpa.get("time_relative_h")
                )
            except Exception as exc:
                logger.warning("AIS continuity audit unavailable: %s", exc)
                spoofing_audit = {"status": "UNAVAILABLE", "integrity_rating": "NOT_ASSESSED"}

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
                    "proximity_score": s_prox,
                    "temporal_score": s_time,
                    "speed_anomaly_score": s_speed,
                    "course_anomaly_score": s_course,
                    "vessel_type_score": None
                },
                "closest_approach": {
                    "distance_nm": dist_nm,
                    "distance_km": dist_nm * 1.852 if dist_nm is not None and dist_nm <= 1e307 else None,
                    "time_diff_h": time_diff_h,
                    "time_relative_h": FalsificationAndAbstentionEngine.finite_number(cpa.get("time_relative_h")),
                    "point": candidate_release_point,
                },
                "kinematics": kinematics,
                "spoofing_audit": spoofing_audit,
                "full_trajectory": self._valid_position_track(v.get("trajectory")),
                "data_origin": v.get("data_origin", "Not supplied; source provenance requires verification"),
                "is_real_ais": v.get("is_real_ais")
            }
            ranked_vessels.append(v_result)

        # Multi-Criteria Decision Analysis (TOPSIS + Borda Count)
        ranked_vessels = self.compute_topsis_rankings(ranked_vessels)

        # Sort by composite suspect score descending
        ranked_vessels.sort(key=lambda x: x["lead_priority_score"] if x["lead_priority_score"] is not None else -1, reverse=True)

        # Sensitivity/integrity checks precede the final evidence-state decision.
        try:
            for v in ranked_vessels:
                v["composite_score"] = v.get("lead_priority_score")
                v["adversarial_stress_test"] = FalsificationAndAbstentionEngine.run_adversarial_stress_test(
                    v, origin_lat, origin_lon, current_speed_knots=1.5, wind_speed_knots=12.0
                )
            group_verdict = FalsificationAndAbstentionEngine.evaluate_decision_theoretic_abstention(
                ranked_vessels, coverage_validated=coverage_validated, unknown_source_hypothesis=True
            )
            for index, v in enumerate(ranked_vessels):
                verdict = deepcopy(group_verdict)
                integrity = FalsificationAndAbstentionEngine.assess_ais_integrity(v.get("spoofing_audit"))
                verdict["candidate_mmsi"] = v.get("mmsi")
                verdict["candidate_is_leading"] = index == group_verdict.get("leading_candidate_index")
                verdict["candidate_priority_weight"] = v.get("lead_priority_weight")
                verdict["integrity_status"] = integrity["status"]
                verdict["stress_status"] = v["adversarial_stress_test"]["status"]
                if not integrity["passed"]:
                    v["lead_integrity_warning"] = integrity["reason"]
                    verdict.update(is_abstention=True, decision="INSUFFICIENT_EVIDENCE",
                                   reason=f"Candidate integrity hold: {integrity['reason']} {group_verdict['reason']}")
                    if integrity["status"] in ("NOT_ASSESSED", "UNAVAILABLE"):
                        verdict["status"] = integrity["status"]
                elif not v["adversarial_stress_test"].get("stress_passed"):
                    verdict.update(is_abstention=True, decision="INSUFFICIENT_EVIDENCE",
                                   reason="Candidate sensitivity checks failed. " + group_verdict["reason"])
                elif not verdict["candidate_is_leading"]:
                    verdict.update(is_abstention=True, decision="LOWER_PRIORITY_SCREENING_CANDIDATE",
                                   reason="Candidate is not the leading hypothesis. " + group_verdict["reason"])
                v["abstention_verdict"] = verdict
                v["assessment_status"] = verdict["status"]
        except Exception as exc:
            logger.warning("AIS evidence gate unavailable: %s", exc)
            for v in ranked_vessels:
                verdict = FalsificationAndAbstentionEngine.unavailable_verdict(
                    f"Evidence gate or sensitivity evaluation failed ({type(exc).__name__}).", ranked_vessels)
                verdict["candidate_mmsi"] = v.get("mmsi")
                v["lead_priority_weight"] = None
                v["abstention_verdict"] = verdict
                v["assessment_status"] = "UNAVAILABLE"
                v.setdefault("adversarial_stress_test", {
                    "status": "UNAVAILABLE", "stress_passed": False,
                    "verdict": "SENSITIVITY_CHECK_UNAVAILABLE", "challenges": [],
                    "adversarial_robustness_score": None,
                })

        return _json_safe(ranked_vessels)

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
        radar_targets: Optional[List[Dict[str, Any]]] = None,
        *,
        coverage_validated: bool = False,
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
            filtered, origin_lat, origin_lon, origin_time_relative_h,
            coverage_validated=coverage_validated,
        )

        primary_suspect = ranked[0] if ranked else None
        if primary_suspect:
            screening_gate = deepcopy(primary_suspect["abstention_verdict"])
        else:
            try:
                screening_gate = FalsificationAndAbstentionEngine.evaluate_decision_theoretic_abstention(
                    [], coverage_validated=coverage_validated)
            except Exception as exc:
                logger.warning("Empty-corridor evidence gate unavailable: %s", exc)
                screening_gate = FalsificationAndAbstentionEngine.unavailable_verdict(
                    f"Evidence gate failed ({type(exc).__name__}).")

        # Dark ship detection
        dark_vessels = []
        matched_radar = []
        if radar_targets:
            matched_radar, dark_vessels = self.correlate_radar_targets_with_ais(
                radar_targets, raw_vessels, origin_lat, origin_lon
            )

        return {
            "status": screening_gate["status"],
            "is_abstention": screening_gate["is_abstention"],
            "total_vessels_in_region": len(raw_vessels),
            "vessels_evaluated_in_corridor": len(ranked),
            "primary_review_lead": primary_suspect,
            "evidentiary_lead": primary_suspect if screening_gate["is_abstention"] is False else None,
            "primary_culprit": None,
            "ranked_suspects": ranked,
            "dark_vessels_detected": dark_vessels,
            "dark_vessels_count": len(dark_vessels),
            "screening_gate": screening_gate,
            "bayesian_legal_gate": deepcopy(screening_gate),  # Legacy UI alias, not a Bayesian/legal claim
            "compatibility_keys": {"bayesian_legal_gate": "legacy alias of screening_gate; uncalibrated screening only"},
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
