"""
AIS Engine: Automatic Identification System Maritime Traffic Correlation & Culprit Attribution
Ingests historical vessel transponder trajectories, filters irrelevant maritime traffic,
reconstructs spatio-temporal vessel positions around the spill origin window (x0, y0, t0),
and computes multi-criteria behavioral anomaly scores to identify and rank culprit vessels.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class AISEngine:
    """
    Spatio-Temporal AIS Trajectory Ingestion, Spatial Indexing,
    Kinematic Anomaly Detection, and Culprit Vessel Attribution.
    """

    VESSEL_TYPE_WEIGHTS = {
        "Crude Oil Tanker": 95.0,
        "Chemical Tanker": 90.0,
        "Product Tanker": 88.0,
        "Bulk Carrier": 80.0,
        "Container Ship": 65.0,
        "General Cargo": 55.0,
        "Offshore Supply Vessel": 30.0,
        "Tug / Workboat": 20.0,
        "Fishing Vessel": 10.0,
        "Pleasure Craft": 5.0,
        "Other / Unknown": 40.0
    }

    def __init__(
        self,
        cpa_distance_sigma_nm: float = 1.8,   # Closest Point of Approach scale in Nautical Miles
        temporal_sigma_hours: float = 1.2,     # Time difference decay parameter
        weight_proximity: float = 0.35,
        weight_temporal: float = 0.25,
        weight_speed_anomaly: float = 0.20,
        weight_course_anomaly: float = 0.08,
        weight_vessel_type: float = 0.12
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
        r = 6371.0  # Earth mean radius in km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
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
            track = v.get("trajectory", [])
            if not track:
                continue

            min_dist_nm = float("inf")
            best_point = None
            best_origin_time = origin_time_relative_h

            if hindcast_trajectory:
                # Full space-time cross-correlation along the drift path
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
                                best_origin_time = step_t

            # Fallback to single epicenter (origin_lat, origin_lon)
            if best_point is None:
                for pt in track:
                    d_nm = self.haversine_distance_nm(pt["lat"], pt["lon"], origin_lat, origin_lon)
                    if d_nm < min_dist_nm:
                        min_dist_nm = d_nm
                        best_point = pt
                best_origin_time = origin_time_relative_h

            # Check if vessel traversed within corridor
            if min_dist_nm <= spatial_radius_nm and best_point is not None:
                pt_rel_t = best_point.get("relative_time_hours", 0.0)
                time_diff_h = abs(pt_rel_t - best_origin_time)

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
        Analyzes vessel telemetry for behavioral anomalies:
        1. Speed drops (cruising at 15 kts -> dropping to 4-7 kts during covert bilge discharge)
        2. Sudden course alterations / erratic maneuvering
        3. AIS transponder gaps (turning off AIS while dumping)
        """
        track = vessel.get("trajectory", [])
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

        # Speed anomaly scoring:
        # Cruising tankers usually don't drop from 15 kts to 5 kts in open sea unless maneuvering/discharging
        if speed_drop >= 7.0 and 3.5 <= min_near_speed <= 8.5:
            speed_score = 96.0  # Classic illegal discharge profile
            speed_comment = f"Severe speed drop ({cruise_speed:.1f} -> {min_near_speed:.1f} kts) matching illegal bilge purge profile"
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

        # Check for AIS gap (> 45 min gap between reports in open sea)
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

    def score_and_rank_suspects(
        self,
        candidate_vessels: List[Dict[str, Any]],
        origin_lat: float,
        origin_lon: float,
        origin_time_relative_h: float
    ) -> List[Dict[str, Any]]:
        """
        Computes composite attribution score for all candidate vessels:
        Score = w_prox*S_prox + w_time*S_time + w_speed*S_speed + w_course*S_course + w_type*S_type
        Returns ranked list of vessels from highest probability suspect to lowest.
        """
        ranked_vessels = []

        for v in candidate_vessels:
            cpa = v["closest_approach"]
            dist_nm = cpa["distance_nm"]
            time_diff_h = cpa["time_diff_h"]

            # 1. Proximity score (Gaussian spatial decay)
            s_prox = 100.0 * math.exp(-(dist_nm ** 2) / (2.0 * (self.cpa_sigma_nm ** 2)))

            # 2. Temporal coincidence score (Gaussian temporal decay)
            s_time = 100.0 * math.exp(-(time_diff_h ** 2) / (2.0 * (self.temporal_sigma_hours ** 2)))

            # 3. Kinematics analysis
            kinematics = self.analyze_vessel_kinematics(v, origin_time_relative_h)
            s_speed = kinematics["speed_anomaly_score"]
            s_course = kinematics["course_anomaly_score"]

            # Bonus penalty if transponder had suspicious blackout gap near origin
            if kinematics["has_ais_gap"]:
                s_speed = min(100.0, s_speed + 15.0)

            # 4. Vessel type risk factor
            v_type = v.get("vessel_type", "Other / Unknown")
            s_type = self.VESSEL_TYPE_WEIGHTS.get(v_type, 40.0)

            # Composite Score
            composite_score = (
                self.w_prox * s_prox +
                self.w_time * s_time +
                self.w_speed * s_speed +
                self.w_course * s_course +
                self.w_type * s_type
            )
            composite_score = round(min(max(composite_score, 0.0), 99.8), 1)

            # Attribution confidence tier
            if composite_score >= 80.0:
                attribution_tier = "PRIMARY SUSPECT (HIGH CERTAINTY)"
                flag_color = "#ff3366"  # Red
            elif composite_score >= 60.0:
                attribution_tier = "PERSON OF INTEREST (MODERATE)"
                flag_color = "#ffaa00"  # Amber
            else:
                attribution_tier = "UNLIKELY / INCIDENTAL TRAFFIC"
                flag_color = "#05d6a0"  # Green

            v_result = {
                "mmsi": v.get("mmsi"),
                "imo": v.get("imo"),
                "vessel_name": v.get("vessel_name", "UNKNOWN VESSEL"),
                "call_sign": v.get("call_sign", "N/A"),
                "flag_state": v.get("flag_state", "Unknown"),
                "vessel_type": v_type,
                "length_m": v.get("length_m", 180),
                "width_m": v.get("width_m", 32),
                "dwt_tonnes": v.get("dwt_tonnes", 45000),
                "composite_suspect_score": composite_score,
                "attribution_tier": attribution_tier,
                "flag_color": flag_color,
                "score_breakdown": {
                    "proximity_score": round(s_prox, 1),
                    "temporal_score": round(s_time, 1),
                    "speed_anomaly_score": round(s_speed, 1),
                    "course_anomaly_score": round(s_course, 1),
                    "vessel_type_score": round(s_type, 1)
                },
                "closest_approach": cpa,
                "kinematics": kinematics,
                "full_trajectory": v.get("trajectory", [])
            }
            ranked_vessels.append(v_result)

        # Sort by composite suspect score descending
        ranked_vessels.sort(key=lambda x: x["composite_suspect_score"], reverse=True)
        return ranked_vessels

    def attribute_oil_spill(
        self,
        raw_vessels: List[Dict[str, Any]],
        origin_lat: float,
        origin_lon: float,
        origin_time_relative_h: float,
        spatial_radius_nm: float = 30.0,
        temporal_window_h: float = 5.0,
        hindcast_trajectory: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Complete end-to-end attribution workflow:
        1. Filters out irrelevant traffic (using full trajectory matching if provided)
        2. Reconstructs space-time corridor
        3. Analyzes kinematic anomalies
        4. Identifies and ranks culprit vessels
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

        return {
            "total_vessels_in_region": len(raw_vessels),
            "vessels_evaluated_in_corridor": len(ranked),
            "primary_culprit": primary_suspect,
            "ranked_suspects": ranked,
            "origin_query": {
                "origin_lat": origin_lat,
                "origin_lon": origin_lon,
                "origin_time_relative_h": origin_time_relative_h
            }
        }
