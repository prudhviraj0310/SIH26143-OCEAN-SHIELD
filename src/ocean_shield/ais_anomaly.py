"""
AIS Anomaly Detection Engine — Vessel Behavior Analysis

Adapted from Sandia National Laboratories' maritime-trajectory-anomaly-detection
(github.com/sandialabs/maritime-trajectory-anomaly-detection)

Provides real-time AIS anomaly detection including:
- Overspeed detection with statistical thresholding
- AIS gap detection (dark-vessel screening)
- Course deviation anomalies
- Loitering behavior detection
- Proximity alert generation (vessel-to-spill)

Reference: Sandia National Laboratories, BSD-3 License
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Vessel class speed limits (knots) — derived from IMO and Sandia research
VESSEL_SPEED_LIMITS: Dict[str, float] = {
    "cargo": 18.0,
    "tanker": 16.0,
    "crude / product tanker": 16.0,
    "bulk cargo carrier": 15.0,
    "container": 25.0,
    "passenger": 30.0,
    "fishing": 12.0,
    "commercial fishing vessel": 12.0,
    "tug": 14.0,
    "towing vessel": 14.0,
    "pilot vessel": 20.0,
    "search and rescue vessel": 35.0,
    "military vessel": 35.0,
    "pleasure craft": 40.0,
    "high speed vessel": 50.0,
}

# Default for unknown vessel types
DEFAULT_SPEED_LIMIT_KNOTS = 20.0

# Statistical thresholds
OVERSPEED_SIGMA_MULTIPLIER = 2.5  # Flag if speed > mean + 2.5σ
AIS_GAP_THRESHOLD_MINUTES = 60    # Flag if no AIS report for 60+ min
LOITERING_RADIUS_NM = 1.0         # Vessel stays within 1 NM for extended period
LOITERING_DURATION_HOURS = 4.0    # Must loiter for 4+ hours
COURSE_DEVIATION_THRESHOLD = 45.0 # Degrees sudden course change


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in nautical miles."""
    R_NM = 3440.065  # Earth radius in nautical miles
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return R_NM * 2 * math.asin(min(1.0, math.sqrt(a)))


def compute_speed_knots(lat1: float, lon1: float, t1: datetime,
                        lat2: float, lon2: float, t2: datetime) -> Optional[float]:
    """Compute speed in knots between two AIS reports."""
    dt_hours = (t2 - t1).total_seconds() / 3600.0
    if dt_hours <= 0:
        return None
    dist_nm = haversine_nm(lat1, lon1, lat2, lon2)
    return dist_nm / dt_hours


def compute_course_deg(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """Compute initial bearing (course) in degrees [0, 360)."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)
    x = math.sin(dlon_r) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r) -
         math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r))
    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360.0


class AnomalyResult:
    """A single detected anomaly event."""

    __slots__ = ("anomaly_type", "severity", "vessel_mmsi", "vessel_name",
                 "timestamp", "lat", "lon", "details", "confidence")

    def __init__(self, anomaly_type: str, severity: str,
                 vessel_mmsi: str, vessel_name: str,
                 timestamp: str, lat: float, lon: float,
                 details: str, confidence: float):
        self.anomaly_type = anomaly_type
        self.severity = severity
        self.vessel_mmsi = vessel_mmsi
        self.vessel_name = vessel_name
        self.timestamp = timestamp
        self.lat = lat
        self.lon = lon
        self.details = details
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "vessel_mmsi": self.vessel_mmsi,
            "vessel_name": self.vessel_name,
            "timestamp": self.timestamp,
            "position": {"lat": self.lat, "lon": self.lon},
            "details": self.details,
            "confidence_score": round(self.confidence, 3),
        }


class AISAnomalyDetector:
    """Multi-pattern AIS anomaly detection engine.

    Adapted from Sandia National Labs' benchmarking framework with
    domain-specific enhancements for maritime oil spill investigations.
    """

    def __init__(self, custom_speed_limits: Optional[Dict[str, float]] = None):
        self.speed_limits = {**VESSEL_SPEED_LIMITS}
        if custom_speed_limits:
            self.speed_limits.update(custom_speed_limits)

    def get_speed_limit(self, vessel_type: str) -> float:
        """Look up speed limit for a vessel type."""
        vt = (vessel_type or "").lower().strip()
        for key, limit in self.speed_limits.items():
            if key in vt or vt in key:
                return limit
        return DEFAULT_SPEED_LIMIT_KNOTS

    def detect_overspeed(self, track: List[Dict[str, Any]],
                         vessel_type: str = "") -> List[AnomalyResult]:
        """Detect overspeed events in a vessel track.

        Uses both absolute speed limits (from IMO vessel class) and
        statistical outlier detection (Sandia method: mean + 2.5σ).
        """
        anomalies = []
        if len(track) < 2:
            return anomalies

        limit = self.get_speed_limit(vessel_type)
        speeds = []
        mmsi = str(track[0].get("mmsi", "UNKNOWN"))
        name = track[0].get("name", track[0].get("vessel_name", "Unknown"))

        # Compute inter-report speeds
        for i in range(len(track) - 1):
            p1, p2 = track[i], track[i + 1]
            try:
                t1 = _parse_time(p1.get("timestamp", p1.get("BaseDateTime", "")))
                t2 = _parse_time(p2.get("timestamp", p2.get("BaseDateTime", "")))
                lat1 = float(p1.get("lat", p1.get("LAT", 0)))
                lon1 = float(p1.get("lon", p1.get("LON", 0)))
                lat2 = float(p2.get("lat", p2.get("LAT", 0)))
                lon2 = float(p2.get("lon", p2.get("LON", 0)))
            except (ValueError, TypeError):
                continue

            speed = compute_speed_knots(lat1, lon1, t1, lat2, lon2, t2)
            if speed is not None and math.isfinite(speed):
                speeds.append((speed, i, lat2, lon2, str(t2)))

        if not speeds:
            return anomalies

        # Statistical threshold (Sandia method)
        speed_values = [s[0] for s in speeds]
        mean_speed = sum(speed_values) / len(speed_values)
        variance = sum((s - mean_speed) ** 2 for s in speed_values) / max(len(speed_values) - 1, 1)
        std_speed = math.sqrt(variance)
        stat_threshold = mean_speed + OVERSPEED_SIGMA_MULTIPLIER * std_speed

        for speed, idx, lat, lon, ts in speeds:
            exceeded_absolute = speed > limit
            exceeded_statistical = speed > stat_threshold and std_speed > 0.5

            if exceeded_absolute or exceeded_statistical:
                severity = "CRITICAL" if speed > limit * 1.5 else "HIGH" if exceeded_absolute else "MEDIUM"
                confidence = min(1.0, (speed - limit) / limit) if exceeded_absolute else 0.6
                details = (f"Speed {speed:.1f} kts exceeds "
                           f"{'class limit ' + str(limit) + ' kts' if exceeded_absolute else ''}"
                           f"{'and ' if exceeded_absolute and exceeded_statistical else ''}"
                           f"{'statistical threshold ' + f'{stat_threshold:.1f} kts' if exceeded_statistical else ''}")
                anomalies.append(AnomalyResult(
                    anomaly_type="OVERSPEED",
                    severity=severity,
                    vessel_mmsi=mmsi,
                    vessel_name=name,
                    timestamp=ts,
                    lat=lat, lon=lon,
                    details=details,
                    confidence=confidence,
                ))

        return anomalies

    def detect_ais_gaps(self, track: List[Dict[str, Any]],
                        gap_threshold_min: float = AIS_GAP_THRESHOLD_MINUTES
                        ) -> List[AnomalyResult]:
        """Detect AIS transmission gaps (potential dark-vessel activity).

        A gap > threshold_min between consecutive AIS reports is flagged.
        This is a key indicator for vessels trying to hide their position
        near oil spill locations.
        """
        anomalies = []
        if len(track) < 2:
            return anomalies

        mmsi = str(track[0].get("mmsi", "UNKNOWN"))
        name = track[0].get("name", track[0].get("vessel_name", "Unknown"))

        for i in range(len(track) - 1):
            p1, p2 = track[i], track[i + 1]
            try:
                t1 = _parse_time(p1.get("timestamp", p1.get("BaseDateTime", "")))
                t2 = _parse_time(p2.get("timestamp", p2.get("BaseDateTime", "")))
                lat = float(p2.get("lat", p2.get("LAT", 0)))
                lon = float(p2.get("lon", p2.get("LON", 0)))
            except (ValueError, TypeError):
                continue

            gap_min = (t2 - t1).total_seconds() / 60.0
            if gap_min > gap_threshold_min:
                gap_hours = gap_min / 60.0
                severity = "CRITICAL" if gap_hours > 12 else "HIGH" if gap_hours > 4 else "MEDIUM"
                confidence = min(1.0, gap_hours / 24.0)
                anomalies.append(AnomalyResult(
                    anomaly_type="AIS_GAP",
                    severity=severity,
                    vessel_mmsi=mmsi,
                    vessel_name=name,
                    timestamp=str(t2),
                    lat=lat, lon=lon,
                    details=f"AIS gap of {gap_hours:.1f} hours detected between reports",
                    confidence=confidence,
                ))

        return anomalies

    def detect_course_deviation(self, track: List[Dict[str, Any]],
                                threshold_deg: float = COURSE_DEVIATION_THRESHOLD
                                ) -> List[AnomalyResult]:
        """Detect sudden course changes that may indicate evasive maneuvers."""
        anomalies = []
        if len(track) < 3:
            return anomalies

        mmsi = str(track[0].get("mmsi", "UNKNOWN"))
        name = track[0].get("name", track[0].get("vessel_name", "Unknown"))

        for i in range(len(track) - 2):
            try:
                lat1 = float(track[i].get("lat", track[i].get("LAT", 0)))
                lon1 = float(track[i].get("lon", track[i].get("LON", 0)))
                lat2 = float(track[i+1].get("lat", track[i+1].get("LAT", 0)))
                lon2 = float(track[i+1].get("lon", track[i+1].get("LON", 0)))
                lat3 = float(track[i+2].get("lat", track[i+2].get("LAT", 0)))
                lon3 = float(track[i+2].get("lon", track[i+2].get("LON", 0)))
                ts = track[i+1].get("timestamp", track[i+1].get("BaseDateTime", ""))
            except (ValueError, TypeError):
                continue

            course1 = compute_course_deg(lat1, lon1, lat2, lon2)
            course2 = compute_course_deg(lat2, lon2, lat3, lon3)
            delta = abs(course2 - course1)
            if delta > 180:
                delta = 360 - delta

            if delta > threshold_deg:
                severity = "HIGH" if delta > 90 else "MEDIUM"
                confidence = min(1.0, delta / 180.0)
                anomalies.append(AnomalyResult(
                    anomaly_type="COURSE_DEVIATION",
                    severity=severity,
                    vessel_mmsi=mmsi,
                    vessel_name=name,
                    timestamp=str(ts),
                    lat=lat2, lon=lon2,
                    details=f"Course deviation of {delta:.1f}° (from {course1:.0f}° to {course2:.0f}°)",
                    confidence=confidence,
                ))

        return anomalies

    def detect_loitering(self, track: List[Dict[str, Any]],
                         radius_nm: float = LOITERING_RADIUS_NM,
                         duration_hours: float = LOITERING_DURATION_HOURS
                         ) -> List[AnomalyResult]:
        """Detect vessel loitering (circling/drifting in a small area).

        Suspicious near oil spill sites — may indicate the vessel
        was involved in the discharge or attempting cleanup/evidence destruction.
        """
        anomalies = []
        if len(track) < 4:
            return anomalies

        mmsi = str(track[0].get("mmsi", "UNKNOWN"))
        name = track[0].get("name", track[0].get("vessel_name", "Unknown"))

        i = 0
        while i < len(track):
            try:
                anchor_lat = float(track[i].get("lat", track[i].get("LAT", 0)))
                anchor_lon = float(track[i].get("lon", track[i].get("LON", 0)))
                t_start = _parse_time(track[i].get("timestamp", track[i].get("BaseDateTime", "")))
            except (ValueError, TypeError):
                i += 1
                continue

            j = i + 1
            while j < len(track):
                try:
                    lat_j = float(track[j].get("lat", track[j].get("LAT", 0)))
                    lon_j = float(track[j].get("lon", track[j].get("LON", 0)))
                except (ValueError, TypeError):
                    j += 1
                    continue

                if haversine_nm(anchor_lat, anchor_lon, lat_j, lon_j) > radius_nm:
                    break
                j += 1

            try:
                t_end = _parse_time(track[min(j, len(track) - 1)].get(
                    "timestamp", track[min(j, len(track) - 1)].get("BaseDateTime", "")))
                loiter_hours = (t_end - t_start).total_seconds() / 3600.0
            except (ValueError, TypeError):
                i = j
                continue

            if loiter_hours >= duration_hours:
                severity = "HIGH" if loiter_hours > 12 else "MEDIUM"
                confidence = min(1.0, loiter_hours / 24.0)
                anomalies.append(AnomalyResult(
                    anomaly_type="LOITERING",
                    severity=severity,
                    vessel_mmsi=mmsi,
                    vessel_name=name,
                    timestamp=str(t_start),
                    lat=anchor_lat, lon=anchor_lon,
                    details=f"Vessel loitered for {loiter_hours:.1f} hours within {radius_nm} NM radius",
                    confidence=confidence,
                ))

            i = j

        return anomalies

    def detect_proximity_to_spill(self, track: List[Dict[str, Any]],
                                  spill_lat: float, spill_lon: float,
                                  radius_nm: float = 5.0
                                  ) -> List[AnomalyResult]:
        """Detect vessels that passed near the oil spill location."""
        anomalies = []
        mmsi = str(track[0].get("mmsi", "UNKNOWN")) if track else "UNKNOWN"
        name = track[0].get("name", track[0].get("vessel_name", "Unknown")) if track else "Unknown"

        min_dist = float("inf")
        closest_point = None

        for point in track:
            try:
                lat = float(point.get("lat", point.get("LAT", 0)))
                lon = float(point.get("lon", point.get("LON", 0)))
                ts = point.get("timestamp", point.get("BaseDateTime", ""))
            except (ValueError, TypeError):
                continue

            dist = haversine_nm(spill_lat, spill_lon, lat, lon)
            if dist < min_dist:
                min_dist = dist
                closest_point = (lat, lon, ts)

        if min_dist <= radius_nm and closest_point:
            severity = "CRITICAL" if min_dist < 1.0 else "HIGH" if min_dist < 3.0 else "MEDIUM"
            confidence = max(0.3, 1.0 - (min_dist / radius_nm))
            anomalies.append(AnomalyResult(
                anomaly_type="PROXIMITY_TO_SPILL",
                severity=severity,
                vessel_mmsi=mmsi,
                vessel_name=name,
                timestamp=str(closest_point[2]),
                lat=closest_point[0], lon=closest_point[1],
                details=f"Vessel passed within {min_dist:.2f} NM of spill at ({spill_lat:.4f}, {spill_lon:.4f})",
                confidence=confidence,
            ))

        return anomalies

    def run_full_screening(self, vessels: List[Dict[str, Any]],
                           spill_lat: Optional[float] = None,
                           spill_lon: Optional[float] = None
                           ) -> Dict[str, Any]:
        """Run complete anomaly screening on all vessels.

        Returns a structured report with all detected anomalies
        grouped by vessel and ranked by severity.
        """
        all_anomalies: List[Dict[str, Any]] = []
        vessel_summaries: List[Dict[str, Any]] = []

        for vessel in vessels:
            mmsi = str(vessel.get("mmsi", vessel.get("MMSI", "UNKNOWN")))
            name = vessel.get("name", vessel.get("vessel_name", vessel.get("VesselName", "Unknown")))
            vtype = vessel.get("type", vessel.get("vessel_type", vessel.get("VesselType", "")))
            track = vessel.get("track", vessel.get("positions", []))

            # If no track, build from vessel's single position
            if not track and vessel.get("lat") is not None:
                track = [vessel]

            vessel_anomalies = []
            vessel_anomalies.extend(self.detect_overspeed(track, vtype))
            vessel_anomalies.extend(self.detect_ais_gaps(track))
            vessel_anomalies.extend(self.detect_course_deviation(track))
            vessel_anomalies.extend(self.detect_loitering(track))

            if spill_lat is not None and spill_lon is not None:
                vessel_anomalies.extend(
                    self.detect_proximity_to_spill(track, spill_lat, spill_lon))

            anomaly_dicts = [a.to_dict() for a in vessel_anomalies]
            all_anomalies.extend(anomaly_dicts)

            # Compute vessel risk score
            risk_score = _compute_vessel_risk_score(vessel_anomalies)

            vessel_summaries.append({
                "mmsi": mmsi,
                "name": name,
                "vessel_type": vtype,
                "anomaly_count": len(vessel_anomalies),
                "risk_score": round(risk_score, 3),
                "risk_level": _risk_level(risk_score),
                "anomalies": anomaly_dicts,
            })

        # Sort vessels by risk score (highest first)
        vessel_summaries.sort(key=lambda v: v["risk_score"], reverse=True)

        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for a in all_anomalies:
            sev = a.get("severity", "LOW")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "total_vessels_screened": len(vessels),
            "total_anomalies_detected": len(all_anomalies),
            "severity_breakdown": severity_counts,
            "vessels": vessel_summaries,
            "all_anomalies": sorted(all_anomalies,
                                    key=lambda a: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(a["severity"], 4)),
            "screening_engine": "OCEAN-SHIELD AIS Anomaly Detector v1.0 (adapted from Sandia NL methodology)",
        }


def _compute_vessel_risk_score(anomalies: List[AnomalyResult]) -> float:
    """Compute aggregate risk score [0.0, 1.0] for a vessel."""
    if not anomalies:
        return 0.0
    weights = {"CRITICAL": 1.0, "HIGH": 0.7, "MEDIUM": 0.4, "LOW": 0.2}
    total = sum(weights.get(a.severity, 0.2) * a.confidence for a in anomalies)
    return min(1.0, total / max(len(anomalies), 1))


def _risk_level(score: float) -> str:
    if score >= 0.75:
        return "CRITICAL"
    elif score >= 0.5:
        return "HIGH"
    elif score >= 0.25:
        return "MEDIUM"
    return "LOW"


def _parse_time(ts: Any) -> datetime:
    """Parse various AIS timestamp formats."""
    if isinstance(ts, datetime):
        return ts
    ts_str = str(ts).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts_str}")
