"""Validated parsing for MarineCadastre-style historic AIS CSV exports."""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import datetime, timezone
from typing import Any, Dict, List


MAX_AIS_UPLOAD_BYTES = 25 * 1024 * 1024


def _value(row: Dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _number(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    timestamp = datetime.fromisoformat(normalized)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _normalized_vessel_type(raw_type: str, cargo: str) -> str:
    cargo_l = cargo.lower()
    raw_l = raw_type.lower()
    if "crude" in cargo_l:
        return "Crude Oil Tanker"
    if "petroleum" in cargo_l or "product" in cargo_l or raw_l in {"80", "81", "82", "tanker"}:
        return "Product Tanker"
    if "cargo" in raw_l:
        return "General Cargo"
    if "fishing" in raw_l:
        return "Fishing Vessel"
    return raw_type or "Other / Unknown"


def parse_marinecadastre_csv(content: bytes, filename: str = "ais.csv") -> Dict[str, Any]:
    """Parse and time-normalize a CSV without inventing missing AIS telemetry.

    Relative time is referenced to the newest valid ping in the uploaded file, which
    becomes T=0.  The caller can use this reference for transparent correlation.
    """
    if not content:
        raise ValueError("The AIS CSV is empty.")
    if len(content) > MAX_AIS_UPLOAD_BYTES:
        raise ValueError("AIS CSV exceeds the 25 MB upload limit.")

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("AIS CSV must be UTF-8 encoded.") from exc

    reader = csv.DictReader(io.StringIO(text))
    required = {"MMSI", "BaseDateTime", "LAT", "LON"}
    headers = set(reader.fieldnames or [])
    if not required.issubset(headers):
        raise ValueError("AIS CSV requires MMSI, BaseDateTime, LAT, and LON columns.")

    vessel_rows: Dict[int, Dict[str, Any]] = {}
    rejected_rows = 0
    valid_pings = 0

    for row in reader:
        try:
            mmsi = _integer(_value(row, "MMSI", "mmsi"))
            lat = _number(_value(row, "LAT", "lat"), default=999.0)
            lon = _number(_value(row, "LON", "lon"), default=999.0)
            observed_at = _parse_timestamp(_value(row, "BaseDateTime", "base_datetime"))
            if not (100_000_000 <= mmsi <= 999_999_999 and -90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("invalid AIS coordinates or MMSI")

            vessel_name = _value(row, "VesselName", "vessel_name", default=f"VESSEL-{mmsi}")
            cargo = _value(row, "Cargo", "cargo")
            vessel_type = _normalized_vessel_type(_value(row, "VesselType", "vessel_type"), cargo)
            vessel = vessel_rows.setdefault(mmsi, {
                "mmsi": mmsi,
                "imo": _integer(_value(row, "IMO", "imo")),
                "vessel_name": vessel_name,
                "call_sign": _value(row, "CallSign", "call_sign", default="N/A"),
                "flag_state": "Not supplied by source",
                "vessel_type": vessel_type,
                "length_m": _number(_value(row, "Length", "length")),
                "width_m": _number(_value(row, "Width", "width")),
                "cargo": cargo or "Not supplied by source",
                "trajectory": [],
            })
            vessel["trajectory"].append({
                "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
                "_observed_timestamp": observed_at.timestamp(),
                "lat": lat,
                "lon": lon,
                "sog_knots": _number(_value(row, "SOG", "sog")),
                "cog_degrees": _number(_value(row, "COG", "cog")),
            })
            valid_pings += 1
        except (TypeError, ValueError, OverflowError):
            rejected_rows += 1

    if not vessel_rows:
        raise ValueError("No valid, time-stamped AIS pings were found in the CSV.")

    all_timestamps = [pt["_observed_timestamp"] for vessel in vessel_rows.values() for pt in vessel["trajectory"]]
    reference_timestamp = max(all_timestamps)
    for vessel in vessel_rows.values():
        vessel["trajectory"].sort(key=lambda pt: pt["_observed_timestamp"])
        for point in vessel["trajectory"]:
            point["relative_time_hours"] = round((point["_observed_timestamp"] - reference_timestamp) / 3600.0, 4)
            del point["_observed_timestamp"]

    vessels = sorted(vessel_rows.values(), key=lambda vessel: vessel["mmsi"])
    return {
        "vessels": vessels,
        "provenance": {
            "source_kind": "user_uploaded_historic_ais_csv",
            "source_filename": filename or "ais.csv",
            "sha256": hashlib.sha256(content).hexdigest(),
            "reference_time_utc": datetime.fromtimestamp(reference_timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "valid_pings": valid_pings,
            "rejected_rows": rejected_rows,
            "vessels": len(vessels),
        },
    }
