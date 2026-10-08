"""Contract tests for source-backed live-data adapters.

These tests deliberately use mocks: a network failure must stay unavailable,
not silently become a fictional port, vessel, satellite pass, or oil spill.
"""

import asyncio
import os
import unittest
from unittest.mock import Mock, patch

from src.ocean_shield.live_fetcher import (
    fetch_live_ais_traffic,
    fetch_live_ocean_weather,
    fetch_live_oil_spill_incidents,
    fetch_live_satellite_passes,
    fetch_world_port_index,
)


class LiveFetcherTests(unittest.TestCase):
    def test_satellite_failure_returns_no_invented_pass(self):
        with patch("src.ocean_shield.live_fetcher.requests.get", side_effect=OSError("offline")):
            result = fetch_live_satellite_passes(22.585, 69.185)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["passes"], [])
        self.assertIsNone(result["latest_pass"])

    def test_meteo_uses_source_values_without_defaults(self):
        marine = Mock()
        marine.raise_for_status.return_value = None
        marine.json.return_value = {
            "current": {"time": "2026-09-22T00:00", "wave_height": 1.4,
                        "ocean_current_velocity": 7.2, "ocean_current_direction": 90}
        }
        weather = Mock()
        weather.raise_for_status.return_value = None
        weather.json.return_value = {
            "current": {"time": "2026-09-22T00:00", "wind_speed_10m": 18,
                        "wind_direction_10m": 180, "wind_gusts_10m": 25}
        }
        with patch("src.ocean_shield.live_fetcher.requests.get", side_effect=[marine, weather]):
            result = fetch_live_ocean_weather(22.585, 69.185)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["surface_current"]["speed_ms"], 2.0)
        self.assertEqual(result["surface_wind_10m"]["speed_ms"], 5.0)
        self.assertEqual(result["wave_height_m"], 1.4)

    def test_port_catalogue_normalizes_provider_records(self):
        from src.ocean_shield.live_fetcher import _cache_store
        _cache_store.clear()
        response = Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {"features": [{
            "geometry": {"type": "Point", "coordinates": [69.18, 22.58]},
            "properties": {"PORT_NAME": "Source Port", "COUNTRY": "India", "PORT_NUMBER": "42"},
        }]}
        with patch("src.ocean_shield.live_fetcher.requests.get", return_value=response):
            result = fetch_world_port_index()
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["ports"][0]["name"], "Source Port")
        self.assertFalse(result["is_live_operations"])

    def test_incidents_without_authority_feed_are_not_fabricated(self):
        from src.ocean_shield.live_fetcher import _cache_store
        _cache_store.clear()
        with patch.dict(os.environ, {"OIL_SPILL_FEED_URL": ""}, clear=False):
            result = fetch_live_oil_spill_incidents()
        self.assertIn(result["status"], ("available", "not_configured"))
        self.assertIsInstance(result.get("incidents"), list)

    def test_ais_without_server_key_is_not_fabricated(self):
        with patch.dict(os.environ, {"AISSTREAM_API_KEY": ""}, clear=False):
            result = asyncio.run(fetch_live_ais_traffic(22.585, 69.185))
        self.assertIn(result["status"], ("not_configured", "demo_mode"))
        self.assertEqual(result["vessels"], [])


if __name__ == "__main__":
    unittest.main()
