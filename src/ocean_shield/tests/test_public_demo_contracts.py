"""Public demo semantics and local frontend assets, using only ASGI requests."""

from pathlib import Path
import re
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.ocean_shield import server


class PublicDemoContracts(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)

    def test_health_means_reachability_and_never_initializes_inference_models(self):
        with patch.object(server.sar_engine, "_ensure_unet_loaded", side_effect=AssertionError("Health must not load models")):
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "reachable")
        self.assertEqual(body["certification_status"], "NOT_CERTIFIED")
        self.assertEqual(body["agency_affiliation"], "NONE_CLAIMED")
        self.assertIn("reachability only", body["status_scope"])
        self.assertNotIn("NTRO", body["system"])
        self.assertNotIn("Indian Coast Guard", body["system"])

    def test_visible_brand_and_chamber_do_not_claim_agency_or_measured_contrast(self):
        page = self.client.get("/").text
        self.assertIn("INDEPENDENT RESEARCH &bull; NOT CERTIFIED", page)
        self.assertIn("ILLUSTRATIVE BENCHMARK CHRONOLOGY", page)
        self.assertNotIn("-10.2 dB", page)
        self.assertNotIn("NTRO MARITIME ENVIRONMENTAL SECURITY", page)
        self.assertNotIn("FORENSIC REWIND LOOP CLOSED", page)
        self.assertNotIn("automatically filters out innocent traffic", page)
        self.assertNotIn("Speed_drop_delta = 0.0", page)
        self.assertNotIn("Live Rows Sample", page)

    def test_leaflet_and_icons_are_locally_served_without_remote_font_dependency(self):
        page = self.client.get("/").text
        self.assertIn('href="/static/vendor/leaflet.css"', page)
        self.assertIn('src="/static/vendor/leaflet.js"', page)
        css = self.client.get("/static/css/dashboard.css").text
        techstack = self.client.get("/techstack").text
        for content in (page, css, techstack):
            self.assertNotIn("fonts.googleapis.com", content)
            self.assertNotIn("fonts.gstatic.com", content)
            self.assertNotIn("unpkg.com", content)
        vendor_css = self.client.get("/static/vendor/leaflet.css")
        self.assertEqual(vendor_css.status_code, 200)
        self.assertEqual(self.client.get("/static/vendor/leaflet.js").status_code, 200)
        for icon in re.findall(r'url\([\'"]?(images/[^)\'"\s]+)', vendor_css.text):
            self.assertEqual(self.client.get("/static/vendor/" + icon).status_code, 200)
        # The source-backed tile providers have not been replaced by fake data.
        script = Path(server.STATIC_DIR, "js/app.js").read_text()
        self.assertIn("server.arcgisonline.com", script)

    def test_historic_csv_upload_remains_freshness_unverified(self):
        content = ("MMSI,BaseDateTime,LAT,LON,SOG,COG,VesselName\n"
                   "419001234,2017-01-01T00:00:00Z,0,0,6,90,Historic fixture\n"
                   "419001234,2017-01-01T00:05:00Z,0,0.01,6,90,Historic fixture\n").encode()
        response = self.client.post("/api/upload-ais-csv", content=content,
                                    headers={"X-File-Name": "historic-fixture.csv"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["provenance"]["freshness_status"], "UNVERIFIED")
        self.assertIn("freshness is unverified", response.json()["screening_notice"])


if __name__ == "__main__":
    unittest.main()
