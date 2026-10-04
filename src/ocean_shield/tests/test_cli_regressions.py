"""Headless integration must preserve nulls, explicit demo inputs and evidence holds."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CLIRegressions(unittest.TestCase):
    root = Path(__file__).resolve().parents[3]

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "ocean_shield_cli.py", *args], cwd=self.root,
                              env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                              capture_output=True, text=True, timeout=45)

    def test_missing_demo_or_age_is_not_silently_fabricated(self):
        for args in [[], ["--demo"], ["--demo", "--age-hours", "nan"],
                     ["--demo", "--age-hours", "1", "--ais-csv", "not-read.csv"]]:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("Traceback", result.stderr)

    def test_actual_sar_and_eo_demo_and_pdf_preserve_holds(self):
        with tempfile.TemporaryDirectory(prefix="ocean-cli-regression-") as directory:
            result = self.run_cli("--demo", "--age-hours", "0.1", "--engine", "cfar_edge",
                                  "--sensor", "both", "--export-pdf", "--output-dir", directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("DEMONSTRATION ONLY", result.stdout)
            self.assertIn("Actual integrated clock: -0.1h", result.stdout)
            self.assertIn("origin confidence NOT ESTIMATED", result.stdout)
            self.assertIn("Weathering NOT_ASSESSED", result.stdout)
            self.assertNotIn("Oil Confidence Score:", result.stdout)
            files = list(Path(directory).glob("*.pdf"))
            self.assertEqual(len(files), 1)
            self.assertGreater(files[0].stat().st_size, 5000)


if __name__ == "__main__":
    unittest.main()
