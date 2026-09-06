"""
OCEAN-SHIELD: Satellite SAR Oil Spill Detection & AIS Rogue Vessel Attribution
National Technical Research Organisation (NTRO) / Indian Coast Guard
Smart India Hackathon 2026 - Problem Statement SIH26143
"""

__version__ = "1.0.0"

from .sar_engine import SAREngine
from .eo_engine import EOEngine
from .drift_engine import DriftEngine
from .ais_engine import AISEngine
from .report_generator import DossierReportGenerator
