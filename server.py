"""
OCEAN-SHIELD: Root Gateway & Uvicorn Application Entrypoint
NTRO / Smart India Hackathon 2026 (Problem Statement SIH26143)
"""

import sys
import os

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ocean_shield.server import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.ocean_shield.server:app", host="127.0.0.1", port=8090, reload=True)

