"""
OCEAN-SHIELD: Root Gateway & Uvicorn Application Entrypoint
NTRO / Smart India Hackathon 2026 (Problem Statement SIH26143)

Optimized for Render Free Tier (512 MB RAM, 0.1 vCPU):
- Single worker, single thread
- Keep-alive pinger to prevent spin-down
- Lazy model loading (PyTorch loads on first request, not at startup)
"""

import sys
import os

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Reduce PyTorch thread overhead for Render free-tier
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from src.ocean_shield.server import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8090))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(
        "src.ocean_shield.server:app",
        host=host,
        port=port,
        workers=1,           # Single worker: saves ~200 MB RAM
        timeout_keep_alive=120,
        limit_concurrency=5,  # Prevent OOM from parallel requests
    )
