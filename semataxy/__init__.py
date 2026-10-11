"""Semataxy overlay on upstream GovSim.

Upstream code (third_party/GovSim and its pathfinder submodule) is imported
unmodified as the ``simulation`` and ``pathfinder`` packages. Everything that
differs from upstream lives in this package and is listed in docs/INVENTORY.md.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOVSIM_ROOT = os.path.join(REPO_ROOT, "third_party", "GovSim")
PATHFINDER_ROOT = os.path.join(GOVSIM_ROOT, "pathfinder")

for _p in (GOVSIM_ROOT, PATHFINDER_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Upstream's hydra searchpath entries are relative to the working directory.
# Our compose call passes absolute file:// paths instead, so the cwd does not matter.
