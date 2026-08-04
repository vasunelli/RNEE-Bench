from __future__ import annotations

import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "rnee_theory_validation"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
