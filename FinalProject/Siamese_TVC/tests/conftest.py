import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / 'official'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OFFICIAL) not in sys.path:
    sys.path.insert(0, str(OFFICIAL))
