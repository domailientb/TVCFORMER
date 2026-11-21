import importlib.util
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = REPOSITORY_ROOT / "official"
official_path = str(OFFICIAL_ROOT)
if official_path in sys.path:
    sys.path.remove(official_path)
sys.path.insert(0, official_path)

_model_path = OFFICIAL_ROOT / "models" / "TVCFormer.py"
_spec = importlib.util.spec_from_file_location(
    "_tvcformer_official_model", str(_model_path)
)
if _spec is None or _spec.loader is None:
    raise ImportError("could not load official TVCFormer from {}".format(_model_path))
TVCFormer = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = TVCFormer
_spec.loader.exec_module(TVCFormer)

__all__ = ["TVCFormer"]
