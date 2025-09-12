import importlib.util as _util
from pathlib import Path as _Path

_spec = _util.spec_from_file_location(
    "_stack_metrics_impl", _Path(__file__).resolve().parent.parent / "stack_metrics.py"
)
_module = _util.module_from_spec(_spec)
_spec.loader.exec_module(_module)  # type: ignore[attr-defined]

analyze_lineup = _module.analyze_lineup
compute_presence_and_counts = _module.compute_presence_and_counts
compute_features = _module.compute_features
classify_bucket = _module.classify_bucket

__all__ = [
    "analyze_lineup",
    "compute_presence_and_counts",
    "compute_features",
    "classify_bucket",
]
