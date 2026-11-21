"""Detector adapters, imported lazily so each one's heavy dependencies stay optional."""

from importlib import import_module
from typing import Any

from .base import Detector

# CLI name -> "module:class". The CLI name is also the predictions directory.
# Add MiniCheck and Lynx here once their adapters are implemented.
DETECTORS = {
    "hhem": "medical_hallucinations.detectors.hhem:HHEMDetector",
    "lettucedetect": "medical_hallucinations.detectors.lettucedetect:LettuceDetectV2",
    "lettucedetect-v1": "medical_hallucinations.detectors.lettucedetect:LettuceDetectV1",
}


def load_detector(name: str, **options: Any) -> Detector:
    """Instantiate a registered detector (this loads its model weights).

    ``options`` (e.g. ``device="cuda"``) are passed to the adapter's constructor.
    """
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector {name!r}; choose from {sorted(DETECTORS)}")
    module_name, class_name = DETECTORS[name].split(":")
    return getattr(import_module(module_name), class_name)(**options)
