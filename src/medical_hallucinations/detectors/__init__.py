"""Detector adapters, imported lazily so each one's heavy dependencies stay optional."""

from importlib import import_module

from .base import Detector

# CLI name -> "module:class". Add MiniCheck, Lynx, and LettuceDetect here.
DETECTORS = {
    "hhem": "medical_hallucinations.detectors.hhem:HHEMDetector",
}


def load_detector(name: str) -> Detector:
    """Instantiate a registered detector (this loads its model weights)."""
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector {name!r}; choose from {sorted(DETECTORS)}")
    module_name, class_name = DETECTORS[name].split(":")
    return getattr(import_module(module_name), class_name)()
