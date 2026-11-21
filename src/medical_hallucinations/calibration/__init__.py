"""Post-hoc calibrators that remap frozen detector scores (RQ3)."""

from .base import Calibrator
from .isotonic import IsotonicCalibration
from .platt import PlattScaling
from .temperature import TemperatureScaling

CALIBRATORS: dict[str, type[Calibrator]] = {
    "platt": PlattScaling,
    "temperature": TemperatureScaling,
    "isotonic": IsotonicCalibration,
}

__all__ = [
    "CALIBRATORS", "Calibrator", "IsotonicCalibration", "PlattScaling", "TemperatureScaling",
]
