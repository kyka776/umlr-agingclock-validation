"""Independent validation tools for the UMLR aging-clock method.

This project intentionally does not expose a production calibrator.  The official
method acts while fitting a clock from biomarkers; it is not a post-hoc transform
of already predicted ages.
"""

from .metrics import diagnostic_metrics, downstream_association
from .models import ConstrainedLassoOracle, LinearRecalibration, ResidualCorrection

__all__ = [
    "ConstrainedLassoOracle",
    "LinearRecalibration",
    "ResidualCorrection",
    "diagnostic_metrics",
    "downstream_association",
]

__version__ = "0.1.0"
