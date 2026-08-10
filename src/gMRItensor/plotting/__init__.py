"""Plotting utilities for gMRItensor decomposition results."""
from gMRItensor.plotting.spatial_mode import plot_spatial_mode
from gMRItensor.plotting.subject_mode import plot_subject_mode
from gMRItensor.plotting.subject_mode import plot_subject_mode_correlation

__all__ = [
    "plot_subject_mode",
    "plot_subject_mode_correlation",
    "plot_spatial_mode",
]
