from typing import Callable

import numpy as np
import pandas as pd


def scale_mode(arr):
    return arr / np.linalg.norm(arr, ax=0)[None, :]


def plot_subject_mode(subject_mode: np.ndarray, subject_info: pd.DataFrame):
    raise NotImplementedError


def plot_time_mode(time_mode: np.ndarray, time_points: list):
    raise NotImplementedError


def plot_spatial_mode(
    spatial_mode: np.ndarray,
    index_list,
    csf_indices,
    parenchyma_indices,
    background,
    slices: list,
    transform: Callable,
):
    raise NotImplementedError
