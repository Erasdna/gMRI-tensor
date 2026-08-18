from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import nibabel as nib
import numexpr as ne
import numpy as np
import pandas as pd
from scipy.ndimage import labeled_comprehension
from tqdm.contrib import tenumerate


def compute_tracer(baseline: np.ndarray, post_injection: np.ndarray, signal_type: str):

    if signal_type == "T1map":
        expr = "where((abs(post_injection) < 1e-6) | (abs(baseline) < 1e-6), nan, (1 / post_injection) - (1 / baseline))"  # noqa: E501
    elif signal_type == "T1w":
        expr = "where(abs(baseline) < 1e-6, nan, post_injection / baseline)"
    else:
        raise ValueError("Invalid argument value for 'signal_type'")

    return ne.evaluate(
        expr,
        local_dict={
            "baseline": baseline,
            "post_injection": post_injection,
            "nan": np.nan,
        },
    )


def compute_tracer_from_image(
    baseline_path: Path,
    post_injection_path: Path,
    signal_type: str,
    mask_path: Path,
    segmentation_path: Path | None = None,
    func: Callable = np.nanmedian,
):
    baseline_nifti = nib.load(baseline_path)
    post_injection_nifti = nib.load(post_injection_path)
    mask_nifti = nib.load(mask_path)
    segmentation_nifti = nib.load(segmentation_path)

    # Verify that images align so that we get sensible results
    assert np.allclose(baseline_nifti.affine, post_injection_nifti.affine)
    assert np.allclose(baseline_nifti.affine, mask_nifti.affine)

    mask = mask_nifti.get_fdata()
    tracer = compute_tracer(
        baseline_nifti.get_fdata()[mask > 0],
        post_injection_nifti.get_fdata()[mask > 0],
        signal_type,
    )

    if segmentation_path is not None:
        assert np.allclose(baseline_nifti.affine, segmentation_nifti.affine)
        segmentation = segmentation_nifti.get_fdata()[mask > 0]
        unique_labels = np.unique(segmentation)
        unique_labels = unique_labels[unique_labels > 1e-6]
        values = labeled_comprehension(
            tracer,
            segmentation,
            unique_labels,
            func,
            default=np.nan,
            out_dtype=float,
        )
        return np.rint(unique_labels), values
    else:
        return None, tracer


def _compute_tracer_worker(args):
    return compute_tracer_from_image(
        args["baseline_path"],
        args["post_injection_path"],
        args["signal_type"],
        args["mask_path"],
        args["segmentation_path"],
        func=args["func"],
    )


def compute_tracer_parallel(args_list, n_procs: int = 5):

    results_dict = []
    if n_procs == 1:
        for i, args in tenumerate(
            args_list,
            desc="Computing tracer signal sequential",
        ):
            labels, values = _compute_tracer_worker(args)
            tmp_dict = {
                "labels": labels,
                "values": values,
                "subject": args["subject"],
                "time_point": args["time_point"],
            }

            results_dict.append(pd.DataFrame(tmp_dict))
    else:
        ne.set_num_threads(1)
        with Pool(n_procs) as pool:
            for i, (labels, values) in tenumerate(
                pool.imap(_compute_tracer_worker, args_list),
                total=len(args_list),
                desc="Computing tracer signal in parallel",
            ):
                tmp_dict = {
                    "labels": labels,
                    "values": values,
                    "subject": args_list[i]["subject"],
                    "time_point": args_list[i]["time_point"],
                }

                results_dict.append(pd.DataFrame(tmp_dict))
    return pd.concat(results_dict, ignore_index=True)


def prepare_tensor(df: pd.DataFrame, group_filtering: tuple[str, str] | None = None):
    if group_filtering is not None:
        df = df.query(f"{group_filtering[0]}=='{group_filtering[1]}'")

    # Make pivot table
    pivot_df = df.pivot_table(
        index=["subject", "time_point"],
        columns="labels",
        values="values",
        aggfunc="first",  # Handles single value per cell
    )

    # Find all subjects with all time points
    subjects_by_tp = pivot_df.unstack(level="time_point")
    complete_subjects_mask = ~subjects_by_tp.isna().all(axis=1)
    valid_subjects = subjects_by_tp[complete_subjects_mask].index

    # Filter the pivoted dataframe to valid subjects only
    pivot_df = pivot_df.loc[
        pivot_df.index.get_level_values("subject").isin(valid_subjects)
    ]

    # Drop nan values
    valid_pivot = pivot_df.dropna(axis=1, how="any")

    # Find all subjects, time points and labels remaining
    retained_subjects = sorted(valid_pivot.index.get_level_values("subject").unique())
    retained_timepoints = sorted(
        valid_pivot.index.get_level_values("time_point").unique(),
    )
    retained_labels = valid_pivot.columns.tolist()

    # Reshape to (subjects x time_points x labels)
    numpy_3d = valid_pivot.to_numpy().reshape(
        len(retained_subjects),
        len(retained_timepoints),
        len(retained_labels),
    )
    return (
        numpy_3d,
        np.array(retained_subjects).astype(str),
        np.array(retained_timepoints).astype(int),
        np.array(retained_labels).astype(int),
    )
