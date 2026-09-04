from multiprocessing import Pool
from pathlib import Path
from typing import Callable
from typing import cast

import nibabel as nib
import numexpr as ne
import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image
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
    baseline_nifti = cast(Nifti1Image, nib.load(baseline_path))
    post_injection_nifti = cast(Nifti1Image, nib.load(post_injection_path))
    mask_nifti = cast(Nifti1Image, nib.load(mask_path))

    # Verify that images align so that we get sensible results
    if not np.allclose(baseline_nifti.affine, post_injection_nifti.affine):
        raise ValueError("Baseline and post-injection images are not aligned")
    if not np.allclose(baseline_nifti.affine, mask_nifti.affine):
        raise ValueError("Baseline and mask images are not aligned")

    mask = mask_nifti.get_fdata()
    tracer = compute_tracer(
        baseline_nifti.get_fdata()[mask > 0],
        post_injection_nifti.get_fdata()[mask > 0],
        signal_type,
    )

    if segmentation_path is not None:
        segmentation_nifti = cast(Nifti1Image, nib.load(segmentation_path))
        if not np.allclose(baseline_nifti.affine, segmentation_nifti.affine):
            raise ValueError("Baseline and segmentation images are not aligned")
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


def _pivot_tracer_df(
    df: pd.DataFrame,
    group_filtering: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Filter by group and pivot the long-format tracer DataFrame.

    Shared first step of `prepare_tensor` and `prepare_parafac2_slices`:
    optionally filters `df` to one group, then pivots to a DataFrame indexed
    by (subject, time_point) with one column per label. Only (subject,
    time_point) combinations actually present in `df` get a row -- no
    NaN-filled rows are introduced for combinations that were never observed.
    """
    if group_filtering is not None:
        df = df.query(f"{group_filtering[0]}=='{group_filtering[1]}'")

    return df.pivot_table(
        index=["subject", "time_point"],
        columns="labels",
        values="values",
        aggfunc="first",  # Handles single value per cell
    )


def prepare_tensor(df: pd.DataFrame, group_filtering: tuple[str, str] | None = None):
    pivot_df = _pivot_tracer_df(df, group_filtering)

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


def prepare_parafac2_slices(
    df: pd.DataFrame,
    group_filtering: tuple[str, str] | None = None,
    min_timepoints: int = 2,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    """Prepare ragged per-subject tensor slices for PARAFAC2 decomposition.

    Unlike `prepare_tensor`, subjects are not forced to share the same set of
    time points: only (subject, time_point) combinations that are actually
    observed are used, so one subject's missing scan no longer forces a
    spatial region to be dropped for every subject. The label (region) set
    still needs to be complete and shared across all subjects/time points --
    that constraint is unavoidable, since PARAFAC2 still requires a regular
    mode 2 (region).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format tracer DataFrame (as produced by `compute_tracer_parallel`),
        with `subject`, `time_point`, `labels`, `values` columns.
    group_filtering : tuple[str, str] | None, optional
        `(column, value)` to filter `df` to a single group before pivoting.
        By default None.
    min_timepoints : int, optional
        Minimum number of observed time points required to keep a subject;
        subjects with fewer are dropped and reported (PARAFAC2's evolving
        mode needs at least a couple of points per subject to be meaningful).
        By default 2.

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]
        `(slices, subjects, timepoints_per_subject, labels)`. `slices[i]` has
        shape `(len(timepoints_per_subject[i]), len(labels))`, ordered by
        `timepoints_per_subject[i]`.
    """
    pivot_df = _pivot_tracer_df(df, group_filtering)

    # Drop any label with a NaN among the *actually observed* (subject,
    # time_point) rows -- unlike prepare_tensor, a subject's missing time
    # point never shows up here as a NaN row, so it can no longer force an
    # otherwise well-observed label to be dropped for every subject.
    valid_pivot = pivot_df.dropna(axis=1, how="any")
    labels = np.array(valid_pivot.columns.tolist()).astype(int)

    subjects = []
    timepoints_per_subject = []
    slices = []
    dropped_subjects = []
    for subject, subject_df in valid_pivot.groupby(level="subject"):
        time_points = subject_df.index.get_level_values("time_point").to_numpy()
        order = np.argsort(time_points)
        if len(order) < min_timepoints:
            dropped_subjects.append((subject, len(order)))
            continue
        subjects.append(subject)
        timepoints_per_subject.append(time_points[order].astype(int))
        slices.append(subject_df.to_numpy()[order])

    if dropped_subjects:
        print(
            f"prepare_parafac2_slices: dropped {len(dropped_subjects)} subject(s) "
            f"with fewer than {min_timepoints} observed time points: {dropped_subjects}",
        )

    return slices, np.array(subjects).astype(str), timepoints_per_subject, labels
