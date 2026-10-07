"""Construct monotonic OCV-SOC curves from extracted OCV points."""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

_QUALITY_COLUMNS = [
    "rest_duration_s",
    "tail_samples",
    "tail_coverage_s",
    "tail_voltage_span_mV",
    "tail_voltage_slope_mV_per_min",
]


def _weighted_pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Fit a non-decreasing sequence using weighted isotonic regression.
    
    Adjacent blocks that violate the monotonicity constraint are merged and replaced by their weighted mean
    until the sequence is non-decreasing.
    In the OCV workflow, each weight corresponds to the number of measurements in the associated SOC bin."""
    
    block_values: list[float] = []
    block_weights: list[float] = []
    block_starts: list[int] = []
    block_ends: list[int] = []

    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        
        # Treat each observation as an individual block initially.
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_starts.append(index)
        block_ends.append(index)

        # Merge adjacent violating blocks until the sequence is non-decreasing.
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (block_values[-2] * block_weights[-2] + block_values[-1] * block_weights[-1]) / merged_weight
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            block_starts[-2:] = [block_starts[-2]]
            block_ends[-2:] = [block_ends[-1]]

    fitted = np.empty(len(values), dtype=float)

    # Expand fitted blocks back to the original indices.
    for value, start, end in zip(block_values, block_starts, block_ends, strict=True):
        fitted[start : end + 1] = value
    return fitted


def fit_ocv_soc_curve(
    df_ocv_points: pd.DataFrame,
    bin_width: float = 0.01,
    grid_step: float = 0.001,
    soc_column: str = "SOC",
    voltage_column: str = "Voltage[V]",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """Build a monotonic OCV-SOC curve from extracted OCV points.

   OCV points are grouped into SOC bins, and the median voltages are made non-decreasing using weighted PAVA.
   The resulting values are interpolated with PCHIP over the observed SOC range.

   Charge and discharge points should be processed separately to preserve voltage hysteresis.
    """

    # Validate input data and fitting parameters
    if not isinstance(df_ocv_points, pd.DataFrame):
        raise ValueError("df_ocv_points must be a pandas DataFrame.")
    missing = [column for column in (soc_column, voltage_column) if column not in df_ocv_points.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not 0 < bin_width <= 0.1:
        raise ValueError("bin_width must be greater than 0 and at most 0.1.")
    if not 0 < grid_step <= bin_width:
        raise ValueError("grid_step must be greater than 0 and at most bin_width.")

    # Keep only finite OCV points within the normalized SOC range.
    points = df_ocv_points[[soc_column, voltage_column]].copy()
    points[soc_column] = pd.to_numeric(points[soc_column], errors="coerce")
    points[voltage_column] = pd.to_numeric(points[voltage_column], errors="coerce")
    finite = np.isfinite(points[soc_column]) & np.isfinite(points[voltage_column])
    points = points.loc[finite & points[soc_column].between(0.0, 1.0, inclusive="both")].copy()
    if points.empty:
        raise ValueError("No finite OCV points within the normalized SOC range [0, 1].")

    # Group OCV point into SOC bins and calculate robust voltage statistics.
    edges = np.arange(0.0, 1.0, bin_width)
    edges = np.append(edges, 1.0)
    edges = np.unique(np.clip(edges, 0.0, 1.0))
    points["soc_bin"] = pd.cut(
        points[soc_column],
        bins=edges,
        include_lowest=True,
        labels=False,
    )
    binned = (
        points.groupby("soc_bin", observed=True)[voltage_column]
        .agg(
            n="size",
            ocv_median_V="median",
            ocv_q25_V=lambda values: values.quantile(0.25),
            ocv_q75_V=lambda values: values.quantile(0.75),
        )
        .reset_index()
        .sort_values("soc_bin")
        .reset_index(drop=True)
    )
    if len(binned) < 2:
        raise ValueError("At least two SOC bins must contain usable OCV points.")

    # Enforce a non-decreasing OCV-SOC relationship using weighted PAVA.
    bin_indices = binned["soc_bin"].astype(int).to_numpy()
    binned["SOC"] = (edges[bin_indices] + edges[bin_indices + 1]) / 2.0
    binned["ocv_monotonic_V"] = _weighted_pava(
        binned["ocv_median_V"].to_numpy(float),
        binned["n"].to_numpy(float),
    )

    # Build and evalute the PCHIP curve over the observed SOC range.
    soc = binned["SOC"].to_numpy(float)
    voltage = binned["ocv_monotonic_V"].to_numpy(float)
    grid = np.arange(soc.min(), soc.max() + grid_step * 0.5, grid_step)
    grid = grid[grid <= soc.max() + np.finfo(float).eps * 8]
    if grid[-1] < soc.max() - np.finfo(float).eps * 8:
        grid = np.append(grid, soc.max())

    curve = pd.DataFrame(
        {
            "SOC": grid,
            "OCV[V]": PchipInterpolator(soc, voltage, extrapolate=False)(grid),
        }
    )
    return binned, curve


def _rest_quality_metrics(
    df_primitives: pd.DataFrame,
    point_ids: set[object],
    tail_window_s: float,
    time_column: str,
    voltage_column: str,
) -> pd.DataFrame:
    
    """Calculate rest-end stability metrics for extracted OCV points.
    For each selected rest segment, the total rest duration and voltage stability over the final tail_window_s seconds are calculated. 
    
    This function only calculates quality metrics and does not accept or reject OCV points.
    """

    # Select only the rest segments associated with extracted OCV points.
    rest = df_primitives.loc[
        df_primitives["ID"].isin(point_ids) & df_primitives["Type"].eq("Rest"),
        ["ID", time_column, voltage_column],
    ].copy()

    # Remove samples whose time or voltage cannot be interpreted numerically.
    rest[time_column] = pd.to_numeric(rest[time_column], errors="coerce")
    rest[voltage_column] = pd.to_numeric(rest[voltage_column], errors="coerce")
    rest = rest.dropna(subset=[time_column, voltage_column])

    rows: list[dict[str, float | int | object]] = []
    for segment_id, group in rest.groupby("ID", sort=False):
        group = group.sort_values(time_column)
        time_s = group[time_column].to_numpy(float)
        voltage_v = group[voltage_column].to_numpy(float)
        end_s = float(time_s[-1])

        # Select samples within the final tail_window_s seconds.
        tail = time_s >= end_s - tail_window_s
        tail_time = time_s[tail]
        tail_voltage = voltage_v[tail]
        

        # Fit the voltage slope only when at least three samples and two distinct timestamps are available.
        slope = np.nan
        if len(tail_time) >= 3 and np.ptp(tail_time) > 0:
            slope = float(np.polyfit(tail_time - tail_time[0], tail_voltage, 1)[0] * 1000.0 * 60.0)
        rows.append(
            {
                "ID": segment_id,
                "rest_duration_s": float(time_s[-1] - time_s[0]),
                "tail_samples": len(tail_time),
                "tail_coverage_s": float(tail_time[-1] - tail_time[0]),
                "tail_voltage_span_mV": float(np.ptp(tail_voltage) * 1000.0),
                "tail_voltage_slope_mV_per_min": slope,
            }
        )
    return pd.DataFrame(rows, columns=["ID", *_QUALITY_COLUMNS])


def _screen_ocv_points(
    points: pd.DataFrame,
    min_rest_duration_s: float,
    min_tail_coverage_s: float,
    min_tail_samples: int,
    max_tail_voltage_span_mV: float,
    max_abs_tail_voltage_slope_mV_per_min: float,
    max_endpoint_current_A: float,
    soc_column: str,
    voltage_column: str,
    current_column: str,
) -> pd.DataFrame:

    """Evaluate extracted OCV points against quality criteria.
    
    Each quality check is stored as a qc_* flag. The function also adds an overall validity flag
    and records the reasons for failed checks.
    
    All input points are retained. Failed points are annotated rather than removed.
    """

    # Work on a copy so QC annotations do not modify the input table.
    result = points.copy()

    # Convert relevant columns to numeric values for consistent QC checks.
    soc = pd.to_numeric(result[soc_column], errors="coerce")
    voltage = pd.to_numeric(result[voltage_column], errors="coerce")
    current = pd.to_numeric(result[current_column], errors="coerce")
    rest_duration = pd.to_numeric(result["rest_duration_s"], errors="coerce")
    tail_samples = pd.to_numeric(result["tail_samples"], errors="coerce")
    tail_coverage = pd.to_numeric(result["tail_coverage_s"], errors="coerce")
    tail_span = pd.to_numeric(result["tail_voltage_span_mV"], errors="coerce")
    tail_slope = pd.to_numeric(result["tail_voltage_slope_mV_per_min"], errors="coerce")

    # Define the failure condition for each quality criterion. 
    checks = {
        "soc_missing_or_outside_0_1": ~np.isfinite(soc) | ~soc.between(0.0, 1.0, inclusive="both"),
        "voltage_missing": ~np.isfinite(voltage),
        "endpoint_current_too_large": ~np.isfinite(current) | current.abs().gt(max_endpoint_current_A),
        "rest_too_short": ~np.isfinite(rest_duration) | rest_duration.lt(min_rest_duration_s),
        "tail_coverage_too_short": ~np.isfinite(tail_coverage) | tail_coverage.lt(min_tail_coverage_s),
        "too_few_tail_samples": ~np.isfinite(tail_samples) | tail_samples.lt(min_tail_samples),
        "tail_voltage_span_too_large": ~np.isfinite(tail_span) | tail_span.gt(max_tail_voltage_span_mV),
        "tail_voltage_slope_too_large": (
            ~np.isfinite(tail_slope) | tail_slope.abs().gt(max_abs_tail_voltage_slope_mV_per_min)
        ),
        "unknown_direction": ~result["direction"].isin({"charge", "discharge"}),
    }

    # Store each failed quality check as a separate qc_* flag.
    for name, failed in checks.items():
        result[f"qc_{name}"] = failed

    qc_columns = [f"qc_{name}" for name in checks]

    # A point is valid only if none of the quality checks has failed.
    result["is_valid_for_ocv"] = ~result[qc_columns].any(axis=1)
    reason_by_column = {f"qc_{name}": name for name in checks}

    # Record all failed checks as exclusion reasons for traceability.
    result["exclusion_reason"] = result.apply(
        lambda row: ";".join(reason for column, reason in reason_by_column.items() if bool(row[column])),
        axis=1,
    )
    return result


def _plot_ocv_curves(
    points: pd.DataFrame,
    curves: pd.DataFrame,
    soc_column: str,
    voltage_column: str,
) -> None:
    """Show screened points and fitted curves separately for each iOCV block."""
    block_ids = points["ocv_block_id"].drop_duplicates().tolist()
    ncols = min(2, len(block_ids))
    nrows = (len(block_ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    styles = {"charge": ("#0072B2", "-"), "discharge": ("#D55E00", "--")}

    for ax, block_id in zip(axes.flat, block_ids, strict=False):
        block_points = points.loc[points["ocv_block_id"].eq(block_id)]
        block_curves = curves.loc[curves["ocv_block_id"].eq(block_id)]
        for direction, group in block_points.groupby("direction", sort=True):
            color, linestyle = styles.get(direction, ("#777777", ":"))
            soc = pd.to_numeric(group[soc_column], errors="coerce")
            voltage = pd.to_numeric(group[voltage_column], errors="coerce")
            finite = np.isfinite(soc) & np.isfinite(voltage)
            kept = finite & group["is_valid_for_ocv"]
            excluded = finite & ~group["is_valid_for_ocv"]
            if kept.any():
                ax.scatter(
                    soc.loc[kept],
                    voltage.loc[kept],
                    facecolors="none",
                    edgecolors=color,
                    s=24,
                    label=f"{direction.capitalize()} kept points",
                    zorder=3,
                )
            if excluded.any():
                ax.scatter(
                    soc.loc[excluded],
                    voltage.loc[excluded],
                    color="#777777",
                    marker="x",
                    s=32,
                    label=f"{direction.capitalize()} excluded points",
                    zorder=3,
                )
            fitted = block_curves.loc[block_curves["direction"].eq(direction)].sort_values("SOC")
            if not fitted.empty:
                ax.plot(
                    fitted["SOC"],
                    fitted["OCV[V]"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=f"{direction.capitalize()} fit",
                )
        ax.set_title(f"iOCV block {block_id}")
        ax.set_xlabel("SOC")
        ax.set_ylabel("OCV [V]")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(frameon=False)

    for ax in axes.flat[len(block_ids) :]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.show()


def create_ocv_curves(
    ocv_blocks: list[pd.DataFrame],
    df_primitives: pd.DataFrame,
    tail_window_s: float = 60.0,
    min_rest_duration_s: float = 300.0,
    min_tail_coverage_s: float = 55.0,
    min_tail_samples: int = 3,
    max_tail_voltage_span_mV: float = 3.0,
    max_abs_tail_voltage_slope_mV_per_min: float = 1.5,
    max_endpoint_current_A: float = 0.075,
    bin_width: float = 0.01,
    grid_step: float = 0.001,
    soc_column: str = "SOC",
    voltage_column: str = "Voltage[V]",
    current_column: str = "Current[A]",
    time_column: str = "Test_Time[s]",
    visualize: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quality-screen iOCV blocks and build separate charge/discharge curves.

    The input blocks are the output of :func:`extract_ocv_iocv`. For every
    block, the function measures voltage stability over the end of each rest,
    applies configurable point-level quality checks, and fits a monotonic
    OCV-SOC curve to the remaining points with :func:`fit_ocv_soc_curve`.

    Returns the screened points, SOC-bin statistics, and interpolated curves.
    Every output retains ``ocv_block_id`` and ``direction`` so charge,
    discharge, and separate iOCV check-ups are never mixed.
    """
    if not isinstance(ocv_blocks, list) or not ocv_blocks:
        raise ValueError("ocv_blocks must be a non-empty list of DataFrames.")
    if not isinstance(df_primitives, pd.DataFrame):
        raise ValueError("df_primitives must be a pandas DataFrame.")

    numeric_limits = {
        "tail_window_s": tail_window_s,
        "min_rest_duration_s": min_rest_duration_s,
        "min_tail_coverage_s": min_tail_coverage_s,
        "max_tail_voltage_span_mV": max_tail_voltage_span_mV,
        "max_abs_tail_voltage_slope_mV_per_min": max_abs_tail_voltage_slope_mV_per_min,
        "max_endpoint_current_A": max_endpoint_current_A,
    }
    if tail_window_s <= 0:
        raise ValueError("tail_window_s must be greater than 0.")
    if any(value < 0 for name, value in numeric_limits.items() if name != "tail_window_s"):
        raise ValueError("OCV quality limits must be non-negative.")
    if min_tail_coverage_s > tail_window_s:
        raise ValueError("min_tail_coverage_s must not exceed tail_window_s.")
    if not isinstance(min_tail_samples, int) or min_tail_samples < 1:
        raise ValueError("min_tail_samples must be a positive integer.")

    required_primitive_columns = {"ID", "Type", time_column, voltage_column}
    missing_primitives = sorted(required_primitive_columns - set(df_primitives.columns))
    if missing_primitives:
        raise ValueError(f"df_primitives is missing required columns: {missing_primitives}")

    required_point_columns = {"ID", "iOCV_type", soc_column, voltage_column, current_column}
    prepared_blocks: list[pd.DataFrame] = []
    for block_id, block in enumerate(ocv_blocks, start=1):
        if not isinstance(block, pd.DataFrame) or block.empty:
            raise ValueError(f"OCV block {block_id} must be a non-empty DataFrame.")
        missing_points = sorted(required_point_columns - set(block.columns))
        if missing_points:
            raise ValueError(f"OCV block {block_id} is missing required columns: {missing_points}")
        prepared = block.copy()
        prepared.insert(0, "ocv_block_id", block_id)
        prepared["direction"] = prepared["iOCV_type"].astype("string").str.lower().fillna("unknown")
        prepared_blocks.append(prepared)

    points = pd.concat(prepared_blocks, ignore_index=True)
    metrics = _rest_quality_metrics(
        df_primitives=df_primitives,
        point_ids=set(points["ID"].dropna()),
        tail_window_s=tail_window_s,
        time_column=time_column,
        voltage_column=voltage_column,
    )
    points = points.merge(metrics, on="ID", how="left", validate="many_to_one")
    points = _screen_ocv_points(
        points=points,
        min_rest_duration_s=min_rest_duration_s,
        min_tail_coverage_s=min_tail_coverage_s,
        min_tail_samples=min_tail_samples,
        max_tail_voltage_span_mV=max_tail_voltage_span_mV,
        max_abs_tail_voltage_slope_mV_per_min=max_abs_tail_voltage_slope_mV_per_min,
        max_endpoint_current_A=max_endpoint_current_A,
        soc_column=soc_column,
        voltage_column=voltage_column,
        current_column=current_column,
    )

    binned_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []
    for (block_id, direction), group in points.groupby(["ocv_block_id", "direction"], sort=True):
        valid = group.loc[group["is_valid_for_ocv"]]
        try:
            binned, curve = fit_ocv_soc_curve(
                valid,
                bin_width=bin_width,
                grid_step=grid_step,
                soc_column=soc_column,
                voltage_column=voltage_column,
            )
        except ValueError as exc:
            raise ValueError(f"Could not fit OCV block {block_id} ({direction}): {exc}") from exc
        for table in (binned, curve):
            table.insert(0, "direction", direction)
            table.insert(0, "ocv_block_id", int(block_id))
        binned_parts.append(binned)
        curve_parts.append(curve)

    return (
        points,
        pd.concat(binned_parts, ignore_index=True),
        pd.concat(curve_parts, ignore_index=True),
    )
