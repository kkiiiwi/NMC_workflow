"""Construct monotonic OCV-SOC curves from extracted OCV points."""

from __future__ import annotations

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
    """Return the weighted non-decreasing isotonic fit of ``values``."""
    block_values: list[float] = []
    block_weights: list[float] = []
    block_starts: list[int] = []
    block_ends: list[int] = []

    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_starts.append(index)
        block_ends.append(index)

        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (block_values[-2] * block_weights[-2] + block_values[-1] * block_weights[-1]) / merged_weight
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            block_starts[-2:] = [block_starts[-2]]
            block_ends[-2:] = [block_ends[-1]]

    fitted = np.empty(len(values), dtype=float)
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

    Points are grouped into SOC bins. The median voltage in every occupied bin
    is made non-decreasing with weighted pool-adjacent-violators regression
    (PAVA), using the number of points in each bin as its weight. A
    shape-preserving PCHIP interpolator then evaluates the curve only across
    the observed SOC range; no extrapolation is performed.

    Charge and discharge points should be passed separately so their voltage
    hysteresis remains visible.

    Parameters
    ----------
    df_ocv_points : pandas.DataFrame
        Extracted OCV points containing SOC and voltage columns.
    bin_width : float, default 0.01
        Width of the SOC bins, expressed on the normalized interval [0, 1].
    grid_step : float, default 0.001
        SOC spacing of the interpolated output curve. Must not exceed
        ``bin_width``.
    soc_column : str, default "SOC"
        Name of the normalized SOC column.
    voltage_column : str, default "Voltage[V]"
        Name of the OCV-point voltage column.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The first DataFrame contains bin statistics and the PAVA-adjusted
        voltage. The second contains the PCHIP-interpolated ``SOC`` and
        ``OCV[V]`` curve.

    Raises
    ------
    ValueError
        If parameters or required columns are invalid, or fewer than two SOC
        bins contain usable finite data.
    """
    if not isinstance(df_ocv_points, pd.DataFrame):
        raise ValueError("df_ocv_points must be a pandas DataFrame.")
    missing = [column for column in (soc_column, voltage_column) if column not in df_ocv_points.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not 0 < bin_width <= 0.1:
        raise ValueError("bin_width must be greater than 0 and at most 0.1.")
    if not 0 < grid_step <= bin_width:
        raise ValueError("grid_step must be greater than 0 and at most bin_width.")

    points = df_ocv_points[[soc_column, voltage_column]].copy()
    points[soc_column] = pd.to_numeric(points[soc_column], errors="coerce")
    points[voltage_column] = pd.to_numeric(points[voltage_column], errors="coerce")
    finite = np.isfinite(points[soc_column]) & np.isfinite(points[voltage_column])
    points = points.loc[finite & points[soc_column].between(0.0, 1.0, inclusive="both")].copy()
    if points.empty:
        raise ValueError("No finite OCV points within the normalized SOC range [0, 1].")

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

    bin_indices = binned["soc_bin"].astype(int).to_numpy()
    binned["SOC"] = (edges[bin_indices] + edges[bin_indices + 1]) / 2.0
    binned["ocv_monotonic_V"] = _weighted_pava(
        binned["ocv_median_V"].to_numpy(float),
        binned["n"].to_numpy(float),
    )

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
    rest = df_primitives.loc[
        df_primitives["ID"].isin(point_ids) & df_primitives["Type"].eq("Rest"),
        ["ID", time_column, voltage_column],
    ].copy()
    rest[time_column] = pd.to_numeric(rest[time_column], errors="coerce")
    rest[voltage_column] = pd.to_numeric(rest[voltage_column], errors="coerce")
    rest = rest.dropna(subset=[time_column, voltage_column])

    rows: list[dict[str, float | int | object]] = []
    for segment_id, group in rest.groupby("ID", sort=False):
        group = group.sort_values(time_column)
        time_s = group[time_column].to_numpy(float)
        voltage_v = group[voltage_column].to_numpy(float)
        end_s = float(time_s[-1])
        tail = time_s >= end_s - tail_window_s
        tail_time = time_s[tail]
        tail_voltage = voltage_v[tail]

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
    result = points.copy()
    soc = pd.to_numeric(result[soc_column], errors="coerce")
    voltage = pd.to_numeric(result[voltage_column], errors="coerce")
    current = pd.to_numeric(result[current_column], errors="coerce")
    rest_duration = pd.to_numeric(result["rest_duration_s"], errors="coerce")
    tail_samples = pd.to_numeric(result["tail_samples"], errors="coerce")
    tail_coverage = pd.to_numeric(result["tail_coverage_s"], errors="coerce")
    tail_span = pd.to_numeric(result["tail_voltage_span_mV"], errors="coerce")
    tail_slope = pd.to_numeric(result["tail_voltage_slope_mV_per_min"], errors="coerce")

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
    for name, failed in checks.items():
        result[f"qc_{name}"] = failed

    qc_columns = [f"qc_{name}" for name in checks]
    result["is_valid_for_ocv"] = ~result[qc_columns].any(axis=1)
    reason_by_column = {f"qc_{name}": name for name in checks}
    result["exclusion_reason"] = result.apply(
        lambda row: ";".join(reason for column, reason in reason_by_column.items() if bool(row[column])),
        axis=1,
    )
    return result


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
