"""Construct monotonic OCV-SOC curves from extracted OCV points."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


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
