"""Segment HPPC measurement data by current state.
    
    This module identifies consecutive positive current, negative current,
    and rest segments in preidentified HPPC measurement data. It supports
    batch-wise processing and extracts segment-level statictics, including
    duration, current, voltage, and SOC information.
    
    The module also provides a helper to locate current transitions near
    nominal pulse start and stop times.
    
    It does not identify HPPC protocols or calculate higher level HPPC
    parameters such as a pulse resistance or power capacity.
    """

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _current_state(
    current_A: np.ndarray,
    threshold_A: float,
) -> np.ndarray:
    """Classify current samples as negative, rest, or positive current.

    Values below ``-threshold_A`` are classified as ``-1``, values above
    ``threshold_A`` as ``1``, and values within the threshold as ``0``.
    """

    return np.where(
        current_A > threshold_A,
        1,
        np.where(current_A < -threshold_A, -1, 0),
    ).astype(np.int8)


def _aggregate_batch_segments(
    test_index: np.ndarray,
    time_s: np.ndarray,
    current_A: np.ndarray,
    voltage_V: np.ndarray,
    soc: np.ndarray,
    threshold_A: float,
) -> list[dict[str, float | int]]:
    """Compress consecutive current states into measurement segments.

    A new segment starts when the TestIndex changes, the classified current
    state changes, or the measurement time moves backwards.
    """

    valid = np.isfinite(test_index) & np.isfinite(time_s) & np.isfinite(current_A) & np.isfinite(voltage_V)
    if not valid.any():
        return []

    test_index = test_index[valid]
    time_s = time_s[valid]
    current_A = current_A[valid]
    voltage_V = voltage_V[valid]
    soc = soc[valid]

    states = _current_state(
        current_A,
        threshold_A,
    )

    boundary_mask = (test_index[1:] != test_index[:-1]) | (states[1:] != states[:-1]) | (time_s[1:] < time_s[:-1])
    starts = np.r_[
        0,
        np.flatnonzero(boundary_mask) + 1,
    ]
    ends = np.r_[
        starts[1:],
        len(states),
    ]

    segments: list[dict[str, float | int]] = []

    for start, end in zip(
        starts,
        ends,
        strict=True,
    ):
        currents = current_A[start:end]
        soc_values = soc[start:end]
        finite_soc = soc_values[np.isfinite(soc_values)]

        segments.append(
            {
                "TestIndex": float(test_index[start]),
                "state": int(states[start]),
                "start_time_s": float(time_s[start]),
                "last_time_s": float(time_s[end - 1]),
                "n_samples": int(end - start),
                "current_sum": float(currents.sum()),
                "current_sumsq": float(np.square(currents).sum()),
                "current_min_A": float(currents.min()),
                "current_max_A": float(currents.max()),
                "voltage_start_V": float(voltage_V[start]),
                "voltage_end_V": float(voltage_V[end - 1]),
                "soc_start": (float(soc_values[0]) if np.isfinite(soc_values[0]) else np.nan),
                "soc_end": (float(soc_values[-1]) if np.isfinite(soc_values[-1]) else np.nan),
                "soc_valid_count": int(len(finite_soc)),
                "soc_physical_count": int(np.count_nonzero((finite_soc >= 0.0) & (finite_soc <= 1.0))),
                "soc_min": (float(finite_soc.min()) if len(finite_soc) else np.nan),
                "soc_max": (float(finite_soc.max()) if len(finite_soc) else np.nan),
            }
        )

    return segments


def _merge_segment(
    active: dict[str, float | int] | None,
    segment: dict[str, float | int],
) -> tuple[
    dict[str, float | int] | None,
    dict[str, float | int] | None,
]:
    """Merge matching segments that continue across data batches.

    The first returned value is the currently active segment. The second
    returned value is a completed segment, or ``None`` if the active segment
    is still continuing.
    """

    if active is None:
        return segment.copy(), None

    same_run = (
        float(active["TestIndex"]) == float(segment["TestIndex"])
        and int(active["state"]) == int(segment["state"])
        and float(segment["start_time_s"]) >= float(active["last_time_s"])
    )

    if same_run:
        active["last_time_s"] = segment["last_time_s"]
        active["n_samples"] = int(active["n_samples"]) + int(segment["n_samples"])
        active["current_sum"] = float(active["current_sum"]) + float(segment["current_sum"])
        active["current_sumsq"] = float(active["current_sumsq"]) + float(segment["current_sumsq"])
        active["current_min_A"] = min(
            float(active["current_min_A"]),
            float(segment["current_min_A"]),
        )
        active["current_max_A"] = max(
            float(active["current_max_A"]),
            float(segment["current_max_A"]),
        )
        active["voltage_end_V"] = segment["voltage_end_V"]
        active["soc_end"] = segment["soc_end"]
        active["soc_valid_count"] = int(active["soc_valid_count"]) + int(segment["soc_valid_count"])
        active["soc_physical_count"] = int(active["soc_physical_count"]) + int(segment["soc_physical_count"])

        if np.isfinite(float(segment["soc_min"])):
            active["soc_min"] = (
                min(
                    float(active["soc_min"]),
                    float(segment["soc_min"]),
                )
                if np.isfinite(float(active["soc_min"]))
                else segment["soc_min"]
            )
            active["soc_max"] = (
                max(
                    float(active["soc_max"]),
                    float(segment["soc_max"]),
                )
                if np.isfinite(float(active["soc_max"]))
                else segment["soc_max"]
            )

        return active, None

    finished = active

    # Use the next segment's start time as the end boundary of the
    # preceding segment when both belong to the same TestIndex.
    if float(active["TestIndex"]) == float(segment["TestIndex"]):
        finished["duration_s"] = max(
            0.0,
            float(segment["start_time_s"]) - float(active["start_time_s"]),
        )
    else:
        finished["duration_s"] = max(
            0.0,
            float(active["last_time_s"]) - float(active["start_time_s"]),
        )

    return segment.copy(), finished


def _finalize_run(
    run: dict[str, float | int],
    cell_id: str,
    run_index: int,
) -> dict[str, float | int | str]:
    """Calculate final statistics for one current-state segment."""

    # Prevent division by zero if the sample count is unexpectedly zero.
    n_samples = max(
        1,
        int(run["n_samples"]),
    )
    mean_current_A = float(run["current_sum"]) / n_samples

    # Clip small negative variance values caused by flooting-point errors.
    variance = max(
        0.0,
        float(run["current_sumsq"]) / n_samples - mean_current_A**2,
    )
    duration_s = float(
        run.get(
            "duration_s",
            max(
                0.0,
                float(run["last_time_s"]) - float(run["start_time_s"]),
            ),
        )
    )

    state = int(run["state"])
    state_labels = {
        -1: "negative",
        0: "rest",
        1: "positive",
    }

    return {
        "cell_id": cell_id,
        "TestIndex": float(run["TestIndex"]),
        "run_index": run_index,
        "state": state,
        "state_label": state_labels[state],
        "start_time_s": float(run["start_time_s"]),
        "end_time_s": float(run["last_time_s"]),
        "duration_s": duration_s,
        "n_samples": n_samples,
        "mean_current_A": mean_current_A,
        "current_std_A": float(np.sqrt(variance)),
        "current_min_A": float(run["current_min_A"]),
        "current_max_A": float(run["current_max_A"]),
        "voltage_start_V": float(run["voltage_start_V"]),
        "voltage_end_V": float(run["voltage_end_V"]),
        "soc_start": float(run["soc_start"]),
        "soc_end": float(run["soc_end"]),
        "soc_valid_count": int(run["soc_valid_count"]),
        "soc_physical_count": int(run["soc_physical_count"]),
        "soc_min": float(run["soc_min"]),
        "soc_max": float(run["soc_max"]),
    }


def extract_current_segments(
    df: pd.DataFrame,
    cell_id: str,
    current_threshold_A: float = 0.05,
    batch_size: int = 500_000,
) -> pd.DataFrame:
    """Extract consecutive current-state segments from measurement data.

    The input must use the standardized PyDPEET columns ``TestIndex``,
    ``Test_Time[s]``, ``Current[A]``, and ``Voltage[V]``. The ``SOC``
    column is optional. Input row order is preserved.

    The caller must provide data that has already been identified as an HPPC block.
    This function does not infer the measurement protocol.

    Parameters
    ----------
    df: pd. DataFrame
        Input HPPC measurement data.
    cell_id: str
        Identifier or the battery cell.
    current_threshold_A: float, default 0.05
        Absolute current threshold used to classify rest states.
    batch_size: int, default 500_000
        Number of input rows processed per batch.

    Returns
    -------
    pd. DateFrame
        Segment-level statistics, including current state, duration,
        current statistics, boundary voltages, and SOC information.
    """

    # Validate input data and parameters before processing.
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError(
            "df must contain at least one HPPC measurement row."
        )

    required_columns = {
        "TestIndex",
        "Test_Time[s]",
        "Current[A]",
        "Voltage[V]",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if current_threshold_A <= 0:
        raise ValueError("current_threshold_A must be greater than 0.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    

    # Select only the required measurement columns to limit memory usgae.
    columns = [
        "TestIndex",
        "Test_Time[s]",
        "Current[A]",
        "Voltage[V]",
    ]
    has_soc = "SOC" in df.columns
    if has_soc:
        columns.append("SOC")

    data = df[columns].copy()

    # Convert measurement values to numeric types, treating invalid
    # values and missing SOC data as NaN.
    for column in [
        "TestIndex",
        "Test_Time[s]",
        "Current[A]",
        "Voltage[V]",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if has_soc:
        data["SOC"] = pd.to_numeric(
            data["SOC"],
            errors="coerce",
        )
    else:
        data["SOC"] = np.nan

    active: dict[str, float | int] | None = None
    rows: list[dict[str, float | int | str]] = []
    run_counter: defaultdict[float, int] = defaultdict(int)


    # Process data in batches to limit temporary memory usage.
    # Segement continuity is maintained across batch boundaried.
    for start in range(
        0,
        len(data),
        batch_size,
    ):
        batch = data.iloc[start : start + batch_size]

        test_index = batch["TestIndex"].to_numpy(dtype=float)
        time_s = batch["Test_Time[s]"].to_numpy(dtype=float)
        current_A = batch["Current[A]"].to_numpy(dtype=float)
        voltage_V = batch["Voltage[V]"].to_numpy(dtype=float)
        soc = batch["SOC"].to_numpy(dtype=float)

        segments = _aggregate_batch_segments(
            test_index=test_index,
            time_s=time_s,
            current_A=current_A,
            voltage_V=voltage_V,
            soc=soc,
            threshold_A=current_threshold_A,
        )

        for segment in segments:
            active, finished = _merge_segment(
                active,
                segment,
            )

            # Finalize a segment when it can no longer be merged
            # with the next segment.
            if finished is not None:
                test_index_value = float(finished["TestIndex"])
                run_counter[test_index_value] += 1
                rows.append(
                    _finalize_run(
                        finished,
                        cell_id,
                        run_counter[test_index_value],
                    )
                )

    # Finalize the last active segment, which has no subsequent
    # segment to trigger its completion.
    if active is not None:
        test_index_value = float(active["TestIndex"])
        run_counter[test_index_value] += 1
        rows.append(
            _finalize_run(
                active,
                cell_id,
                run_counter[test_index_value],
            )
        )

    return pd.DataFrame(rows)


def _event_target_position(
    time_s: np.ndarray,
    current_A: np.ndarray,
    event_time_s: float,
    event_kind: str,
    pulse_current_A: float,
    search_before_s: float = 1.0,
    search_after_s: float = 5.0,
    min_current_step_A: float = 0.5,
) -> int | None:
    """Locate the actual current transition near a nominal pulse event.
    
    Search for significant current changes near the cexpected pulse
    start or stop time. A valid transition must move toward the pulse
    current at the start, or away from it at the stop.
    
    The input time array must be sorted in ascending order.
    
    Parameters
    ----------
    time_s: np.ndarray
        Measurement timestamps in seconds.
    current_A: np.ndarray
        Measured current values in amperes.
    event_kind: str
        Event type, either "start" or "stop".
    pulse_current_A: float
        Reference current of the pulse in amperes. 
    search_before_s: float, default 1.0
        Search window before the nominal event time in seconds.
    search_after_s: float, default 5.0
        Search window after the nominal event time in seconds.
    min_current_step_A: float, default 0.5
        Minimum absolute current change required for a valid transition.
        
    Returns
    -------
    int or None
        Index of the nearest matching current transition, or None
        if no valid transition is found.
    """

    if event_kind not in {"start", "stop"}:
        raise ValueError(
            "event_kind must be either 'start' or 'stop'."
        )

    lower = max(
        1,
        int(
            np.searchsorted(
                time_s,
                event_time_s - search_before_s,
                side="left",
            )
        ),
    )
    upper = min(
        len(time_s),
        int(
            np.searchsorted(
                time_s,
                event_time_s + search_after_s,
                side="right",
            )
        ),
    )

    if lower >= upper:
        return None

    candidates = np.arange(
        lower,
        upper,
        dtype=int,
    )
    delta_current_A = (
        current_A[candidates]
        - current_A[candidates - 1]
    )
    before_distance_A = np.abs(
        current_A[candidates - 1]
        - pulse_current_A
    )
    after_distance_A = np.abs(
        current_A[candidates]
        - pulse_current_A
    )

    significant = (
        np.abs(delta_current_A)
        >= min_current_step_A
    )

    if event_kind == "start":
        matches_pulse = (
            after_distance_A
            < before_distance_A
        )
    else:
        matches_pulse = (
            before_distance_A
            < after_distance_A
        )

    candidates = candidates[
        significant & matches_pulse
    ]

    if len(candidates) == 0:
        return None

    nearest = np.argmin(
        np.abs(
            time_s[candidates]
            - event_time_s
        )
    )

    return int(candidates[nearest])