"""Detect HPPC regions in complete battery measurement data.
    
    This module first divides complete measurement data into consecutive
    positive-current, negative-current, and rest segments. It then identifies
    HPPC regions from repeated pulse-rest-opposite-pulse patterns.

    Pulse duration, rest duration, current threshold, and repetition
    requirements can be configured by the caller. The module also provides a
    helper for locating current transitions near nominal pulse events.
    """

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

SEGMENT_COLUMNS = [
    "cell_id",
    "TestIndex",
    "run_index",
    "state",
    "state_label",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "n_samples",
    "mean_current_A",
    "current_std_A",
    "current_min_A",
    "current_max_A",
    "voltage_start_V",
    "voltage_end_V",
    "soc_start",
    "soc_end",
    "soc_valid_count",
    "soc_physical_count",
    "soc_min",
    "soc_max",
]

DETECTION_COLUMNS = [
    "hppc_block_id",
    "hppc_role",
    "hppc_pair_count",
    "detected_pair_count",
    "block_start_time_s",
    "block_end_time_s",
]


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

    # Clip small negative variance values caused by floating-point errors.
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


def _segment_complete_data(
    df: pd.DataFrame,
    cell_id: str,
    current_threshold_A: float,
    batch_size: int,
) -> pd.DataFrame:
    """Extract consecutive current-state segments from measurement data.

    The input must use the standardized PyDPEET columns ``TestIndex``,
    ``Test_Time[s]``, ``Current[A]``, and ``Voltage[V]``. The ``SOC``
    column is optional. Input row order is preserved.

    The input may contain complete measurement data.
    This function segments current states. HPPC detection is
    performed later.
    This function does not infer the measurement protocol.

    Parameters
    ----------
    df: pd.DataFrame
        Complete battery measurement data.
    cell_id: str
        Identifier of the battery cell.

    Returns
    -------
    pd.DataFrame
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
    

    # Select only the required measurement columns to limit memory usage.
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
    # Segment continuity is maintained across batch boundaries.
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

    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)


def _validate_range(
    name: str,
    values: tuple[float, float],
) -> tuple[float, float]:
    """Validate and return a duration range."""

    if len(values) != 2:
        raise ValueError(
            f"{name} must contain exactly two values."
        )

    lower, upper = map(float, values)

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(
            f"{name} values must be finite."
        )

    if lower < 0.0 or upper < lower:
        raise ValueError(
            f"{name} must satisfy 0 <= lower <= upper."
        )

    return lower, upper


def _similar_value(
    left: float,
    right: float,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    """Check whether two values are sufficiently similar."""

    return abs(left - right) <= max(
        absolute_tolerance,
        relative_tolerance * max(
            abs(left),
            abs(right),
        ),
    )


def _cluster_profile_pairs(
    pairs: pd.DataFrame,
    relative_tolerance: float,
    absolute_tolerance_s: float,
) -> pd.DataFrame:
    """Cluster pulse profiles by direction, duration, and current."""

    if pairs.empty:
        result = pairs.copy()
        result["cluster_id"] = pd.Series(dtype="Int64")
        result["cluster_size"] = pd.Series(dtype="Int64")
        return result

    clusters: list[list[int]] = []
    centers: list[dict[str, float]] = []

    for idx, row in pairs.sort_values(
        "first_start_time_s"
    ).iterrows():
        assigned: int | None = None

        for cluster_id, center in enumerate(
            centers,
            start=1,
        ):
            duration_match = all(
                _similar_value(
                    float(row[column]),
                    float(center[column]),
                    relative_tolerance,
                    absolute_tolerance_s,
                )
                for column in (
                    "first_duration_s",
                    "middle_rest_duration_s",
                    "second_duration_s",
                )
            )

            current_match = all(
                _similar_value(
                    abs(float(row[column])),
                    abs(float(center[column])),
                    relative_tolerance,
                    0.1,
                )
                for column in (
                    "first_current_A",
                    "second_current_A",
                )
            )

            if (
                int(row["first_state"])
                == int(center["first_state"])
                and duration_match
                and current_match
            ):
                assigned = cluster_id
                break

        if assigned is None:
            clusters.append([idx])

            centers.append(
                {
                    "first_state": float(
                        row["first_state"]
                    ),
                    "first_duration_s": float(
                        row["first_duration_s"]
                    ),
                    "middle_rest_duration_s": float(
                        row["middle_rest_duration_s"]
                    ),
                    "second_duration_s": float(
                        row["second_duration_s"]
                    ),
                    "first_current_A": float(
                        row["first_current_A"]
                    ),
                    "second_current_A": float(
                        row["second_current_A"]
                    ),
                }
            )

        else:
            clusters[assigned - 1].append(idx)

            subset = pairs.loc[
                clusters[assigned - 1]
            ]

            for column in centers[assigned - 1]:
                centers[assigned - 1][column] = float(
                    subset[column].median()
                )

    result = pairs.copy()
    result["cluster_id"] = 0
    result["cluster_size"] = 0

    for cluster_id, indices in enumerate(
        clusters,
        start=1,
    ):
        result.loc[indices, "cluster_id"] = cluster_id
        result.loc[indices, "cluster_size"] = len(indices)

    return result


def _find_profile_pairs(
    runs: pd.DataFrame,
    pulse_duration_range_s: tuple[float, float],
    rest_duration_range_s: tuple[float, float],
    minimum_repetitions: int,
    max_plateau_cv: float,
    cluster_relative_tolerance: float,
    cluster_absolute_tolerance_s: float,
) -> pd.DataFrame:
    """Find repeated pulse-rest-opposite-pulse profiles."""

    pulse_min, pulse_max = pulse_duration_range_s
    rest_min, rest_max = rest_duration_range_s

    runs = runs.sort_values(
        "run_index"
    ).reset_index(drop=True)

    candidates: list[
        dict[str, float | int | str | bool]
    ] = []

    for idx in range(len(runs) - 2):
        first = runs.iloc[idx]
        rest = runs.iloc[idx + 1]
        second = runs.iloc[idx + 2]

        sequence_matches = (
            int(first["state"]) != 0
            and int(rest["state"]) == 0
            and int(second["state"])
            == -int(first["state"])
        )

        duration_matches = (
            pulse_min
            <= float(first["duration_s"])
            <= pulse_max
            and rest_min
            <= float(rest["duration_s"])
            <= rest_max
            and pulse_min
            <= float(second["duration_s"])
            <= pulse_max
        )

        if not sequence_matches or not duration_matches:
            continue

        first_cv = float(
            first["current_std_A"]
        ) / max(
            abs(float(first["mean_current_A"])),
            1e-12,
        )

        second_cv = float(
            second["current_std_A"]
        ) / max(
            abs(float(second["mean_current_A"])),
            1e-12,
        )

        plateau_stable = (
            first_cv <= max_plateau_cv
            and second_cv <= max_plateau_cv
        )

        if not plateau_stable:
            continue

        candidates.append(
            {
                "cell_id": str(first["cell_id"]),
                "TestIndex": float(first["TestIndex"]),
                "first_run_index": int(
                    first["run_index"]
                ),
                "rest_run_index": int(
                    rest["run_index"]
                ),
                "second_run_index": int(
                    second["run_index"]
                ),
                "first_state": int(first["state"]),
                "first_start_time_s": float(
                    first["start_time_s"]
                ),
                "second_end_time_s": float(
                    second["end_time_s"]
                ),
                "first_duration_s": float(
                    first["duration_s"]
                ),
                "middle_rest_duration_s": float(
                    rest["duration_s"]
                ),
                "second_duration_s": float(
                    second["duration_s"]
                ),
                "first_current_A": float(
                    first["mean_current_A"]
                ),
                "second_current_A": float(
                    second["mean_current_A"]
                ),
                "first_current_cv": first_cv,
                "second_current_cv": second_cv,
            }
        )

    clustered = _cluster_profile_pairs(
        pd.DataFrame(candidates),
        relative_tolerance=(
            cluster_relative_tolerance
        ),
        absolute_tolerance_s=(
            cluster_absolute_tolerance_s
        ),
    )

    if clustered.empty:
        return clustered

    accepted = clustered[
        clustered["cluster_size"].ge(
            minimum_repetitions
        )
    ].copy()

    accepted = accepted.sort_values(
        "first_start_time_s"
    ).reset_index(drop=True)

    accepted["hppc_pair_id"] = np.arange(
        1,
        len(accepted) + 1,
        dtype=int,
    )

    return accepted


def _annotate_detected_blocks(
    segments: pd.DataFrame,
    accepted_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate the complete automatically detected HPPC blocks."""

    if accepted_pairs.empty:
        return pd.DataFrame(
            columns=(
                SEGMENT_COLUMNS
                + DETECTION_COLUMNS
            )
        )

    tables: list[pd.DataFrame] = []
    block_id = 0

    for (
        cell_id,
        test_index,
    ), pairs in accepted_pairs.groupby(
        ["cell_id", "TestIndex"],
        sort=True,
    ):
        group = segments[
            segments["cell_id"].eq(cell_id)
            & segments["TestIndex"].eq(
                float(test_index)
            )
        ].sort_values("run_index")

        if group.empty:
            continue

        first_run = int(
            pairs["first_run_index"].min()
        )
        last_run = int(
            pairs["second_run_index"].max()
        )

        # Include an adjacent rest before and after
        # the detected HPPC pulse sequence.
        preceding = group[
            group["run_index"].eq(first_run - 1)
        ]
        following = group[
            group["run_index"].eq(last_run + 1)
        ]

        if (
            not preceding.empty
            and int(preceding.iloc[0]["state"]) == 0
        ):
            first_run -= 1

        if (
            not following.empty
            and int(following.iloc[0]["state"]) == 0
        ):
            last_run += 1

        selected = group[
            group["run_index"].between(
                first_run,
                last_run,
            )
        ].copy()

        if selected.empty:
            continue

        block_id += 1

        pulse_counts: defaultdict[int, int] = (
            defaultdict(int)
        )
        rest_counts: defaultdict[int, int] = (
            defaultdict(int)
        )

        for _, pair in pairs.iterrows():
            pulse_counts[
                int(pair["first_run_index"])
            ] += 1

            pulse_counts[
                int(pair["second_run_index"])
            ] += 1

            rest_counts[
                int(pair["rest_run_index"])
            ] += 1

        selected["hppc_block_id"] = block_id
        selected["hppc_role"] = "between_profiles"

        selected.loc[
            selected["run_index"].isin(
                pulse_counts
            ),
            "hppc_role",
        ] = "pulse"

        selected.loc[
            selected["run_index"].isin(
                rest_counts
            ),
            "hppc_role",
        ] = "rest"

        selected["hppc_pair_count"] = selected[
            "run_index"
        ].map(
            lambda value: (
                pulse_counts[int(value)]
                + rest_counts[int(value)]
            )
        )

        selected["detected_pair_count"] = int(
            len(pairs)
        )

        selected["block_start_time_s"] = float(
            selected["start_time_s"].min()
        )

        selected["block_end_time_s"] = float(
            selected["end_time_s"].max()
        )

        tables.append(selected)

    if not tables:
        return pd.DataFrame(
            columns=(
                SEGMENT_COLUMNS
                + DETECTION_COLUMNS
            )
        )

    return pd.concat(
        tables,
        ignore_index=True,
    )


def extract_hppc_segments(
    df: pd.DataFrame,
    cell_id: str,
    current_threshold_A: float = 0.05,
    pulse_duration_range_s: tuple[float, float] = (5.0, 20.0),
    rest_duration_range_s: tuple[float, float] = (500.0, 700.0),
    minimum_repetitions: int = 3,
    max_plateau_cv: float = 0.10,
    cluster_relative_tolerance: float = 0.20,
    cluster_absolute_tolerance_s: float = 1.0,
    batch_size: int = 500_000,
) -> pd.DataFrame:
    """Identify HPPC regions automatically from complete measurement data."""

    pulse_range = _validate_range(
        "pulse_duration_range_s",
        pulse_duration_range_s,
    )
    rest_range = _validate_range(
        "rest_duration_range_s",
        rest_duration_range_s,
    )

    if minimum_repetitions < 1:
        raise ValueError("minimum_repetitions must be at least 1.")
    if max_plateau_cv < 0.0:
        raise ValueError("max_plateau_cv must be non-negative.")
    if cluster_relative_tolerance < 0.0:
        raise ValueError(
            "cluster_relative_tolerance must be non-negative."
        )
    if cluster_absolute_tolerance_s < 0.0:
        raise ValueError(
            "cluster_absolute_tolerance_s must be non-negative."
        )

    segments = _segment_complete_data(
        df=df,
        cell_id=cell_id,
        current_threshold_A=current_threshold_A,
        batch_size=batch_size,
    )

    pair_tables: list[pd.DataFrame] = []

    for _, group in segments.groupby(
        ["cell_id", "TestIndex"],
        sort=True,
    ):
        accepted = _find_profile_pairs(
            runs=group,
            pulse_duration_range_s=pulse_range,
            rest_duration_range_s=rest_range,
            minimum_repetitions=minimum_repetitions,
            max_plateau_cv=max_plateau_cv,
            cluster_relative_tolerance=cluster_relative_tolerance,
            cluster_absolute_tolerance_s=(
                cluster_absolute_tolerance_s
            ),
        )

        if not accepted.empty:
            pair_tables.append(accepted)

    accepted_pairs = (
        pd.concat(pair_tables, ignore_index=True)
        if pair_tables
        else pd.DataFrame()
    )

    return _annotate_detected_blocks(
        segments,
        accepted_pairs,
    )

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
    
    Search for significant current changes near the expected pulse
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



__all__ = ["extract_hppc_segments"]
