"""Fit second-order RC equivalent-circuit models to battery pulse data.

This module extracts pulse and rest windows from user-supplied HPPC
measurement blocks and estimates the parameters of a second-order RC
equivalent-circuit model.

The extracted parameters include the ohmic resistance R0, polarization
resistances R1 and R2, time constants tau1 and tau2, and capacitances C1
and C2. Fit-quality metrics and quality flags are also returned for validation.

The input data must already be identified as HPPC measurement data.
Optional overview figures show the extracted parameters over SOC, FEC,
and calendar age.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from pydpeet.process.analyze.configs.battery_config import (
    _BatteryConfigClass,
    )
from pydpeet.process.analyze.extract.hppc import (
    _event_target_position,
    )
from pydpeet.process.analyze.resistance import (
    add_resistance_internal,
)



def select_fit_positions(time_s: np.ndarray) -> np.ndarray:
    """Subsample long rest periods without losing the fast early dynamics."""

    # Retain dense sampling immediately after the pulse, where the voltage
    # changes rapidly, and progressively reduce sampling at later times.
    masks = [
        time_s <= 5.0,
        (time_s > 5.0) & (time_s <= 30.0),
        (time_s > 30.0) & (time_s <= 120.0),
        time_s > 120.0,
    ]
    strides = [1, 5, 20, 50]

    selected: list[np.ndarray] = []

    for mask, stride in zip(
        masks,
        strides,
        strict=True,
    ):
        positions = np.flatnonzero(mask)
        selected.append(positions[::stride])

    # Always include the final sample to capture the long-term recovery level.
    selected.append(
        np.array(
            [len(time_s) - 1],
            dtype=int,
        )
    )

    return np.unique(np.concatenate(selected))


def extract_rest_window(
    raw: pd.DataFrame,
    pulse: pd.Series,
    current_threshold_A: float = 0.05,
) -> (
    tuple[
        np.ndarray,
        np.ndarray,
        dict[str, float | int],
    ]
    | None
):
    """Extract the complete rest period following one current pulse.
    
    Locate the actual pulse end and extract the subsequent rest period
    until the next current activity or the end of the HPPC block.
    
    Return the relative time, direction-normalized voltage recovery,
    and summary metadata, or None if no valid rest window is found.
    """

    time_s = raw["Test_Time[s]"].to_numpy(dtype=float)
    current_A = raw["Current[A]"].to_numpy(dtype=float)
    voltage_V = raw["Voltage[V]"].to_numpy(dtype=float)

    # Locate the measured current transition at the end of the pulse.
    stop_position = _event_target_position(
        time_s=time_s,
        current_A=current_A,
        event_time_s=float(pulse["pulse_end_time_s"]),
        event_kind="stop",
        pulse_current_A=float(pulse["mean_current_A"]),
    )

    if stop_position is None:
        return None

    
    # The rest window continues until the next non-rest current segement
    # or until the end of the supplied HPPC block.
    future_active = np.flatnonzero(np.abs(current_A[stop_position + 1 :]) > current_threshold_A)
    rest_end = stop_position + 1 + int(future_active[0]) if len(future_active) else len(raw)

    if rest_end - stop_position < 10:
        return None

    rest_time_s = time_s[stop_position:rest_end]
    rest_voltage_V = voltage_V[stop_position:rest_end]
    shifted_time_s = rest_time_s - rest_time_s[0]


    # Normalize the recovery direction so that charge and discharge
    # pulse can be fitted using the same positive recovery model.
    direction_sign = 1.0 if str(pulse["pulse_direction"]) == "discharge" else -1.0
    recovery_V = direction_sign * (rest_voltage_V - rest_voltage_V[0])

    metadata: dict[str, float | int] = {
        "rest_start_time_s": float(rest_time_s[0]),
        "rest_end_time_s": float(rest_time_s[-1]),
        "rest_duration_s": float(shifted_time_s[-1]),
        "rest_raw_sample_count": int(len(shifted_time_s)),
        "rest_dt_median_s": float(np.median(np.diff(shifted_time_s))),

        # Use the median of the final samples to reduce sensitivity to noise.
        "recovery_final_V": float(
            np.median(
                recovery_V[
                    max(
                        0,
                        len(recovery_V) - 100,
                    ) :
                ]
            )
        ),
    }

    return (
        shifted_time_s,
        recovery_V,
        metadata,
    )



def recovery_model(
    time_s: np.ndarray,
    amplitude_1_V: float,
    tau_1_s: float,
    amplitude_2_V: float,
    tau_2_s: float,
) -> np.ndarray:
    """Evaluate the voltage recovery of a second-order RC model."""

    return amplitude_1_V * (1.0 - np.exp(-time_s / tau_1_s)) + amplitude_2_V * (1.0 - np.exp(-time_s / tau_2_s))


def unpack_parameters(
    parameters: np.ndarray,
) -> tuple[float, float, float, float]:
    """Transform logarithmic parameters into positive model parameters."""

    # Use exponential transforms to ensure positive parameters and
    # enforce tau2 > tau1 through a positive time-constant gap.
    amplitude_1 = float(np.exp(parameters[0]))
    amplitude_2 = float(np.exp(parameters[1]))
    tau_1 = float(np.exp(parameters[2]))
    tau_2 = float(tau_1 + np.exp(parameters[3]))

    return (
        amplitude_1,
        tau_1,
        amplitude_2,
        tau_2,
    )


def residual_log_parameters(
    parameters: np.ndarray,
    time_s: np.ndarray,
    recovery_V: np.ndarray,
) -> np.ndarray:
    """Calculate residuals for logarithmic model parameters."""

    (
        amplitude_1,
        tau_1,
        amplitude_2,
        tau_2,
    ) = unpack_parameters(parameters)

    return (
        recovery_model(
            time_s,
            amplitude_1,
            tau_1,
            amplitude_2,
            tau_2,
        )
        - recovery_V
    )


def initial_guesses(
    final_recovery_V: float,
    rest_duration_s: float,
    profile: str = "full",
) -> list[np.ndarray]:
    """Generate initial parameter guesses for the nonlinear fit.
    
    Use predefined amplitude splits and time-constant pairs to generate
    multiple starting points. Guesses are expressed in log space to ensure
    positive parameters and ordered time constants.
    """

    # Apply a positive lower bound to the recovery amplitude
    # to avoid invalid logarithms when generating initial guesses.
    total = max(
        float(final_recovery_V),
        2e-4,
    )

    # Use a reduced set of amplitude splits and time constants
    # to limit the computational cost of batch fitting.
    if profile == "batch":
        batch_specs = [
            (0.50, 1.0, 25.0),
            (0.60, 3.0, 50.0),
            (0.75, 0.5, 20.0),
            (
                0.50,
                5.0,
                min(
                    100.0,
                    0.8 * rest_duration_s,
                ),
            ),
        ]

        return [
            np.array(
                [
                    np.log(
                        max(
                            total * fraction,
                            1e-6,
                        )
                    ),
                    np.log(
                        max(
                            total * (1.0 - fraction),
                            1e-6,
                        )
                    ),
                    np.log(tau_1),
                    np.log(tau_2 - tau_1),
                ],
                dtype=float,
            )
            for fraction, tau_1, tau_2 in batch_specs
            if tau_2 > tau_1
        ]

    if profile != "full":
        raise ValueError(f"Unknown initial-guess profile: {profile}")

    # Explore a wider range of time-constant pairs and amplitude
    # splits to reduce sensitivity to initial parameter estimates.
    tau_pairs = [
        (0.3, 5.0),
        (0.5, 20.0),
        (1.0, 50.0),
        (2.0, 100.0),
        (
            5.0,
            min(
                300.0,
                0.8 * rest_duration_s,
            ),
        ),
        (
            15.0,
            min(
                500.0,
                0.9 * rest_duration_s,
            ),
        ),
    ]

    guesses: list[np.ndarray] = []

    for fraction in [0.25, 0.5, 0.75]:
        for tau_1, tau_2 in tau_pairs:
            if tau_2 <= tau_1:
                continue
            
            # Express initial guesses in log space, using the positive
            # time-constant gap to preserve tau2 > tau1.
            guesses.append(
                np.array(
                    [
                        np.log(
                            max(
                                total * fraction,
                                1e-6,
                            )
                        ),
                        np.log(
                            max(
                                total * (1.0 - fraction),
                                1e-6,
                            )
                        ),
                        np.log(tau_1),
                        np.log(tau_2 - tau_1),
                    ],
                    dtype=float,
                )
            )

    return guesses


def fit_rest_window(
    time_s: np.ndarray,
    recovery_V: np.ndarray,
    guess_profile: str = "full",
) -> tuple[dict[str, object], np.ndarray]:
    """Fit a second-order RC model to a measured voltage recovery."""


    # Subsample the recovery data while retaining dense early time
    # measurements to capture the first RC response.
    fit_positions = select_fit_positions(time_s)
    fit_time = time_s[fit_positions]
    fit_recovery = recovery_V[fit_positions]

    # Estimate the final recovery amplitude from the final samples,
    # using the median to reduce sensitivity to noise and outliers.
    final_recovery = float(
        np.median(
            recovery_V[
                max(
                    0,
                    len(recovery_V) - 100,
                ) :
            ]
        )
    )

    # Define optimization bounds in log space to constrain the
    # recovery amplitudes and time constants to positive ranges.
    lower = np.log(
        np.array(
            [
                1e-6,
                1e-6,
                max(
                    0.05,
                    np.median(np.diff(time_s)) / 2.0,
                ),
                0.05,
            ]
        )
    )

    upper = np.log(
        np.array(
            [
                1.0,
                1.0,
                max(
                    60.0,
                    time_s[-1],
                ),
                max(
                    6000.0,
                    10.0 * time_s[-1],
                ),
            ]
        )
    )

    # Fit from multiple initial guesses and retain the solution
    # with the lowest robust least-squares cost.
    best = None

    for guess in initial_guesses(
        final_recovery,
        float(time_s[-1]),
        profile=guess_profile,
    ):
        clipped_guess = np.clip(
            guess,
            lower + 1e-9,
            upper - 1e-9,
        )

        candidate = least_squares(
            residual_log_parameters,
            clipped_guess,
            args=(
                fit_time,
                fit_recovery,
            ),
            bounds=(
                lower,
                upper,
            ),

            # Reduce the influence of isolated voltage outliers.
            loss="soft_l1",
            f_scale=2e-4,
            max_nfev=5000,
        )

        if best is None or candidate.cost < best.cost:
            best = candidate

    if best is None:
        raise RuntimeError("No second-order fit candidate was generated.")

    (
        amplitude_1,
        tau_1,
        amplitude_2,
        tau_2,
    ) = unpack_parameters(best.x)

    # Evaluate the fitted model over the full recovery window,
    # including samples excluded from the optimization.
    predicted = recovery_model(
        time_s,
        amplitude_1,
        tau_1,
        amplitude_2,
        tau_2,
    )

    residual = predicted - recovery_V

    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    centered = recovery_V - np.mean(recovery_V)
    denominator = float(np.sum(centered**2))

    r_squared = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else np.nan

    # A large condition number indicates weak parameter identifiability.
    try:
        jacobian_condition = float(np.linalg.cond(best.jac.T @ best.jac))
    except np.linalg.LinAlgError:
        jacobian_condition = np.inf

    result = {
        "optimizer_success": bool(best.success),
        "optimizer_status": int(best.status),
        "optimizer_message": str(best.message),
        "fit_sample_count": int(len(fit_positions)),
        "amplitude_1_V": amplitude_1,
        "tau_1_s": tau_1,
        "amplitude_2_V": amplitude_2,
        "tau_2_s": tau_2,
        "tau_ratio": float(tau_2 / tau_1),
        "fit_rmse_mV": rmse * 1000.0,
        "fit_mae_mV": mae * 1000.0,
        "fit_R_squared": r_squared,
        "jacobian_condition": jacobian_condition,
        "optimizer_nfev": int(best.nfev),
    }

    return result, predicted



def quality_flags(
    result: dict[str, object],
) -> str:
    """Return transparent quality warnings without removing fit results."""

    flags: list[str] = []

    if not bool(result["optimizer_success"]):
        flags.append("optimizer_not_converged")

    r_squared = float (result["fit_R_squared"])

    if not np.isfinite(r_squared):
        flags.append("R_squared_not_finite")
    elif r_squared < 0.98:
        flags.append("R_squared_below_0.98")
        
    if float(result["fit_rmse_mV"]) > 1.0:
        flags.append("RMSE_above_1mV")
    if float(result["tau_ratio"]) < 2.0:
        flags.append("time_constants_not_separated")
    if float(result["tau_2_s"]) > float(result["rest_duration_s"]):
        flags.append("tau2_exceeds_rest_window")
    if float(result["jacobian_condition"]) > 1e10:
        flags.append("ill_conditioned_jacobian")

    return "|".join(flags)


def fit_ecm_pulse(
    raw: pd.DataFrame,
    pulse: pd.Series,
    pulse_row_id: int,
    config: _BatteryConfigClass,
    current_threshold_A: float = 0.05,
    guess_profile: str = "full",
) -> dict[str, object]:
    """Fit second-order RC parameters from one post-pulse rest period."""

    row = pulse.to_dict()
    row["pulse_row_id"] = pulse_row_id
    row["fit_method"] = "rest_relaxation"

    # Calculate R0 using the existing PyDPEET resistance function.
    r0_result = calculate_pulse_r0(
        raw=raw,
        pulse=pulse,
        config=config,
    )
    row.update(r0_result)

    if not bool(r0_result["R0_valid"]):
        row.update(
            {
                "fit_valid": False,
                "fit_exclusion_reason": "R0_not_available",
                "quality_flags": "R0_not_available",
            }
        )
        return row

    # Extract the complete rest period following the pulse.
    rest = extract_rest_window(
        raw=raw,
        pulse=pulse,
        current_threshold_A=current_threshold_A,
    )

    if rest is None:
        row.update(
            {
                "fit_valid": False,
                "fit_exclusion_reason": "rest_window_not_found",
                "quality_flags": "rest_window_not_found",
            }
        )
        return row

    time_s, recovery_V, rest_metadata = rest
    row.update(rest_metadata)

    try:
        relaxation_fit, _ = fit_rest_window(
            time_s=time_s,
            recovery_V=recovery_V,
            guess_profile=guess_profile,
        )
    except Exception as error:
        row.update(
            {
                "fit_valid": False,
                "fit_exclusion_reason": f"rest_fit_failed:{error}",
                "quality_flags": "rest_fit_failed",
            }
        )
        return row

    row.update(relaxation_fit)

    current_abs_A = abs(float(pulse["mean_current_A"]))
    pulse_duration_s = float(pulse["pulse_duration_s"])

    if current_abs_A <= 0.0 or pulse_duration_s <= 0.0:
        row.update(
            {
                "fit_valid": False,
                "fit_exclusion_reason": "invalid_pulse_current_or_duration",
                "quality_flags": "invalid_pulse_current_or_duration",
            }
        )
        return row

    tau_1_s = float(relaxation_fit["tau_1_s"])
    tau_2_s = float(relaxation_fit["tau_2_s"])

    denominator_1 = current_abs_A * (
        1.0 - np.exp(-pulse_duration_s / tau_1_s)
    )
    denominator_2 = current_abs_A * (
        1.0 - np.exp(-pulse_duration_s / tau_2_s)
    )

    if denominator_1 <= 0.0 or denominator_2 <= 0.0:
        row.update(
            {
                "fit_valid": False,
                "fit_exclusion_reason": "invalid_RC_denominator",
                "quality_flags": "invalid_RC_denominator",
            }
        )
        return row

    resistance_1_ohm = (
        float(relaxation_fit["amplitude_1_V"])
        / denominator_1
    )
    resistance_2_ohm = (
        float(relaxation_fit["amplitude_2_V"])
        / denominator_2
    )

    capacitance_1_F = tau_1_s / resistance_1_ohm
    capacitance_2_F = tau_2_s / resistance_2_ohm

    row.update(
        {
            "R0_mohm": float(row["R0_ohm"]) * 1000.0,
            "R1_ohm": resistance_1_ohm,
            "R2_ohm": resistance_2_ohm,
            "R1_mohm": resistance_1_ohm * 1000.0,
            "R2_mohm": resistance_2_ohm * 1000.0,
            "Rsum_mohm": (resistance_1_ohm + resistance_2_ohm) * 1000.0,
            "C1_F": capacitance_1_F,
            "C2_F": capacitance_2_F,
            "pulse_duration_actual_s": pulse_duration_s,
        }
    )

    required_parameters = np.asarray(
        [
            row["R0_ohm"],
            row["R1_ohm"],
            row["R2_ohm"],
            row["C1_F"],
            row["C2_F"],
            row["tau_1_s"],
            row["tau_2_s"],
        ],
        dtype=float,
    )

    row["fit_valid"] = bool(
        row["optimizer_success"]
        and np.isfinite(required_parameters).all()
        and np.all(required_parameters > 0.0)
        and float(row["tau_2_s"]) > float(row["tau_1_s"])
    )

    row["fit_exclusion_reason"] = (
        ""
        if row["fit_valid"]
        else "nonphysical_fit"
    )
    row["quality_flags"] = (
        quality_flags(row)
        if row["fit_valid"]
        else "nonphysical_fit"
    )

    return row

_PARAMETER_PLOT_COLUMNS = (
    ("R0_mohm", r"$R_0$ [m$\Omega$]"),
    ("R1_mohm", r"$R_1$ [m$\Omega$]"),
    ("R2_mohm", r"$R_2$ [m$\Omega$]"),
    ("Rsum_mohm", r"$R_1+R_2$ [m$\Omega$]"),
    ("C1_F", r"$C_1$ [F]"),
    ("C2_F", r"$C_2$ [F]"),
    ("tau_1_s", r"$\tau_1$ [s]"),
    ("tau_2_s", r"$\tau_2$ [s]"),
)


def _annotate_quality_status(parameters: pd.DataFrame) -> pd.DataFrame:
    """Classify fits as pass, review, or excluded."""

    rows = parameters.copy()
    valid = rows.get(
        "fit_valid",
        pd.Series(False, index=rows.index),
    ).fillna(False).astype(bool)
    flags = rows.get(
        "quality_flags",
        pd.Series("", index=rows.index),
    ).fillna("").astype(str).str.strip()

    # valid fits pass without warnings and require review when warnings are present.
    rows["quality_status"] = "excluded"
    rows.loc[valid & flags.eq(""), "quality_status"] = "pass"
    rows.loc[valid & flags.ne(""), "quality_status"] = "review"
    return rows


def _annotate_soc_paths(parameters: pd.DataFrame) -> pd.DataFrame:
    """Identify descending and ascending SOC paths within each HPPC block."""

    rows = parameters.copy()
    rows["soc_path"] = "unclear"
    rows["sweep_id"] = 0

    required = {
        "cell_id",
        "TestIndex",
        "hppc_block_id",
        "pulse_start_time_s",
        "SOC",
    }
    if rows.empty or not required.issubset(rows.columns):
        return rows

    group_columns = ["cell_id", "TestIndex", "hppc_block_id"]
    for _, sequence in rows.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        ordered = sequence.copy()
        ordered["_time"] = pd.to_numeric(
            ordered["pulse_start_time_s"],
            errors="coerce",
        )
        ordered["_soc"] = pd.to_numeric(
            ordered["SOC"],
            errors="coerce",
        )
        ordered = ordered.dropna(subset=["_time", "_soc"])
        ordered = ordered.loc[ordered["_soc"].between(0.0, 1.0)]
        ordered = ordered.sort_values("_time", kind="stable")
        if ordered.empty:
            continue

        soc = ordered["_soc"].to_numpy(dtype=float)
        first_soc = float(soc[0])
        last_soc = float(soc[-1])
        minimum_position = int(np.argmin(soc))
        maximum_position = int(np.argmax(soc))
        minimum_soc = float(np.min(soc))
        maximum_soc = float(np.max(soc))

        # Identify a turning only when SOC changes by at least
        # 20 percentage points on both sides of the extremum.
        down_up = (
            0 < minimum_position < len(ordered) - 1
            and first_soc - minimum_soc >= 0.20
            and last_soc - minimum_soc >= 0.20
        )
        up_down = (
            0 < maximum_position < len(ordered) - 1
            and maximum_soc - first_soc >= 0.20
            and maximum_soc - last_soc >= 0.20
        )

        if down_up:
            descending = ordered.index[: minimum_position + 1]
            ascending = ordered.index[minimum_position + 1 :]
            rows.loc[descending, ["soc_path", "sweep_id"]] = (
                "descending",
                1,
            )
            rows.loc[ascending, ["soc_path", "sweep_id"]] = (
                "ascending",
                2,
            )
        elif up_down:
            ascending = ordered.index[: maximum_position + 1]
            descending = ordered.index[maximum_position + 1 :]
            rows.loc[ascending, ["soc_path", "sweep_id"]] = (
                "ascending",
                1,
            )
            rows.loc[descending, ["soc_path", "sweep_id"]] = (
                "descending",
                2,
            )
        else:
            path = "ascending" if last_soc > first_soc else "descending"
            rows.loc[ordered.index, ["soc_path", "sweep_id"]] = (
                path,
                1,
            )

    rows["sweep_id"] = pd.to_numeric(
        rows["sweep_id"],
        errors="coerce",
    ).fillna(0).astype(int)
    return rows


def _filter_parameter_rows(
    parameters: pd.DataFrame,
    quality_statuses: tuple[str, ...],
    pulse_directions: tuple[str, ...],
    soc_paths: tuple[str, ...],
    target_current_abs_A: float | None,
    current_tolerance_A: float,
) -> pd.DataFrame:
    """Apply the NMC workflow selection rules used for parameter plots."""

    rows = parameters.copy()
    for column, allowed in (
        ("quality_status", quality_statuses),
        ("pulse_direction", pulse_directions),
        ("soc_path", soc_paths),
    ):
        if column not in rows.columns:
            return rows.iloc[0:0].copy()
        rows = rows.loc[rows[column].isin(allowed)]

    if target_current_abs_A is not None:
        current = pd.to_numeric(
            rows["mean_current_A"],
            errors="coerce",
        ).abs()
        rows = rows.loc[
            (current - float(target_current_abs_A)).abs()
            <= float(current_tolerance_A)
        ]

    return rows.copy()


def _block_plot_label(block: pd.DataFrame) -> str:
    """Create a compact label for one HPPC block."""

    cell_id = str(block["cell_id"].iloc[0])
    block_id = block["hppc_block_id"].iloc[0]

    for column, prefix in (
        ("FEC", "FEC"),
        ("calendar_age_days", "day"),
    ):
        if column not in block.columns:
            continue
        value = pd.to_numeric(block[column], errors="coerce").dropna()
        if not value.empty:
            return f"{cell_id}, {prefix} {float(value.iloc[0]):g}"

    return f"{cell_id}, block {block_id}"


def _plot_parameters_over_soc(
    parameters: pd.DataFrame,
    pulse_direction: str,
    soc_path: str,
) -> None:
    """Plot one pulse direction and one SOC path without mixing curves."""

    rows = parameters.loc[
        parameters["pulse_direction"].eq(pulse_direction)
        & parameters["soc_path"].eq(soc_path)
    ].copy()
    if rows.empty or "SOC" not in rows.columns:
        return

    rows["SOC"] = pd.to_numeric(rows["SOC"], errors="coerce")
    rows = rows.dropna(subset=["SOC"])
    series_size = rows.groupby(
        ["cell_id", "hppc_block_id"],
        dropna=False,
    )["SOC"].transform("size")
    # Omit isolated points becasuse a trend requires at least two points.
    rows = rows.loc[series_size >= 2]
    if rows.empty:
        return

    specs = [
        (column, label)
        for column, label in _PARAMETER_PLOT_COLUMNS
        if column in rows.columns
    ]
    if not specs:
        return

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12.0, 9.0),
        constrained_layout=False,
    )
    flat_axes = axes.ravel()

    for _, block in rows.groupby(
        ["cell_id", "hppc_block_id"],
        sort=True,
        dropna=False,
    ):
        block = block.copy()
        block["SOC"] = pd.to_numeric(block["SOC"], errors="coerce")
        block = block.dropna(subset=["SOC"]).sort_values("SOC")
        if block.empty:
            continue

        label = _block_plot_label(block)
        for axis, (column, y_label) in zip(
            flat_axes,
            specs,
            strict=False,
        ):
            values = pd.to_numeric(block[column], errors="coerce")
            finite = np.isfinite(block["SOC"]) & np.isfinite(values)
            if int(finite.sum()) < 2:
                continue
            axis.plot(
                block.loc[finite, "SOC"],
                values.loc[finite],
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                label=label,
            )
            axis.set_xlabel("SOC [-]")
            axis.set_ylabel(y_label)
            axis.grid(alpha=0.25)

    for axis in flat_axes[len(specs):]:
        axis.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.82, 0.5),
            frameon=False,
        )
    quality_label = ", ".join(
        sorted(rows["quality_status"].astype(str).unique())
    )
    fig.suptitle(
        "Second-order RC parameters over SOC\n"
        f"{pulse_direction} pulse, SOC {soc_path}, quality: {quality_label}"
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 0.96))
    plt.show()


def _select_target_soc_rows(
    parameters: pd.DataFrame,
    target_socs: tuple[float, ...],
    soc_tolerance: float,
) -> pd.DataFrame:
    """Select the nearest measured point at each requested diagnostic SOC."""

    selected_rows: list[pd.Series] = []
    group_columns = [
        "cell_id",
        "hppc_block_id",
        "pulse_direction",
        "soc_path",
    ]
    for _, block in parameters.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        block = block.copy()
        block["SOC"] = pd.to_numeric(block["SOC"], errors="coerce")
        block = block.loc[block["SOC"].between(0.0, 1.0)]
        if block.empty:
            continue

        for target_soc in target_socs:
            distance = (block["SOC"] - float(target_soc)).abs()
            candidates = block.loc[
                distance <= float(soc_tolerance) + 1e-12
            ].copy()
            if candidates.empty:
                continue
            candidates["SOC_distance"] = distance.loc[candidates.index]
            order_columns = ["SOC_distance"]
            if "fit_rmse_mV" in candidates.columns:
                order_columns.append("fit_rmse_mV")
            row = candidates.sort_values(order_columns, kind="stable").iloc[0].copy()
            row["target_SOC"] = float(target_soc)
            selected_rows.append(row)

    if not selected_rows:
        return parameters.iloc[0:0].assign(
            target_SOC=pd.Series(dtype=float),
            SOC_distance=pd.Series(dtype=float),
        )
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _plot_parameters_over_age(
    parameters: pd.DataFrame,
    x_column: str,
    x_label: str,
    pulse_direction: str,
    soc_path: str,
    target_socs: tuple[float, ...],
    soc_tolerance: float,
    normalize_to_first: bool = False,
) -> None:
    """Plot ageing at fixed SOC, optionally relative to each first valid value.

    Normalization is separate for every cell, target SOC, pulse direction,
    SOC path and parameter. The baseline is the earliest available finite
    value, which need not be an age-zero measurement. A zero baseline is
    skipped, matching the NMC workflow's normalization rule.
    """

    rows = parameters.loc[
        parameters["pulse_direction"].eq(pulse_direction)
        & parameters["soc_path"].eq(soc_path)
    ].copy()
    if x_column not in rows.columns or "SOC" not in rows.columns:
        return

    rows = _select_target_soc_rows(
        parameters=rows,
        target_socs=target_socs,
        soc_tolerance=soc_tolerance,
    )
    rows[x_column] = pd.to_numeric(rows[x_column], errors="coerce")
    rows = rows.loc[np.isfinite(rows[x_column])]
    series_size = rows.groupby(
        ["cell_id", "target_SOC"],
        dropna=False,
    )[x_column].transform("nunique")
    # An ageing curve requires at least two distinct ageing states.
    rows = rows.loc[series_size >= 2]
    if rows.empty:
        return

    specs = [
        (column, label)
        for column, label in _PARAMETER_PLOT_COLUMNS
        if column in rows.columns
    ]
    if not specs:
        return

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12.0, 9.0),
        constrained_layout=False,
    )
    flat_axes = axes.ravel()
    for axis, (column, y_label) in zip(
        flat_axes,
        specs,
        strict=False,
    ):
        for (cell_id, target_soc), series in rows.groupby(
            ["cell_id", "target_SOC"],
            dropna=False,
            sort=True,
        ):
            series = series.copy()
            series["_value"] = pd.to_numeric(
                series[column],
                errors="coerce",
            )
            series = series.dropna(subset=[x_column, "_value"])
            series = series.loc[np.isfinite(series["_value"])]
            order_columns = [x_column]
            if "TestIndex" in series.columns:
                order_columns.append("TestIndex")
            series = series.sort_values(order_columns, kind="stable")
            series = series.drop_duplicates(
                subset=[x_column],
                keep = "last",
            )
            
            if len(series) < 2:
                continue
            label = f"{cell_id}, SOC {100.0 * float(target_soc):g}%"
            if normalize_to_first:
                baseline = float(series["_value"].iloc[0])
                if baseline == 0.0:
                    continue
                series["_value"] = 100.0 * series["_value"] / baseline
                reference_age = float(series[x_column].iloc[0])
                label += f", ref. {reference_age:g}"
            axis.plot(
                series[x_column],
                series["_value"],
                marker="o",
                markersize=3.5,
                linewidth=1.2,
                label=label,
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(
            f"{y_label.split(' [')[0]} [% of first valid]"
            if normalize_to_first else y_label
        )
        if normalize_to_first:
            axis.axhline(100.0, color="0.65", linewidth=0.7, linestyle="--")
        axis.grid(alpha=0.25)

    for axis in flat_axes[len(specs):]:
        axis.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.82, 0.5),
            frameon=False,
        )
    quality_label = ", ".join(
        sorted(rows["quality_status"].astype(str).unique())
    )
    fig.suptitle(
        f"{'Normalized ' if normalize_to_first else ''}Second-order RC parameters over {x_label}\n"
        f"{pulse_direction} pulse, SOC {soc_path}, quality: {quality_label}"
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 0.94))
    plt.show()


def _plot_ecm_parameters(
    parameters: pd.DataFrame,
    quality_statuses: tuple[str, ...],
    pulse_directions: tuple[str, ...],
    soc_paths: tuple[str, ...],
    target_current_abs_A: float | None,
    current_tolerance_A: float,
    target_socs: tuple[float, ...],
    soc_tolerance: float,
    normalize_to_first: bool = False,
) -> None:
    """Create filtered NMC-style SOC and ageing parameter figures."""

    rows = _filter_parameter_rows(
        parameters=parameters,
        quality_statuses=quality_statuses,
        pulse_directions=pulse_directions,
        soc_paths=soc_paths,
        target_current_abs_A=target_current_abs_A,
        current_tolerance_A=current_tolerance_A,
    )
    if rows.empty:
        return

    # Plot each pulse direction and SOC path separately to avoid
    # connecting points from different paths at the same SOC.
    for pulse_direction in pulse_directions:
        for soc_path in soc_paths:
            combination = rows.loc[
                rows["pulse_direction"].eq(pulse_direction)
                & rows["soc_path"].eq(soc_path)
            ]
            if combination.empty:
                continue
            _plot_parameters_over_soc(
                rows,
                pulse_direction,
                soc_path,
            )
            _plot_parameters_over_age(
                rows,
                "FEC",
                "FEC [-]",
                pulse_direction,
                soc_path,
                target_socs,
                soc_tolerance,
                normalize_to_first,
            )
            _plot_parameters_over_age(
                rows,
                "calendar_age_days",
                "Calendar age [days]",
                pulse_direction,
                soc_path,
                target_socs,
                soc_tolerance,
                normalize_to_first,
            )


def create_ecm_parameters(
    df: pd.DataFrame,
    hppc_segments: pd.DataFrame,
    config: _BatteryConfigClass,
    block_metadata: list[dict[str, object]] | None = None,
    current_threshold_A: float = 0.05,
    guess_profile: str = "full",
    visualize: bool = False,
    plot_quality_statuses: tuple[str, ...] = ("pass",),
    plot_pulse_directions: tuple[str, ...] = (
        "charge",
        "discharge",
    ),
    plot_soc_paths: tuple[str, ...] = (
        "descending",
        "ascending",
    ),
    plot_target_current_abs_A: float | None = 4.8,
    plot_current_tolerance_A: float = 0.25,
    plot_target_socs: tuple[float, ...] = (0.25, 0.50, 0.75),
    plot_soc_tolerance: float = 0.02,
    plot_normalize_to_first: bool = True,
) -> pd.DataFrame:
    """Fit 2RC parameters using automatically detected HPPC segments.

    ``df`` contains the complete measurement data. ``hppc_segments`` is
    the table returned by ``extract_hppc_segments``. Optional metadata,
    including FEC and calendar age, is copied to every pulse belonging
    to the corresponding HPPC block. The returned table includes
    ``quality_status``, ``soc_path``, and ``sweep_id``. By default,
    visualization uses only pass-quality pulses at 4.8 +/- 0.25 A,
    separates charge/discharge pulses and ascending/descending SOC paths,
    and uses fixed SOC values of 25%, 50%, and 75% for ageing plots.
    Ageing plots default to 100% at each series' first valid measurement;
    set ``plot_normalize_to_first=False`` to display absolute parameters.
    Parameter-over-SOC plots retain their physical units.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")
    if not isinstance(hppc_segments, pd.DataFrame):
        raise ValueError(
            "hppc_segments must be a pandas DataFrame."
        )
    if not isinstance(visualize, bool):
        raise ValueError("visualize must be a boolean.")
    if not isinstance(plot_normalize_to_first, bool):
        raise ValueError("plot_normalize_to_first must be a boolean.")

    allowed_quality = {"pass", "review", "excluded"}
    allowed_directions = {"charge", "discharge"}
    allowed_paths = {"descending", "ascending", "unclear"}
    if (
        not plot_quality_statuses
        or not set(plot_quality_statuses).issubset(allowed_quality)
    ):
        raise ValueError(
            "plot_quality_statuses may contain pass, review, or excluded."
        )
    if not set(plot_pulse_directions).issubset(allowed_directions):
        raise ValueError(
            "plot_pulse_directions may contain charge or discharge."
        )
    if not set(plot_soc_paths).issubset(allowed_paths):
        raise ValueError(
            "plot_soc_paths may contain descending, ascending, or unclear."
        )
    if (
        plot_target_current_abs_A is not None
        and float(plot_target_current_abs_A) <= 0.0
    ):
        raise ValueError("plot_target_current_abs_A must be positive or None.")
    if float(plot_current_tolerance_A) < 0.0:
        raise ValueError("plot_current_tolerance_A must be non-negative.")
    if not plot_target_socs or any(
        not 0.0 <= float(value) <= 1.0
        for value in plot_target_socs
    ):
        raise ValueError("plot_target_socs must contain values from 0 to 1.")
    if float(plot_soc_tolerance) < 0.0:
        raise ValueError("plot_soc_tolerance must be non-negative.")

    if hppc_segments.empty:
        return pd.DataFrame()

    required_raw_columns = {
        "TestIndex",
        "Test_Time[s]",
        "Current[A]",
        "Voltage[V]",
    }
    missing_raw = sorted(
        required_raw_columns - set(df.columns)
    )
    if missing_raw:
        raise ValueError(
            f"df is missing required columns: {missing_raw}"
        )

    required_segment_columns = {
        "cell_id",
        "TestIndex",
        "run_index",
        "state",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "mean_current_A",
        "soc_start",
        "hppc_block_id",
        "hppc_role",
    }
    missing_segments = sorted(
        required_segment_columns - set(hppc_segments.columns)
    )
    if missing_segments:
        raise ValueError(
            "hppc_segments is missing required columns: "
            f"{missing_segments}"
        )

    block_ids = sorted(
        pd.to_numeric(
            hppc_segments["hppc_block_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not block_ids:
        return pd.DataFrame()

    if block_metadata is None:
        metadata_by_block = {
            block_id: {}
            for block_id in block_ids
        }
    else:
        if len(block_metadata) != len(block_ids):
            raise ValueError(
                "block_metadata must contain one entry "
                "per HPPC block."
            )

        metadata_by_block = dict(
            zip(
                block_ids,
                block_metadata,
                strict=True,
            )
        )

    numeric_raw_test_index = pd.to_numeric(
        df["TestIndex"],
        errors="coerce",
    )
    numeric_block_id = pd.to_numeric(
        hppc_segments["hppc_block_id"],
        errors="coerce",
    )

    result_rows: list[dict[str, object]] = []
    pulse_row_id = 0

    for block_id in block_ids:
        block_segments = hppc_segments.loc[
            numeric_block_id.eq(block_id)
        ].copy()

        block_info = metadata_by_block[block_id]

        for test_index, segments in block_segments.groupby(
            "TestIndex",
            sort=True,
        ):
            segments = segments.sort_values(
                "run_index"
            ).reset_index(drop=True)

            following_state = segments["state"].shift(-1)

            pulse_mask = (
                segments["hppc_role"].eq("pulse")
                & segments["state"].ne(0)
                & following_state.eq(0)
            )
            pulse_segments = segments.loc[pulse_mask]

            numeric_test_index = pd.to_numeric(
                pd.Series([test_index]),
                errors="coerce",
            ).iloc[0]

            if pd.isna(numeric_test_index):
                continue

            raw_test = df.loc[
                numeric_raw_test_index.eq(
                    float(numeric_test_index)
                )
            ].copy()

            if raw_test.empty:
                continue

            raw_test = raw_test.sort_values(
                "Test_Time[s]"
            ).reset_index(drop=True)

            for _, segment in pulse_segments.iterrows():
                pulse = segment.copy()

                pulse["pulse_start_time_s"] = float(
                    segment["start_time_s"]
                )
                pulse["pulse_end_time_s"] = float(
                    segment["end_time_s"]
                )
                pulse["pulse_duration_s"] = float(
                    segment["duration_s"]
                )
                pulse["pulse_direction"] = (
                    "charge"
                    if float(segment["mean_current_A"]) > 0.0
                    else "discharge"
                )
                pulse["SOC"] = float(
                    segment["soc_start"]
                )
                pulse["SOC_is_physical"] = bool(
                    np.isfinite(pulse["SOC"])
                    and 0.0 <= pulse["SOC"] <= 1.0
                )

                result = fit_ecm_pulse(
                    raw=raw_test,
                    pulse=pulse,
                    pulse_row_id=pulse_row_id,
                    config=config,
                    current_threshold_A=(
                        current_threshold_A
                    ),
                    guess_profile=guess_profile,
                )

                result["hppc_block_id"] = block_id

                # Preserve FEC, calendar age, and other metadata.
                for key, value in block_info.items():
                    if key not in result:
                        result[key] = value

                result_rows.append(result)
                pulse_row_id += 1

    results = pd.DataFrame(result_rows)

    if not results.empty:
        results = _annotate_quality_status(results)
        results = _annotate_soc_paths(results)

    if visualize and not results.empty:
        _plot_ecm_parameters(
            parameters=results,
            quality_statuses=tuple(plot_quality_statuses),
            pulse_directions=tuple(plot_pulse_directions),
            soc_paths=tuple(plot_soc_paths),
            target_current_abs_A=plot_target_current_abs_A,
            current_tolerance_A=plot_current_tolerance_A,
            target_socs=tuple(float(value) for value in plot_target_socs),
            soc_tolerance=float(plot_soc_tolerance),
            normalize_to_first=plot_normalize_to_first,
        )

    return results

def calculate_pulse_r0(
    raw: pd.DataFrame,
    pulse: pd.Series,
    config: _BatteryConfigClass,
) -> dict[str, float | str | bool]:
    """Calculate R0 at the start and stop of one current pulse.
    
    The internal resistance calculation is delegated to
    
    "add_resistance_internal". This function locates the pulse-start and
    pulse-stop transistions, extracts their resistance values, and selects
    the pulse-stop value with the pulse-start values as a fallback.
    """

    required_columns = [
        "Test_Time[s]",
        "Current[A]",
        "Voltage[V]",
    ]
    work = raw[required_columns].copy()

    for column in required_columns:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    time_s = work["Test_Time[s]"].to_numpy(dtype=float)
    current_A = work["Current[A]"].to_numpy(dtype=float)
    pulse_current_A = float(pulse["mean_current_A"])

    # Locate the measured current transitions at the pulse start and stop.
    start_position = _event_target_position(
        time_s=time_s,
        current_A=current_A,
        event_time_s=float(pulse["pulse_start_time_s"]),
        event_kind="start",
        pulse_current_A=pulse_current_A,
    )
    stop_position = _event_target_position(
        time_s=time_s,
        current_A=current_A,
        event_time_s=float(pulse["pulse_end_time_s"]),
        event_kind="stop",
        pulse_current_A=pulse_current_A,
    )

    # Reuse the existing PyDPEET resistance calculation to obtain
    # internal resistance estimates at the detected pulse transistions.
    calculated = add_resistance_internal(
        work,
        config=config,
        verbose=False,
    )

    def resistance_at(
        position: int | None,
    ) -> float:
        if position is None:
            return np.nan

        value = calculated.iloc[position]["InternalResistance[ohm]"]
        return float(value) if pd.notna(value) else np.nan

    r0_start_ohm = resistance_at(start_position)
    r0_stop_ohm = resistance_at(stop_position)

    # Prefer a valid resistance estimate at pulse stop, falling
    # back to the pulse-start estimate when neccessary.
    if np.isfinite(r0_stop_ohm) and r0_stop_ohm > 0:
        r0_ohm = r0_stop_ohm
        r0_source = "stop"
    elif np.isfinite(r0_start_ohm) and r0_start_ohm > 0:
        r0_ohm = r0_start_ohm
        r0_source = "start"
    else:
        r0_ohm = np.nan
        r0_source = "none"

    return {
        "R0_start_ohm": r0_start_ohm,
        "R0_stop_ohm": r0_stop_ohm,
        "R0_ohm": r0_ohm,
        "R0_source": r0_source,
        "R0_valid": bool(np.isfinite(r0_ohm) and r0_ohm > 0),
    }


__all__ = ["create_ecm_parameters"]
