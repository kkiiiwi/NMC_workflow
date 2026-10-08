"""Fit second-order RC equivalent-circuit models to battery pulse data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pydpeet.process.analyze.configs.battery_config import (
    _BatteryConfigClass,
)
from pydpeet.process.analyze.extract.hppc import _event_target_position, extract_current_segments
from pydpeet.process.analyze.resistance import (
    add_resistance_internal,
)


@dataclass
class FitCurve:
    """Measured and fitted voltage recovery for one pulse."""

    pulse_row_id: int
    time_s: np.ndarray
    recovery_measured_V: np.ndarray
    recovery_fitted_V: np.ndarray


@dataclass
class LoadRestWindow:
    """Current, voltage, and OCV data for one pulse and subsequent rest."""

    time_s: np.ndarray
    current_A: np.ndarray
    voltage_V: np.ndarray
    ocv_path_V: np.ndarray
    pulse_positions: np.ndarray
    rest_positions: np.ndarray
    fit_positions: np.ndarray
    pre_rest_duration_s: float
    pulse_duration_actual_s: float
    rest_duration_s: float
    ocv_before_V: float
    ocv_after_V: float


def select_fit_positions(time_s: np.ndarray) -> np.ndarray:
    """Subsample long rest periods without losing the fast early dynamics."""

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
    """Extract the complete rest period following one current pulse."""

    time_s = raw["Test_Time[s]"].to_numpy(dtype=float)
    current_A = raw["Current[A]"].to_numpy(dtype=float)
    voltage_V = raw["Voltage[V]"].to_numpy(dtype=float)

    stop_position = _event_target_position(
        time_s=time_s,
        current_A=current_A,
        event_time_s=float(pulse["pulse_end_time_s"]),
        event_kind="stop",
        pulse_current_A=float(pulse["mean_current_A"]),
    )

    if stop_position is None:
        return None

    future_active = np.flatnonzero(np.abs(current_A[stop_position + 1 :]) > current_threshold_A)
    rest_end = stop_position + 1 + int(future_active[0]) if len(future_active) else len(raw)

    if rest_end - stop_position < 10:
        return None

    rest_time_s = time_s[stop_position:rest_end]
    rest_voltage_V = voltage_V[stop_position:rest_end]
    shifted_time_s = rest_time_s - rest_time_s[0]

    direction_sign = 1.0 if str(pulse["pulse_direction"]) == "discharge" else -1.0
    recovery_V = direction_sign * (rest_voltage_V - rest_voltage_V[0])

    metadata: dict[str, float | int] = {
        "rest_start_time_s": float(rest_time_s[0]),
        "rest_end_time_s": float(rest_time_s[-1]),
        "rest_duration_s": float(shifted_time_s[-1]),
        "rest_raw_sample_count": int(len(shifted_time_s)),
        "rest_dt_median_s": float(np.median(np.diff(shifted_time_s))),
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


def extract_load_and_rest_window(
    raw: pd.DataFrame,
    pulse: pd.Series,
    current_threshold_A: float = 0.05,
) -> LoadRestWindow | None:
    """Extract one measured current pulse and the complete following rest.

    Durations are inferred from current transitions in the supplied HPPC
    block. No nominal pulse or rest duration is assumed.
    """

    time_s = raw["Test_Time[s]"].to_numpy(dtype=float)
    current_A = raw["Current[A]"].to_numpy(dtype=float)
    voltage_V = raw["Voltage[V]"].to_numpy(dtype=float)
    pulse_current_A = float(pulse["mean_current_A"])

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

    if start_position is None or stop_position is None or stop_position <= start_position + 2:
        return None

    future_active = np.flatnonzero(np.abs(current_A[stop_position + 1 :]) > current_threshold_A)
    rest_end = stop_position + 1 + int(future_active[0]) if len(future_active) else len(raw)

    if rest_end - stop_position < 10:
        return None

    pre_end = start_position - 1
    pre_start = pre_end

    while pre_start > 0 and abs(current_A[pre_start - 1]) <= current_threshold_A:
        pre_start -= 1

    pre_rest_duration_s = float(time_s[pre_end] - time_s[pre_start])
    pre_tail_mask = (
        time_s
        >= max(
            time_s[pre_start],
            time_s[start_position] - 30.0,
        )
    ) & (time_s < time_s[start_position])

    if not np.any(pre_tail_mask):
        return None

    ocv_before_V = float(np.median(voltage_V[pre_tail_mask]))

    selection = slice(
        start_position,
        rest_end,
    )
    window_time_s = time_s[selection]
    window_current_A = current_A[selection]
    window_voltage_V = voltage_V[selection]

    local_time_s = window_time_s - window_time_s[0]
    local_stop = stop_position - start_position

    pulse_positions = np.arange(
        0,
        local_stop,
        dtype=int,
    )
    rest_positions = np.arange(
        local_stop,
        len(local_time_s),
        dtype=int,
    )

    if len(pulse_positions) < 3 or len(rest_positions) < 10:
        return None

    rest_time_s = local_time_s[rest_positions] - local_time_s[rest_positions[0]]
    rest_duration_s = float(rest_time_s[-1])
    tail_start_s = max(
        0.0,
        rest_duration_s - 30.0,
    )
    tail_positions = rest_positions[rest_time_s >= tail_start_s]
    ocv_after_V = float(np.median(window_voltage_V[tail_positions]))

    cumulative_charge_As = np.zeros(
        len(local_time_s),
        dtype=float,
    )
    if len(local_time_s) > 1:
        cumulative_charge_As[1:] = np.cumsum(window_current_A[:-1] * np.diff(local_time_s))

    pulse_charge_As = float(cumulative_charge_As[local_stop])
    if abs(pulse_charge_As) < 1e-9:
        return None

    ocv_fraction = np.clip(
        cumulative_charge_As / pulse_charge_As,
        0.0,
        1.0,
    )
    ocv_fraction[rest_positions] = 1.0
    ocv_path_V = ocv_before_V + (ocv_after_V - ocv_before_V) * ocv_fraction

    load_fit_positions = pulse_positions[local_time_s[pulse_positions] >= 0.5]
    selected_rest = select_fit_positions(rest_time_s)
    selected_rest = selected_rest[rest_time_s[selected_rest] >= 0.5]
    rest_fit_positions = rest_positions[selected_rest]
    fit_positions = np.unique(
        np.concatenate(
            [
                load_fit_positions,
                rest_fit_positions,
            ]
        )
    )

    if len(fit_positions) < 12:
        return None

    return LoadRestWindow(
        time_s=local_time_s,
        current_A=window_current_A,
        voltage_V=window_voltage_V,
        ocv_path_V=ocv_path_V,
        pulse_positions=pulse_positions,
        rest_positions=rest_positions,
        fit_positions=fit_positions,
        pre_rest_duration_s=(pre_rest_duration_s),
        pulse_duration_actual_s=float(time_s[stop_position] - time_s[start_position]),
        rest_duration_s=rest_duration_s,
        ocv_before_V=ocv_before_V,
        ocv_after_V=ocv_after_V,
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
    """Generate initial parameter guesses for the nonlinear fit."""

    total = max(
        float(final_recovery_V),
        2e-4,
    )

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

    fit_positions = select_fit_positions(time_s)
    fit_time = time_s[fit_positions]
    fit_recovery = recovery_V[fit_positions]

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


def unpack_direct_rc_parameters(
    parameters: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return positive RC parameters with ordered time constants."""

    resistance_1 = float(np.exp(parameters[0]))
    resistance_2 = float(np.exp(parameters[1]))
    tau_1 = float(np.exp(parameters[2]))
    tau_2 = float(tau_1 + np.exp(parameters[3]))

    return (
        resistance_1,
        resistance_2,
        tau_1,
        tau_2,
    )


def simulate_load_and_rest(
    parameters: np.ndarray,
    window: LoadRestWindow,
    resistance_0_ohm: float,
) -> np.ndarray:
    """Simulate a measured, potentially irregular current time series."""

    (
        resistance_1,
        resistance_2,
        tau_1,
        tau_2,
    ) = unpack_direct_rc_parameters(parameters)

    predicted = np.empty(
        len(window.time_s),
        dtype=float,
    )

    voltage_rc_1 = 0.0
    voltage_rc_2 = 0.0

    for index in range(len(window.time_s)):
        current = float(window.current_A[index])

        predicted[index] = float(window.ocv_path_V[index]) + resistance_0_ohm * current + voltage_rc_1 + voltage_rc_2

        if index + 1 >= len(window.time_s):
            continue

        dt = float(window.time_s[index + 1] - window.time_s[index])

        if dt <= 0:
            continue

        decay_1 = float(np.exp(-dt / tau_1))
        decay_2 = float(np.exp(-dt / tau_2))

        voltage_rc_1 = decay_1 * voltage_rc_1 + resistance_1 * (1.0 - decay_1) * current

        voltage_rc_2 = decay_2 * voltage_rc_2 + resistance_2 * (1.0 - decay_2) * current

    return predicted


def fit_load_and_rest_window(
    window: LoadRestWindow,
    resistance_0_ohm: float,
    initial_rc: tuple[float, float, float, float],
) -> tuple[dict[str, object], np.ndarray]:
    """Fit R1, R2, tau1, and tau2 to a load-and-rest window."""

    def residual(
        parameters: np.ndarray,
    ) -> np.ndarray:
        predicted = simulate_load_and_rest(
            parameters=parameters,
            window=window,
            resistance_0_ohm=resistance_0_ohm,
        )

        return predicted[window.fit_positions] - window.voltage_V[window.fit_positions]

    median_dt = float(np.median(np.diff(window.time_s)))
    minimum_tau = max(
        0.05,
        median_dt / 2.0,
    )
    maximum_tau_1 = max(
        60.0,
        window.rest_duration_s,
    )
    maximum_tau_gap = max(
        6000.0,
        10.0 * window.rest_duration_s,
    )

    lower = np.log(
        np.array(
            [
                1e-6,
                1e-6,
                minimum_tau,
                0.05,
            ],
            dtype=float,
        )
    )
    upper = np.log(
        np.array(
            [
                0.5,
                0.5,
                maximum_tau_1,
                maximum_tau_gap,
            ],
            dtype=float,
        )
    )

    resistance_1, resistance_2, tau_1, tau_2 = initial_rc

    seeds = [
        (
            resistance_1,
            resistance_2,
            tau_1,
            tau_2,
        ),
        (
            0.7 * resistance_1,
            1.3 * resistance_2,
            0.7 * tau_1,
            1.3 * tau_2,
        ),
        (
            1.3 * resistance_1,
            0.7 * resistance_2,
            1.3 * tau_1,
            max(
                1.5 * tau_1,
                0.8 * tau_2,
            ),
        ),
    ]

    best = None

    for (
        seed_resistance_1,
        seed_resistance_2,
        seed_tau_1,
        seed_tau_2,
    ) in seeds:
        seed_tau_1 = max(
            seed_tau_1,
            minimum_tau * 1.01,
        )
        seed_tau_2 = max(
            seed_tau_2,
            seed_tau_1 + 0.051,
        )

        guess = np.log(
            np.array(
                [
                    seed_resistance_1,
                    seed_resistance_2,
                    seed_tau_1,
                    seed_tau_2 - seed_tau_1,
                ],
                dtype=float,
            )
        )

        guess = np.clip(
            guess,
            lower + 1e-9,
            upper - 1e-9,
        )

        candidate = least_squares(
            residual,
            guess,
            bounds=(
                lower,
                upper,
            ),
            loss="soft_l1",
            f_scale=5e-4,
            max_nfev=3000,
        )

        if best is None or candidate.cost < best.cost:
            best = candidate

    if best is None:
        raise RuntimeError("No load-and-rest fit candidate was generated.")

    (
        resistance_1,
        resistance_2,
        tau_1,
        tau_2,
    ) = unpack_direct_rc_parameters(best.x)

    predicted = simulate_load_and_rest(
        parameters=best.x,
        window=window,
        resistance_0_ohm=resistance_0_ohm,
    )

    residual_all = predicted - window.voltage_V

    rest_time = window.time_s[window.rest_positions] - window.time_s[window.rest_positions[0]]

    load_eval = window.pulse_positions[window.time_s[window.pulse_positions] >= 0.5]
    rest_eval = window.rest_positions[rest_time >= 0.5]
    overall_eval = np.unique(
        np.concatenate(
            [
                load_eval,
                rest_eval,
            ]
        )
    )

    def metrics(
        positions: np.ndarray,
    ) -> dict[str, float]:
        values = residual_all[positions]
        measured = window.voltage_V[positions]

        denominator = float(np.sum((measured - np.mean(measured)) ** 2))

        return {
            "rmse_mV": float(np.sqrt(np.mean(values**2)) * 1000.0),
            "mae_mV": float(np.mean(np.abs(values)) * 1000.0),
            "R_squared": (float(1.0 - np.sum(values**2) / denominator) if denominator > 0 else np.nan),
        }

    overall = metrics(overall_eval)
    load = metrics(load_eval)
    rest = metrics(rest_eval)

    try:
        jacobian_condition = float(np.linalg.cond(best.jac.T @ best.jac))
    except np.linalg.LinAlgError:
        jacobian_condition = np.inf

    pulse_current = float(np.median(window.current_A[window.pulse_positions]))
    pulse_duration = window.pulse_duration_actual_s

    amplitude_1 = abs(pulse_current) * resistance_1 * (1.0 - np.exp(-pulse_duration / tau_1))
    amplitude_2 = abs(pulse_current) * resistance_2 * (1.0 - np.exp(-pulse_duration / tau_2))

    result = {
        "fit_method": "load_and_relaxation",
        "optimizer_success": bool(best.success),
        "optimizer_status": int(best.status),
        "optimizer_message": str(best.message),
        "optimizer_nfev": int(best.nfev),
        "fit_sample_count": int(len(window.fit_positions)),
        "amplitude_1_V": float(amplitude_1),
        "amplitude_2_V": float(amplitude_2),
        "R1_ohm": resistance_1,
        "R2_ohm": resistance_2,
        "C1_F": float(tau_1 / resistance_1),
        "C2_F": float(tau_2 / resistance_2),
        "R1_mohm": float(resistance_1 * 1000.0),
        "R2_mohm": float(resistance_2 * 1000.0),
        "tau_1_s": tau_1,
        "tau_2_s": tau_2,
        "tau_ratio": float(tau_2 / tau_1),
        "fit_rmse_mV": overall["rmse_mV"],
        "fit_mae_mV": overall["mae_mV"],
        "fit_R_squared": overall["R_squared"],
        "load_rmse_mV": load["rmse_mV"],
        "load_mae_mV": load["mae_mV"],
        "load_R_squared": load["R_squared"],
        "rest_rmse_mV": rest["rmse_mV"],
        "rest_mae_mV": rest["mae_mV"],
        "rest_R_squared": rest["R_squared"],
        "jacobian_condition": jacobian_condition,
        "pre_rest_duration_s": (window.pre_rest_duration_s),
        "rest_start_time_s": float(window.time_s[window.rest_positions[0]]),
        "rest_end_time_s": float(window.time_s[-1]),
        "rest_duration_s": (window.rest_duration_s),
        "rest_raw_sample_count": int(len(window.rest_positions)),
        "rest_dt_median_s": float(np.median(np.diff(rest_time))),
        "recovery_final_V": float(amplitude_1 + amplitude_2),
        "pulse_duration_actual_s": (pulse_duration),
        "ocv_before_V": window.ocv_before_V,
        "ocv_after_V": window.ocv_after_V,
        "ocv_change_mV": float((window.ocv_after_V - window.ocv_before_V) * 1000.0),
        "initial_RC_states": ("zero_after_detected_pre_rest"),
    }

    return result, predicted


def quality_flags(
    result: dict[str, object],
) -> str:
    """Return transparent quality warnings without removing fit results."""

    flags: list[str] = []

    if not bool(result["optimizer_success"]):
        flags.append("optimizer_not_converged")
    if float(result["fit_R_squared"]) < 0.98:
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
    guess_profile: str = "full",
) -> tuple[dict[str, object], FitCurve | None]:
    """Fit a complete second-order RC model to one HPPC pulse."""

    row = pulse.to_dict()
    row["pulse_row_id"] = pulse_row_id

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
        return row, None

    # Fit the voltage relaxation first to obtain initial RC estimates.
    rest = extract_rest_window(
        raw=raw,
        pulse=pulse,
    )

    if rest is None:
        row.update(
            {
                "fit_valid": False,
                "fit_exclusion_reason": "rest_window_not_found",
                "quality_flags": "rest_window_not_found",
            }
        )
        return row, None

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
        return row, None

    current_abs_A = abs(float(pulse["mean_current_A"]))
    pulse_duration_s = float(pulse["pulse_duration_s"])

    denominator_1 = current_abs_A * (1.0 - np.exp(-pulse_duration_s / float(relaxation_fit["tau_1_s"])))
    denominator_2 = current_abs_A * (1.0 - np.exp(-pulse_duration_s / float(relaxation_fit["tau_2_s"])))

    if denominator_1 <= 0.0 or denominator_2 <= 0.0:
        row.update(
            {
                "fit_valid": False,
                "fit_exclusion_reason": "invalid_initial_RC_estimate",
                "quality_flags": "invalid_initial_RC_estimate",
            }
        )
        return row, None

    seed_r1_ohm = float(relaxation_fit["amplitude_1_V"]) / denominator_1
    seed_r2_ohm = float(relaxation_fit["amplitude_2_V"]) / denominator_2

    # Fit the measured pulse and following rest together.
    window = extract_load_and_rest_window(
        raw=raw,
        pulse=pulse,
    )

    if window is None:
        row.update(
            {
                "fit_valid": False,
                "fit_exclusion_reason": ("load_and_rest_window_not_found"),
                "quality_flags": ("load_and_rest_window_not_found"),
            }
        )
        return row, None

    try:
        fit_result, predicted_voltage_V = fit_load_and_rest_window(
            window=window,
            resistance_0_ohm=float(r0_result["R0_ohm"]),
            initial_rc=(
                seed_r1_ohm,
                seed_r2_ohm,
                float(relaxation_fit["tau_1_s"]),
                float(relaxation_fit["tau_2_s"]),
            ),
        )
    except Exception as error:
        row.update(
            {
                "fit_valid": False,
                "fit_exclusion_reason": (f"load_and_rest_fit_failed:{error}"),
                "quality_flags": "load_and_rest_fit_failed",
            }
        )
        return row, None

    row.update(fit_result)

    required_parameters = np.asarray(
        [
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

    row["fit_exclusion_reason"] = "" if row["fit_valid"] else "nonphysical_fit"
    row["quality_flags"] = quality_flags(row) if row["fit_valid"] else "nonphysical_fit"

    rest_positions = window.rest_positions
    rest_voltage_V = window.voltage_V[rest_positions]
    fitted_rest_voltage_V = predicted_voltage_V[rest_positions]

    direction_sign = 1.0 if str(pulse["pulse_direction"]) == "discharge" else -1.0

    curve = FitCurve(
        pulse_row_id=pulse_row_id,
        time_s=(window.time_s[rest_positions] - window.time_s[rest_positions[0]]),
        recovery_measured_V=direction_sign * (rest_voltage_V - rest_voltage_V[0]),
        recovery_fitted_V=direction_sign * (fitted_rest_voltage_V - fitted_rest_voltage_V[0]),
    )

    return row, curve


def create_ecm_parameters(
    hppc_blocks: pd.DataFrame | list[pd.DataFrame],
    config: _BatteryConfigClass,
    cell_id: str,
    block_metadata: list[dict[str, object]] | None = None,
    current_threshold_A: float = 0.05,
    guess_profile: str = "full",
) -> tuple[pd.DataFrame, list[FitCurve]]:
    """Fit 2RC parameters for user-supplied HPPC blocks.

    The caller must identify the HPPC blocks. Within each block, consecutive
    current segments are extracted automatically. Every non-rest segment that
    is followed by a rest segment is treated as one pulse candidate.

    Optional block metadata, such as checkup number, calendar age, or FEC, is
    copied into the corresponding result rows.
    """

    blocks = [hppc_blocks] if isinstance(hppc_blocks, pd.DataFrame) else list(hppc_blocks)

    if not blocks:
        return pd.DataFrame(), []

    if block_metadata is None:
        metadata = [{} for _ in blocks]
    else:
        if len(block_metadata) != len(blocks):
            raise ValueError("block_metadata must contain one entry per HPPC block.")
        metadata = block_metadata

    result_rows: list[dict[str, object]] = []
    fit_curves: list[FitCurve] = []
    pulse_row_id = 0

    for block_id, (raw, block_info) in enumerate(
        zip(
            blocks,
            metadata,
            strict=True,
        )
    ):
        segments = extract_current_segments(
            df=raw,
            cell_id=cell_id,
            current_threshold_A=current_threshold_A,
        )

        if segments.empty:
            continue

        following_state = segments["state"].shift(-1)
        following_test_index = segments["TestIndex"].shift(-1)

        pulse_mask = segments["state"].ne(0) & following_state.eq(0) & segments["TestIndex"].eq(following_test_index)

        pulse_segments = segments.loc[pulse_mask]

        numeric_test_index = pd.to_numeric(
            raw["TestIndex"],
            errors="coerce",
        )

        for _, segment in pulse_segments.iterrows():
            pulse = segment.copy()

            pulse["pulse_start_time_s"] = float(segment["start_time_s"])
            pulse["pulse_end_time_s"] = float(segment["end_time_s"])
            pulse["pulse_duration_s"] = float(segment["duration_s"])
            pulse["pulse_direction"] = "charge" if float(segment["mean_current_A"]) > 0.0 else "discharge"
            pulse["SOC"] = float(segment["soc_start"])
            pulse["SOC_is_physical"] = bool(np.isfinite(pulse["SOC"]) and 0.0 <= pulse["SOC"] <= 1.0)

            test_index = float(segment["TestIndex"])
            raw_test = raw.loc[numeric_test_index.eq(test_index)].copy()

            result, curve = fit_ecm_pulse(
                raw=raw_test,
                pulse=pulse,
                pulse_row_id=pulse_row_id,
                config=config,
                guess_profile=guess_profile,
            )

            result["hppc_block_id"] = block_id

            for key, value in block_info.items():
                if key not in result:
                    result[key] = value

            result_rows.append(result)

            if curve is not None:
                fit_curves.append(curve)

            pulse_row_id += 1

    return pd.DataFrame(result_rows), fit_curves


def calculate_pulse_r0(
    raw: pd.DataFrame,
    pulse: pd.Series,
    config: _BatteryConfigClass,
) -> dict[str, float | str | bool]:
    """Calculate R0 at the start and stop of one current pulse."""

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
