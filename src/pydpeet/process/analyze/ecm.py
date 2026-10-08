"""Fit second-order RC equivalent-circuit models to battery pulse data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


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
