from dataclasses import replace

import numpy as np
import pytest

from pydpeet.process.analyze.ecm import (
    LoadRestWindow,
    fit_load_and_rest_window,
    fit_rest_window,
    recovery_model,
    simulate_load_and_rest,
)


def test_fit_rest_window_recovers_known_parameters() -> None:
    time_s = np.linspace(
        0.0,
        600.0,
        601,
    )

    recovery_v = recovery_model(
        time_s=time_s,
        amplitude_1_V=0.025,
        tau_1_s=3.0,
        amplitude_2_V=0.015,
        tau_2_s=80.0,
    )

    result, predicted = fit_rest_window(
        time_s=time_s,
        recovery_V=recovery_v,
    )

    assert result["optimizer_success"]
    assert result["amplitude_1_V"] == pytest.approx(
        0.025,
        rel=0.05,
    )
    assert result["tau_1_s"] == pytest.approx(
        3.0,
        rel=0.05,
    )
    assert result["amplitude_2_V"] == pytest.approx(
        0.015,
        rel=0.05,
    )
    assert result["tau_2_s"] == pytest.approx(
        80.0,
        rel=0.05,
    )
    assert result["tau_1_s"] < result["tau_2_s"]
    assert result["fit_R_squared"] > 0.999

    np.testing.assert_allclose(
        predicted,
        recovery_v,
        atol=1e-5,
    )


def test_fit_load_and_rest_window_recovers_known_parameters() -> None:
    time_s = np.arange(
        0.0,
        661.0,
        1.0,
    )

    # A 2 A discharge pulse followed by a 600 s rest.
    current_a = np.zeros_like(time_s)
    current_a[time_s < 60.0] = -2.0

    ocv_v = np.full_like(
        time_s,
        4.0,
    )

    pulse_positions = np.flatnonzero(time_s < 60.0)
    rest_positions = np.flatnonzero(time_s >= 60.0)

    # Skip the initial sample during fitting.
    fit_positions = np.arange(
        1,
        len(time_s),
        dtype=int,
    )

    window = LoadRestWindow(
        time_s=time_s,
        current_A=current_a,
        voltage_V=np.zeros_like(time_s),
        ocv_path_V=ocv_v,
        pulse_positions=pulse_positions,
        rest_positions=rest_positions,
        fit_positions=fit_positions,
        pre_rest_duration_s=300.0,
        pulse_duration_actual_s=60.0,
        rest_duration_s=600.0,
        ocv_before_V=4.0,
        ocv_after_V=4.0,
    )

    resistance_0_ohm = 0.010
    resistance_1_ohm = 0.020
    resistance_2_ohm = 0.010
    tau_1_s = 5.0
    tau_2_s = 80.0

    true_parameters = np.log(
        np.array(
            [
                resistance_1_ohm,
                resistance_2_ohm,
                tau_1_s,
                tau_2_s - tau_1_s,
            ],
            dtype=float,
        )
    )

    measured_voltage_v = simulate_load_and_rest(
        parameters=true_parameters,
        window=window,
        resistance_0_ohm=resistance_0_ohm,
    )

    # Replace the placeholder voltage with the simulated measurement.
    window = replace(
        window,
        voltage_V=measured_voltage_v,
    )

    result, predicted_voltage_v = fit_load_and_rest_window(
        window=window,
        resistance_0_ohm=resistance_0_ohm,
        initial_rc=(
            0.015,
            0.012,
            4.0,
            100.0,
        ),
    )

    assert result["optimizer_success"]
    assert result["R1_ohm"] == pytest.approx(
        resistance_1_ohm,
        rel=0.05,
    )
    assert result["R2_ohm"] == pytest.approx(
        resistance_2_ohm,
        rel=0.05,
    )
    assert result["tau_1_s"] == pytest.approx(
        tau_1_s,
        rel=0.05,
    )
    assert result["tau_2_s"] == pytest.approx(
        tau_2_s,
        rel=0.05,
    )
    assert result["tau_1_s"] < result["tau_2_s"]
    assert result["fit_R_squared"] > 0.999

    np.testing.assert_allclose(
        predicted_voltage_v,
        measured_voltage_v,
        atol=1e-5,
    )
