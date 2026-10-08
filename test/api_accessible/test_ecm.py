from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pydpeet.process.analyze.configs.battery_config import (
    battery_config_wrapper,
)
from pydpeet.process.analyze.ecm import (
    FitCurve,
    LoadRestWindow,
    calculate_pulse_r0,
    create_ecm_parameters,
    extract_load_and_rest_window,
    fit_ecm_pulse,
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


def test_extract_load_and_rest_window() -> None:
    time_s = np.arange(
        0.0,
        101.0,
        1.0,
    )
    current_A = np.zeros_like(time_s)
    current_A[10:20] = 1.0

    voltage_V = np.full_like(
        time_s,
        3.7,
    )
    voltage_V[10:20] = 3.6
    voltage_V[20:] = 3.65 + 0.05 * (1.0 - np.exp(-(time_s[20:] - 20.0) / 10.0))

    raw = pd.DataFrame(
        {
            "Test_Time[s]": time_s,
            "Current[A]": current_A,
            "Voltage[V]": voltage_V,
        }
    )
    pulse = pd.Series(
        {
            "pulse_start_time_s": 10.0,
            "pulse_end_time_s": 19.0,
            "mean_current_A": 1.0,
        }
    )

    window = extract_load_and_rest_window(
        raw=raw,
        pulse=pulse,
    )

    assert window is not None
    assert len(window.pulse_positions) == 10
    assert len(window.rest_positions) == 81
    assert np.isclose(
        window.pulse_duration_actual_s,
        10.0,
    )
    assert np.all(window.current_A[window.pulse_positions] == 1.0)
    assert np.all(window.current_A[window.rest_positions] == 0.0)


def test_calculate_pulse_r0_prefers_stop_value() -> None:
    raw = pd.DataFrame(
        {
            "Test_Time[s]": np.arange(
                31,
                dtype=float,
            ),
            "Current[A]": np.r_[
                np.zeros(10),
                -np.ones(10),
                np.zeros(11),
            ],
            "Voltage[V]": np.r_[
                np.full(10, 3.70),
                np.full(10, 3.60),
                np.full(11, 3.65),
            ],
        }
    )
    pulse = pd.Series(
        {
            "pulse_start_time_s": 10.0,
            "pulse_end_time_s": 19.0,
            "mean_current_A": -1.0,
        }
    )
    config = battery_config_wrapper(
        min_current_diff=0.5,
        max_time_diff=2.0,
        min_voltage_diff=0.001,
        ignore_negative_resistance_values=True,
    )

    result = calculate_pulse_r0(
        raw=raw,
        pulse=pulse,
        config=config,
    )

    assert result["R0_valid"]
    assert result["R0_source"] == "stop"
    assert result["R0_stop_ohm"] == pytest.approx(0.05)
    assert result["R0_ohm"] == pytest.approx(0.05)


def test_fit_ecm_pulse_connects_complete_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = pd.DataFrame(
        {
            "Test_Time[s]": [0.0, 1.0, 2.0, 3.0],
            "Current[A]": [1.0, 1.0, 0.0, 0.0],
            "Voltage[V]": [4.0, 3.9, 3.95, 3.98],
        }
    )

    pulse = pd.Series(
        {
            "pulse_start_time_s": 0.0,
            "pulse_end_time_s": 2.0,
            "pulse_duration_s": 2.0,
            "mean_current_A": 1.0,
            "pulse_direction": "charge",
        }
    )

    window = LoadRestWindow(
        time_s=np.array([0.0, 1.0, 2.0, 3.0]),
        current_A=np.array([1.0, 1.0, 0.0, 0.0]),
        voltage_V=np.array([4.0, 3.9, 3.95, 3.98]),
        ocv_path_V=np.array([4.0, 4.0, 4.0, 4.0]),
        pulse_positions=np.array([0, 1]),
        rest_positions=np.array([2, 3]),
        fit_positions=np.array([0, 1, 2, 3]),
        pre_rest_duration_s=30.0,
        pulse_duration_actual_s=2.0,
        rest_duration_s=1.0,
        ocv_before_V=4.0,
        ocv_after_V=3.98,
    )

    monkeypatch.setattr(
        "pydpeet.process.analyze.ecm.calculate_pulse_r0",
        lambda **kwargs: {
            "R0_start_ohm": 0.05,
            "R0_stop_ohm": 0.05,
            "R0_ohm": 0.05,
            "R0_source": "stop",
            "R0_valid": True,
        },
    )

    monkeypatch.setattr(
        "pydpeet.process.analyze.ecm.extract_rest_window",
        lambda **kwargs: (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.01, 0.015]),
            {
                "rest_start_time_s": 2.0,
                "rest_end_time_s": 4.0,
                "rest_duration_s": 2.0,
                "rest_raw_sample_count": 3,
                "rest_dt_median_s": 1.0,
                "recovery_final_V": 0.015,
            },
        ),
    )

    monkeypatch.setattr(
        "pydpeet.process.analyze.ecm.fit_rest_window",
        lambda **kwargs: (
            {
                "amplitude_1_V": 0.01,
                "amplitude_2_V": 0.005,
                "tau_1_s": 1.0,
                "tau_2_s": 10.0,
            },
            np.array([0.0, 0.01, 0.015]),
        ),
    )

    monkeypatch.setattr(
        "pydpeet.process.analyze.ecm.extract_load_and_rest_window",
        lambda **kwargs: window,
    )

    monkeypatch.setattr(
        "pydpeet.process.analyze.ecm.fit_load_and_rest_window",
        lambda **kwargs: (
            {
                "optimizer_success": True,
                "fit_method": "load_and_relaxation",
                "R1_ohm": 0.01,
                "R2_ohm": 0.02,
                "C1_F": 100.0,
                "C2_F": 1000.0,
                "R1_mohm": 10.0,
                "R2_mohm": 20.0,
                "tau_1_s": 1.0,
                "tau_2_s": 20.0,
                "tau_ratio": 20.0,
                "fit_rmse_mV": 0.5,
                "fit_R_squared": 0.99,
                "jacobian_condition": 100.0,
                "rest_duration_s": 30.0,
            },
            np.array([4.0, 3.9, 3.95, 3.98]),
        ),
    )

    result, curve = fit_ecm_pulse(
        raw=raw,
        pulse=pulse,
        pulse_row_id=7,
        config=object(),
    )

    assert result["fit_valid"]
    assert result["pulse_row_id"] == 7
    assert result["R0_ohm"] == pytest.approx(0.05)
    assert result["R1_ohm"] == pytest.approx(0.01)
    assert result["R2_ohm"] == pytest.approx(0.02)
    assert result["C1_F"] == pytest.approx(100.0)
    assert result["C2_F"] == pytest.approx(1000.0)
    assert result["quality_flags"] == ""
    assert curve is not None
    assert curve.pulse_row_id == 7


def test_create_ecm_parameters_processes_all_pulses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = pd.DataFrame(
        {
            "TestIndex": [1.0, 1.0, 1.0, 1.0],
            "Test_Time[s]": [0.0, 10.0, 20.0, 30.0],
            "Current[A]": [1.0, 0.0, -1.0, 0.0],
            "Voltage[V]": [4.0, 4.1, 4.0, 3.9],
        }
    )

    segments = pd.DataFrame(
        {
            "TestIndex": [1.0, 1.0, 1.0, 1.0],
            "state": [1, 0, -1, 0],
            "start_time_s": [0.0, 10.0, 20.0, 30.0],
            "end_time_s": [9.0, 19.0, 29.0, 39.0],
            "duration_s": [10.0, 10.0, 10.0, 10.0],
            "mean_current_A": [1.0, 0.0, -1.0, 0.0],
            "soc_start": [0.8, 0.8, 0.6, 0.6],
        }
    )

    monkeypatch.setattr(
        "pydpeet.process.analyze.ecm.extract_current_segments",
        lambda **kwargs: segments,
    )

    def fake_fit_ecm_pulse(
        **kwargs: object,
    ) -> tuple[dict[str, object], FitCurve]:
        pulse = kwargs["pulse"]
        pulse_row_id = int(kwargs["pulse_row_id"])

        assert isinstance(pulse, pd.Series)

        curve = FitCurve(
            pulse_row_id=pulse_row_id,
            time_s=np.array([0.0, 1.0]),
            recovery_measured_V=np.array([0.0, 0.01]),
            recovery_fitted_V=np.array([0.0, 0.01]),
        )

        return (
            {
                "pulse_row_id": pulse_row_id,
                "pulse_direction": pulse["pulse_direction"],
                "SOC": pulse["SOC"],
                "fit_valid": True,
            },
            curve,
        )

    monkeypatch.setattr(
        "pydpeet.process.analyze.ecm.fit_ecm_pulse",
        fake_fit_ecm_pulse,
    )

    results, curves = create_ecm_parameters(
        hppc_blocks=raw,
        config=object(),
        cell_id="test-cell",
        block_metadata=[
            {
                "checkup_id": 3,
                "calendar_age_days": 120.0,
            }
        ],
    )

    assert len(results) == 2
    assert len(curves) == 2
    assert results["pulse_direction"].tolist() == [
        "charge",
        "discharge",
    ]
    assert results["SOC"].tolist() == pytest.approx([0.8, 0.6])
    assert results["checkup_id"].tolist() == [3, 3]
    assert results["calendar_age_days"].tolist() == pytest.approx([120.0, 120.0])
    assert results["hppc_block_id"].tolist() == [0, 0]
