import numpy as np
import pandas as pd
import pytest

from pydpeet.process.analyze.ecm import (
    _annotate_quality_status,
    _annotate_soc_paths,
    _filter_parameter_rows,
    _plot_parameters_over_age,
    _plot_parameters_over_soc,
    _select_target_soc_rows,
    create_ecm_parameters,
    fit_rest_window,
    recovery_model,
)


def test_fit_rest_window_recovers_known_parameters() -> None:
    time_s = np.linspace(0.0, 600.0, 601)
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
    assert result["amplitude_1_V"] == pytest.approx(0.025, rel=0.05)
    assert result["tau_1_s"] == pytest.approx(3.0, rel=0.05)
    assert result["amplitude_2_V"] == pytest.approx(0.015, rel=0.05)
    assert result["tau_2_s"] == pytest.approx(80.0, rel=0.05)
    assert result["fit_R_squared"] > 0.999
    np.testing.assert_allclose(predicted, recovery_v, atol=1e-5)


def test_quality_status_uses_pass_review_and_excluded() -> None:
    rows = pd.DataFrame(
        {
            "fit_valid": [True, True, False],
            "quality_flags": ["", "R_squared_below_0.98", "nonphysical_fit"],
        }
    )

    result = _annotate_quality_status(rows)

    assert result["quality_status"].tolist() == [
        "pass",
        "review",
        "excluded",
    ]


def test_soc_paths_are_split_at_the_soc_turning_point() -> None:
    rows = pd.DataFrame(
        {
            "cell_id": ["cell"] * 7,
            "TestIndex": [1.0] * 7,
            "hppc_block_id": [1] * 7,
            "pulse_start_time_s": np.arange(7, dtype=float),
            "SOC": [0.90, 0.60, 0.30, 0.10, 0.35, 0.65, 0.90],
        }
    )

    result = _annotate_soc_paths(rows)

    assert result["soc_path"].tolist() == [
        "descending",
        "descending",
        "descending",
        "descending",
        "ascending",
        "ascending",
        "ascending",
    ]
    assert result["sweep_id"].tolist() == [1, 1, 1, 1, 2, 2, 2]


def test_plot_filter_defaults_keep_only_matching_pass_rows() -> None:
    rows = pd.DataFrame(
        {
            "quality_status": ["pass", "review", "pass", "pass"],
            "pulse_direction": ["charge", "charge", "charge", "discharge"],
            "soc_path": ["ascending"] * 4,
            "mean_current_A": [4.8, 4.8, 7.2, -4.8],
        }
    )

    result = _filter_parameter_rows(
        parameters=rows,
        quality_statuses=("pass",),
        pulse_directions=("charge",),
        soc_paths=("ascending",),
        target_current_abs_A=4.8,
        current_tolerance_A=0.25,
    )

    assert result.index.tolist() == [0]


def test_target_soc_selection_is_per_block_and_within_tolerance() -> None:
    rows = pd.DataFrame(
        {
            "cell_id": ["cell"] * 4,
            "hppc_block_id": [1, 1, 2, 2],
            "pulse_direction": ["charge"] * 4,
            "soc_path": ["ascending"] * 4,
            "SOC": [0.24, 0.52, 0.26, 0.70],
            "R0_mohm": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = _select_target_soc_rows(
        parameters=rows,
        target_socs=(0.25, 0.50),
        soc_tolerance=0.02,
    )

    assert result["hppc_block_id"].tolist() == [1, 1, 2]
    assert result["target_SOC"].tolist() == pytest.approx([0.25, 0.50, 0.25])


def test_single_point_series_are_not_plotted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = pd.DataFrame(
        {
            "cell_id": ["cell"],
            "hppc_block_id": [1],
            "pulse_direction": ["charge"],
            "soc_path": ["ascending"],
            "quality_status": ["pass"],
            "SOC": [0.50],
            "calendar_age_days": [10.0],
            "R0_mohm": [20.0],
        }
    )
    show_calls: list[bool] = []
    monkeypatch.setattr(
        "pydpeet.process.analyze.ecm.plt.show",
        lambda: show_calls.append(True),
    )

    _plot_parameters_over_soc(rows, "charge", "ascending")
    _plot_parameters_over_age(
        rows,
        "calendar_age_days",
        "Calendar age [days]",
        "charge",
        "ascending",
        (0.50,),
        0.02,
    )

    assert show_calls == []


def test_create_ecm_parameters_adds_quality_and_soc_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pulse_socs = [0.90, 0.60, 0.10, 0.40, 0.80]
    segment_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, float]] = []
    for pulse_index, soc in enumerate(pulse_socs):
        pulse_time = float(20 * pulse_index)
        segment_rows.extend(
            [
                {
                    "cell_id": "cell",
                    "TestIndex": 1.0,
                    "run_index": 2 * pulse_index,
                    "state": 1,
                    "start_time_s": pulse_time,
                    "end_time_s": pulse_time + 9.0,
                    "duration_s": 10.0,
                    "mean_current_A": 4.8,
                    "soc_start": soc,
                    "hppc_block_id": 1,
                    "hppc_role": "pulse",
                },
                {
                    "cell_id": "cell",
                    "TestIndex": 1.0,
                    "run_index": 2 * pulse_index + 1,
                    "state": 0,
                    "start_time_s": pulse_time + 10.0,
                    "end_time_s": pulse_time + 19.0,
                    "duration_s": 10.0,
                    "mean_current_A": 0.0,
                    "soc_start": soc,
                    "hppc_block_id": 1,
                    "hppc_role": "rest",
                },
            ]
        )
        raw_rows.append(
            {
                "TestIndex": 1.0,
                "Test_Time[s]": pulse_time,
                "Current[A]": 4.8,
                "Voltage[V]": 4.0,
            }
        )

    def fake_fit_ecm_pulse(**kwargs: object) -> dict[str, object]:
        pulse = kwargs["pulse"]
        assert isinstance(pulse, pd.Series)
        return {
            **pulse.to_dict(),
            "fit_valid": True,
            "quality_flags": "",
        }

    monkeypatch.setattr(
        "pydpeet.process.analyze.ecm.fit_ecm_pulse",
        fake_fit_ecm_pulse,
    )

    result = create_ecm_parameters(
        df=pd.DataFrame(raw_rows),
        hppc_segments=pd.DataFrame(segment_rows),
        config=object(),
    )

    assert result["quality_status"].eq("pass").all()
    assert result["soc_path"].tolist() == [
        "descending",
        "descending",
        "descending",
        "ascending",
        "ascending",
    ]
