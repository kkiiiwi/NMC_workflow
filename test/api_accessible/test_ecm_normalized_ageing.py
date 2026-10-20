import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pydpeet.process.analyze.ecm import _plot_parameters_over_age, _select_target_soc_rows


def test_target_soc_tie_uses_lower_fit_error():
    rows = pd.DataFrame({
        "cell_id": ["cell", "cell"], "hppc_block_id": [1, 1],
        "pulse_direction": ["charge", "charge"], "soc_path": ["ascending", "ascending"],
        "SOC": [.24, .26], "fit_rmse_mV": [2., .3], "source_pulse_id": [1, 2],
    })
    result = _select_target_soc_rows(rows, (.25,), .02)
    assert result["source_pulse_id"].tolist() == [2]


@pytest.mark.parametrize("normalized", [True, False])
def test_age_plot_keeps_independent_nonzero_age_baselines(monkeypatch, normalized):
    rows = pd.DataFrame({
        "cell_id": ["cell"] * 4,
        "hppc_block_id": [1, 2, 1, 2],
        "TestIndex": [1, 2, 1, 2],
        "pulse_direction": ["charge"] * 4,
        "soc_path": ["ascending"] * 4,
        "quality_status": ["pass"] * 4,
        "SOC": [.25, .25, .75, .75],
        "calendar_age_days": [10., 30., 10., 30.],
        "R0_mohm": [2., 4., 10., 5.],
    })
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        _plot_parameters_over_age(rows, "calendar_age_days", "Calendar age [days]",
                                  "charge", "ascending", (.25, .75), .02,
                                  normalize_to_first=normalized)
        axis = plt.gcf().axes[0]
        curves = [line for line in axis.lines if line.get_marker() == "o"]
        assert len(curves) == 2
        expected = [[100., 200.], [100., 50.]] if normalized else [[2., 4.], [10., 5.]]
        for line, values in zip(curves, expected, strict=True):
            np.testing.assert_allclose(line.get_xdata(), [10., 30.])
            np.testing.assert_allclose(line.get_ydata(), values)
        if normalized:
            assert "first valid" in axis.get_ylabel()
            assert all("ref. 10" in line.get_label() for line in curves)
    finally:
        plt.close("all")


def test_age_normalization_does_not_replace_zero_baseline(monkeypatch):
    rows = pd.DataFrame({
        "cell_id": ["cell", "cell"], "hppc_block_id": [1, 2],
        "pulse_direction": ["charge", "charge"], "soc_path": ["ascending", "ascending"],
        "quality_status": ["pass", "pass"], "SOC": [.5, .5],
        "FEC": [0., 50.], "R0_mohm": [0., 2.],
    })
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        _plot_parameters_over_age(rows, "FEC", "FEC [-]", "charge", "ascending", (.5,), .02,
                                  normalize_to_first=True)
        assert not [line for line in plt.gcf().axes[0].lines if line.get_marker() == "o"]
    finally:
        plt.close("all")
