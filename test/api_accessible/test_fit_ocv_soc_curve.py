import numpy as np
import pandas as pd
import pytest

from pydpeet.process.analyze.extract.ocv_curve import fit_ocv_soc_curve


def test_fit_ocv_soc_curve_enforces_monotonic_voltage() -> None:
    points = pd.DataFrame(
        {
            "SOC": [0.05, 0.15, 0.25, 0.35],
            "Voltage[V]": [3.0, 3.2, 3.1, 3.5],
        }
    )

    binned, curve = fit_ocv_soc_curve(points, bin_width=0.1, grid_step=0.05)

    np.testing.assert_allclose(
        binned["ocv_monotonic_V"],
        [3.0, 3.15, 3.15, 3.5],
    )
    assert np.all(np.diff(curve["OCV[V]"]) >= -1e-12)
    assert curve["SOC"].min() == pytest.approx(binned["SOC"].min())
    assert curve["SOC"].max() == pytest.approx(binned["SOC"].max())


def test_fit_ocv_soc_curve_uses_bin_counts_as_pava_weights() -> None:
    points = pd.DataFrame(
        {
            "SOC": [0.05, 0.15, 0.16, 0.25, 0.35],
            "Voltage[V]": [3.0, 3.2, 3.2, 3.0, 3.5],
        }
    )

    binned, _ = fit_ocv_soc_curve(points, bin_width=0.1, grid_step=0.05)

    expected_merged_voltage = (2 * 3.2 + 3.0) / 3
    assert binned.loc[1, "ocv_monotonic_V"] == pytest.approx(expected_merged_voltage)
    assert binned.loc[2, "ocv_monotonic_V"] == pytest.approx(expected_merged_voltage)


@pytest.mark.parametrize(
    ("bin_width", "grid_step"),
    [(0.0, 0.001), (0.11, 0.001), (0.01, 0.0), (0.01, 0.02)],
)
def test_fit_ocv_soc_curve_rejects_invalid_resolution(bin_width: float, grid_step: float) -> None:
    points = pd.DataFrame({"SOC": [0.1, 0.2], "Voltage[V]": [3.1, 3.2]})

    with pytest.raises(ValueError):
        fit_ocv_soc_curve(points, bin_width=bin_width, grid_step=grid_step)


def test_fit_ocv_soc_curve_requires_two_occupied_bins() -> None:
    points = pd.DataFrame({"SOC": [0.11, 0.12], "Voltage[V]": [3.1, 3.2]})

    with pytest.raises(ValueError, match="At least two SOC bins"):
        fit_ocv_soc_curve(points, bin_width=0.1)


def test_fit_ocv_soc_curve_requires_input_columns() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        fit_ocv_soc_curve(pd.DataFrame({"SOC": [0.1, 0.2]}))
