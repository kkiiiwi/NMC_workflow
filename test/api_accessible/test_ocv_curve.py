import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pydpeet.process.analyze.extract.ocv_curve import (
    create_ocv_curves,
    fit_ocv_soc_curve,
)


def test_fit_ocv_soc_curve_enforces_monotonic_voltage() -> None:
    points = pd.DataFrame(
        {
            "SOC": [0.05, 0.15, 0.25, 0.35],
            "Voltage[V]": [3.0, 3.2, 3.1, 3.5],
        }
    )

    binned, curve = fit_ocv_soc_curve(
        points,
        bin_width=0.1,
        grid_step=0.03,
    )

    np.testing.assert_allclose(
        binned["ocv_monotonic_V"],
        [3.0, 3.15, 3.15, 3.5],
    )

    # The fitted voltage must remain non-decreasing.
    assert np.all(
        np.diff(curve["OCV[V]"].to_numpy()) >= -1e-12
    )

    # The interpolation grid must contain the exact observed endpoints.
    assert curve["SOC"].iloc[0] == binned["SOC"].min()
    assert curve["SOC"].iloc[-1] == binned["SOC"].max()

    # PCHIP must not produce missing values within the observed range.
    assert np.isfinite(
        curve[["SOC", "OCV[V]"]].to_numpy()
    ).all()


def test_fit_ocv_soc_curve_uses_bin_counts_as_pava_weights() -> None:
    points = pd.DataFrame(
        {
            "SOC": [0.05, 0.15, 0.16, 0.25, 0.35],
            "Voltage[V]": [3.0, 3.2, 3.2, 3.0, 3.5],
        }
    )

    binned, _ = fit_ocv_soc_curve(
        points,
        bin_width=0.1,
        grid_step=0.05,
    )

    expected_merged_voltage = (2 * 3.2 + 3.0) / 3

    assert binned.loc[
        1,
        "ocv_monotonic_V",
    ] == pytest.approx(expected_merged_voltage)

    assert binned.loc[
        2,
        "ocv_monotonic_V",
    ] == pytest.approx(expected_merged_voltage)


@pytest.mark.parametrize(
    ("bin_width", "grid_step"),
    [
        (0.0, 0.001),
        (0.11, 0.001),
        (0.01, 0.0),
        (0.01, 0.02),
    ],
)
def test_fit_ocv_soc_curve_rejects_invalid_resolution(
    bin_width: float,
    grid_step: float,
) -> None:
    points = pd.DataFrame(
        {
            "SOC": [0.1, 0.2],
            "Voltage[V]": [3.1, 3.2],
        }
    )

    with pytest.raises(ValueError):
        fit_ocv_soc_curve(
            points,
            bin_width=bin_width,
            grid_step=grid_step,
        )


def test_fit_ocv_soc_curve_requires_two_occupied_bins() -> None:
    points = pd.DataFrame(
        {
            "SOC": [0.11, 0.12],
            "Voltage[V]": [3.1, 3.2],
        }
    )

    with pytest.raises(
        ValueError,
        match="At least two SOC bins",
    ):
        fit_ocv_soc_curve(
            points,
            bin_width=0.1,
        )


def test_fit_ocv_soc_curve_requires_input_columns() -> None:
    points = pd.DataFrame(
        {
            "SOC": [0.1, 0.2],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        fit_ocv_soc_curve(points)


def _make_ocv_test_data() -> tuple[
    list[pd.DataFrame],
    pd.DataFrame,
]:
    """Create small synthetic iOCV blocks and corresponding rest data."""
    ocv_blocks: list[pd.DataFrame] = []
    primitive_rows: list[dict[str, object]] = []

    directions = [
        "Charge",
        "Discharge",
    ]
    soc_values = [
        0.1,
        0.3,
        0.5,
        0.7,
    ]
    voltage_values = [
        3.2,
        3.5,
        3.45,
        3.9,
    ]

    for block_id, direction in enumerate(
        directions,
        start=1,
    ):
        block_rows: list[dict[str, object]] = []

        for point_index, (soc, voltage) in enumerate(
            zip(
                soc_values,
                voltage_values,
                strict=True,
            )
        ):
            segment_id = block_id * 100 + point_index

            block_rows.append(
                {
                    "ID": segment_id,
                    "iOCV_type": direction,
                    "SOC": soc,
                    "Voltage[V]": voltage,
                    "Current[A]": 0.0,
                }
            )

            # Generate a sufficiently long and stable rest segment.
            for time_s in (
                0.0,
                300.0,
                330.0,
                360.0,
            ):
                primitive_rows.append(
                    {
                        "ID": segment_id,
                        "Type": "Rest",
                        "Test_Time[s]": time_s,
                        "Voltage[V]": voltage,
                    }
                )

        ocv_blocks.append(
            pd.DataFrame(block_rows)
        )

    return (
        ocv_blocks,
        pd.DataFrame(primitive_rows),
    )


def test_create_ocv_curves_returns_screened_results() -> None:
    ocv_blocks, df_primitives = _make_ocv_test_data()

    points, binned, curves = create_ocv_curves(
        ocv_blocks=ocv_blocks,
        df_primitives=df_primitives,
        visualize=False,
    )

    assert not points.empty
    assert not binned.empty
    assert not curves.empty

    # All synthetic rest points satisfy the configured QC criteria.
    assert points["is_valid_for_ocv"].all()

    assert set(curves["ocv_block_id"]) == {
        1,
        2,
    }
    assert set(curves["direction"]) == {
        "charge",
        "discharge",
    }

    assert np.isfinite(
        curves[["SOC", "OCV[V]"]].to_numpy()
    ).all()


def test_create_ocv_curves_uses_default_legend_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ocv_blocks, df_primitives = _make_ocv_test_data()

    # Use a non-interactive backend and prevent a visible window.
    plt.switch_backend("Agg")
    monkeypatch.setattr(
        plt,
        "show",
        lambda: None,
    )
    plt.close("all")

    create_ocv_curves(
        ocv_blocks=ocv_blocks,
        df_primitives=df_primitives,
        visualize=True,
    )

    figure_numbers = plt.get_fignums()

    # One figure for charge and one for discharge.
    assert len(figure_numbers) == 2

    legend_labels: set[str] = set()

    for figure_number in figure_numbers:
        figure = plt.figure(figure_number)
        legend = figure.axes[0].get_legend()

        legend_labels.update(
            text.get_text()
            for text in legend.get_texts()
        )

    assert legend_labels == {
        "iOCV block 1",
        "iOCV block 2",
    }

    plt.close("all")


def test_create_ocv_curves_uses_custom_and_fallback_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ocv_blocks, df_primitives = _make_ocv_test_data()

    plt.switch_backend("Agg")
    monkeypatch.setattr(
        plt,
        "show",
        lambda: None,
    )
    plt.close("all")

    create_ocv_curves(
        ocv_blocks=ocv_blocks,
        df_primitives=df_primitives,
        visualize=True,
        curve_labels={
            1: "Initial state",
        },
        legend_title="Alterungszustand",
    )

    figure_numbers = plt.get_fignums()
    assert len(figure_numbers) == 2

    legend_labels: set[str] = set()
    legend_titles: set[str] = set()

    for figure_number in figure_numbers:
        figure = plt.figure(figure_number)
        legend = figure.axes[0].get_legend()

        legend_labels.update(
            text.get_text()
            for text in legend.get_texts()
        )
        legend_titles.add(
            legend.get_title().get_text()
        )

    # Block 1 uses the user-defined label; block 2 uses the fallback.
    assert legend_labels == {
        "Initial state",
        "iOCV block 2",
    }
    assert legend_titles == {
        "Alterungszustand",
    }

    plt.close("all")
