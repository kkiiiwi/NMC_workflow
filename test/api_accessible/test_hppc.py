import numpy as np
import pandas as pd
import pytest

from pydpeet.process.analyze.extract.hppc import (
    extract_current_segments,
)


def test_extract_current_segments_merges_across_batches() -> None:
    data = pd.DataFrame(
        {
            "TestIndex": [0.0] * 7,
            "Test_Time[s]": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ],
            "Current[A]": [
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
            ],
            "Voltage[V]": [
                3.70,
                3.70,
                3.65,
                3.64,
                3.63,
                3.68,
                3.69,
            ],
            "SOC": [
                0.50,
                0.50,
                0.49,
                0.49,
                0.48,
                0.48,
                0.48,
            ],
        }
    )

    segments = extract_current_segments(
        df=data,
        cell_id="test-cell",
        current_threshold_A=0.05,
        batch_size=3,
    )

    assert segments["state"].tolist() == [
        0,
        1,
        0,
    ]
    assert segments["state_label"].tolist() == [
        "rest",
        "positive",
        "rest",
    ]
    assert segments["n_samples"].tolist() == [
        2,
        3,
        2,
    ]
    assert segments["run_index"].tolist() == [
        1,
        2,
        3,
    ]
    assert segments["cell_id"].tolist() == [
        "test-cell",
        "test-cell",
        "test-cell",
    ]

    np.testing.assert_allclose(
        segments["duration_s"],
        [2.0, 3.0, 1.0],
    )
    np.testing.assert_allclose(
        segments["mean_current_A"],
        [0.0, 1.0, 0.0],
    )


def test_extract_current_segments_requires_columns() -> None:
    data = pd.DataFrame(
        {
            "TestIndex": [0.0],
            "Current[A]": [0.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        extract_current_segments(
            df=data,
            cell_id="test-cell",
        )
