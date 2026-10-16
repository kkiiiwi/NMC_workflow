import numpy as np
import pandas as pd
import pytest

import pydpeet as eet


def test_equivalent_full_cycles_from_selected_cycles():
    cycles = pd.DataFrame(
        {
            "cycle_number": [1, 2, 3],
            "charge_ah": [1.0, 0.5, 0.25],
            "discharge_ah": [1.0, 0.5, 0.25],
        }
    )
    config = eet.battery_config_wrapper(c_ref=1.0)

    result = eet.add_equivalent_full_cycles_from_cycles(
        cycles,
        config=config,
        verbose=False,
    )

    np.testing.assert_allclose(
        result["AbsoluteChargeThroughput[Ah]"],
        [2.0, 3.0, 3.5],
    )
    np.testing.assert_allclose(
        result["EquivalentFullCycles"],
        [1.0, 1.5, 1.75],
    )


def test_equivalent_full_cycles_from_selected_cycles_accepts_signed_values():
    cycles = pd.DataFrame(
        {
            "charge_ah": [1.0, 1.0],
            "discharge_ah": [-1.0, -1.0],
        }
    )
    config = eet.battery_config_wrapper(c_ref=2.0)

    result = eet.add_equivalent_full_cycles_from_cycles(
        cycles,
        config=config,
        verbose=False,
    )

    np.testing.assert_allclose(result["EquivalentFullCycles"], [0.5, 1.0])


def test_equivalent_full_cycles_from_selected_cycles_rejects_missing_columns():
    config = eet.battery_config_wrapper(c_ref=1.0)

    with pytest.raises(KeyError, match="discharge_ah"):
        eet.add_equivalent_full_cycles_from_cycles(
            pd.DataFrame({"charge_ah": [1.0]}),
            config=config,
            verbose=False,
        )


def test_equivalent_full_cycles_from_selected_cycles_rejects_invalid_values():
    config = eet.battery_config_wrapper(c_ref=1.0)

    with pytest.raises(ValueError, match="finite"):
        eet.add_equivalent_full_cycles_from_cycles(
            pd.DataFrame(
                {
                    "charge_ah": [1.0, np.nan],
                    "discharge_ah": [1.0, 1.0],
                }
            ),
            config=config,
            verbose=False,
        )
