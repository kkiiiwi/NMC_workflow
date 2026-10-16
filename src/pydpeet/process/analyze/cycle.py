import inspect
import logging

import numpy as np
import pandas as pd

from pydpeet.process.analyze.capacity import add_charge_throughput
from pydpeet.process.analyze.configs.battery_config import _BatteryConfigClass
from pydpeet.process.analyze.utils import _StepTimer


def add_equivalent_full_cycles(
    df: pd.DataFrame,
    config: _BatteryConfigClass = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calculate equivalent full cycles from absolute charge throughput and capacity reference.

    Equivalent full cycles are calculated as the absolute charge throughput divided by two times the capacity reference,
    so that every full cycle is counted as one cycle.

    For protocol-aware cycling-ageing FEC, first detect complete ageing cycles
    and call ``add_equivalent_full_cycles_from_cycles``. Filtering raw samples
    and concatenating them is unsafe because integration could bridge gaps
    between separate cycling windows.

    Parameters:
        df (pandas.DataFrame): Input DataFrame containing battery test data
        config (BatteryConfig, optional): Configuration for the battery. Defaults to BATTERY_DEFAULT

    Returns:
        pandas.DataFrame: DataFrame with added 'EquivalentFullCycles' column
    """
    if config is None:
        frame = inspect.currentframe()
        assert frame is not None
        func_name = frame.f_code.co_name
        raise ValueError(f"config is None, please provide a valid config for {func_name}")

    df_mod = df.copy()
    logging.info(f"Starting Equivalent Full Cycles computation on dataframe of size {len(df_mod)}...")

    c_ref = config.c_ref
    if c_ref is None:
        logging.info("c_ref is None, attempting to use first valid capacity value as reference")

        first_valid_idx = df_mod["Capacity[Ah]"].first_valid_index()

        if first_valid_idx is None:
            logging.warning("No valid capacity values found — returning DataFrame with empty SOH column")
            df_mod["SOH"] = np.nan  # or whatever column you intend to fill
            return df_mod
        else:
            c_ref = df_mod.at[first_valid_idx, "Capacity[Ah]"]
            logging.info(f"Using first valid capacity value ({c_ref:.4f} Ah) as reference")

    if "AbsoluteChargeThroughput[Ah]" not in df_mod.columns:
        logging.info(
            "AbsoluteChargeThroughput[Ah] not found, adding AbsoluteChargeThroughput[Ah] column with add_charge_throughput function..."
        )
        with _StepTimer(verbose) as st:
            df_mod = add_charge_throughput(df_mod)
            st._log("added charge throughput column to df")

    with _StepTimer(verbose) as st:
        df_mod["EquivalentFullCycles"] = df_mod["AbsoluteChargeThroughput[Ah]"] / (c_ref * 2)
        st._log("computed Equivalent Full Cycles")

    return df_mod


def add_equivalent_full_cycles_from_cycles(
    cycles: pd.DataFrame,
    config: _BatteryConfigClass = None,
    charge_column: str = "charge_ah",
    discharge_column: str = "discharge_ah",
    verbose: bool = True,
) -> pd.DataFrame:
    """Calculate FEC from complete cycles selected by an upstream detector.

    The input must contain one row per accepted cycling-ageing cycle and the
    charge/discharge throughput of that cycle. Diagnostic measurements are not
    counted because they are excluded before this function is called. The
    cumulative selected throughput is then passed to
    ``add_equivalent_full_cycles`` for the standard PyDPEET FEC calculation.

    Parameters:
        cycles: One row per accepted cycling-ageing cycle.
        config: Battery configuration containing the reference capacity.
        charge_column: Column containing charge throughput magnitudes in Ah.
        discharge_column: Column containing discharge throughput magnitudes in Ah.
        verbose: Enable timing messages.

    Returns:
        A copy of ``cycles`` with ``AbsoluteChargeThroughput[Ah]`` and
        ``EquivalentFullCycles`` columns.
    """
    required = {charge_column, discharge_column}
    missing = required - set(cycles.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    result = cycles.copy()
    charge_ah = pd.to_numeric(result[charge_column], errors="coerce").to_numpy(dtype=float)
    discharge_ah = pd.to_numeric(result[discharge_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(charge_ah).all() or not np.isfinite(discharge_ah).all():
        raise ValueError("Cycle charge/discharge throughput must be finite")

    result["AbsoluteChargeThroughput[Ah]"] = np.cumsum(np.abs(charge_ah) + np.abs(discharge_ah))
    return add_equivalent_full_cycles(result, config=config, verbose=verbose)
