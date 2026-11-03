# HPPC, ECM and OCV–SOC: quick start

Use this guide to detect HPPC regions, fit 2RC parameters and create OCV–SOC
curves. Before starting, follow the project's [installation guide](../installation.md)
to set up PyDPEET and ensure your version includes the features described here.

## 1. Prepare the input

Supply data for one battery cell with `TestIndex`, `Test_Time[s]`, `Current[A]`
and `Voltage[V]`. Current is positive for charging and negative for discharging.
For SOC plots, provide `SOC` as a fraction from 0 to 1. Calculate missing SOC
with `eet.add_soc()` on complete measurement sequences before cropping HPPC
windows. Rename source columns such as `Testtime[s]` where necessary.

## 2. Detect HPPC and fit parameters

Import `create_ecm_parameters` from its module and call it directly, as shown below. This example assumes that the input data already contain an `SOC` column with values between 0 and 1. If SOC is missing, calculate it first using `eet.add_soc()` on the complete measurement sequence. The example produces parameter plots against SOC. To also plot parameters against FEC, supply the external FEC information through `block_metadata`, as described in Section 3.

```python
import pandas as pd
import pydpeet as eet
from pydpeet.process.analyze.extract.hppc import extract_hppc_segments
from pydpeet.process.analyze.ecm import create_ecm_parameters

config = eet.BatteryConfig.LGM50LT_NMC_4800
df = pd.read_parquet("cell_with_soc.parquet")
segments = extract_hppc_segments(df=df, cell_id="cell_A")
if segments.empty:
    raise ValueError("No HPPC regions detected; check the detection settings.")

parameters = create_ecm_parameters(
    df=df, hppc_segments=segments, config=config,
    guess_profile="batch", visualize=True,
)
parameters.to_csv("ecm_parameters.csv", index=False)
```

Select a battery configuration appropriate for your cell. Detection defaults to
5–20 s pulses, 500–700 s rests and at least three repetitions; adjust these for
your protocol. Results include R0, R1, R2, C1, C2, tau1, tau2 and quality fields.
`visualize=True` displays eligible figures through Matplotlib. Saving images
and choosing output directories are handled by your application.

## 3. Add external FEC or calendar age

Ageing information is optional for SOC plots and required for ageing plots.
Provide cumulative FEC (equivalent full cycles) or calendar-age days externally.
`TestIndex` is a measurement identifier, not FEC.

For example, an external CSV can contain these columns and illustrative values:

```csv
cell_id,TestIndex,FEC,calendar_age_days
cell_A,0,0,
cell_A,1,50,
cell_B,0,,0
cell_B,1,,30
```

If each cell/TestIndex has one ageing state, map it to the detected blocks:

```python
ageing = pd.read_csv("ageing_metadata.csv").set_index(
    ["cell_id", "TestIndex"], verify_integrity=True
)
metadata = []
for block_id in sorted(segments["hppc_block_id"].unique()):
    block = segments.loc[segments["hppc_block_id"].eq(block_id)]
    row = ageing.loc[(block["cell_id"].iloc[0], block["TestIndex"].iloc[0])]
    info = {name: float(row[name]) for name in ("FEC", "calendar_age_days")
            if name in row.index and pd.notna(row[name])}
    if not info:
        raise ValueError("Missing ageing information")
    metadata.append(info)

parameters = create_ecm_parameters(
    df=df, hppc_segments=segments, config=config,
    block_metadata=metadata, guess_profile="batch", visualize=True,
    plot_normalize_to_first=True,
)
```

`block_metadata` requires one dictionary per block, in ascending numerical
`hppc_block_id` order. If ageing changes within a TestIndex, match individual
blocks by time instead. For cycle-level data, use the cumulative FEC of the last
cycle completed at or before the block's first pulse. Measurement and cycle
times must share a reference. Resolve missing or conflicting metadata explicitly;
only assume zero before the first cycle when the complete cycle history is known.

## 4. Select plots and normalization

| Setting | Default |
| --- | --- |
| `plot_quality_statuses` | `("pass",)`; other results remain in the parameter table. |
| `plot_target_current_abs_A` / `plot_current_tolerance_A` | 4.8 ± 0.25 A. |
| `plot_pulse_directions` / `plot_soc_paths` | Separate charge/discharge and descending/ascending SOC. |
| `plot_target_socs` / `plot_soc_tolerance` | 25%, 50%, 75% SOC, within ±0.02 fractional SOC. |
| `plot_normalize_to_first` | `True` for FEC/calendar-age plots; `False` retains physical units. |

**SOC–parameter plots always retain physical units.** Ageing normalization uses
`100 * value / first_valid_value`, independently for each cell, target SOC,
pulse direction, SOC path and parameter. The reference need not be at zero age;
zero-reference series are skipped. Returned fitted parameters are unchanged.

**Plot filters do not restrict which pulses are fitted.** To reduce runtime,
an application can preselect pulse/rest pairs while preserving complete rests,
measured transition neighbours and original timestamps. Calculate SOC first.
If a plot is missing, inspect quality flags and available points: SOC curves
need at least two eligible points, ageing curves two distinct ageing states.

## 5. Create OCV–SOC curves

OCV analysis is independent of ECM. For previously extracted and quality-screened
points of one direction, use:

```python
from pydpeet.process.analyze.extract.ocv_curve import (
    fit_ocv_soc_curve, create_ocv_curves,
)

points = pd.read_csv("charge_ocv_points.csv")
binned, curve = fit_ocv_soc_curve(points)
curve.to_csv("charge_ocv_curve.csv", index=False)
```

Points require `SOC` and `Voltage[V]`; process charge and discharge separately.
For rest-quality screening, pass blocks from `eet.extract_ocv_iocv()` and their
primitive data to `create_ocv_curves(ocv_blocks, df_primitives, visualize=True)`.
It returns screened points, bin statistics and curves. Blocks need `ID`,
`iOCV_type`, `SOC`, `Voltage[V]`, `Current[A]`; primitive data need `ID`, `Type`,
`Test_Time[s]`, `Voltage[V]`. Keep primitive IDs unique across combined checkups.

## 6. Run the API tests

With the development test dependencies installed, run from the repository root:

```text
python -m pytest test/api_accessible/test_hppc.py test/api_accessible/test_ecm.py test/api_accessible/test_ecm_normalized_ageing.py test/api_accessible/test_ocv_curve.py
```

Supply your own measurement and ageing files. See the [developer guide](../developer.md)
and `CONTRIBUTING.md` for the project review and CI process.
