import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).parents[2] / "src" / "rnee_build" / "02_audit_energy_target.py"
SPEC = importlib.util.spec_from_file_location("audit_energy_target", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_direct_fuel_interval_liters() -> None:
    assert np.isclose(MODULE.direct_fuel_liters(3.6, 2.0), 0.002)


def test_maf_fuel_interval_liters() -> None:
    # 14.08 g/s air at stoichiometry -> 1 g/s fuel; 745 g/L for 1 s.
    value = MODULE.maf_fuel_liters(14.08, 0.0, 0.0, 1.0, 14.08, 745.0)
    assert np.isclose(value, 1.0 / 745.0)


def test_battery_sign_and_actual_dt() -> None:
    # Source convention: -10 A is discharge. Released target is positive discharge.
    assert np.isclose(MODULE.battery_net_wh(-10.0, 300.0, 2.0), 5.0 / 3.0)
    assert np.isclose(MODULE.battery_net_wh(10.0, 300.0, 2.0), -5.0 / 3.0)


def test_old_eved_electric_formula_assumes_one_second() -> None:
    assert np.isclose(MODULE.old_eved_electric_kwh(-10.0, 300.0), 3.0 / 3600.0)


def test_old_combustion_formula_contains_speed_and_gallon_factor() -> None:
    observed = MODULE.old_eved_combustion_kwh(
        speed_kmh=36.0,
        maf_g_s=10.0,
        stft_pct=0.0,
        ltft_pct=0.0,
        afr=14.08,
    )
    expected = (36.0 / 3600.0) / 100.0 * (10.0 * 3.78541) / 14.08 / 0.1123
    assert np.isclose(observed, expected)


def test_direct_fuel_precedes_maf_and_missing_is_explicit() -> None:
    direct = pd.Series([0.01, np.nan, np.nan])
    maf = pd.Series([0.02, 0.03, np.nan])
    values, source = MODULE.choose_fuel_liters(
        direct,
        pd.Series([True, False, False]),
        maf,
        pd.Series([True, True, False]),
    )
    assert values.iloc[:2].tolist() == [0.01, 0.03]
    assert np.isnan(values.iloc[2])
    assert source.tolist() == ["direct", "maf", "missing"]


def test_identifier_normalization_rejects_rounding() -> None:
    with pytest.raises(ValueError):
        MODULE.normalize_id(pd.Series([1.2]))


def test_exact_timestamp_rejects_fractional_milliseconds() -> None:
    with pytest.raises(ValueError):
        MODULE.exact_integer_values(pd.Series([1000.5]), "Timestamp(ms)")


def test_dt_policy_boundaries() -> None:
    dt = pd.Series([0.0, 0.1, 10.0, 10.1, np.nan])
    valid = dt.gt(0.0) & dt.le(10.0)
    assert valid.tolist() == [False, True, True, False, False]


def test_production_vectorized_target_path() -> None:
    plausibility = {
        "fuel_rate_l_h": [0.0, 100.0],
        "maf_g_s": [0.0, 500.0],
        "trim_correction": [0.1, 2.0],
        "battery_voltage_v": [50.0, 1000.0],
        "battery_current_abs_a_max": 1500.0,
        "battery_power_abs_kw_max": 500.0,
    }
    result = MODULE.compute_interval_targets(
        dt=pd.Series([2.0, 1.0, 11.0]),
        fuel_rate=pd.Series([3.6, np.nan, 3.6]),
        maf=pd.Series([14.08, 14.08, 14.08]),
        correction=pd.Series([1.0, 1.0, 1.0]),
        current=pd.Series([-10.0, 10.0, -10.0]),
        voltage=pd.Series([300.0, 300.0, 300.0]),
        max_dt=10.0,
        afr=14.08,
        density=745.0,
        plausibility=plausibility,
    )
    assert np.isclose(result["fuel_l"].iloc[0], 0.002)
    assert np.isclose(result["fuel_l"].iloc[1], 1.0 / 745.0)
    assert np.isnan(result["fuel_l"].iloc[2])
    assert result["fuel_source"].tolist() == ["direct", "maf", "missing"]


def test_gate_fails_when_any_blocking_check_fails() -> None:
    assert MODULE.decide_gate({"source": True}, {"target": True}) == (
        "PASS_OPERATIONAL_TARGET_ONLY"
    )
    assert MODULE.decide_gate({"source": False}, {"target": True}) == (
        "FAIL_TARGET_VALIDITY"
    )
    assert MODULE.decide_gate({"source": True}, {"target": False}) == (
        "FAIL_TARGET_VALIDITY"
    )
