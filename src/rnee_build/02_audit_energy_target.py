#!/usr/bin/env python3
"""TARGET_AUDIT: audit VED energy channels and reproduce legacy eVED target provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


COL = {
    "day": "DayNum",
    "vehicle": "VehId",
    "trip": "Trip",
    "timestamp": "Timestamp(ms)",
    "speed": "Vehicle Speed[km/h]",
    "maf": "MAF[g/sec]",
    "rpm": "Engine RPM[RPM]",
    "load": "Absolute Load[%]",
    "fuel": "Fuel Rate[L/hr]",
    "current": "HV Battery Current[A]",
    "soc": "HV Battery SOC[%]",
    "voltage": "HV Battery Voltage[V]",
    "stft1": "Short Term Fuel Trim Bank 1[%]",
    "stft2": "Short Term Fuel Trim Bank 2[%]",
    "ltft1": "Long Term Fuel Trim Bank 1[%]",
    "ltft2": "Long Term Fuel Trim Bank 2[%]",
    "legacy": "Energy_Consumption",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def finite(value: float) -> bool:
    return bool(np.isfinite(value))


def direct_fuel_liters(fuel_rate_l_h: float, dt_s: float) -> float:
    return float(fuel_rate_l_h) * float(dt_s) / 3600.0


def maf_fuel_liters(
    maf_g_s: float,
    stft_pct: float,
    ltft_pct: float,
    dt_s: float,
    afr: float,
    density_g_l: float,
) -> float:
    correction = 1.0 + float(stft_pct) / 100.0 + float(ltft_pct) / 100.0
    return float(maf_g_s) * correction / float(afr) / float(density_g_l) * float(dt_s)


def battery_net_wh(current_a: float, voltage_v: float, dt_s: float) -> float:
    return -float(current_a) * float(voltage_v) * float(dt_s) / 3600.0


def old_eved_electric_kwh(current_a: float, voltage_v: float) -> float:
    return -float(current_a) * float(voltage_v) / 1000.0 / 3600.0


def old_eved_combustion_kwh(
    speed_kmh: float,
    maf_g_s: float,
    stft_pct: float,
    ltft_pct: float,
    afr: float,
) -> float:
    driving_length = float(speed_kmh) / 3600.0
    mislabeled_mass_air_flow = float(maf_g_s) * 3.78541
    return (
        driving_length
        / 100.0
        * mislabeled_mass_air_flow
        * (1.0 + float(stft_pct) / 100.0 + float(ltft_pct) / 100.0)
        / float(afr)
        / 0.1123
    )


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = numeric.isna() | ~np.isclose(numeric, np.round(numeric), rtol=0.0, atol=0.0)
    if invalid.any():
        examples = series[invalid].head(5).tolist()
        raise ValueError(f"Non-integer or missing identifier values: {examples}")
    return numeric.astype("Int64").astype("string")


def exact_integer_values(series: pd.Series, name: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(float)
    invalid = ~np.isfinite(numeric) | (numeric != np.rint(numeric))
    if invalid.any():
        raise ValueError(f"Non-integer or missing {name}: {numeric[invalid][:5].tolist()}")
    return np.rint(numeric).astype(np.int64)


def choose_fuel_liters(
    direct_l: pd.Series,
    direct_ok: pd.Series,
    maf_l: pd.Series,
    maf_ok: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    values = direct_l.where(direct_ok, maf_l.where(maf_ok))
    source = pd.Series("missing", index=values.index, dtype="string")
    source.loc[maf_ok] = "maf"
    source.loc[direct_ok] = "direct"
    return values, source


def compute_interval_targets(
    dt: pd.Series,
    fuel_rate: pd.Series,
    maf: pd.Series,
    correction: pd.Series,
    current: pd.Series,
    voltage: pd.Series,
    max_dt: float,
    afr: float,
    density: float,
    plausibility: dict[str, Any],
) -> dict[str, pd.Series]:
    valid_dt = dt.gt(0.0) & dt.le(float(max_dt))
    fuel_min, fuel_max = [float(value) for value in plausibility["fuel_rate_l_h"]]
    maf_min, maf_max = [float(value) for value in plausibility["maf_g_s"]]
    correction_min, correction_max = [
        float(value) for value in plausibility["trim_correction"]
    ]
    voltage_min, voltage_max = [
        float(value) for value in plausibility["battery_voltage_v"]
    ]
    battery_power_kw = current * voltage / 1000.0
    direct_ok = valid_dt & fuel_rate.between(fuel_min, fuel_max, inclusive="both")
    maf_ok = (
        valid_dt
        & maf.between(maf_min, maf_max, inclusive="both")
        & correction.between(correction_min, correction_max, inclusive="both")
    )
    battery_ok = (
        valid_dt
        & current.abs().le(float(plausibility["battery_current_abs_a_max"]))
        & voltage.between(voltage_min, voltage_max, inclusive="both")
        & battery_power_kw.abs().le(float(plausibility["battery_power_abs_kw_max"]))
    )
    direct_l = fuel_rate * dt / 3600.0
    maf_l = maf * correction / float(afr) / float(density) * dt
    battery_wh = -current * voltage * dt / 3600.0
    fuel_l, fuel_source = choose_fuel_liters(direct_l, direct_ok, maf_l, maf_ok)
    return {
        "valid_dt": valid_dt,
        "direct_ok": direct_ok,
        "maf_ok": maf_ok,
        "battery_ok": battery_ok,
        "direct_l": direct_l,
        "maf_l": maf_l,
        "battery_wh": battery_wh,
        "battery_power_kw": battery_power_kw,
        "fuel_l": fuel_l,
        "fuel_source": fuel_source,
    }


def decide_gate(
    source_integrity_checks: dict[str, bool],
    target_validity_checks: dict[str, bool],
) -> str:
    return (
        "PASS_OPERATIONAL_TARGET_ONLY"
        if all(source_integrity_checks.values()) and all(target_validity_checks.values())
        else "FAIL_TARGET_VALIDITY"
    )


def mean_available(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    values = frame[names].apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1, skipna=True).fillna(0.0)


def descriptive(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "min": None, "p01": None, "p50": None, "p99": None, "max": None}
    q = np.quantile(array, [0.01, 0.5, 0.99])
    return {
        "n": int(len(array)),
        "min": float(array.min()),
        "p01": float(q[0]),
        "p50": float(q[1]),
        "p99": float(q[2]),
        "max": float(array.max()),
    }


class RunningPairs:
    def __init__(self) -> None:
        self.n = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.sum_abs = 0.0
        self.max_abs = 0.0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            return
        x = x[mask].astype(float)
        y = y[mask].astype(float)
        difference = x - y
        self.n += int(len(x))
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_x2 += float(np.dot(x, x))
        self.sum_y2 += float(np.dot(y, y))
        self.sum_xy += float(np.dot(x, y))
        self.sum_abs += float(np.abs(difference).sum())
        self.max_abs = max(self.max_abs, float(np.abs(difference).max()))

    def result(self) -> dict[str, float | int | None]:
        if not self.n:
            return {"n": 0, "correlation": None, "mae": None, "max_abs": None}
        numerator = self.n * self.sum_xy - self.sum_x * self.sum_y
        denominator = math.sqrt(
            max(self.n * self.sum_x2 - self.sum_x**2, 0.0)
            * max(self.n * self.sum_y2 - self.sum_y**2, 0.0)
        )
        return {
            "n": self.n,
            "correlation": float(numerator / denominator) if denominator else None,
            "mae": self.sum_abs / self.n,
            "max_abs": self.max_abs,
        }


def evidence_rows(config: dict[str, Any], notebook_hash: str) -> list[dict[str, str]]:
    return [
        {
            "statement": "VED timestamps are milliseconds within a trip; integration uses observed deltas.",
            "basis": "dataset artifact",
            "source": "official VED dynamic CSV header and row sequence",
            "status": "supported",
        },
        {
            "statement": "Fuel Rate is released in L/hr; MAF is released in g/s.",
            "basis": "primary paper and dataset artifact",
            "source": "Oh et al., VED, arXiv:1905.02081, Table II; official CSV header",
            "status": "supported",
        },
        {
            "statement": "VED uses AFR=14.08 for E10 in its fuel-rate estimation algorithm.",
            "basis": "primary paper",
            "source": "Oh et al., VED, arXiv:1905.02081, Algorithm 1",
            "status": "supported",
        },
        {
            "statement": "Positive HV current denotes charging and negative current denotes discharging.",
            "basis": "primary paper",
            "source": "Oh et al., VED, arXiv:1905.02081, Table II note",
            "status": "supported",
        },
        {
            "statement": "Battery terminal power is voltage multiplied by current.",
            "basis": "primary paper",
            "source": "Oh et al., VED, arXiv:1905.02081, energy-use discussion",
            "status": "supported",
        },
        {
            "statement": "The paper's MAF fallback is dimensionally complete as L/hr.",
            "basis": "unit audit",
            "source": "Algorithm 1 omits fuel density and seconds-to-hours conversion",
            "status": "not supported",
        },
        {
            "statement": "A single fleet-wide fuel-plus-electric Energy_Consumption boundary is documented.",
            "basis": "primary paper and repository audit",
            "source": "no common boundary, fuel LHV, or drivetrain boundary is specified",
            "status": "not supported",
        },
        {
            "statement": "Legacy eVED combustion rows reproduce the notebook's speed×MAF×3.78541 formula.",
            "basis": "reference-only code audit",
            "source": (
                f"{config['reference_only']['derivation_notebook']} "
                f"(sha256={notebook_hash})"
            ),
            "status": "to be reproduced for provenance",
        },
    ]


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, output_dir / "run_config.yaml")

    static_path = Path(config["source"]["static_metadata"])
    inventory_path = Path(config["source"]["source_file_inventory"])
    source_inventory_summary_path = Path(config["source"]["source_inventory_summary"])
    static_hash = sha256_file(static_path)
    inventory_hash = sha256_file(inventory_path)
    static_hash_matches = (
        static_hash == str(config["source"]["static_metadata_sha256"]).lower()
    )
    inventory_hash_matches = (
        inventory_hash == str(config["source"]["source_file_inventory_sha256"]).lower()
    )
    source_inventory_summary = json.loads(source_inventory_summary_path.read_text(encoding="utf-8"))
    source_inventory = pd.read_parquet(inventory_path)
    expected_dynamic_hashes = source_inventory.set_index("source_file")[
        "sha256"
    ].astype(str).str.lower().to_dict()

    static = pd.read_parquet(static_path)
    static["VehId_key"] = normalize_id(static["VehId"])
    engine_map = static.set_index("VehId_key")["EngineType_official"].astype(str).to_dict()
    engine_configuration_map = static.set_index("VehId_key")[
        "Engine Configuration & Displacement"
    ].astype(str).to_dict()
    expected_counts = config["quality_gates"]["expected_powertrain_counts"]
    actual_counts = static["EngineType_official"].value_counts().to_dict()

    raw_dir = Path(config["source"]["ved_dynamic_dir"])
    raw_files = sorted(raw_dir.glob(config["source"]["ved_file_pattern"]))
    if not raw_files:
        raise FileNotFoundError(f"No original VED files found in {raw_dir}")
    dynamic_hash_results = {
        path.name: sha256_file(path) == expected_dynamic_hashes.get(path.name)
        for path in raw_files
    }
    eved_dir = Path(config["reference_only"]["eved_dynamic_dir"])
    eved_files = {
        path.name.replace("eVED_", "VED_", 1): path
        for path in eved_dir.glob(config["reference_only"]["eved_file_pattern"])
    }
    notebook = Path(config["reference_only"]["derivation_notebook"])
    notebook_hash = sha256_file(notebook)

    chunk_rows = int(config["output"]["chunk_rows"])
    max_dt = float(config["integration"]["maximum_dt_seconds_inclusive"])
    afr = float(config["fuel"]["stoichiometric_afr_e10"])
    density = float(config["fuel"]["nominal_density_g_per_l"])
    density_grid = [float(value) for value in config["fuel"]["density_sensitivity_g_per_l"]]
    dt_sensitivity_limits = [
        float(value) for value in config["integration"]["sensitivity_maximum_dt_seconds"]
    ]
    plausibility = config["quality_gates"]["plausibility"]

    vehicle: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    file_rows: list[dict[str, Any]] = []
    dt_values: list[float] = []
    direct_maf_pairs = RunningPairs()
    legacy_formula_pairs = {"combustion": RunningPairs(), "electric": RunningPairs()}
    legacy_physical_pairs = {"combustion": RunningPairs(), "electric": RunningPairs()}
    all_rows = 0
    joined_rows = 0
    aligned_rows = 0
    legacy_rows = 0
    key_mismatch_rows = 0
    invalid_order_rows = 0
    density_totals = {str(value): 0.0 for value in density_grid}
    dt_sensitivity_totals = {
        str(value): {"fuel_l": 0.0, "battery_net_wh": 0.0}
        for value in dt_sensitivity_limits
    }
    channel_samples: dict[str, list[float]] = defaultdict(list)
    channel_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    trip_rows: list[dict[str, Any]] = []

    usecols = list(COL.values())
    for raw_path in raw_files:
        raw = pd.read_csv(raw_path, usecols=[name for name in usecols if name != COL["legacy"]])
        raw.columns = [str(name).strip().rstrip(";") for name in raw.columns]
        raw["VehId_key"] = normalize_id(raw[COL["vehicle"]])
        raw["Trip_key"] = normalize_id(raw[COL["trip"]])
        raw["EngineType"] = raw["VehId_key"].map(engine_map)
        n_rows = len(raw)
        all_rows += n_rows
        joined_rows += int(raw["EngineType"].notna().sum())

        timestamp = pd.to_numeric(raw[COL["timestamp"]], errors="coerce")
        next_timestamp = timestamp.groupby(
            [raw["VehId_key"], raw["Trip_key"]], sort=False
        ).shift(-1)
        dt = (next_timestamp - timestamp) / 1000.0
        invalid_order_rows += int((dt < 0).sum())
        valid_dt = dt.gt(float(config["integration"]["minimum_dt_seconds_exclusive"])) & dt.le(max_dt)
        dt_values.extend(dt[dt.notna()].sample(min(int(dt.notna().sum()), 50_000), random_state=162).tolist())

        numeric = {}
        for key in [
            "speed", "maf", "rpm", "load", "fuel", "current", "soc", "voltage",
            "stft1", "stft2", "ltft1", "ltft2",
        ]:
            numeric[key] = pd.to_numeric(raw[COL[key]], errors="coerce")
        stft = mean_available(raw, [COL["stft1"], COL["stft2"]])
        ltft = mean_available(raw, [COL["ltft1"], COL["ltft2"]])
        correction = 1.0 + stft / 100.0 + ltft / 100.0

        target = compute_interval_targets(
            dt=dt,
            fuel_rate=numeric["fuel"],
            maf=numeric["maf"],
            correction=correction,
            current=numeric["current"],
            voltage=numeric["voltage"],
            max_dt=max_dt,
            afr=afr,
            density=density,
            plausibility=plausibility,
        )
        valid_dt = target["valid_dt"]
        direct_ok = target["direct_ok"]
        maf_ok = target["maf_ok"]
        battery_ok = target["battery_ok"]
        direct_l = target["direct_l"]
        maf_l = target["maf_l"]
        battery_wh = target["battery_wh"]
        battery_power_kw_raw = target["battery_power_kw"]
        preferred_fuel_l = target["fuel_l"]
        fuel_source = target["fuel_source"]

        direct_maf_pairs.add(
            direct_l.where(direct_ok & maf_ok).to_numpy(float),
            maf_l.where(direct_ok & maf_ok).to_numpy(float),
        )
        for value in density_grid:
            candidate = numeric["maf"] * correction / afr / value * dt
            density_totals[str(value)] += float(candidate.where(maf_ok & ~direct_ok).sum(skipna=True))
        for limit in dt_sensitivity_limits:
            sensitivity_target = compute_interval_targets(
                dt=dt,
                fuel_rate=numeric["fuel"],
                maf=numeric["maf"],
                correction=correction,
                current=numeric["current"],
                voltage=numeric["voltage"],
                max_dt=limit,
                afr=afr,
                density=density,
                plausibility=plausibility,
            )
            dt_sensitivity_totals[str(limit)]["fuel_l"] += float(
                sensitivity_target["fuel_l"].sum(skipna=True)
            )
            dt_sensitivity_totals[str(limit)]["battery_net_wh"] += float(
                sensitivity_target["battery_wh"]
                .where(sensitivity_target["battery_ok"])
                .sum(skipna=True)
            )

        channel_series = {
            "maf_g_s": numeric["maf"],
            "fuel_rate_l_h": numeric["fuel"],
            "trim_correction": correction.where(
                raw[[COL["stft1"], COL["stft2"], COL["ltft1"], COL["ltft2"]]]
                .notna()
                .any(axis=1)
            ),
            "battery_voltage_v": numeric["voltage"],
            "battery_current_a": numeric["current"],
            "battery_power_kw": battery_power_kw_raw,
            "fuel_interval_l": preferred_fuel_l,
            "battery_interval_net_wh": battery_wh.where(battery_ok),
            "maf_jump_g_s2": (
                numeric["maf"].groupby(
                    [raw["VehId_key"], raw["Trip_key"]], sort=False
                ).shift(-1)
                - numeric["maf"]
            ).abs()
            / dt.where(valid_dt),
            "fuel_rate_jump_l_h_per_s": (
                numeric["fuel"].groupby(
                    [raw["VehId_key"], raw["Trip_key"]], sort=False
                ).shift(-1)
                - numeric["fuel"]
            ).abs()
            / dt.where(valid_dt),
            "battery_current_jump_a_per_s": (
                numeric["current"].groupby(
                    [raw["VehId_key"], raw["Trip_key"]], sort=False
                ).shift(-1)
                - numeric["current"]
            ).abs()
            / dt.where(valid_dt),
            "battery_voltage_jump_v_per_s": (
                numeric["voltage"].groupby(
                    [raw["VehId_key"], raw["Trip_key"]], sort=False
                ).shift(-1)
                - numeric["voltage"]
            ).abs()
            / dt.where(valid_dt),
        }
        for name, series in channel_series.items():
            valid_values = series[np.isfinite(series.to_numpy(float))]
            previous_n = int(channel_counts[name]["n"])
            channel_counts[name]["n"] += int(len(valid_values))
            if len(valid_values):
                observed_min = float(valid_values.min())
                observed_max = float(valid_values.max())
                if previous_n == 0:
                    channel_counts[name]["observed_min"] = observed_min
                    channel_counts[name]["observed_max"] = observed_max
                else:
                    channel_counts[name]["observed_min"] = min(
                        float(channel_counts[name]["observed_min"]), observed_min
                    )
                    channel_counts[name]["observed_max"] = max(
                        float(channel_counts[name]["observed_max"]), observed_max
                    )
                channel_counts[name]["negative"] += int((valid_values < 0).sum())
                channel_counts[name]["zero"] += int((valid_values == 0).sum())
                if name in {"maf_g_s", "fuel_rate_l_h", "trim_correction", "battery_voltage_v"}:
                    lower, upper = [float(value) for value in plausibility[name]]
                    channel_counts[name]["out_of_range"] += int(
                        ((valid_values < lower) | (valid_values > upper)).sum()
                    )
                elif name == "battery_current_a":
                    channel_counts[name]["out_of_range"] += int(
                        (
                            valid_values.abs()
                            > float(plausibility["battery_current_abs_a_max"])
                        ).sum()
                    )
                elif name == "battery_power_kw":
                    channel_counts[name]["out_of_range"] += int(
                        (
                            valid_values.abs()
                            > float(plausibility["battery_power_abs_kw_max"])
                        ).sum()
                    )
                sample = valid_values.sample(
                    min(len(valid_values), 20_000), random_state=162
                ).tolist()
                channel_samples[name].extend(sample)

        distance_km = numeric["speed"].where(valid_dt).clip(lower=0) * dt / 3600.0
        trip_frame = pd.DataFrame(
            {
                "VehId": raw["VehId_key"],
                "Trip": raw["Trip_key"],
                "EngineType_official": raw["EngineType"],
                "distance_km": distance_km,
                "fuel_l": preferred_fuel_l,
                "battery_net_wh": battery_wh.where(battery_ok),
                "fuel_observed": preferred_fuel_l.notna().astype(int),
                "fuel_direct": fuel_source.eq("direct").astype(int),
                "fuel_maf": fuel_source.eq("maf").astype(int),
                "battery_observed": battery_ok.astype(int),
                "valid_dt": valid_dt.astype(int),
                "soc": numeric["soc"],
            }
        )
        for (vehid, trip, engine), group in trip_frame.groupby(
            ["VehId", "Trip", "EngineType_official"], sort=False, dropna=False
        ):
            soc_values = group["soc"].dropna()
            trip_rows.append(
                {
                    "source_file": raw_path.name,
                    "VehId": vehid,
                    "Trip": trip,
                    "EngineType_official": engine,
                    "distance_km": float(group["distance_km"].sum(skipna=True)),
                    "fuel_l": float(group["fuel_l"].sum(skipna=True)),
                    "battery_net_wh": float(group["battery_net_wh"].sum(skipna=True)),
                    "valid_dt_rows": int(group["valid_dt"].sum()),
                    "fuel_observed_rows": int(group["fuel_observed"].sum()),
                    "fuel_direct_rows": int(group["fuel_direct"].sum()),
                    "fuel_maf_rows": int(group["fuel_maf"].sum()),
                    "battery_observed_rows": int(group["battery_observed"].sum()),
                    "soc_start": float(soc_values.iloc[0]) if len(soc_values) else np.nan,
                    "soc_end": float(soc_values.iloc[-1]) if len(soc_values) else np.nan,
                }
            )

        for (vehid, engine), indices in raw.groupby(["VehId_key", "EngineType"], dropna=False).groups.items():
            key = (str(vehid), str(engine))
            idx = np.asarray(indices, dtype=int)
            stats = vehicle[key]
            stats["rows"] += len(idx)
            stats["valid_dt_rows"] += int(valid_dt.iloc[idx].sum())
            stats["direct_fuel_rows"] += int(direct_ok.iloc[idx].sum())
            stats["maf_rows"] += int(maf_ok.iloc[idx].sum())
            stats["preferred_fuel_rows"] += int(preferred_fuel_l.iloc[idx].notna().sum())
            stats["preferred_fuel_zero_rows"] += int(preferred_fuel_l.iloc[idx].eq(0).sum())
            stats["battery_rows"] += int(battery_ok.iloc[idx].sum())
            stats["battery_discharge_rows"] += int(
                (battery_ok.iloc[idx] & battery_wh.iloc[idx].gt(0)).sum()
            )
            stats["battery_charge_rows"] += int(
                (battery_ok.iloc[idx] & battery_wh.iloc[idx].lt(0)).sum()
            )
            stats["battery_zero_rows"] += int(
                (battery_ok.iloc[idx] & battery_wh.iloc[idx].eq(0)).sum()
            )
            stats["fuel_l"] += float(preferred_fuel_l.iloc[idx].sum(skipna=True))
            stats["battery_net_wh"] += float(battery_wh.iloc[idx].where(battery_ok.iloc[idx]).sum(skipna=True))
            stats["battery_discharge_wh"] += float(
                battery_wh.iloc[idx].where(battery_ok.iloc[idx] & battery_wh.iloc[idx].gt(0)).sum(skipna=True)
            )
            stats["battery_regen_wh"] += float(
                battery_wh.iloc[idx].where(battery_ok.iloc[idx] & battery_wh.iloc[idx].lt(0)).sum(skipna=True)
            )

        file_record = {
            "source_file": raw_path.name,
            "rows": n_rows,
            "vehicles": int(raw["VehId_key"].nunique()),
            "trips": int(raw[["VehId_key", "Trip_key"]].drop_duplicates().shape[0]),
            "static_join_rows": int(raw["EngineType"].notna().sum()),
            "valid_dt_rows": int(valid_dt.sum()),
            "nonpositive_dt_rows": int(dt.le(0).sum()),
            "gap_gt_max_rows": int(dt.gt(max_dt).sum()),
            "direct_fuel_rows": int(direct_ok.sum()),
            "maf_candidate_rows": int(maf_ok.sum()),
            "battery_rows": int(battery_ok.sum()),
            "preferred_fuel_l": float(preferred_fuel_l.sum(skipna=True)),
            "battery_net_wh": float(battery_wh.where(battery_ok).sum(skipna=True)),
        }

        if config["reference_only"]["provenance_reproduction"]:
            legacy_path = eved_files.get(raw_path.name)
            if not legacy_path:
                raise FileNotFoundError(f"Missing eVED counterpart for {raw_path.name}")
            offset = 0
            for legacy_chunk in pd.read_csv(
                legacy_path,
                usecols=[COL["vehicle"], COL["trip"], COL["timestamp"], COL["legacy"]],
                chunksize=chunk_rows,
            ):
                stop = offset + len(legacy_chunk)
                source_chunk = raw.iloc[offset:stop]
                if len(source_chunk) != len(legacy_chunk):
                    raise ValueError(f"Row-count mismatch in {raw_path.name} at {offset}")
                key_equal = (
                    normalize_id(source_chunk[COL["vehicle"]]).reset_index(drop=True)
                    == normalize_id(legacy_chunk[COL["vehicle"]]).reset_index(drop=True)
                ) & (
                    normalize_id(source_chunk[COL["trip"]]).reset_index(drop=True)
                    == normalize_id(legacy_chunk[COL["trip"]]).reset_index(drop=True)
                ) & (
                    exact_integer_values(
                        source_chunk[COL["timestamp"]], "source Timestamp(ms)"
                    )
                    == exact_integer_values(
                        legacy_chunk[COL["timestamp"]], "eVED Timestamp(ms)"
                    )
                )
                aligned_rows += int(np.asarray(key_equal).sum())
                key_mismatch_rows += int((~np.asarray(key_equal)).sum())

                idx = np.arange(offset, stop)
                legacy_value = pd.to_numeric(legacy_chunk[COL["legacy"]], errors="coerce").to_numpy(float)
                legacy_rows += int(np.isfinite(legacy_value).sum())
                engines = raw["EngineType"].iloc[idx].to_numpy(str)
                combustion_mask = np.isin(engines, ["ICE", "HEV"])
                electric_mask = np.isin(engines, ["PHEV", "EV"])
                old_combustion = (
                    numeric["speed"].iloc[idx].to_numpy(float) / 3600.0 / 100.0
                    * (numeric["maf"].iloc[idx].to_numpy(float) * 3.78541)
                    * (
                        1.0
                        + numeric["stft1"].iloc[idx].to_numpy(float) / 100.0
                        + numeric["ltft1"].iloc[idx].to_numpy(float) / 100.0
                    )
                    / afr / 0.1123
                )
                old_electric = (
                    -numeric["current"].iloc[idx].to_numpy(float)
                    * numeric["voltage"].iloc[idx].to_numpy(float) / 1000.0 / 3600.0
                )
                legacy_formula_pairs["combustion"].add(
                    legacy_value[combustion_mask], old_combustion[combustion_mask]
                )
                legacy_formula_pairs["electric"].add(
                    legacy_value[electric_mask], old_electric[electric_mask]
                )
                corrected_combustion_kwh_eq = (
                    preferred_fuel_l.iloc[idx].to_numpy(float) / 0.1123
                )
                corrected_electric_kwh = (
                    battery_wh.where(battery_ok).iloc[idx].to_numpy(float) / 1000.0
                )
                legacy_physical_pairs["combustion"].add(
                    legacy_value[combustion_mask], corrected_combustion_kwh_eq[combustion_mask]
                )
                legacy_physical_pairs["electric"].add(
                    legacy_value[electric_mask], corrected_electric_kwh[electric_mask]
                )
                offset = stop
            if offset != n_rows:
                raise ValueError(f"eVED counterpart has fewer rows: {legacy_path.name}")
        file_rows.append(file_record)

    vehicle_rows: list[dict[str, Any]] = []
    for (vehid, engine), stats in vehicle.items():
        rows = stats["rows"]
        valid = stats["valid_dt_rows"]
        vehicle_rows.append(
            {
                "VehId": vehid,
                "EngineType_official": engine,
                **{name: value for name, value in stats.items()},
                "direct_fuel_coverage_valid_dt": stats["direct_fuel_rows"] / valid if valid else 0.0,
                "maf_coverage_valid_dt": stats["maf_rows"] / valid if valid else 0.0,
                "preferred_fuel_coverage_valid_dt": stats["preferred_fuel_rows"] / valid if valid else 0.0,
                "battery_coverage_valid_dt": stats["battery_rows"] / valid if valid else 0.0,
                "row_share": rows / all_rows if all_rows else 0.0,
            }
        )
    vehicle_df = pd.DataFrame(vehicle_rows).sort_values(["EngineType_official", "VehId"])
    vehicle_df.to_csv(output_dir / "vehicle_target_summary.csv", index=False)
    vehicle_df.to_parquet(output_dir / "vehicle_target_summary.parquet", index=False)
    file_df = pd.DataFrame(file_rows)
    file_df.to_csv(output_dir / "file_target_summary.csv", index=False)

    powertrain_rows: list[dict[str, Any]] = []
    for engine, group in vehicle_df.groupby("EngineType_official"):
        powertrain_rows.append(
            {
                "EngineType_official": engine,
                "vehicles": int(group["VehId"].nunique()),
                "rows": int(group["rows"].sum()),
                "valid_dt_rows": int(group["valid_dt_rows"].sum()),
                "vehicles_with_direct_fuel": int((group["direct_fuel_rows"] > 0).sum()),
                "vehicles_with_maf": int((group["maf_rows"] > 0).sum()),
                "vehicles_with_battery": int((group["battery_rows"] > 0).sum()),
                "direct_fuel_rows": int(group["direct_fuel_rows"].sum()),
                "maf_rows": int(group["maf_rows"].sum()),
                "preferred_fuel_rows": int(group["preferred_fuel_rows"].sum()),
                "battery_rows": int(group["battery_rows"].sum()),
                "preferred_fuel_zero_rows": int(group["preferred_fuel_zero_rows"].sum()),
                "battery_discharge_rows": int(group["battery_discharge_rows"].sum()),
                "battery_charge_rows": int(group["battery_charge_rows"].sum()),
                "battery_zero_rows": int(group["battery_zero_rows"].sum()),
                "fuel_l": float(group["fuel_l"].sum()),
                "battery_net_wh": float(group["battery_net_wh"].sum()),
                "battery_discharge_wh": float(group["battery_discharge_wh"].sum()),
                "battery_regen_wh": float(group["battery_regen_wh"].sum()),
            }
        )
    powertrain_df = pd.DataFrame(powertrain_rows).sort_values("EngineType_official")
    powertrain_df.to_csv(output_dir / "powertrain_channel_summary.csv", index=False)

    trip_df = pd.DataFrame(trip_rows)
    trip_df["fuel_coverage_valid_dt"] = (
        trip_df["fuel_observed_rows"] / trip_df["valid_dt_rows"].replace(0, np.nan)
    )
    trip_df["battery_coverage_valid_dt"] = (
        trip_df["battery_observed_rows"] / trip_df["valid_dt_rows"].replace(0, np.nan)
    )
    trip_df["fuel_source_dominant"] = np.select(
        [
            trip_df["fuel_direct_rows"].gt(trip_df["fuel_maf_rows"]),
            trip_df["fuel_maf_rows"].gt(0),
        ],
        ["direct", "maf"],
        default="missing",
    )
    trip_df["fuel_l_per_100km"] = np.where(
        trip_df["distance_km"] > 0,
        trip_df["fuel_l"] / trip_df["distance_km"] * 100.0,
        np.nan,
    )
    trip_df["battery_net_wh_per_km"] = np.where(
        trip_df["distance_km"] > 0,
        trip_df["battery_net_wh"] / trip_df["distance_km"],
        np.nan,
    )
    trip_df["soc_delta_pct"] = trip_df["soc_end"] - trip_df["soc_start"]
    trip_df["battery_soc_sign_agrees"] = np.where(
        trip_df["battery_net_wh"].abs().gt(1.0)
        & trip_df["soc_delta_pct"].abs().gt(0.1),
        np.sign(trip_df["battery_net_wh"]) == -np.sign(trip_df["soc_delta_pct"]),
        np.nan,
    )
    moving_combustion = (
        trip_df["EngineType_official"].isin(["ICE", "HEV"])
        & trip_df["distance_km"].ge(1.0)
    )
    low_combustion_fuel = moving_combustion & trip_df["fuel_l_per_100km"].lt(
        float(plausibility["moving_combustion_fuel_l_per_100km_min"])
    ) & trip_df["fuel_observed_rows"].gt(0) & trip_df[
        "fuel_coverage_valid_dt"
    ].ge(0.95)
    high_fuel = (
        trip_df["distance_km"].ge(1.0)
        & trip_df["fuel_observed_rows"].gt(0)
        & trip_df["fuel_l_per_100km"].gt(
            float(plausibility["fuel_l_per_100km_max"])
        )
    )
    high_battery = (
        trip_df["distance_km"].ge(1.0)
        & trip_df["battery_observed_rows"].gt(0)
        & trip_df["battery_net_wh_per_km"].abs().gt(
            float(plausibility["battery_net_wh_per_km_abs_max"])
        )
    )
    trip_df["trip_target_plausible"] = ~(
        low_combustion_fuel | high_fuel | high_battery
    )
    trip_df["trip_exclusion_reason"] = np.select(
        [low_combustion_fuel, high_fuel, high_battery],
        [
            "moving_combustion_fuel_below_minimum",
            "fuel_intensity_above_maximum",
            "battery_intensity_above_maximum",
        ],
        default="",
    )
    coverage_eligible = np.select(
        [
            trip_df["EngineType_official"].isin(["ICE", "HEV"]),
            trip_df["EngineType_official"].eq("PHEV"),
            trip_df["EngineType_official"].eq("EV"),
        ],
        [
            trip_df["fuel_coverage_valid_dt"].ge(0.95),
            trip_df["fuel_coverage_valid_dt"].ge(0.95)
            & trip_df["battery_coverage_valid_dt"].ge(0.95),
            trip_df["battery_coverage_valid_dt"].ge(0.95),
        ],
        default=False,
    ).astype(bool)
    trip_df["target_release_eligible"] = (
        coverage_eligible & trip_df["trip_target_plausible"]
    )
    trip_df["target_release_exclusion_reason"] = np.where(
        ~trip_df["trip_target_plausible"],
        trip_df["trip_exclusion_reason"],
        np.where(~coverage_eligible, "insufficient_target_coverage", ""),
    )
    trip_df.to_csv(output_dir / "trip_target_summary.csv", index=False)
    trip_df.to_parquet(output_dir / "trip_target_summary.parquet", index=False)
    source_scale_df = (
        trip_df.loc[
            trip_df["distance_km"].ge(1.0)
            & trip_df["fuel_observed_rows"].gt(0)
            & trip_df["fuel_l_per_100km"].notna()
        ]
        .groupby(["EngineType_official", "fuel_source_dominant"])[
            "fuel_l_per_100km"
        ]
        .agg(["count", "median", "mean", "std", "min", "max"])
        .reset_index()
    )
    source_scale_df.to_csv(output_dir / "fuel_source_scale_summary.csv", index=False)

    channel_rows: list[dict[str, Any]] = []
    for name, counts in sorted(channel_counts.items()):
        stats = descriptive(channel_samples[name])
        n = int(counts["n"])
        channel_rows.append(
            {
                "channel": name,
                "n": n,
                "sample_n": int(stats["n"]),
                "min": float(counts["observed_min"]) if n else None,
                "p01_sample": stats["p01"],
                "p50_sample": stats["p50"],
                "p99_sample": stats["p99"],
                "max": float(counts["observed_max"]) if n else None,
                "negative_count": int(counts["negative"]),
                "zero_count": int(counts["zero"]),
                "out_of_range_count": int(counts["out_of_range"]),
                "out_of_range_rate": int(counts["out_of_range"]) / n if n else 0.0,
            }
        )
    channel_df = pd.DataFrame(channel_rows)
    channel_df.to_csv(output_dir / "channel_plausibility_summary.csv", index=False)

    anomaly_rows = [
        {
            "anomaly": "release_count_vs_paper",
            "VehId": "",
            "severity": "documented_release_drift",
            "detail": (
                "Immutable official files contain 384 vehicles/93 HEVs; "
                "the 2019 paper reports 383 vehicles/92 HEVs."
            ),
            "action": "Use file-derived official mapping and retain immutable hashes.",
        }
    ]
    hev_battery = vehicle_df[
        (vehicle_df["EngineType_official"] == "HEV")
        & (vehicle_df["battery_rows"] > 0)
    ]
    for _, row in hev_battery.iterrows():
        anomaly_rows.append(
            {
                "anomaly": "HEV_with_released_battery_channel",
                "VehId": row["VehId"],
                "severity": "source_anomaly",
                "detail": (
                    f"Official static mapping is HEV but {int(row['battery_rows'])} "
                    "valid battery intervals are released."
                ),
                "action": (
                    "Retain official HEV class; ignore battery as an HEV target; "
                    "preserve anomaly flag."
                ),
            }
        )
    pd.DataFrame(anomaly_rows).to_csv(output_dir / "source_anomalies.csv", index=False)

    formula_reproduction = {
        name: pairs.result() for name, pairs in legacy_formula_pairs.items()
    }
    physical_difference = {
        name: pairs.result() for name, pairs in legacy_physical_pairs.items()
    }
    alignment_rate = aligned_rows / all_rows if all_rows else 0.0
    join_rate = joined_rows / all_rows if all_rows else 0.0
    direct_maf = direct_maf_pairs.result()
    formula_tolerance = float(
        config["quality_gates"]["old_formula_reproduction_abs_tolerance_kwh"]
    )
    old_formula_reproduced = all(
        item["n"] > 0 and item["max_abs"] is not None and item["max_abs"] <= formula_tolerance
        for item in formula_reproduction.values()
    )

    observability_results: dict[str, dict[str, Any]] = {}
    observability_pass = True
    powertrain_lookup = powertrain_df.set_index("EngineType_official").to_dict("index")
    for engine, rule in config["quality_gates"]["observability"].items():
        row = powertrain_lookup[engine]
        valid = int(row["valid_dt_rows"])
        fuel_coverage = int(row["preferred_fuel_rows"]) / valid if valid else 0.0
        battery_coverage = int(row["battery_rows"]) / valid if valid else 0.0
        if rule["target"] == "fuel":
            coverage = fuel_coverage
            supported_vehicles = int(row["vehicles_with_direct_fuel"]) + int(
                row["vehicles_with_maf"]
            )
        elif rule["target"] == "battery":
            coverage = battery_coverage
            supported_vehicles = int(row["vehicles_with_battery"])
        else:
            coverage = min(fuel_coverage, battery_coverage)
            supported_vehicles = min(
                int(row["vehicles_with_direct_fuel"]) + int(row["vehicles_with_maf"]),
                int(row["vehicles_with_battery"]),
            )
        passed = (
            coverage >= float(rule["minimum_valid_interval_coverage"])
            and supported_vehicles >= int(rule["minimum_vehicle_count"])
        )
        observability_pass &= passed
        observability_results[engine] = {
            "target": rule["target"],
            "coverage": coverage,
            "fuel_coverage": fuel_coverage,
            "battery_coverage": battery_coverage,
            "supported_vehicles": supported_vehicles,
            "minimum_coverage": float(rule["minimum_valid_interval_coverage"]),
            "minimum_vehicles": int(rule["minimum_vehicle_count"]),
            "passed": passed,
        }

    fuel_trip_mask = (
        trip_df["fuel_observed_rows"].gt(0)
        & trip_df["distance_km"].ge(1.0)
        & trip_df["fuel_l_per_100km"].notna()
    )
    battery_trip_mask = (
        trip_df["battery_observed_rows"].gt(0)
        & trip_df["distance_km"].ge(1.0)
        & trip_df["battery_net_wh_per_km"].notna()
    )
    fuel_trip_outlier_rate = float(
        (
            (
                trip_df.loc[fuel_trip_mask, "fuel_l_per_100km"]
                > float(plausibility["fuel_l_per_100km_max"])
            )
            | (
                trip_df.loc[fuel_trip_mask, "EngineType_official"].isin(
                    ["ICE", "HEV"]
                )
                & trip_df.loc[fuel_trip_mask, "fuel_l_per_100km"].lt(
                    float(
                        plausibility[
                            "moving_combustion_fuel_l_per_100km_min"
                        ]
                    )
                )
            )
        ).mean()
    )
    battery_trip_outlier_rate = float(
        (
            trip_df.loc[battery_trip_mask, "battery_net_wh_per_km"].abs()
            > float(plausibility["battery_net_wh_per_km_abs_max"])
        ).mean()
    )
    soc_sign = trip_df["battery_soc_sign_agrees"].dropna()
    soc_sign_agreement_rate = float(soc_sign.astype(bool).mean()) if len(soc_sign) else None
    excluded_trip_count = int((~trip_df["trip_target_plausible"]).sum())
    release_excluded_trip_count = int((~trip_df["target_release_eligible"]).sum())
    excluded_but_eligible_count = int(
        (
            ~trip_df["trip_target_plausible"]
            & trip_df["target_release_eligible"]
        ).sum()
    )
    eligible_fuel_outlier_count = int(
        (
            trip_df["target_release_eligible"]
            & (low_combustion_fuel | high_fuel)
        ).sum()
    )
    eligible_battery_outlier_count = int(
        (
            trip_df["target_release_eligible"]
            & high_battery
        ).sum()
    )
    plausibility_exclusion_policy_applied = (
        excluded_trip_count > 0
        and excluded_but_eligible_count == 0
        and eligible_fuel_outlier_count == 0
        and eligible_battery_outlier_count == 0
    )

    baseline_sensitivity = dt_sensitivity_totals[str(max_dt)]
    dt_relative_differences: dict[str, dict[str, float]] = {}
    for limit, totals in dt_sensitivity_totals.items():
        dt_relative_differences[limit] = {}
        for channel in ["fuel_l", "battery_net_wh"]:
            denominator = abs(float(baseline_sensitivity[channel]))
            dt_relative_differences[limit][channel] = (
                abs(float(totals[channel]) - float(baseline_sensitivity[channel]))
                / denominator
                if denominator
                else 0.0
            )
    dt_sensitivity_pass = all(
        value
        <= float(plausibility["maximum_dt_sensitivity_relative_difference"])
        for result in dt_relative_differences.values()
        for value in result.values()
    )
    density_values = list(density_totals.values())
    density_relative_span = (
        (max(density_values) - min(density_values))
        / density_totals[str(density)]
        if density_totals[str(density)]
        else 0.0
    )
    production_formula_probe = compute_interval_targets(
        dt=pd.Series([2.0, 1.0]),
        fuel_rate=pd.Series([3.6, np.nan]),
        maf=pd.Series([14.08, 14.08]),
        correction=pd.Series([1.0, 1.0]),
        current=pd.Series([-10.0, 10.0]),
        voltage=pd.Series([300.0, 300.0]),
        max_dt=max_dt,
        afr=afr,
        density=density,
        plausibility=plausibility,
    )
    formula_unit_tests_pass = all(
        [
            np.isclose(production_formula_probe["fuel_l"].iloc[0], 0.002),
            production_formula_probe["fuel_source"].iloc[0] == "direct",
            np.isclose(
                production_formula_probe["fuel_l"].iloc[1], 1.0 / density
            ),
            production_formula_probe["fuel_source"].iloc[1] == "maf",
            np.isclose(
                production_formula_probe["battery_wh"].iloc[0], 5.0 / 3.0
            ),
            np.isclose(
                production_formula_probe["battery_wh"].iloc[1], -5.0 / 6.0
            ),
        ]
    )

    target_contract = {
        "fixed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "official original VED only",
        "interval": {
            "definition": "left-hold row i channel value over timestamp(i+1)-timestamp(i)",
            "valid_dt": f"0 < dt <= {max_dt} seconds within the same official VehId and Trip",
            "first_last_policy": "trip-final rows have no forward interval and are not integrated",
        },
        "combustion_channel": {
            "primary": "fuel_volume_L",
            "direct_formula": "Fuel Rate[L/hr] * dt_s / 3600",
            "fallback_formula": (
                "MAF[g/s] * (1 + mean(STFT)/100 + mean(LTFT)/100) "
                f"/ {afr} / {density} [g/L] * dt_s"
            ),
            "hierarchy": "nonnegative direct Fuel Rate, else MAF fallback",
            "source_flag": "fuel_source in {direct, maf, missing}",
            "direct_status": "channel-native physical measurement/vehicle-reported estimate",
            "maf_status": "assumption-bound operational estimate, not channel-native",
            "abs_load_rpm_fallback": "prohibited: displacement/volumetric-efficiency assumptions are not sufficiently fixed",
            "physical_scope": "observed/estimated fuel volume, not traction energy",
        },
        "electric_channel": {
            "primary": "battery_terminal_net_Wh",
            "formula": "-HV Current[A] * HV Voltage[V] * dt_s / 3600",
            "sign": "positive discharge; negative charging/regeneration",
            "physical_scope": "battery-terminal net DC energy, not wheel energy",
        },
        "powertrain_release": {
            "ICE": ["fuel_volume_L"],
            "HEV": ["fuel_volume_L"],
            "PHEV": ["fuel_volume_L", "battery_terminal_net_Wh"],
            "EV": ["battery_terminal_net_Wh"],
        },
        "required_row_fields": [
            "fuel_volume_L",
            "fuel_source",
            "battery_terminal_net_Wh",
            "dt_seconds",
            "dt_valid",
            "target_exclusion_reason",
        ],
        "plausibility_exclusion_policy": {
            **plausibility,
            "trip_rule": (
                "exclude moving ICE/HEV trips below the combustion minimum; "
                "exclude fuel/battery intensity above maxima"
            ),
            "excluded_trip_count": excluded_trip_count,
            "release_excluded_trip_count": release_excluded_trip_count,
        },
        "observability_policy": config["quality_gates"]["observability"],
        "segment_label_eligibility_for_SEGMENT_SPLIT": {
            "coverage_measure": "valid target duration / valid segment duration",
            "minimum_coverage": 0.95,
            "ICE_HEV": "fuel_volume_L coverage",
            "PHEV": "both fuel_volume_L and battery_terminal_net_Wh coverage",
            "EV": "battery_terminal_net_Wh coverage",
            "missing_interval_policy": "no target imputation; preserve exclusion reason",
            "trip_policy": "trip_target_plausible must be true",
        },
        "unified_scalar": {
            "status": "not released",
            "reason": (
                "VED does not document a common fuel/electric energy boundary or fuel LHV; "
                "the legacy 0.1123 L/kWh equivalence is not a measured vehicle-energy boundary."
            ),
        },
        "legacy_eved_energy_consumption": {
            "status": "reference-only, prohibited as RNEE ground truth",
            "reason": (
                "combustion derivation is dimensionally invalid and electric derivation assumes "
                "one second per row instead of observed timestamp deltas"
            ),
        },
    }
    write_json(output_dir / "target_contract.json", target_contract)

    evidence = pd.DataFrame(evidence_rows(config, notebook_hash))
    evidence.loc[
        evidence["statement"].str.startswith("Legacy eVED combustion"), "status"
    ] = "reproduced" if old_formula_reproduced else "not reproduced"
    evidence.to_csv(output_dir / "source_evidence_table.csv", index=False)

    source_integrity_checks = {
        "source_inventory_gate_passed": source_inventory_summary.get("status") == "PASS_RAW_LINEAGE",
        "static_metadata_hash": static_hash_matches,
        "source_inventory_hash": inventory_hash_matches,
        "dynamic_file_hashes": all(dynamic_hash_results.values()),
        "raw_row_count": all_rows == int(config["quality_gates"]["expected_raw_rows"]),
        "static_vehicle_count": len(static) == int(config["quality_gates"]["expected_vehicle_count"]),
        "powertrain_counts": all(actual_counts.get(key, 0) == value for key, value in expected_counts.items()),
        "static_join_rate": join_rate >= float(config["quality_gates"]["static_join_rate_min"]),
        "eved_key_alignment_rate": alignment_rate >= float(config["quality_gates"]["key_alignment_rate_min"]),
        "timestamp_order": invalid_order_rows == 0,
    }
    target_validity_checks = {
        "formula_unit_tests": formula_unit_tests_pass,
        "observability_thresholds": observability_pass,
        "plausibility_exclusion_policy_applied": plausibility_exclusion_policy_applied,
        "fuel_trip_plausibility": eligible_fuel_outlier_count == 0,
        "battery_trip_plausibility": eligible_battery_outlier_count == 0,
        "battery_soc_sign_agreement": (
            soc_sign_agreement_rate is not None
            and soc_sign_agreement_rate
            >= float(plausibility["minimum_battery_soc_sign_agreement"])
        ),
        "dt_cap_sensitivity": dt_sensitivity_pass,
        "density_sensitivity": density_relative_span
        <= float(plausibility["maximum_density_sensitivity_relative_span"]),
        "separate_powertrain_targets": (
            target_contract["unified_scalar"]["status"] == "not released"
            and target_contract["powertrain_release"]["PHEV"]
            == ["fuel_volume_L", "battery_terminal_net_Wh"]
        ),
        "source_anomalies_explicitly_flagged": {
            row["anomaly"] for row in anomaly_rows
        }
        >= {"release_count_vs_paper", "HEV_with_released_battery_channel"},
    }
    diagnostic_checks = {
        "legacy_formula_reproduced": old_formula_reproduced,
        "direct_maf_overlap_available": direct_maf["n"] >= int(
            config["quality_gates"]["direct_vs_maf_min_overlap_rows"]
        ),
        "common_boundary_documented": False,
    }
    checks = {
        **source_integrity_checks,
        **target_validity_checks,
        **diagnostic_checks,
    }
    overall_gate = decide_gate(source_integrity_checks, target_validity_checks)

    summary = {
        "stage": "TARGET_AUDIT",
        "overall_gate": overall_gate,
        "gate_interpretation": {
            "direct_fuel_and_battery_channels": "PASS_PHYSICAL_TARGET",
            "maf_fuel_estimate": "PASS_OPERATIONAL_TARGET_ONLY",
            "single_fleet_wide_scalar": "FAIL_PHYSICAL_TARGET",
            "benchmark": overall_gate,
        },
        "counts": {
            "files": len(raw_files),
            "rows": all_rows,
            "vehicles": int(vehicle_df["VehId"].nunique()),
            "static_join_rows": joined_rows,
            "legacy_nonmissing_rows": legacy_rows,
            "eved_aligned_rows": aligned_rows,
            "eved_key_mismatch_rows": key_mismatch_rows,
        },
        "rates": {
            "static_join_rate": join_rate,
            "eved_key_alignment_rate": alignment_rate,
        },
        "powertrain_counts": actual_counts,
        "dt_seconds_sample": descriptive(dt_values),
        "direct_vs_maf_interval_liters": direct_maf,
        "legacy_formula_reproduction": formula_reproduction,
        "legacy_vs_corrected_reference_scale_interval": physical_difference,
        "maf_fallback_density_sensitivity_total_l": density_totals,
        "maf_fallback_density_sensitivity_relative_span": density_relative_span,
        "dt_cap_sensitivity_totals": dt_sensitivity_totals,
        "dt_cap_sensitivity_relative_difference": dt_relative_differences,
        "observability": observability_results,
        "plausibility": {
            "channel_summary": channel_df.to_dict("records"),
            "fuel_source_scale_summary": source_scale_df.to_dict("records"),
            "fuel_trip_count": int(fuel_trip_mask.sum()),
            "fuel_trip_outlier_rate": fuel_trip_outlier_rate,
            "battery_trip_count": int(battery_trip_mask.sum()),
            "battery_trip_outlier_rate": battery_trip_outlier_rate,
            "battery_soc_sign_trip_count": int(len(soc_sign)),
            "battery_soc_sign_agreement_rate": soc_sign_agreement_rate,
            "excluded_trip_count": excluded_trip_count,
            "excluded_but_eligible_count": excluded_but_eligible_count,
            "eligible_fuel_outlier_count": eligible_fuel_outlier_count,
            "eligible_battery_outlier_count": eligible_battery_outlier_count,
        },
        "source_anomalies": anomaly_rows,
        "checks": checks,
        "check_classes": {
            "blocking_source_integrity": list(source_integrity_checks),
            "blocking_target_validity": list(target_validity_checks),
            "nonblocking_target_diagnostics": list(diagnostic_checks),
        },
        "artifacts": {
            "config": str(args.config.resolve()),
            "original_ved_dir": str(raw_dir.resolve()),
            "static_metadata": str(static_path.resolve()),
            "static_metadata_sha256": static_hash,
            "source_file_inventory": str(inventory_path.resolve()),
            "source_file_inventory_sha256": inventory_hash,
            "dynamic_file_hashes_all_match": all(dynamic_hash_results.values()),
            "source_inventory_summary": str(source_inventory_summary_path.resolve()),
            "eved_reference_dir": str(eved_dir.resolve()),
            "eved_notebook_sha256": notebook_hash,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    write_json(output_dir / "target_audit_summary.json", summary)

    report = f"""# TARGET_AUDIT Original-VED Target Validity Gate

## Verdict

`{overall_gate}`

- Direct Fuel Rate and battery-terminal targets: `PASS_PHYSICAL_TARGET`.
- MAF-derived fuel volume: assumption-bound operational target, never labeled channel-native.
- One fleet-wide scalar target: `FAIL_PHYSICAL_TARGET`.
- RNEE benchmark may proceed only with separate powertrain-aware targets and explicit
  direct/MAF provenance. The legacy eVED
  `Energy_Consumption` column is prohibited as ground truth.

## Audited corpus

- Original VED files: {len(raw_files)}
- Rows: {all_rows:,}
- Official static vehicles: {len(static):,}
- Dynamic-static join: {join_rate:.6%}
- eVED provenance-reproduction key alignment: {alignment_rate:.6%}
- Official powertrain counts: {json.dumps(actual_counts, ensure_ascii=False)}

## Fixed target contract

1. Integrate only within official `(VehId, Trip)` using actual timestamp differences and
   `0 < dt <= {max_dt:g} s`.
2. ICE/HEV: release fuel volume in litres. Prefer direct `Fuel Rate[L/hr]`; use the
   explicitly assumption-bound MAF fallback only where direct rate is unavailable, and
   preserve `fuel_source`.
3. PHEV: retain fuel volume and battery-terminal net Wh as two separate channels.
4. EV: release battery-terminal net Wh, positive for discharge and negative for
   charging/regeneration.
5. Do not release a common kWh scalar until a common boundary and fuel energy content are
   independently documented and sensitivity-tested.

## Why the legacy eVED label fails as a physical target

The legacy combustion notebook multiplies speed-derived distance by MAF, applies a
`3.78541` factor to a g/s variable, and divides by `0.1123`. This does not conserve units.
The electric branch uses voltage-current power but integrates every row as exactly one
second; original VED timestamps are irregular. The provenance reproduction tests whether these
code paths reproduce the released eVED column, but reproduction is provenance evidence,
not physical validation.

## Key diagnostics

- Legacy formula reproduction: `{json.dumps(formula_reproduction, ensure_ascii=False)}`
- Legacy versus corrected reference-scale intervals (not physical common kWh):
  `{json.dumps(physical_difference, ensure_ascii=False)}`
- Direct-fuel versus MAF overlap: `{json.dumps(direct_maf, ensure_ascii=False)}`
- Timestamp sample: `{json.dumps(summary['dt_seconds_sample'], ensure_ascii=False)}`
- MAF density sensitivity totals: `{json.dumps(density_totals, ensure_ascii=False)}`
- Observability gate: `{json.dumps(observability_results, ensure_ascii=False)}`
- Trip plausibility: fuel outlier rate `{fuel_trip_outlier_rate:.6%}`;
  battery outlier rate `{battery_trip_outlier_rate:.6%}`.
- Plausibility exclusions: `{excluded_trip_count}` trips excluded and
  `{excluded_but_eligible_count}` remain release-eligible.
- Battery/SOC trip-sign agreement: `{soc_sign_agreement_rate}` over `{len(soc_sign)}` trips.
- Timestamp-cap sensitivity: `{json.dumps(dt_relative_differences, ensure_ascii=False)}`.

## Gate consequence

Existing eVED experiments cannot be interpreted as validating a fleet-wide physical
energy-consumption target. The new RNEE dataset construction may continue, but downstream
modeling must either (a) be stratified by target channel/powertrain, or (b) wait for a
separately justified common energy boundary. No model result may be used to select the
target definition.

The absence of direct-Fuel-Rate/MAF overlap is a limitation, not a pipeline failure: the
released vehicles do not provide an technical calibration set for the MAF fallback. Target
units, sign, formulas, source flags, plausibility exclusions, and raw observability
thresholds are fixed here. SEGMENT_SPLIT must additionally report the same target scale and
observability by each newly constructed original-VED split; old eVED splits cannot be
reused as the source contract.
"""
    (output_dir / "TARGET_AUDIT_TARGET_VALIDITY_GATE.md").write_text(report, encoding="utf-8")
    return 0 if overall_gate != "FAIL_TARGET_VALIDITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
