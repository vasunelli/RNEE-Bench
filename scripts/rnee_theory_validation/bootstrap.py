from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from evaluation import SYSTEM_TO_PREDICTION


EFFECT_ORDER = (
    "gain_real",
    "gain_noise",
    "gain_permutation",
    "real_minus_noise",
    "real_minus_permutation",
)


def simultaneous_max_t_intervals(
    point: np.ndarray,
    bootstrap: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    point = np.asarray(point, dtype=float)
    bootstrap = np.asarray(bootstrap, dtype=float)
    standard_error = np.nanstd(bootstrap, axis=0, ddof=1)
    safe = np.where(standard_error > 0, standard_error, 1.0)
    standardized = np.abs((bootstrap - point[None, :]) / safe[None, :])
    standardized[:, standard_error <= 0] = 0.0
    maximum = np.nanmax(standardized, axis=1)
    critical = float(np.nanquantile(maximum, 1.0 - alpha))
    half_width = critical * standard_error
    return point - half_width, point + half_width, critical


def classify_gain(lower: float, upper: float, sesoi: float) -> str:
    if lower > sesoi:
        return "PRACTICAL_POSITIVE"
    if lower > -sesoi and upper < sesoi:
        return "EQUIVALENT_NO_MEANINGFUL_EFFECT"
    if upper < -sesoi:
        return "PRACTICAL_NEGATIVE"
    return "UNRESOLVED"


def classify_specificity(lower: float, upper: float, sesoi: float) -> str:
    if lower > 0:
        return "REAL_SUPERIOR"
    if lower > -sesoi and upper < sesoi:
        return "EQUIVALENT_WITHIN_SESOI"
    if upper < 0:
        return "CONTROL_SUPERIOR"
    return "UNRESOLVED"


def _effect_arrays(mae: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for sensor, prefix in (("low", "L"), ("high", "H")):
        baseline = mae[f"{prefix}-R0"]
        real = (baseline - mae[f"{prefix}-R-real"]) / baseline
        noise = (baseline - mae[f"{prefix}-R-noise"]) / baseline
        permutation = (baseline - mae[f"{prefix}-R-perm"]) / baseline
        result[f"{sensor}|gain_real"] = real
        result[f"{sensor}|gain_noise"] = noise
        result[f"{sensor}|gain_permutation"] = permutation
        result[f"{sensor}|real_minus_noise"] = real - noise
        result[f"{sensor}|real_minus_permutation"] = real - permutation
    return result


def vehicle_cluster_bootstrap(
    environment_frames: dict[str, pd.DataFrame],
    repetitions: int,
    seed: int,
    alpha: float,
    sesoi: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Joint highest-level vehicle bootstrap across all frozen environments."""
    environments = list(environment_frames)
    vehicles = sorted(
        {
            str(vehicle)
            for frame in environment_frames.values()
            for vehicle in frame["VehId"].astype(str).unique()
        }
    )
    if len(vehicles) < 2:
        raise RuntimeError("Vehicle-cluster bootstrap needs at least two vehicles.")
    vehicle_index = {vehicle: index for index, vehicle in enumerate(vehicles)}
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(vehicles),
        np.full(len(vehicles), 1.0 / len(vehicles)),
        size=repetitions,
    ).astype(float)

    point_mae: dict[str, list[float]] = {
        system: [] for system in SYSTEM_TO_PREDICTION
    }
    bootstrap_mae: dict[str, list[np.ndarray]] = {
        system: [] for system in SYSTEM_TO_PREDICTION
    }
    for environment in environments:
        frame = environment_frames[environment].copy()
        frame["VehId"] = frame["VehId"].astype(str)
        aggregations: dict[str, tuple[str, str]] = {
            "count": ("segment_id", "size")
        }
        for system, prediction in SYSTEM_TO_PREDICTION.items():
            ae_column = f"ae__{system}"
            frame[ae_column] = np.abs(
                frame["target_l"].to_numpy(float)
                - frame[prediction].to_numpy(float)
            )
            aggregations[f"sum__{system}"] = (ae_column, "sum")
        grouped = frame.groupby("VehId", sort=False).agg(**aggregations)
        counts = np.zeros(len(vehicles), dtype=float)
        sums = {
            system: np.zeros(len(vehicles), dtype=float)
            for system in SYSTEM_TO_PREDICTION
        }
        for vehicle, row in grouped.iterrows():
            position = vehicle_index[str(vehicle)]
            counts[position] = float(row["count"])
            for system in SYSTEM_TO_PREDICTION:
                sums[system][position] = float(row[f"sum__{system}"])
        denominators = weights @ counts
        if np.any(denominators <= 0):
            raise RuntimeError(f"Empty bootstrap support in {environment}.")
        for system in SYSTEM_TO_PREDICTION:
            point_mae[system].append(float(sums[system].sum() / counts.sum()))
            bootstrap_mae[system].append((weights @ sums[system]) / denominators)

    point_arrays = {
        system: np.asarray(values, dtype=float)
        for system, values in point_mae.items()
    }
    bootstrap_arrays = {
        system: np.column_stack(values)
        for system, values in bootstrap_mae.items()
    }
    point_effects = _effect_arrays(point_arrays)
    bootstrap_effects = _effect_arrays(bootstrap_arrays)

    # Arrays are environment-indexed; flatten explicitly into frozen families.
    gain_members: list[tuple[str, str, str]] = []
    specificity_members: list[tuple[str, str, str]] = []
    for environment in environments:
        for sensor in ("low", "high"):
            for effect in EFFECT_ORDER[:3]:
                gain_members.append((environment, sensor, effect))
            for effect in EFFECT_ORDER[3:]:
                specificity_members.append((environment, sensor, effect))

    def collect(
        members: list[tuple[str, str, str]]
    ) -> tuple[np.ndarray, np.ndarray]:
        points: list[float] = []
        samples: list[np.ndarray] = []
        for environment, sensor, effect in members:
            environment_index = environments.index(environment)
            key = f"{sensor}|{effect}"
            points.append(float(point_effects[key][environment_index]))
            samples.append(bootstrap_effects[key][:, environment_index])
        return np.asarray(points), np.column_stack(samples)

    family_results: dict[str, Any] = {}
    interval_lookup: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for family_name, members in (
        ("road_gain", gain_members),
        ("semantic_specificity", specificity_members),
    ):
        family_point, family_bootstrap = collect(members)
        lower, upper, critical = simultaneous_max_t_intervals(
            family_point, family_bootstrap, alpha
        )
        family_results[family_name] = {
            "member_count": len(members),
            "critical": critical,
        }
        for index, member in enumerate(members):
            interval_lookup[member] = (
                float(lower[index]),
                float(upper[index]),
                critical,
            )

    effect_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for environment_index, environment in enumerate(environments):
        for sensor in ("low", "high"):
            for effect in EFFECT_ORDER:
                key = f"{sensor}|{effect}"
                values = bootstrap_effects[key][:, environment_index]
                estimate = float(point_effects[key][environment_index])
                lower, upper, critical = interval_lookup[
                    (environment, sensor, effect)
                ]
                state = (
                    classify_gain(lower, upper, sesoi)
                    if effect.startswith("gain_")
                    else classify_specificity(lower, upper, sesoi)
                )
                effect_rows.append(
                    {
                        "environment": environment,
                        "sensor": sensor,
                        "effect_type": effect,
                        "estimate": estimate,
                        "pointwise_lcb": float(np.quantile(values, alpha / 2)),
                        "pointwise_ucb": float(np.quantile(values, 1 - alpha / 2)),
                        "simultaneous_lcb": lower,
                        "simultaneous_ucb": upper,
                        "simultaneous_critical": critical,
                        "state": state,
                        "family": (
                            "road_gain"
                            if effect.startswith("gain_")
                            else "semantic_specificity"
                        ),
                    }
                )
                bootstrap_rows.extend(
                    {
                        "replicate": int(replicate),
                        "environment": environment,
                        "sensor": sensor,
                        "effect_type": effect,
                        "value": float(value),
                    }
                    for replicate, value in enumerate(values)
                )
    diagnostics = {
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_seed": int(seed),
        "cluster_level": "vehicle",
        "vehicle_union_count": int(len(vehicles)),
        "environment_order": environments,
        "alpha": float(alpha),
        "relative_sesoi": float(sesoi),
        "families": family_results,
    }
    return (
        pd.DataFrame(effect_rows),
        pd.DataFrame(bootstrap_rows),
        diagnostics,
    )


def final_decision(
    effects: pd.DataFrame,
    sesoi: float,
    minimum_environment_passes: int = 3,
) -> dict[str, Any]:
    cell_rows: list[dict[str, Any]] = []
    for environment in effects["environment"].drop_duplicates():
        for sensor in ("low", "high"):
            cell = effects[
                effects["environment"].eq(environment)
                & effects["sensor"].eq(sensor)
            ].set_index("effect_type")
            real = cell.loc["gain_real"]
            noise = cell.loc["real_minus_noise"]
            permutation = cell.loc["real_minus_permutation"]
            passed = bool(
                float(real["simultaneous_lcb"]) > sesoi
                and float(noise["simultaneous_lcb"]) > 0
                and float(permutation["simultaneous_lcb"]) > 0
            )
            cell_rows.append(
                {
                    "environment": environment,
                    "sensor": sensor,
                    "semantic_specificity_pass": passed,
                    "real_practical_negative": bool(
                        float(real["simultaneous_ucb"]) < -sesoi
                    ),
                    "noise_state": noise["state"],
                    "permutation_state": permutation["state"],
                }
            )
    cells = pd.DataFrame(cell_rows)
    sensor_routes: dict[str, Any] = {}
    for sensor in ("low", "high"):
        subset = cells[cells["sensor"].eq(sensor)]
        pass_count = int(subset["semantic_specificity_pass"].sum())
        negative_count = int(subset["real_practical_negative"].sum())
        sensor_routes[sensor] = {
            "cell_pass_count": pass_count,
            "practical_negative_count": negative_count,
            "cross_ood_pass": (
                pass_count >= minimum_environment_passes and negative_count == 0
            ),
        }
    if any(item["cross_ood_pass"] for item in sensor_routes.values()):
        route = "PASS_SEMANTIC_SPECIFICITY"
        reason = "A frozen sensor regime passes the 3-of-4 rule with no practical negative environment."
    elif bool(cells["semantic_specificity_pass"].any()):
        route = "CONTEXT_DEPENDENT_ONLY"
        reason = "Semantic specificity is supported only in selected frozen environment-sensor cells."
    elif bool(
        cells["noise_state"].eq("EQUIVALENT_WITHIN_SESOI").all()
    ):
        route = "FAIL_DIMENSIONAL_EFFECT"
        reason = "Real-minus-noise is simultaneously equivalent within the frozen SESOI in all primary cells."
    elif bool(
        cells["permutation_state"].eq("EQUIVALENT_WITHIN_SESOI").all()
    ):
        route = "FAIL_ALIGNMENT"
        reason = "Real-minus-permutation is simultaneously equivalent within the frozen SESOI in all primary cells."
    else:
        route = "CONTEXT_DEPENDENT_ONLY"
        reason = "The complete frozen matrix is heterogeneous or unresolved and does not pass a global specificity route."
    return {
        "route": route,
        "reason": reason,
        "sensor_routes": sensor_routes,
        "cell_results": cell_rows,
    }
