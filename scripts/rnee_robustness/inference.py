from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd


MODELS = ("hgb", "ridge", "catboost")
SENSORS = ("low", "high")
NULL_TYPES = ("noise", "permutation")


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


def _gain(baseline: np.ndarray, comparison: np.ndarray) -> np.ndarray:
    if np.any(baseline <= 0):
        raise RuntimeError("Relative MAE gain requires positive baseline MAE.")
    return (baseline - comparison) / baseline


def _state(effect_type: str, lower: float, upper: float, sesoi: float) -> str:
    if effect_type.startswith("gain_"):
        if lower > sesoi:
            return "PRACTICAL_POSITIVE"
        if upper < -sesoi:
            return "PRACTICAL_NEGATIVE"
        if lower > -sesoi and upper < sesoi:
            return "EQUIVALENT_WITHIN_SESOI"
        return "UNRESOLVED"
    if lower > 0:
        return "REAL_SUPERIOR"
    if upper < 0:
        return "CONTROL_SUPERIOR"
    if lower > -sesoi and upper < sesoi:
        return "EQUIVALENT_WITHIN_SESOI"
    return "UNRESOLVED"


def _vehicle_mae(
    frame: pd.DataFrame,
    prediction_columns: dict[str, str],
    vehicles: list[str],
    weights: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    working = frame[["segment_id", "VehId", "target_l", *prediction_columns.values()]]
    working = working.copy()
    working["VehId"] = working["VehId"].astype(str)
    aggregations: dict[str, tuple[str, str]] = {"count": ("segment_id", "size")}
    for key, column in prediction_columns.items():
        ae_column = f"ae__{len(aggregations)}"
        working[ae_column] = np.abs(
            working["target_l"].to_numpy(float)
            - working[column].to_numpy(float)
        )
        aggregations[f"sum__{key}"] = (ae_column, "sum")
    grouped = working.groupby("VehId", sort=False).agg(**aggregations)
    positions = {vehicle: index for index, vehicle in enumerate(vehicles)}
    counts = np.zeros(len(vehicles), dtype=float)
    sums = {
        key: np.zeros(len(vehicles), dtype=float) for key in prediction_columns
    }
    for vehicle, row in grouped.iterrows():
        position = positions[str(vehicle)]
        counts[position] = float(row["count"])
        for key in prediction_columns:
            sums[key][position] = float(row[f"sum__{key}"])
    denominators = weights @ counts
    if np.any(denominators <= 0):
        raise RuntimeError("Vehicle bootstrap produced empty support.")
    point = {
        key: float(values.sum() / counts.sum()) for key, values in sums.items()
    }
    samples = {key: (weights @ values) / denominators for key, values in sums.items()}
    return point, samples


def bootstrap_target(
    target_id: str,
    environment_frames: dict[str, pd.DataFrame],
    prediction_columns: dict[str, dict[str, str]],
    draws: int,
    repetitions: int,
    seed: int,
    alpha: float,
    sesoi: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    environments = list(environment_frames)
    vehicles = sorted(
        {
            str(vehicle)
            for frame in environment_frames.values()
            for vehicle in frame["VehId"].astype(str).unique()
        }
    )
    if len(vehicles) < 2:
        raise RuntimeError("Vehicle bootstrap needs at least two vehicles.")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(vehicles),
        np.full(len(vehicles), 1.0 / len(vehicles)),
        size=repetitions,
    ).astype(float)

    point_mae: dict[str, dict[str, float]] = {}
    bootstrap_mae: dict[str, dict[str, np.ndarray]] = {}
    for environment in environments:
        point_mae[environment], bootstrap_mae[environment] = _vehicle_mae(
            environment_frames[environment],
            prediction_columns[environment],
            vehicles,
            weights,
        )

    members: list[dict[str, Any]] = []

    def add_member(
        family: str,
        environment: str,
        model: str,
        sensor: str,
        effect_type: str,
        point: float,
        samples: np.ndarray,
        null_type: str = "",
        draw: int | None = None,
    ) -> None:
        draw_label = "" if draw is None else f"{draw:02d}"
        effect_id = "|".join(
            [
                target_id,
                family,
                environment,
                model,
                sensor,
                effect_type,
                null_type,
                draw_label,
            ]
        )
        members.append(
            {
                "effect_id": effect_id,
                "target_id": target_id,
                "family": family,
                "environment": environment,
                "model": model,
                "sensor": sensor,
                "effect_type": effect_type,
                "null_type": null_type,
                "draw": draw,
                "estimate": float(point),
                "samples": np.asarray(samples, dtype=float),
            }
        )

    for environment in environments:
        point = point_mae[environment]
        samples = bootstrap_mae[environment]
        for sensor in SENSORS:
            hgb_base_key = f"hgb|{sensor}|baseline"
            hgb_real_key = f"hgb|{sensor}|real"
            hgb_real_point = float(
                _gain(
                    np.asarray([point[hgb_base_key]]),
                    np.asarray([point[hgb_real_key]]),
                )[0]
            )
            hgb_real_samples = _gain(
                samples[hgb_base_key], samples[hgb_real_key]
            )
            add_member(
                "hgb_null_gain",
                environment,
                "hgb",
                sensor,
                "gain_real",
                hgb_real_point,
                hgb_real_samples,
            )
            for null_type in NULL_TYPES:
                for draw in range(1, draws + 1):
                    null_key = f"hgb|{sensor}|{null_type}|{draw:02d}"
                    null_point = float(
                        _gain(
                            np.asarray([point[hgb_base_key]]),
                            np.asarray([point[null_key]]),
                        )[0]
                    )
                    null_samples = _gain(
                        samples[hgb_base_key], samples[null_key]
                    )
                    add_member(
                        "hgb_null_gain",
                        environment,
                        "hgb",
                        sensor,
                        f"gain_{null_type}",
                        null_point,
                        null_samples,
                        null_type,
                        draw,
                    )
                    add_member(
                        "hgb_semantic_specificity",
                        environment,
                        "hgb",
                        sensor,
                        f"real_minus_{null_type}",
                        hgb_real_point - null_point,
                        hgb_real_samples - null_samples,
                        null_type,
                        draw,
                    )
            for model in MODELS:
                base_key = f"{model}|{sensor}|baseline"
                real_key = f"{model}|{sensor}|real"
                model_point = float(
                    _gain(
                        np.asarray([point[base_key]]),
                        np.asarray([point[real_key]]),
                    )[0]
                )
                model_samples = _gain(samples[base_key], samples[real_key])
                add_member(
                    "model_real_gain",
                    environment,
                    model,
                    sensor,
                    "gain_real",
                    model_point,
                    model_samples,
                )

    expected_sizes = {
        "hgb_null_gain": len(environments) * len(SENSORS) * (1 + 2 * draws),
        "hgb_semantic_specificity": (
            len(environments) * len(SENSORS) * 2 * draws
        ),
        "model_real_gain": len(environments) * len(SENSORS) * len(MODELS),
    }
    effect_rows: list[dict[str, Any]] = []
    bootstrap_values: dict[str, np.ndarray] = {}
    family_diagnostics: dict[str, Any] = {}
    for family, family_members_iter in itertools.groupby(
        sorted(members, key=lambda item: item["family"]),
        key=lambda item: item["family"],
    ):
        family_members = list(family_members_iter)
        if len(family_members) != expected_sizes[family]:
            raise RuntimeError(
                f"{target_id}/{family}: {len(family_members)} != "
                f"{expected_sizes[family]}"
            )
        points = np.asarray([item["estimate"] for item in family_members])
        sample_matrix = np.column_stack([item["samples"] for item in family_members])
        lower, upper, critical = simultaneous_max_t_intervals(
            points, sample_matrix, alpha
        )
        family_diagnostics[family] = {
            "member_count": len(family_members),
            "simultaneous_critical": critical,
        }
        for index, item in enumerate(family_members):
            values = item.pop("samples")
            effect_rows.append(
                {
                    **item,
                    "pointwise_lcb": float(np.quantile(values, alpha / 2)),
                    "pointwise_ucb": float(np.quantile(values, 1 - alpha / 2)),
                    "simultaneous_lcb": float(lower[index]),
                    "simultaneous_ucb": float(upper[index]),
                    "simultaneous_critical": critical,
                    "state": _state(
                        item["effect_type"],
                        float(lower[index]),
                        float(upper[index]),
                        sesoi,
                    ),
                }
            )
            bootstrap_values[item["effect_id"]] = values
    diagnostics = {
        "target_id": target_id,
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "cluster_level": "vehicle",
        "vehicle_union_count": len(vehicles),
        "environment_order": environments,
        "alpha": alpha,
        "relative_sesoi": sesoi,
        "families": family_diagnostics,
    }
    return pd.DataFrame(effect_rows), bootstrap_values, diagnostics


def summarize_null_distributions(effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target_id, environment, sensor), cell in effects[
        effects["family"].isin(["hgb_null_gain", "hgb_semantic_specificity"])
    ].groupby(["target_id", "environment", "sensor"], sort=False):
        real = cell[
            cell["family"].eq("hgb_null_gain")
            & cell["effect_type"].eq("gain_real")
        ].iloc[0]
        for null_type in NULL_TYPES:
            gains = cell[
                cell["family"].eq("hgb_null_gain")
                & cell["null_type"].eq(null_type)
            ].sort_values("draw")
            specificity = cell[
                cell["family"].eq("hgb_semantic_specificity")
                & cell["null_type"].eq(null_type)
            ].sort_values("draw")
            values = gains["estimate"].to_numpy(float)
            contrasts = specificity["estimate"].to_numpy(float)
            rows.append(
                {
                    "target_id": target_id,
                    "environment": environment,
                    "sensor": sensor,
                    "null_type": null_type,
                    "draw_count": int(len(gains)),
                    "real_gain": float(real["estimate"]),
                    "null_gain_mean": float(values.mean()),
                    "null_gain_std": float(values.std(ddof=1)),
                    "null_gain_p02_5": float(np.quantile(values, 0.025)),
                    "null_gain_p50": float(np.quantile(values, 0.5)),
                    "null_gain_p97_5": float(np.quantile(values, 0.975)),
                    "null_gain_min": float(values.min()),
                    "null_gain_max": float(values.max()),
                    "probability_real_exceeds_null": float(
                        np.mean(float(real["estimate"]) > values)
                    ),
                    "all_point_contrasts_positive": bool(np.all(contrasts > 0)),
                    "all_simultaneous_lcb_gt_zero": bool(
                        np.all(
                            specificity["simultaneous_lcb"].to_numpy(float) > 0
                        )
                    ),
                    "minimum_specificity_simultaneous_lcb": float(
                        specificity["simultaneous_lcb"].min()
                    ),
                }
            )
    return pd.DataFrame(rows)


def model_comparison(effects: pd.DataFrame) -> pd.DataFrame:
    model_effects = effects[effects["family"].eq("model_real_gain")].copy()
    rows: list[dict[str, Any]] = []
    for (target_id, environment, sensor), cell in model_effects.groupby(
        ["target_id", "environment", "sensor"], sort=False
    ):
        indexed = cell.set_index("model")
        gains = {model: float(indexed.loc[model, "estimate"]) for model in MODELS}
        directions = {
            model: (
                "positive"
                if value > 0
                else "negative"
                if value < 0
                else "zero"
            )
            for model, value in gains.items()
        }
        rows.append(
            {
                "record_type": "cell_agreement",
                "target_id": target_id,
                "environment": environment,
                "sensor": sensor,
                "model_a": "",
                "model_b": "",
                "hgb_gain": gains["hgb"],
                "ridge_gain": gains["ridge"],
                "catboost_gain": gains["catboost"],
                "hgb_direction": directions["hgb"],
                "ridge_direction": directions["ridge"],
                "catboost_direction": directions["catboost"],
                "direction_consistent": len(set(directions.values())) == 1,
                "pearson_effect_correlation": np.nan,
                "spearman_effect_correlation": np.nan,
            }
        )
    for target_id, target in model_effects.groupby("target_id", sort=False):
        pivot = target.pivot_table(
            index=["environment", "sensor"],
            columns="model",
            values="estimate",
            aggfunc="first",
        ).loc[:, list(MODELS)]
        for model_a, model_b in itertools.combinations(MODELS, 2):
            rows.append(
                {
                    "record_type": "pair_correlation",
                    "target_id": target_id,
                    "environment": "",
                    "sensor": "",
                    "model_a": model_a,
                    "model_b": model_b,
                    "hgb_gain": np.nan,
                    "ridge_gain": np.nan,
                    "catboost_gain": np.nan,
                    "hgb_direction": "",
                    "ridge_direction": "",
                    "catboost_direction": "",
                    "direction_consistent": np.nan,
                    "pearson_effect_correlation": float(
                        pivot[model_a].corr(pivot[model_b], method="pearson")
                    ),
                    "spearman_effect_correlation": float(
                        pivot[model_a].corr(pivot[model_b], method="spearman")
                    ),
                }
            )
    return pd.DataFrame(rows)


def final_decision(
    effects: pd.DataFrame,
    null_summary: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    sesoi = float(config["decision"]["relative_sesoi"])
    probability_threshold = float(
        config["decision"]["null_stability"][
            "minimum_probability_real_exceeds_each_null_type"
        ]
    )
    references = config["decision"]["frozen_positive_reference_cells"]
    cell_results: list[dict[str, Any]] = []
    for reference in references:
        target_id = reference["target_id"]
        environment = reference["environment"]
        sensor = reference["sensor"]
        cell_effects = effects[
            effects["target_id"].eq(target_id)
            & effects["environment"].eq(environment)
            & effects["sensor"].eq(sensor)
        ]
        real = cell_effects[
            cell_effects["family"].eq("hgb_null_gain")
            & cell_effects["effect_type"].eq("gain_real")
        ].iloc[0]
        summary = null_summary[
            null_summary["target_id"].eq(target_id)
            & null_summary["environment"].eq(environment)
            & null_summary["sensor"].eq(sensor)
        ].set_index("null_type")
        models = cell_effects[cell_effects["family"].eq("model_real_gain")].set_index(
            "model"
        )
        model_point_positive = bool(
            np.all(models.loc[list(MODELS), "estimate"].to_numpy(float) > 0)
        )
        model_lcb_positive = bool(
            np.all(
                models.loc[list(MODELS), "simultaneous_lcb"].to_numpy(float) > 0
            )
        )
        no_model_practical_negative = bool(
            np.all(
                models.loc[list(MODELS), "simultaneous_ucb"].to_numpy(float)
                >= -sesoi
            )
        )
        descriptive_null_pass = bool(
            np.all(
                summary.loc[
                    list(NULL_TYPES), "probability_real_exceeds_null"
                ].to_numpy(float)
                >= probability_threshold
            )
        )
        strict_null_pass = bool(
            float(real["simultaneous_lcb"]) > sesoi
            and summary.loc[
                list(NULL_TYPES), "all_point_contrasts_positive"
            ].astype(bool).all()
            and summary.loc[
                list(NULL_TYPES), "all_simultaneous_lcb_gt_zero"
            ].astype(bool).all()
        )
        strict_model_pass = bool(
            model_point_positive
            and model_lcb_positive
            and no_model_practical_negative
        )
        cell_results.append(
            {
                **reference,
                "real_gain": float(real["estimate"]),
                "real_gain_simultaneous_lcb": float(real["simultaneous_lcb"]),
                "noise_probability_real_exceeds": float(
                    summary.loc["noise", "probability_real_exceeds_null"]
                ),
                "permutation_probability_real_exceeds": float(
                    summary.loc["permutation", "probability_real_exceeds_null"]
                ),
                "descriptive_null_pass": descriptive_null_pass,
                "strict_null_pass": strict_null_pass,
                "model_point_direction_consistent_positive": model_point_positive,
                "all_model_simultaneous_lcb_gt_zero": model_lcb_positive,
                "no_model_practical_negative": no_model_practical_negative,
                "strict_model_pass": strict_model_pass,
                "strict_cell_pass": strict_null_pass and strict_model_pass,
            }
        )

    primary = effects[effects["target_id"].eq("ICE_fuel_L")]
    hgb_cells: list[dict[str, Any]] = []
    for environment in primary["environment"].drop_duplicates():
        for sensor in SENSORS:
            cell = primary[
                primary["environment"].eq(environment)
                & primary["sensor"].eq(sensor)
            ]
            real = cell[
                cell["family"].eq("hgb_null_gain")
                & cell["effect_type"].eq("gain_real")
            ].iloc[0]
            summary = null_summary[
                null_summary["target_id"].eq("ICE_fuel_L")
                & null_summary["environment"].eq(environment)
                & null_summary["sensor"].eq(sensor)
            ]
            hgb_cells.append(
                {
                    "environment": environment,
                    "sensor": sensor,
                    "strict_null_cell_pass": bool(
                        float(real["simultaneous_lcb"]) > sesoi
                        and summary["all_point_contrasts_positive"].astype(bool).all()
                        and summary[
                            "all_simultaneous_lcb_gt_zero"
                        ].astype(bool).all()
                    ),
                    "real_practical_negative": bool(
                        float(real["simultaneous_ucb"]) < -sesoi
                    ),
                }
            )
    hgb_cell_frame = pd.DataFrame(hgb_cells)
    sensor_routes: dict[str, Any] = {}
    for sensor in SENSORS:
        subset = hgb_cell_frame[hgb_cell_frame["sensor"].eq(sensor)]
        pass_count = int(subset["strict_null_cell_pass"].sum())
        negative_count = int(subset["real_practical_negative"].sum())
        sensor_routes[sensor] = {
            "cell_pass_count": pass_count,
            "practical_negative_count": negative_count,
            "cross_ood_pass": bool(pass_count >= 3 and negative_count == 0),
        }

    if all(item["strict_cell_pass"] for item in cell_results):
        route = "PASS_ROBUST_SEMANTICITY"
        reason = (
            "Both frozen THEORY_VALIDATION-positive ICE low-sensor cells strictly exceed all "
            "20 noise and 20 permutation controls and remain positive under all "
            "three frozen model families."
        )
    elif all(item["descriptive_null_pass"] for item in cell_results):
        route = "PARTIAL_ROBUSTNESS"
        reason = (
            "Both frozen positive cells exceed each null distribution at the "
            "pre-frozen probability threshold, but strict simultaneous or "
            "cross-model conditions are not all satisfied."
        )
    else:
        route = "FAIL_ROBUSTNESS"
        reason = (
            "At least one frozen THEORY_VALIDATION-positive cell does not exceed both null "
            "distributions at the pre-frozen probability threshold."
        )

    any_cross_ood = any(value["cross_ood_pass"] for value in sensor_routes.values())
    if route == "PASS_ROBUST_SEMANTICITY":
        disposition = (
            "UPGRADE_CELL_SPECIFICITY_AND_CROSS_OOD"
            if any_cross_ood
            else "UPGRADE_CELL_SPECIFICITY_ONLY__OVERALL_CONTEXT_DEPENDENT_UNCHANGED"
        )
    elif route == "PARTIAL_ROBUSTNESS":
        disposition = "MAINTAIN_THEORY_VALIDATION_CONTEXT_DEPENDENT_ONLY"
    else:
        disposition = "DOWNGRADE_THEORY_VALIDATION_POSITIVE_CELL_SPECIFICITY"
    return {
        "route": route,
        "reason": reason,
        "theory_validation_disposition": disposition,
        "reference_cell_results": cell_results,
        "hgb_cross_ood_sensor_routes": sensor_routes,
        "primary_target": "ICE_operational_fuel_L",
        "secondary_target_supporting_only": "HEV_operational_fuel_L",
        "causal_interpretation": False,
        "independent_confirmation": False,
    }
