#!/usr/bin/env python3
"""Verify the compact reported-results package using only the standard library."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []


def read_csv(rel: str) -> list[dict[str, str]]:
    path = ROOT / rel
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except Exception as exc:
        fail(rel, "parse", "valid CSV", repr(exc), "repository data contract")
        return []


def read_json(rel: str) -> dict:
    path = ROOT / rel
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(rel, "parse", "valid JSON", repr(exc), "repository data contract")
        return {}


def fail(public_file: str, field: str, expected, observed, source: str) -> None:
    FAILURES.append(
        f"{public_file} | {field} | expected={expected!r} | observed={observed!r} | source={source}"
    )


def close(a, b, tol: float = 1e-10) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return False


def check_equal(rel: str, field: str, expected, observed, source: str) -> None:
    if str(expected) != str(observed):
        fail(rel, field, expected, observed, source)


def check_close(rel: str, field: str, expected, observed, source: str, tol: float = 1e-10) -> None:
    if not close(expected, observed, tol):
        fail(rel, field, expected, observed, source)


def key(rows, fields):
    return {tuple(row[f] for f in fields): row for row in rows}


def check_population_and_primary() -> None:
    print("[1/8] Study population and primary effects")
    pop = read_csv("results/study_population.csv")
    expected_pop = {
        "segments": 112787, "trips": 14647, "vehicles": 197, "maf_only_segments": 112766,
        "direct_fuel_rate_only_segments": 21, "mixed_source_segments": 0,
        "maf_effective_duration_share_pct": 99.9814,
    }
    if not pop:
        return
    cohort = next((r for r in pop if r["record_type"] == "primary_analysis_cohort"), {})
    for field, expected in expected_pop.items():
        check_close("results/study_population.csv", field, expected, cohort.get(field),
                    "Manuscript Sections 2.1-2.2 and Appendix Table A.1")

    rows = read_csv("results/primary_effects.csv")
    if len(rows) != 8:
        fail("results/primary_effects.csv", "row_count", 8, len(rows), "Manuscript Table 3")
    expected = [
        ("Unseen trip", "Lower-information", 7.288906554249152, 5.723739850423097, 8.854073258075209),
        ("Unseen trip", "RPM-augmented", 3.542514840471295, 1.8059522845603346, 5.279077396382255),
        ("October 2018", "Lower-information", 8.963532549623748, 6.526532791310527, 11.40053230793697),
        ("October 2018", "RPM-augmented", 4.487822063153663, 1.6018326077954333, 7.373811518511893),
        ("Single held-out area", "Lower-information", -0.7532842828819598, -4.149247357386863, 2.642678791622944),
        ("Single held-out area", "RPM-augmented", -5.482196295087818, -12.777392942472678, 1.81300035229704),
        ("Motorway-largest-share holdout", "Lower-information", -5.45200174856885, -7.199866789724179, -3.7041367074135216),
        ("Motorway-largest-share holdout", "RPM-augmented", -7.699100677730182, -10.372867445837821, -5.02533390962253),
    ]
    observed = key(rows, ["population", "telemetry"])
    for population, telemetry, effect, lower, upper in expected:
        row = observed.get((population, telemetry))
        label = f"{population}/{telemetry}"
        if row is None:
            fail("results/primary_effects.csv", label, "row present", "missing", "Manuscript Table 3")
            continue
        for field, value in [("effect_pct", effect), ("simultaneous_ci_lower_pct", lower),
                             ("simultaneous_ci_upper_pct", upper)]:
            check_close("results/primary_effects.csv", f"{label}.{field}", value, row[field], "Manuscript Table 3")
        calculated = 100 * (float(row["road_free_mae_L_per_nominal_60s_segment"]) -
                            float(row["road_aware_mae_L_per_nominal_60s_segment"])) / float(
                                row["road_free_mae_L_per_nominal_60s_segment"])
        check_close("results/primary_effects.csv", f"{label}.effect_from_MAEs", row["effect_pct"], calculated,
                    "Manuscript Equation 4", tol=5e-10)


def check_controls() -> None:
    print("[2/8] Correspondence controls")
    draws = read_csv("results/correspondence_control_draws.csv")
    summaries = read_csv("results/correspondence_controls.csv")
    if len(draws) != 320:
        fail("results/correspondence_control_draws.csv", "row_count", 320, len(draws), "20 replicates per control and primary comparison")
    if len(summaries) != 16:
        fail("results/correspondence_controls.csv", "row_count", 16, len(summaries), "Manuscript Appendix Table A.4")
    groups = defaultdict(list)
    for row in draws:
        try:
            groups[(row["population"], row["telemetry"], row["control"])].append(float(row["effect_pct"]))
        except (KeyError, ValueError) as exc:
            fail("results/correspondence_control_draws.csv", "numeric row", "valid draw", repr(exc), "saved control contrasts")
    primary = key(read_csv("results/primary_effects.csv"), ["population", "telemetry"])
    for row in summaries:
        group_key = (row["population"], row["telemetry"], row["control"])
        values = groups.get(group_key, [])
        label = "/".join(group_key)
        check_equal("results/correspondence_controls.csv", f"{label}.replicates", 20, len(values), "control design")
        if values:
            for field, value in [("control_mean_pct", sum(values) / len(values)),
                                 ("control_min_pct", min(values)), ("control_max_pct", max(values))]:
                check_close("results/correspondence_controls.csv", f"{label}.{field}", value, row[field],
                            "Manuscript Appendix Table A.4", tol=2e-10)
        assigned = primary.get((row["population"], row["telemetry"]), {}).get("effect_pct")
        check_close("results/correspondence_controls.csv", f"{label}.assigned_road_effect_pct", assigned,
                    row["assigned_road_effect_pct"], "Manuscript Table 3 and Appendix Table A.4")
    draw_numbers = defaultdict(set)
    for row in draws:
        draw_numbers[(row["population"], row["telemetry"], row["control"])].add(int(row["draw"]))
    for group_key, values in draw_numbers.items():
        if values != set(range(1, 21)):
            fail("results/correspondence_control_draws.csv", "/".join(group_key) + ".draws", "1..20", sorted(values), "control design")


def check_diagnostics() -> None:
    print("[3/8] Model and transfer diagnostics")
    models = read_csv("results/model_sensitivity.csv")
    if len(models) != 24:
        fail("results/model_sensitivity.csv", "row_count", 24, len(models), "Manuscript Appendix Table A.5")
    expected_models = {(p, t, m) for p in ["Unseen trip", "October 2018", "Single held-out area", "Motorway-largest-share holdout"]
                       for t in ["Lower-information", "RPM-augmented"] for m in ["HGB", "Ridge", "CatBoost"]}
    observed_models = set(key(models, ["population", "telemetry", "model"]))
    if observed_models != expected_models:
        fail("results/model_sensitivity.csv", "comparison_keys", sorted(expected_models), sorted(observed_models), "Manuscript Appendix Table A.5")

    rpm = read_csv("results/rpm_diagnostic.csv")
    rpm_expected = [
        ("Unseen trip", 5.44445657803129, 3.2470168596637, 7.64189629639887),
        ("October 2018", 6.32940166529892, 1.33428792630289, 11.32451540429488),
        ("Single held-out area", -3.27093642271334, -7.10245231454297, 0.56057946911624),
        ("Motorway-largest-share holdout", -5.29598694942755, -8.380530554996781, -2.2114433438582304),
    ]
    rpm_by_pop = key(rpm, ["population"])
    for pop, effect, lower, upper in rpm_expected:
        row = rpm_by_pop.get((pop,), {})
        for field, value in [("effect_pct", effect), ("simultaneous_ci_lower_pct", lower),
                             ("simultaneous_ci_upper_pct", upper)]:
            check_close("results/rpm_diagnostic.csv", f"{pop}.{field}", value, row.get(field),
                        "Manuscript Appendix Table A.7 and Figure 6a")

    overlap = read_csv("results/road_attribute_overlap.csv")
    overlap_expected = [
        ("Unseen trip", 122, 0.0498388581952117, 0.5676813608570299, 0.50288104),
        ("October 2018", 121, 0.0538338658146964, 0.5697880811265326, 0.53103516),
        ("Single held-out area", 120, 0.7587677725118483, 1.2438649114584566, 0.99943056),
        ("Motorway-largest-share holdout", 122, 1.0, 15.08099240770186, 0.9999996),
    ]
    overlap_by_pop = key(overlap, ["population"])
    for pop, n_features, exceed, median, auc in overlap_expected:
        row = overlap_by_pop.get((pop,), {})
        for field, value in [("retained_road_attributes", n_features),
                             ("evaluation_fraction_beyond_training_q95", exceed),
                             ("normalized_distance_q50", median), ("domain_classifier_roc_auc", auc)]:
            check_close("results/road_attribute_overlap.csv", f"{pop}.{field}", value, row.get(field),
                        "Manuscript Appendix Table A.8 and Figure 6b")


def check_map_and_config() -> None:
    print("[4/8] Construction, map matching, and inference contracts")
    map_rows = read_csv("results/map_matching_summary.csv")
    observed = {r["metric"]: r["value"] for r in map_rows}
    expected = {
        "source_dynamic_records": 22436808, "matched_points": 22388636,
        "matched_point_fraction": 0.9978529922794722, "edge_associated_points": 22217578,
        "match_distance_q50": 6.054766, "match_distance_q95": 27.891229,
        "match_distance_q99": 53.412187, "abstained_trips": 40, "abstained_rows": 40999,
        "search_radius": 90, "trace_break": 1000,
    }
    for field, value in expected.items():
        check_close("results/map_matching_summary.csv", field, value, observed.get(field),
                    "Manuscript Section 2.3 and Appendix Table A.1")
    cfg = read_json("config/main_analysis.json")
    family_expected = {"primary_HGB": 24, "control_contrasts": 16, "model_sensitivity": 24,
                       "RPM_diagnostic": 7, "supporting_engine_state_and_Ridge": 15}
    family_observed = cfg.get("inference", {}).get("comparison_family_sizes", {})
    if family_observed != family_expected:
        fail("config/main_analysis.json", "inference.comparison_family_sizes", family_expected, family_observed,
             "Manuscript Appendix Table A.3")
    check_equal("config/main_analysis.json", "inference.bootstrap_repetitions", 2000,
                cfg.get("inference", {}).get("bootstrap_repetitions"), "Manuscript Appendix Table A.3")
    features = read_json("config/feature_sets.json")
    check_equal("config/feature_sets.json", "road_attribute_candidates", 142, features.get("road_attribute_candidates"),
                "Manuscript Section 2.3")
    if "Excluded from the primary sensing comparison" not in features.get("absolute_load_status", ""):
        fail("config/feature_sets.json", "absolute_load_status", "excluded from primary sensing comparison",
             features.get("absolute_load_status"), "Manuscript sensing-regime definition")


def check_metadata() -> None:
    print("[5/8] Feature metadata")
    dictionary = read_csv("metadata/variable_dictionary.csv")
    if len(dictionary) != 142 or len({r["feature_name"] for r in dictionary}) != 142:
        fail("metadata/variable_dictionary.csv", "unique_feature_count", 142,
             len({r.get("feature_name") for r in dictionary}), "Manuscript Section 2.3")
    retained = read_csv("metadata/retained_road_attributes.csv")
    counts = defaultdict(int)
    for row in retained:
        if row["retained_from_training_only"] == "true":
            counts[row["population"]] += 1
    expected = {"Unseen trip": 122, "October 2018": 121, "Single held-out area": 120,
                "Motorway-largest-share holdout": 122}
    if dict(counts) != expected:
        fail("metadata/retained_road_attributes.csv", "training_retained_counts", expected, dict(counts),
             "Manuscript Section 3.3")


def normalized(rows: list[dict[str, str]], fields: list[str]) -> list[tuple]:
    return sorted(tuple(row.get(field, "") for field in fields) for row in rows)


def check_figure_crosswalk() -> None:
    print("[6/8] Figure-data crosswalks")
    pairs = [
        ("results/primary_effects.csv", "figure_data/figure_3.csv"),
        ("results/model_sensitivity.csv", "figure_data/figure_5.csv"),
    ]
    for left, right in pairs:
        a, b = read_csv(left), read_csv(right)
        fields = list(a[0]) if a else []
        if normalized(a, fields) != normalized(b, fields):
            fail(right, "rows", f"exact crosswalk to {left}", "different", "publication figure source")
    draws = read_csv("results/correspondence_control_draws.csv")
    fig4 = read_csv("figure_data/figure_4.csv")
    fields = list(draws[0]) if draws else []
    if normalized(draws, fields) != normalized(fig4, fields):
        fail("figure_data/figure_4.csv", "control_draws", "exact crosswalk to results", "different", "Figure 4 source")
    plotted = [r for r in fig4 if r.get("included_in_publication_figure") == "true"]
    if len(plotted) != 160 or {r["telemetry"] for r in plotted} != {"Lower-information"}:
        fail("figure_data/figure_4.csv", "publication_subset", "160 lower-information draws", len(plotted), "Figure 4")
    fig6 = read_csv("figure_data/figure_6.csv")
    if len([r for r in fig6 if r["panel"] == "a"]) != 4 or len([r for r in fig6 if r["panel"] == "b"]) != 4:
        fail("figure_data/figure_6.csv", "panel_rows", "4 rows in each panel", len(fig6), "Figure 6 source")


def check_pdfs() -> None:
    print("[7/8] Publication figure files")
    for n in range(3, 7):
        rel = f"figures/figure_{n}.pdf"
        path = ROOT / rel
        if not path.is_file():
            fail(rel, "file", "present", "missing", f"Publication Figure {n}")
            continue
        data = path.read_bytes()
        if not data.startswith(b"%PDF-") or len(data) < 5000:
            fail(rel, "PDF_signature_and_size", "valid nontrivial PDF", len(data), f"Publication Figure {n}")


def check_portability() -> None:
    print("[8/8] Portability and public-surface scan")
    text_suffixes = {".md", ".txt", ".csv", ".json", ".py"}
    prohibited_terms = [
        "chat" + "gpt", "open" + "ai", "co" + "dex", "cla" + "ude", "gem" + "ini",
        "ai" + "-generated", "review" + "er agent", "phase" + "1", "author input" + " required",
        "zero-context" + " full review", "scientific" + " adjudication",
    ]
    absolute_patterns = [re.compile(r"/mnt/[a-z]/", re.I), re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")]
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        lower = text.lower()
        for term in prohibited_terms:
            if term in lower:
                fail(rel, "public_surface_term", "absent", term, "public release privacy policy")
        for pattern in absolute_patterns:
            match = pattern.search(text)
            if match:
                fail(rel, "absolute_local_path", "absent", match.group(0), "portable repository requirement")
    required = ["README.md", "LICENSE", "LICENSE-CONTENT.md", "requirements.txt",
                "scripts/verify_reported_results.py",
                "scripts/reproduce_tables_and_figures.py", "scripts/run_full_analysis.py"]
    for rel in required:
        if not (ROOT / rel).is_file():
            fail(rel, "required_file", "present", "missing", "repository contract")
    for intentionally_omitted in ["CITATION.cff"]:
        if (ROOT / intentionally_omitted).exists():
            fail(intentionally_omitted, "author_approval", "omitted until approved", "present", "release authorization boundary")


def main() -> int:
    check_population_and_primary()
    check_controls()
    check_diagnostics()
    check_map_and_config()
    check_metadata()
    check_figure_crosswalk()
    check_pdfs()
    check_portability()
    if FAILURES:
        print(f"\nFAILED: {len(FAILURES)} mismatch(es)", file=sys.stderr)
        for item in FAILURES:
            print(f"- {item}", file=sys.stderr)
        return 1
    print("\nPASS: All reported values match the frozen manuscript-facing contracts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
