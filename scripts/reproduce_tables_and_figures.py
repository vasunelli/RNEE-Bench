#!/usr/bin/env python3
"""Recreate reported tables and structural SVG figures from public CSV data."""

from __future__ import annotations

import csv
import html
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated"
POPS = ["Unseen trip", "October 2018", "Single held-out area", "Motorway-largest-share holdout"]


def read_csv(rel: str) -> list[dict[str, str]]:
    with (ROOT / rel).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0])
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row.get(f, "")).replace("|", "\\|") for f in fields) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table(slug: str, rows: list[dict[str, str]]) -> None:
    write_csv(GENERATED / "tables" / f"{slug}.csv", rows)
    write_md(GENERATED / "tables" / f"{slug}.md", rows)


def svg_header(width=1040, height=640):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#1f2933}.title{font-size:22px;font-weight:700}.label{font-size:14px}.small{font-size:12px}.axis{stroke:#52606d;stroke-width:1}.zero{stroke:#9aa5b1;stroke-dasharray:5 4}.ci{stroke:#334e68;stroke-width:3}.point{fill:#0072b2}.rpm{fill:#d55e00}.neg{fill:#cc3311}</style>']


def xscale(value, lo, hi, left=300, right=980):
    return left + (float(value) - lo) * (right - left) / (hi - lo)


def figure3(rows):
    lines = svg_header(height=620)
    lines.append('<text x="30" y="36" class="title">Figure 3 — primary road-context effects</text>')
    lo, hi = -14, 13
    zero = xscale(0, lo, hi)
    lines += [f'<line x1="300" y1="565" x2="980" y2="565" class="axis"/>',
              f'<line x1="{zero}" y1="60" x2="{zero}" y2="550" class="zero"/>']
    for i, row in enumerate(rows):
        y = 82 + i * 59
        label = f'{row["population"]} — {row["telemetry"]}'
        lines.append(f'<text x="25" y="{y+5}" class="label">{html.escape(label)}</text>')
        x1, x2, xp = [xscale(row[f], lo, hi) for f in ["simultaneous_ci_lower_pct", "simultaneous_ci_upper_pct", "effect_pct"]]
        lines.append(f'<line x1="{x1:.2f}" y1="{y}" x2="{x2:.2f}" y2="{y}" class="ci"/>')
        cls = "rpm" if row["telemetry"] == "RPM-augmented" else ("neg" if float(row["effect_pct"]) < 0 else "point")
        lines.append(f'<circle cx="{xp:.2f}" cy="{y}" r="6" class="{cls}"/>')
        lines.append(f'<text x="{min(x2+8,930):.2f}" y="{y+4}" class="small">{float(row["effect_pct"]):.2f}%</text>')
    for tick in [-10, -5, 0, 5, 10]:
        x = xscale(tick, lo, hi); lines.append(f'<text x="{x-10:.2f}" y="590" class="small">{tick}</text>')
    lines.append('<text x="570" y="615" class="label">Relative MAE reduction (%)</text></svg>')
    (GENERATED / "figures/figure_3.svg").write_text("\n".join(lines), encoding="utf-8")


def figure4(draws, primary):
    draws = [r for r in draws if r["telemetry"] == "Lower-information"]
    assigned = {r["population"]: float(r["effect_pct"]) for r in primary if r["telemetry"] == "Lower-information"}
    lines = svg_header(height=620)
    lines.append('<text x="30" y="36" class="title">Figure 4 — correspondence controls (lower-information)</text>')
    lo, hi = -8, 11; zero = xscale(0, lo, hi)
    lines.append(f'<line x1="{zero}" y1="60" x2="{zero}" y2="545" class="zero"/>')
    grouped = {}
    for row in draws: grouped.setdefault((row["population"], row["control"]), []).append(float(row["effect_pct"]))
    for i, pop in enumerate(POPS):
        y0 = 90 + i * 112
        lines.append(f'<text x="25" y="{y0}" class="label">{html.escape(pop)}</text>')
        for j, control in enumerate(["Matched-noise", "Reassigned-road"]):
            y = y0 + 28 + j*27
            vals = grouped[(pop, control)]
            lines.append(f'<text x="85" y="{y+4}" class="small">{control}</text>')
            for d, value in enumerate(vals):
                lines.append(f'<circle cx="{xscale(value,lo,hi):.2f}" cy="{y + ((d%5)-2)*1.5:.2f}" r="2.4" fill="#7b8794"/>')
        xa = xscale(assigned[pop], lo, hi)
        lines.append(f'<polygon points="{xa},{y0+18} {xa-7},{y0+8} {xa+7},{y0+8}" fill="#0072b2"/>')
    lines.append('<text x="545" y="600" class="label">Relative MAE reduction (%)</text></svg>')
    (GENERATED / "figures/figure_4.svg").write_text("\n".join(lines), encoding="utf-8")


def figure5(rows):
    lines = svg_header(height=650)
    lines.append('<text x="30" y="36" class="title">Figure 5 — model sensitivity</text>')
    models = ["HGB", "Ridge", "CatBoost"]
    row_keys = [(p,t) for p in POPS for t in ["Lower-information","RPM-augmented"]]
    values = {(r["population"],r["telemetry"],r["model"]):float(r["effect_pct"]) for r in rows}
    for j,m in enumerate(models): lines.append(f'<text x="{420+j*170}" y="72" class="label">{m}</text>')
    for i,(p,t) in enumerate(row_keys):
        y=95+i*63; lines.append(f'<text x="20" y="{y+25}" class="small">{html.escape(p)} — {t}</text>')
        for j,m in enumerate(models):
            v=values[(p,t,m)]; intensity=min(abs(v)/40,1); color = f'rgba({204 if v<0 else 0},{51 if v<0 else 114},{17 if v<0 else 178},{0.15+0.75*intensity:.3f})'
            x=380+j*170; lines.append(f'<rect x="{x}" y="{y}" width="145" height="45" fill="{color}" stroke="#d9e2ec"/>')
            lines.append(f'<text x="{x+52}" y="{y+28}" class="label">{v:.2f}%</text>')
    lines.append('</svg>'); (GENERATED / "figures/figure_5.svg").write_text("\n".join(lines),encoding="utf-8")


def figure6(rpm, overlap):
    lines=svg_header(width=1200,height=650)
    lines += ['<text x="30" y="36" class="title">Figure 6 — RPM and road-attribute support diagnostics</text>',
              '<text x="40" y="75" class="label">(a) RPM diagnostic effect</text>',
              '<text x="650" y="75" class="label">(b) Predictor overlap</text>']
    lo,hi=-10,13; zero=xscale(0,lo,hi,left=250,right=580); lines.append(f'<line x1="{zero}" y1="90" x2="{zero}" y2="470" class="zero"/>')
    for i,row in enumerate(rpm):
        y=120+i*85; lines.append(f'<text x="40" y="{y+4}" class="small">{html.escape(row["population"])}</text>')
        x1=xscale(row["simultaneous_ci_lower_pct"],lo,hi,left=250,right=580); x2=xscale(row["simultaneous_ci_upper_pct"],lo,hi,left=250,right=580); xp=xscale(row["effect_pct"],lo,hi,left=250,right=580)
        lines += [f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" class="ci"/>',f'<circle cx="{xp}" cy="{y}" r="6" class="point"/>']
    maxdist=16.0
    for i,row in enumerate(overlap):
        y=120+i*85; label=html.escape(row["population"]); median=float(row["normalized_distance_q50"]); exceed=100*float(row["evaluation_fraction_beyond_training_q95"]); auc=float(row["domain_classifier_roc_auc"])
        lines.append(f'<text x="650" y="{y}" class="small">{label}</text>')
        lines.append(f'<rect x="850" y="{y-14}" width="{300*min(median/maxdist,1):.2f}" height="18" fill="#56b4e9"/>')
        lines.append(f'<text x="850" y="{y+22}" class="small">median={median:.3f}; beyond q95={exceed:.2f}%; AUC={auc:.4f}</text>')
    lines.append('</svg>'); (GENERATED / "figures/figure_6.svg").write_text("\n".join(lines),encoding="utf-8")


def main() -> int:
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/verify_reported_results.py")], cwd=ROOT)
    if completed.returncode:
        return completed.returncode
    (GENERATED / "tables").mkdir(parents=True, exist_ok=True)
    (GENERATED / "figures").mkdir(parents=True, exist_ok=True)
    primary=read_csv("results/primary_effects.csv"); controls=read_csv("results/correspondence_controls.csv")
    models=read_csv("results/model_sensitivity.csv"); rpm=read_csv("results/rpm_diagnostic.csv"); overlap=read_csv("results/road_attribute_overlap.csv")
    table("table_3_primary_effects", primary); table("appendix_table_a4_controls", controls)
    table("appendix_table_a5_model_sensitivity", models); table("appendix_table_a7_rpm_diagnostic", rpm)
    figure3(primary); figure4(read_csv("results/correspondence_control_draws.csv"), primary); figure5(models); figure6(rpm,overlap)
    print(f"Generated 8 table files and 4 SVG figures under {GENERATED.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
