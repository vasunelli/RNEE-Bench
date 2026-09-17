# Reported results

`primary_effects.csv` contains all ten setting-by-telemetry rows of Table 3, including unseen vehicles. `estimator_selection.csv` and `selection_by_fold.csv` contain the separate independent-selection sensitivity. `predictive_overlap.csv` contains the kinematic/RPM contrasts and their intervals.

`accounting_table.csv`, `accounting_scales.csv` and `accounting_attenuation.csv` contain Table 4 and the absolute-/squared-error accounting results. These percentage-point gains use observed fuel volume as the denominator; they are distinct from Table 3 relative MAE reductions.

Other CSVs provide road-pairing controls, model sensitivity, distribution/RPM diagnostics, sample counts and map-matching summaries. The four non-vehicle settings in those controls and diagnostics are intentional; they do not replace the five-setting primary result table. Display labels follow the current paper; numerical values are unchanged.

Figure 4 uses the `Base` rows of the control tables. Additional stored telemetry rows remain available with their original comparison families; they are not extra primary results in Table 3.
