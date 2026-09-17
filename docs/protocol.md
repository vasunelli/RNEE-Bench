# Analysis protocol

## Target and predictors

The target is OBD-derived segment fuel volume. Valid Fuel Rate has priority; otherwise MAF with the stated fuel-trim correction is converted using an air-to-fuel mass ratio of 14.08 and fuel density of 745 g/L. Only valid intervals of at most 10 seconds contribute. Segment targets are not rescaled to a nominal duration.

Compact telemetry B contains duration, distance, speed mean/SD, ambient-temperature mean/SD, idle ratio and low-speed ratio. K contains five speed/time-derived kinematic summaries. Z contains RPM mean and SD. R contains 142 predefined historical road-attribute candidates; training-only filtering retains 120–122, including 120–121 across vehicle folds. Exact feature names are in `config/features.json` and the road dictionary. Coordinates, entity/route identifiers and target-source channels are excluded from predictors.

## Evaluation

The five settings and support counts follow Table 2 and `config/evaluation.json`. Unseen trips retain represented vehicles; unseen vehicles use five vehicle-disjoint outer folds; October 2018 is a temporal holdout; the held-out area uses two adjacent 1-km cells and a buffer. Motorway evaluation selects individual eligible segments for which motorway has the largest speed-integrated class share, without requiring a majority. Other segments from those motorway evaluation trips are excluded from development and scoring.

Stored membership files retain their original role codes. The code `calibration` denotes the role called RPM-check in the manuscript; the terminology mapping does not reassign observations. A shared statement that vehicles may recur across roles does not apply to the unseen-vehicle outer evaluation folds.

## Estimation and selection

Primary HGB uses absolute-error loss, 200 iterations, learning rate 0.05, at most 31 leaf nodes, at least 30 observations per leaf, L2 regularisation 1 and no early stopping. Non-vehicle predictions average seeds 0, 1 and 2. Fixed-HGB unseen-vehicle predictions use seed 0. The independent-selection sensitivity performs development-only selection and seed checks within each outer fold; it remains separate from the primary estimator.

The original common-estimator selection pool included units appearing in some later evaluations. The independent unseen-vehicle sensitivity excludes each evaluation vehicle before sampling, filtering, target-IQR calculation, candidate fitting, ranking and seed checks. Four folds select HGB and one LightGBM. The 5.07% primary and 5.09% sensitivity results are separate estimates.

## Intervals and pairing controls

Resampling uses evaluation-vehicle clusters with fixed fits and partitions. Main analyses use 2,000 replicates; speed analyses and diagnostics use 1,000. Non-vehicle primary intervals retain their original simultaneous families; unseen-vehicle primary and independent-selection intervals are separately pointwise.

Compact HGB intervals use the historical 24-contrast family across four non-vehicle settings, two historical input sets and matched/noise/reassigned-road gains. Historical extended-OBD estimates are not current primary results. The separate RPM/model-sensitivity family contains 24 contrasts across four settings, two telemetry sets and three estimators. Compact model sensitivity and replicate-specific matched-minus-control inference retain their own saved families. Figure 4b shows replicate means and full ranges, not confidence intervals for those bars.

Within each core setting, the two richer-input road gains and two sequential attenuation measures share a max-T family. Kinetic-energy and conditional-RPM prediction gains use separate Bonferroni adjustments across core settings. Other reference and boundary comparisons retain their pointwise intervals.

## Fuel-accounting scale

The common-eight cohorts contain 134 trips/78 vehicles for unseen trips, 858/145 for unseen vehicles and 64/41 for October. One-, two-, four- and eight-segment blocks and the full prefix share observations and target volume. Complete trips restore the remaining tails. Eligibility requires all generated segments of each complete trip to be eligible and to have the setting-specific saved prediction.

Table 3 gains are percentages relative to baseline MAE. Table 4 gains are percentage-point differences normalised by observed fuel volume. Residuals satisfy `T = S + CR - C0`; aggregation attenuation and tail restoration have distinct definitions. Six gain endpoints form a simultaneous family within each setting and loss. Attenuation components use pointwise intervals. Tail restoration changes observation support and normalisation as well as cancellation. Spatial common-eight support is insufficient and motorway complete-trip evidence is descriptive.
