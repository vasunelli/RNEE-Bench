# Data provenance and claim boundaries

For the current manuscript's data, tables and figures, see the [paper resource guide](../energy/README.md). This page describes the earlier construction release.

The vehicle-data lineage begins with official VED Dynamic and Static Data at commit `6baa4963782d515a67d32a5490bd5d11f5d9bf0d`. The inventory comprised 54 dynamic files, 22,436,808 records, 384 vehicles, and 32,552 trips. Static metadata supplied powertrain class; the reported primary analysis is restricted to internal-combustion-engine vehicles.

The operational target gives valid direct Fuel Rate priority and otherwise uses MAF with available fuel-trim correction. It is reported in litres per nominal 60-s segment. The analysis contains 112,766 MAF-only segments and 21 direct-rate-only segments, with no mixed-source segments. Fuel Rate, MAF, STFT, and LTFT are excluded from predictors.

Historical road attributes come from the 2017-01-01 and 2018-01-01 Michigan OSM snapshots. Valhalla 3.7.0 used auto costing, a 90-m search radius, and a 1,000-m trace-break rule. Of 22,436,808 source points, 22,388,636 were matched and 22,217,578 received an edge association. Forty abstained trips cover 40,999 preserved records for which road semantics were withheld.

The package intentionally uses “pipeline-assigned road–trajectory correspondence.” It does not claim verified alignment, physical fuel-flow accuracy, external-fleet transfer, fuel savings, emissions reductions, or causal effects of infrastructure. Historical OSM data are © OpenStreetMap contributors and licensed under ODbL; see `third_party/NOTICE.md`.
