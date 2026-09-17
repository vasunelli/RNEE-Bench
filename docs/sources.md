# Source data and code availability

The current manuscript uses the official [Vehicle Energy Dataset](https://github.com/gsoh/VED/tree/6baa4963782d515a67d32a5490bd5d11f5d9bf0d), historical Michigan OpenStreetMap snapshots from 1 January 2017 and 1 January 2018, and Valhalla 3.7.0. Upstream VED contains 22,436,808 records from 384 vehicles and 32,552 trips; the manuscript analysis is the quality-eligible ICE subset described in Table 1.

The [v0.2.0-energy release](https://github.com/vasunelli/RNEE-Bench/releases/tag/v0.2.0-energy) contains current-study predictions, memberships, estimator-selection inputs, kinematic variables, saved resampling data, result tables and the calculation script. These support the current saved-result reproduction commands.

The separately preserved [construction code at v0.1.0-full](https://github.com/vasunelli/RNEE-Bench/tree/v0.1.0-full) and its [data/model release](https://github.com/vasunelli/RNEE-Bench/releases/tag/v0.1.0-full) provide upstream construction resources and earlier model artifacts. Their broader target support, earlier figures and four-setting results are historical resources, not a replacement for the current five-setting paper. The old code, configurations and their dependencies remain together at that fixed version.

For source inspection without mixing the two directory structures:

```bash
git clone -c core.autocrlf=false --branch v0.1.0-full --single-branch https://github.com/vasunelli/RNEE-Bench.git RNEE-construction
```

The historical code is preserved as originally released; it is not a turnkey implementation of every current training task. Current resources reproduce reported calculations from saved predictions. Complete fresh model fitting requires additional full feature inputs and the corresponding model environments. No unavailable model binary or resampling matrix is implied by this archive reference.

VED and OpenStreetMap retain their upstream licence terms. Preserve attribution to the Michigan Vehicle Energy Dataset and © OpenStreetMap contributors. Raw archives, raw OSM snapshots and Valhalla caches are not included in the current scientific data supplement.
