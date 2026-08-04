# Data access

## Vehicle data

The original vehicle inputs are the official VED Dynamic Data and Static Data at commit `6baa4963782d515a67d32a5490bd5d11f5d9bf0d` of the public [VED repository](https://github.com/gsoh/VED). Verify the commit and archive hashes before construction. This package does not redistribute those files.

## Historical road data

Construction used Michigan OpenStreetMap snapshots dated 1 January 2017 and 1 January 2018. Trips recorded in a calendar year used the latest of those snapshots dated on or before travel; future-map fallback was prohibited. Raw OSM files are not redistributed. Obtain historical extracts from a source that can document content and hashes, and retain the Open Database License attribution.

## Expected local layout for full reconstruction

```text
DATA_ROOT/
  ved/
    SOURCE_COMMIT.txt
    raw/
      dynamic/       # official VED dynamic files
      static/        # official VED static files
  osm/
    michigan-2017-01-01.osm.pbf
    michigan-2018-01-01.osm.pbf
```

`SOURCE_COMMIT.txt` must contain the 40-character VED commit above. Archive and extract without modifying source files. The compact verifier does not require these inputs.
