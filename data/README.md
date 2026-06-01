# Data Directory

This repository does not commit raw source data or generated benchmark tables.
Place source files and optional released derived tables under this directory
using the layout documented in `docs/01_DATA.md`.

The expected raw-data layout is:

```text
data/raw/
  digital_typhoon/archive/
  jma/bst_all.txt
  ibtracs/ibtracs.WP.list.v04r01.csv
```

The generated benchmark layout is:

```text
data/processed/
  digital_typhoon/
  benchmark/
```
