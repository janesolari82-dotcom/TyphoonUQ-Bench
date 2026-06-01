# JMA Best Track Notes

The upload bundle treats JMA best track text as a fixed-format raw input, not as a configurable column-mapped CSV source.

Its role in the runtime package is:

- primary storm-level anchor between Digital Typhoon and IBTrACS
- source of native JMA identifiers and time-aligned native pressure / wind values
- transparent quality-control layer alongside the IBTrACS multi-agency labels

Accordingly, parsing is handled by a dedicated adapter rather than by `configs/source_columns.yaml`.

In the cleaned runtime package, `jma_storm_id` refers to the cross-year unique JMA storm key derived from the
best-track header. When the source file provides a separate season-local storm number, it is preserved as
`jma_tc_id` for transparency and traceability.
