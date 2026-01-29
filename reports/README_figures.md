# Figures Overview

This guide lists the publication-ready figures generated from CLAS results and how to reproduce them.

## Generate Figures

```
python3 scripts/plot_clas_results.py
```

Outputs will be written to `reports/figures/` as both PNG and SVG.

## Figure Set
- `overall_delta.svg` — Bar chart of CLAS overall deltas across judged metrics.
- `per_model_deltas.svg` — Grouped bar chart of deltas per model.
- `baseline_vs_steered_<metric>.svg` — Baseline vs. steered means per metric and model.
- `overall_delta_radar.svg` — Radar (polygon) plot of overall CLAS delta across metrics.
- `per_model_delta_radar_<model>.svg` — Radar (polygon) plot per model.

## Notes
- The radar (polygon) plots summarize multi-dimensional shifts: each axis is a judged metric (Assertiveness, Complexity, Emotional Distance, Instructional Density). The radius encodes delta magnitude; filled area provides a quick gestalt of aggregate improvement.
- Use SVG in the paper for crisp rendering and easy styling.