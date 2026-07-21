# INFOCOM figure manifest

All figures are regenerated with:

```bash
venv/bin/python benchmarks/generate_infocom_figures.py
```

The same command exports machine-readable CSV and `booktabs`-compatible
LaTeX tables for the open-loop paired effects and Gate D cross-model effects.
The PNG, CSV, and TeX files are generated artifacts; edit the canonical
JSON or generator rather than changing them by hand.

## open-loop-arrival-qwen1p5b

Sources:

- `data/qwen1.5b_open_loop_arrival_v1/arrival_rate_summary.json`

The three panels show achieved throughput, pooled P95 TTFT, and pooled P95
initial E2E latency against offered load. Pooled tails are descriptive across
all request samples; paired trial-level confidence intervals are reported in
the separate effects figure.

## open-loop-paired-effects

Sources:

- `data/qwen1.5b_open_loop_arrival_v1/arrival_rate_summary.json`
- `data/qwen0.5b_open_loop_arrival_v1/arrival_rate_summary.json`

Points are cuSZp-minus-raw paired means and error bars are paired 95% confidence
intervals across five interleaved trials. Negative handler/E2E differences are
better. Only quality-passing formal runs are included.

## gate-d-cross-model-handler

Sources:

- `data/gate_d_fair_async_probe/gate_d_summary.json`
- each per-model `aggregate.json` referenced by that manifest

Points are fixed-bound cuSZp-minus-raw combined-handler differences and paired
95% confidence intervals. Squares identify per-method quality passes; crosses
identify quality rejects. This figure is boundary evidence and must not be
described as universal speedup.
