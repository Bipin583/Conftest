# Scoring-Path Latency Benchmark

The committed latency benchmark measures model scoring and policy work on already prepared feature rows. It is not an end-to-end commit, API, webhook, or CI latency measurement.

## Measured path

`scripts/run_latency_benchmark.py` repeatedly scores batches of 50 real held-out rows through:

1. array sanitization;
2. five-member ensemble inference;
3. temperature calibration; and
4. selective policy decision.

The run uses 150 timed iterations after warm-up. The first cold batch took 1321.814 ms; timed statistics exclude warm-up.

## Committed results

| Stage | Mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| Array sanitization | 0.033 ms | 0.036 ms | 0.043 ms | 0.057 ms |
| Ensemble inference | 3.246 ms | 3.523 ms | 4.024 ms | 4.309 ms |
| Calibration | 0.036 ms | 0.042 ms | 0.046 ms | 0.068 ms |
| Policy decision | 0.029 ms | 0.032 ms | 0.038 ms | 0.046 ms |
| Scoring total | 3.345 ms | 3.627 ms | 4.139 ms | 4.435 ms |

All timed scoring batches were below 100 ms, so the report records 100% compliance with that scoring-only threshold. It does not establish a sub-100 ms end-to-end service-level objective.

## Excluded work

The figure excludes repository checkout, Git diff mining, pytest discovery, AST parsing, dependency-graph construction, history/database queries, process and model cold start, HTTP/network overhead, persistence, GitHub API work, and test execution. Feature extraction occurs upstream once per change and can dominate scoring.

Hardware, operating-system scheduling, artifact versions, batch size, and warm/cold state also affect results. The committed provenance identifies a five-member, 150-tree ensemble, temperature calibrator, tuned policy configuration, and `data/splits/test.csv` rows.

## Reproduction

```bash
python scripts/evaluate.py --run latency
```

Use `reports/latency_benchmark.json` for exact percentiles and provenance. To make an end-to-end latency claim, instrument the full path separately under representative deployment load, including cold starts and failure modes. See [Architecture](architecture.md) and [Limitations](limitations.md).
