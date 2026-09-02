"""anatid Phase 0 spike benchmark harness.

Modules:
  gen_dataset   deterministic synthetic dataset -> Parquet (numpy seed 42)
  common        shared harness: dataset paths, workload schedule, timing, metrics, result JSON,
                pure-numpy reference implementations of R1 and R2
  report        merges results/*.<scale>.json -> results/REPORT.md with the kill-criterion verdict
  run_<engine>  one runner per engine (owned by other agents)

Runners are executed as scripts from the spike directory, e.g.
  .venv/bin/python bench/run_duckdb_sql.py --scale small
and import the harness with `import common` (bench/ is on sys.path when run as a script)
or `from bench import common` (when the spike directory is on sys.path).
"""
