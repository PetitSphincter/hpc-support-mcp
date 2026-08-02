# Changelog

## 0.1.0 — 2026-08-02

Initial public release, extracted from a production HPC support-assistant MCP server
and fully genericized: English tool descriptions and outputs, standard components only.

- 41 read-only tools in 4 independently-enabled groups (slurm, gpfs, es, prom)
- Anti-false-empty hardening: login resolution guard, SLURM retry on transient
  errors, exact server-side totals, sinfo cross-check in node_health, explicit
  "index missing ≠ no data" handling in every Elasticsearch tool
- Generic Elasticsearch exploration (es_indices, es_fields, es_search,
  es_aggregate, es_tail_logs) with an optional index allowlist — works against
  any index layout
- GPU visibility through the standard NVIDIA dcgm-exporter (gpu_status:
  cluster overview, per-host detail, XID errors)
- Composite investigations (diagnose_job / user_overview / account_overview)
  with failure-cluster detection and representative-log reading
- Interactive/visualization partition awareness (no scancel suggestions on
  idle-by-design jobs)
- Streamable HTTP (optional Bearer auth, pure-ASGI middleware) and stdio
  transports; standard Python packaging (`pipx install hpc-mcp`), configuration
  via environment variables
