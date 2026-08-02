# hpc-support-mcp

A read-only [MCP](https://modelcontextprotocol.io) server exposing **SLURM**, **GPFS**, **Prometheus** (node exporter + **DCGM GPU** metrics) and generic **Elasticsearch** exploration as diagnostic tools for LLM-based HPC support assistants.

Extracted from a production support assistant running on a national-scale HPC cluster, where it backs the daily work of a support team: job diagnostics, quota questions, "why is my job pending", morning health checks. 41 tools, battle-tested against real user tickets. Everything is built on standard components — SLURM client commands, GPFS `mm*` commands, node exporter / dcgm-exporter metrics, plain Elasticsearch APIs — with nothing site-specific.

This is a **support desk** toolkit, not an agentic job-submission framework: every tool reads, none of them acts.

## Design principles

**Read-only by construction.** Only read commands, validated arguments, no shell interpolation. File reads (job logs, scripts) are confined to configured roots via `realpath`. Nothing in this server can modify cluster state — no `scancel`, no writes.

**Anti-false-empty hardening.** The most dangerous failure mode of an LLM tool is not an error — it's an empty result read as truth. A transient NSS/SSSD hiccup makes `squeue -u someone` return nothing, and the model concludes "you have no jobs". This server guards against that class of bug everywhere:

- logins are resolved (`getent passwd` with retry, warming the NSS cache) *before* any per-user SLURM query — an unresolvable login returns an explicit **error**, never an empty list;
- SLURM calls are wrapped with retries on transient failure signatures (`Unable to contact slurm controller`, socket timeouts);
- exact totals are computed server-side (`track_total_hits`, header totals) so the model never sums truncated display rows;
- an Elasticsearch 404 is reported as "index missing or inaccessible", never as "no data", and empty search results point to `es_fields` (a wrong field name being the most common cause of a false empty);
- "not found in metrics" is never presented as "down" — `node_health` cross-checks the Prometheus exporter against `sinfo`, and `gpu_status` states explicitly that a missing DCGM series does not prove the host has no GPUs.

**Deterministic chaining server-side.** Critical sequences (job script → referenced-file existence checks → parallelism analysis → verdicts; overview → failure clustering → representative log) are implemented in code, not left to the model's tool-looping goodwill. Composite tools (`user_overview`, `account_overview`, `diagnose_job`) run the whole investigation in one call with bounded output.

**Interactive-job awareness.** Jobs on interactive/visualization partitions are idle *by design*. Efficiency verdicts and `scancel` suggestions are suppressed for them (`HPC_MCP_INTERACTIVE_PARTITIONS`), so the assistant never tells a user to kill their remote desktop session over low CPU usage.

## Tool groups

Each group can be enabled independently depending on where you deploy (e.g. GPFS `mm*` commands need root on a node that sees the filesystem).

| Group | Env switch | Tools | Requires |
|---|---|---|---|
| `slurm` | `HPC_MCP_ENABLE_SLURM` | 20 — `squeue_jobs`, `sacct_history`, `why_pending`, `fairshare`, `job_priority`, `qos_info`, `job_logs`, `job_script`, `job_efficiency`, `diagnose_job`, `stuck_jobs`, `memory_misuse_scan`, `gpu_usage_by_user`, `latency_probes`, `user_overview`, `account_overview`, … | SLURM client commands, read access to log roots |
| `gpfs` | `HPC_MCP_ENABLE_GPFS` | 11 — `gpfs_filesets_list`, `gpfs_fileset_quota`, `gpfs_all_quotas`, `gpfs_health`, `filesystem_usage`, `list_dir`, `grep_file`, `tail_file`, `stat_file`, … | GPFS `mm*` commands (root) |
| `es` | `HPC_MCP_ENABLE_ES` | 5 — `es_indices`, `es_fields`, `es_search`, `es_aggregate`, `es_tail_logs` | Any Elasticsearch cluster (generic log/index exploration, optional index allowlist) |
| `prom` | `HPC_MCP_ENABLE_PROM` | 5 — `prometheus_query`, `prometheus_range`, `node_health`, `top_loaded_nodes`, `gpu_status` | Prometheus + node exporter; `gpu_status` needs [dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) (standard `DCGM_FI_DEV_*` metrics) |

## Where to run it

This server shells out to `squeue`, `sacct`, `sinfo` and `mm*`. **It must run on a machine that has those clients and the shared filesystem mounted** — a login node, an admin node, or a service node of the cluster. It is not something you install on a laptop and point at a cluster over the network.

Two deployment shapes:

| | Service (HTTP) | Local (stdio) |
|---|---|---|
| Runs on | admin/service node, as a daemon | a login node, on demand |
| Transport | streamable HTTP on `/mcp` | stdio |
| Clients | OpenWebUI, agents, anything HTTP-capable | Claude Desktop and other local MCP clients |
| GPFS group | usable (needs root) | usually disabled |

## Install

### From a wheel (recommended for a service)

Build once where you have network access, then ship the wheel to the cluster:

```bash
python3 -m pip install build
python3 -m build --wheel          # produces dist/hpc_support_mcp-0.1.0-py3-none-any.whl
```

On the target node, install into a dedicated virtualenv **on local disk**:

```bash
python3 -m venv /opt/hpc-support-mcp
/opt/hpc-support-mcp/bin/pip install ./hpc_support_mcp-0.1.0-py3-none-any.whl
/opt/hpc-support-mcp/bin/hpc-support-mcp          # starts on :8765, Ctrl-C to stop
```

> **Keep the venv off the shared filesystem.** If `/opt` is local disk while home directories live on GPFS, this matters: a server whose dependencies sit on GPFS cannot start when GPFS is precisely what you need to diagnose. For the same reason, use the system Python rather than one loaded from a module on the shared tree.

If your site runs an internal package mirror, install from it instead of shipping a file around:

```bash
/opt/hpc-support-mcp/bin/pip install --index-url https://<your-mirror>/simple hpc-support-mcp
```

### From source (development)

```bash
git clone https://github.com/chavaga/hpc-support-mcp
cd hpc-support-mcp
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest        # 13 smoke tests, no cluster access needed
```

### On nodes without internet access

That is the normal case on a cluster, and it is why the wheel path above is the primary one: `pip install` straight from PyPI on a service node generally will not work (no egress, or an authenticating proxy). Build elsewhere and carry the artifact in, or use your mirror.

## Run

### As a service (streamable HTTP)

```ini
# /etc/systemd/system/hpc-support-mcp.service
[Unit]
Description=hpc-support-mcp server
After=network-online.target

[Service]
Type=simple
ExecStart=/opt/hpc-support-mcp/bin/hpc-support-mcp
EnvironmentFile=/etc/hpc-support-mcp.env
Restart=on-failure
RestartSec=5
# The gpfs group needs root for mm* commands. If you disable it
# (HPC_MCP_ENABLE_GPFS=0), run unprivileged instead:
#User=hpcmcp
#Group=hpcmcp

[Install]
WantedBy=multi-user.target
```

```bash
cp config.example.env /etc/hpc-support-mcp.env   # then edit it
chmod 600 /etc/hpc-support-mcp.env               # it holds the bearer token
systemctl enable --now hpc-support-mcp
```

Point your client at `http://<host>:8765/mcp`. Set `HPC_MCP_AUTH_TOKEN` to require `Authorization: Bearer <token>` (constant-time comparison, pure-ASGI middleware — safe with SSE streaming).

### Locally (stdio)

On a login node, for Claude Desktop and other local MCP clients:

```json
{
  "mcpServers": {
    "hpc": {
      "command": "/opt/hpc-support-mcp/bin/hpc-support-mcp",
      "env": {
        "HPC_MCP_TRANSPORT": "stdio",
        "HPC_MCP_ENABLE_GPFS": "0"
      }
    }
  }
}
```

Use the absolute path to the venv binary: a bare `hpc-support-mcp` relies on a `PATH` the client may not share.

## Configuration

Everything is environment variables — see `config.example.env` for a commented template.

| Variable | Default | Purpose |
|---|---|---|
| `HPC_MCP_TRANSPORT` | `http` | `http` (streamable) or `stdio` |
| `HPC_MCP_HOST` / `HPC_MCP_PORT` | `0.0.0.0` / `8765` | HTTP bind |
| `HPC_MCP_AUTH_TOKEN` | *(empty = auth off)* | Bearer token on `/mcp` |
| `HPC_MCP_ENABLE_GPFS/ES/SLURM/PROM` | `1` | Enable tool groups |
| `HPC_MCP_READ_ROOTS` | `/work` | Comma-separated roots allowed for file reads |
| `HPC_MCP_GPFS_DEVICE` | `gpfs` | Device for `mm*` commands |
| `HPC_MCP_ES_URL` | `http://localhost:9200` | Elasticsearch |
| `HPC_MCP_ES_USER` / `HPC_MCP_ES_PASS` | *(none)* | Optional ES basic auth |
| `HPC_MCP_ES_VERIFY` | `1` | Set `0` to skip TLS verification for ES |
| `HPC_MCP_ES_ALLOWED_INDICES` | *(empty = all)* | CSV of fnmatch patterns limiting which indices the tools may touch (e.g. `logs-*,slurm*`) |
| `HPC_MCP_PROM_URL` | `http://localhost:9090` | Prometheus |
| `HPC_MCP_PROM_USER`/`PASS` or `HPC_MCP_PROM_AUTH_FILE` | `~/.config/prometheus_pass` | Basic auth (`user:password`) |
| `HPC_MCP_PROM_VERIFY` | `1` | Set `0` to skip TLS verification (self-signed) |
| `HPC_MCP_INTERACTIVE_PARTITIONS` | `visu` | Partitions where idle jobs are normal |
| `HPC_MCP_MEM_RATIOS` | *(from `sinfo`)* | Override GB-RAM/CPU ratios, e.g. `cpu:7.8,gpu:4.7` |
| `HPC_MCP_PROBE_LS_PATH` / `HPC_MCP_PROBE_DNS_HOST` | first read root / *(off)* | `latency_probes` targets |
| `HPC_MCP_CMD_TIMEOUT` | `30` | Command/HTTP timeout (s) |

The `HPC_MCP_` prefix is kept for every variable regardless of the distribution name: it is short and already deployed in the wild.

## Elasticsearch tools

The `es` group is deliberately generic: it works against **any** Elasticsearch cluster and index layout. `es_indices` discovers what exists, `es_fields` dumps an index's mapping (the reflex when a query comes back empty), `es_search` runs Lucene query strings, `es_aggregate` breaks results down by field with an exact server-side total, and `es_tail_logs` tails any time-based index (configurable timestamp field). `HPC_MCP_ES_ALLOWED_INDICES` restricts the reachable indices when the ES cluster also holds data the assistant should not see.

## Security notes

- Expose the HTTP transport **on internal networks only**, behind the bearer token. The server terminates no TLS: put it behind a reverse proxy if you need HTTPS.
- Argument validation is allowlist-based (`^[A-Za-z0-9_.-]+$` for names, restricted charsets for regex patterns and index names); nothing is passed through a shell.
- The GPFS group requires root: run it only on an admin node, and disable it (`HPC_MCP_ENABLE_GPFS=0`) everywhere else.
- File reads are confined to `HPC_MCP_READ_ROOTS` after `realpath` resolution, so symlinks cannot escape the allowed tree.
- TLS verification is ON by default for both Prometheus and Elasticsearch; opt out explicitly for self-signed internal CAs.

## Compatibility

Requires Python 3.10+ and the MCP Python SDK **1.x** (`mcp[cli]>=1.9,<2`). SDK 2.0 renamed `mcp.server.fastmcp.FastMCP` and changed its API; the pin is deliberate, and lifting it means porting the server.

## License

Apache-2.0.
