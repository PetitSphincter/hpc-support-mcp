#!/usr/bin/env python3
"""hpc-mcp — read-only diagnostic MCP server for HPC clusters.

Exposes SLURM, GPFS, Prometheus (node exporter + DCGM GPU metrics) and
generic Elasticsearch exploration as READ-ONLY MCP tools, designed to be
called by an LLM assisting an HPC support team. Streamable HTTP transport
(natively compatible with OpenWebUI) or stdio.

Tool groups can be enabled independently depending on the deployment node:
    HPC_MCP_ENABLE_GPFS=1   native mm* commands (requires root)
    HPC_MCP_ENABLE_ES=1     generic Elasticsearch exploration (logs, indices)
    HPC_MCP_ENABLE_SLURM=1  squeue/sacct/sinfo + job logs/efficiency/diagnosis
    HPC_MCP_ENABLE_PROM=1   PromQL + node exporter + DCGM GPU

Configuration (environment variables):
    HPC_MCP_TRANSPORT      http (default) or stdio
    HPC_MCP_HOST           bind address (default: 0.0.0.0)
    HPC_MCP_PORT           port (default: 8765)
    HPC_MCP_AUTH_TOKEN     if set, require "Authorization: Bearer <token>" on /mcp
    HPC_MCP_GPFS_DEVICE    GPFS device for mm* commands (default: gpfs)
    HPC_MCP_ES_URL         Elasticsearch URL (default: http://localhost:9200)
    HPC_MCP_ES_USER/HPC_MCP_ES_PASS  optional Elasticsearch basic auth
    HPC_MCP_ES_VERIFY      0 to skip TLS verification for ES (default: 1)
    HPC_MCP_ES_ALLOWED_INDICES  CSV of fnmatch patterns; empty = all indices
    HPC_MCP_PROM_URL       Prometheus URL (default: http://localhost:9090)
    HPC_MCP_PROM_AUTH_FILE "user:password" file (default: ~/.config/prometheus_pass)
    HPC_MCP_PROM_USER/HPC_MCP_PROM_PASS  direct credential override
    HPC_MCP_PROM_VERIFY    0 to skip TLS verification for Prometheus (default: 1)
    HPC_MCP_READ_ROOTS     roots allowed for file reads (default: /work)
    HPC_MCP_CMD_TIMEOUT    command/HTTP timeout in seconds (default: 30)
    HPC_MCP_INTERACTIVE_PARTITIONS  interactive/visu partitions, CSV (default: visu)
    HPC_MCP_MEM_RATIOS     GB-RAM/CPU overrides per partition (e.g. cpu:7.8,gpu:4.7)
    HPC_MCP_PROBE_LS_PATH / HPC_MCP_PROBE_DNS_HOST  latency_probes targets

Security: read-only commands only, validated arguments, no shell. File reads
(job logs) are confined to HPC_MCP_READ_ROOTS via realpath. Expose on
internal networks only.
"""

import fnmatch
import json
import os
import re
import shutil
import subprocess
import urllib3
from collections import deque
from datetime import datetime, timedelta

import requests
from mcp.server.fastmcp import FastMCP


PROBE_DNS_HOST = os.environ.get("HPC_MCP_PROBE_DNS_HOST", "")  # empty = DNS probe disabled
PROBE_LS_PATH = os.environ.get("HPC_MCP_PROBE_LS_PATH", "")  # empty = first entry of READ_ROOTS

HOST = os.environ.get("HPC_MCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("HPC_MCP_PORT", "8765"))
GPFS_DEVICE = os.environ.get("HPC_MCP_GPFS_DEVICE", "gpfs")
ES_URL = os.environ.get("HPC_MCP_ES_URL", "http://localhost:9200").rstrip("/")
ES_VERIFY = os.environ.get("HPC_MCP_ES_VERIFY", "1") == "1"
ES_ALLOWED_INDICES = [
    p.strip() for p in os.environ.get("HPC_MCP_ES_ALLOWED_INDICES", "").split(",") if p.strip()
]
PROM_URL = os.environ.get("HPC_MCP_PROM_URL", "http://localhost:9090")
PROM_AUTH_FILE = os.path.expanduser(
    os.environ.get("HPC_MCP_PROM_AUTH_FILE", "~/.config/prometheus_pass")
)
READ_ROOTS = [
    r.rstrip("/") for r in os.environ.get("HPC_MCP_READ_ROOTS", "/work").split(",") if r
]
CMD_TIMEOUT = int(os.environ.get("HPC_MCP_CMD_TIMEOUT", "30"))

ENABLE_GPFS = os.environ.get("HPC_MCP_ENABLE_GPFS", "1") == "1"
ENABLE_ES = os.environ.get("HPC_MCP_ENABLE_ES", "1") == "1"
ENABLE_SLURM = os.environ.get("HPC_MCP_ENABLE_SLURM", "1") == "1"
ENABLE_PROM = os.environ.get("HPC_MCP_ENABLE_PROM", "1") == "1"

MCP_AUTH_TOKEN = os.environ.get("HPC_MCP_AUTH_TOKEN", "")
# Interactive/visualization partitions: jobs there are idle by design,
# efficiency verdicts are adapted accordingly.
INTERACTIVE_PARTITIONS = [
    p.strip() for p in os.environ.get("HPC_MCP_INTERACTIVE_PARTITIONS", "visu").split(",") if p.strip()
]
PROM_VERIFY = os.environ.get("HPC_MCP_PROM_VERIFY", "1") == "1"

if not (PROM_VERIFY and ES_VERIFY):
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

mcp = FastMCP("hpc-mcp", host=HOST, port=PORT)


class _AuthMiddleware:
    """Static token auth (header "Authorization: Bearer <token>").
    Only active when HPC_MCP_AUTH_TOKEN is set. Constant-time comparison.
    Pure ASGI (no BaseHTTPMiddleware: incompatible with SSE/streaming)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and MCP_AUTH_TOKEN:
            import hmac
            headers = dict(scope.get("headers") or [])
            sent = headers.get(b"authorization", b"").decode("latin-1")
            if not hmac.compare_digest(sent, f"Bearer {MCP_AUTH_TOKEN}"):
                from starlette.responses import JSONResponse
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_PATTERN_RE = re.compile(r"^[A-Za-z0-9_.*+?|:^$()\[\]-]+$")  # Prometheus regex, no quotes/braces (underscore required for some logins)
_INDEX_RE = re.compile(r"^[a-z0-9_.*,-]+$")  # ES index names/patterns

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], max_lines: int = 1000) -> str:
    """Run a command without a shell (no injection possible through args)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=CMD_TIMEOUT)
    except FileNotFoundError:
        return f"ERROR: command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return f"ERROR: timeout ({CMD_TIMEOUT}s) on: {' '.join(cmd)}"
    if result.returncode != 0:
        return f"ERROR (rc={result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
    lines = result.stdout.strip().splitlines()
    if len(lines) > max_lines:
        return (f"⚠ TRUNCATED OUTPUT: showing {max_lines} of {len(lines)} lines. "
                f"Do NOT count the lines below or derive a total from them: this is "
                f"not the full dataset.\n" + "\n".join(lines[:max_lines]))
    return "\n".join(lines)


# --- Anti-false-empty / transient-error hardening ---------------------------
_SLURM_TRANSIENT_RE = re.compile(
    r"unable to contact|slurm_load_\w+ error|connect failure|socket timed out|"
    r"problem (talking|connecting) to|connection (refused|reset)|"
    r"try again|temporarily unavailable",
    re.IGNORECASE,
)

_UNRESOLVED_MSG = (
    "ERROR: user '{u}' could not be resolved through NSS (getent passwd) "
    "after several attempts. Two possible causes: (1) the login is misspelled "
    "or does not exist, (2) a transient NSS/SSSD hiccup. "
    "IMPORTANT: an empty SLURM result filtered with -u for a login that does "
    "not resolve is NOT reliable (the filter matches nothing because resolution "
    "failed) — hence this explicit error instead of a misleading empty list. "
    "Check the spelling of the login; if the user does exist, retry in a few "
    "seconds."
)


def _run_slurm(cmd: list[str], max_lines: int = 1000, retries: int = 2,
               backoff: float = 0.5) -> str:
    """Like _run, but retries on TRANSIENT SLURM failures (controller/slurmdbd
    momentarily unreachable, 'socket timed out'...). Transparent when all is
    well (no retry). Does NOT mask real errors (unknown job, etc.)."""
    import time
    out = _run(cmd, max_lines=max_lines)
    attempt = 0
    while (attempt < retries and out.startswith("ERROR")
           and _SLURM_TRANSIENT_RE.search(out)):
        time.sleep(backoff * (attempt + 1))
        out = _run(cmd, max_lines=max_lines)
        attempt += 1
    return out


def _user_resolves(login: str, retries: int = 2, backoff: float = 0.3) -> bool:
    """Check that a login resolves through NSS (getent passwd), with retry.
    Double purpose: (1) detect a misspelled/unresolved login, (2) warm the
    SSSD cache -> makes the following squeue/sacct -u reliable. The core of
    the anti-false-empty hardening."""
    import time
    for attempt in range(retries + 1):
        r = _run(["getent", "passwd", login])
        if not r.startswith("ERROR") and r.strip():
            return True
        if attempt < retries:
            time.sleep(backoff * (attempt + 1))
    return False


def _parsable_to_dicts(output: str, sep: str = "|") -> list[dict]:
    lines = [l for l in output.splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    headers = lines[0].split(sep)
    return [dict(zip(headers, line.split(sep))) for line in lines[1:]]


def _mmcmd(name: str) -> str:
    """Locate a GPFS binary (PATH or /usr/lpp/mmfs/bin)."""
    found = shutil.which(name)
    if found:
        return found
    candidate = f"/usr/lpp/mmfs/bin/{name}"
    return candidate if os.path.exists(candidate) else name


def _valid_name(value: str) -> bool:
    """Validate a fileset/login/node name (no Lucene/PromQL injection)."""
    return bool(value) and bool(_NAME_RE.match(value))


def _valid_pattern(value: str) -> bool:
    """Validate a regex pattern destined for a Prometheus label matcher."""
    return bool(value) and bool(_PATTERN_RE.match(value))


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(n) < 1024 or unit == "PiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def _parse_slurm_cpu(s: str) -> float | None:
    """'1-02:03:04' / '02:03:04' / '03:04.567' -> seconds."""
    if not s or s in ("INVALID", "Unknown"):
        return None
    days = 0
    if "-" in s:
        d, _, s = s.partition("-")
        try:
            days = int(d)
        except ValueError:
            return None
    parts = s.split(":")
    try:
        parts_f = [float(p) for p in parts]
    except ValueError:
        return None
    sec = 0.0
    for p in parts_f:
        sec = sec * 60 + p
    return days * 86400 + sec


def _parse_slurm_mem(s: str) -> tuple[float, str] | None:
    """'3971344K' / '16G' / '16Gn' / '16Gc' -> (bytes, mode '' | 'n' | 'c')."""
    s = (s or "").strip()
    if not s or s in ("0", "INVALID", "Unknown"):
        return None
    mode = ""
    if s[-1] in "nc":
        mode = s[-1]
        s = s[:-1]
    mult = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    try:
        if s and s[-1].upper() in mult:
            return float(s[:-1]) * mult[s[-1].upper()], mode
        return float(s), mode
    except ValueError:
        return None


def _contained_path(path: str) -> str | None:
    """Resolve the path (symlinks included) and check it sits under an allowed root."""
    if not os.path.isabs(path):
        return None
    real = os.path.realpath(path)
    for root in READ_ROOTS:
        if real == root or real.startswith(root + "/"):
            return real
    return None


def _tail(path: str, lines: int) -> str:
    """Last lines of a file, confinement included."""
    real = _contained_path(path)
    if real is None:
        return f"ERROR: path refused (outside allowed roots {','.join(READ_ROOTS)}): {path}"
    if not os.path.isfile(real):
        return f"ERROR: file not found: {real}"
    lines = max(1, min(lines, 500))
    try:
        with open(real, "r", errors="replace") as f:
            tail = deque(f, maxlen=lines)
    except OSError as e:
        return f"ERROR reading {real}: {e}"
    out = "".join(tail).rstrip()
    if len(out) > 12000:
        out = "...(truncated)...\n" + out[-12000:]
    return out if out else "(empty file)"


def _extract_script_info(script: str) -> tuple[list[str], list[str]]:
    """Extract from an sbatch script the loaded modules and the absolute paths
    it references (under the allowed roots), to enrich the diagnosis."""
    modules = []
    for m in re.finditer(r"^\s*module\s+(?:load|add)\s+(.+)$", script, re.MULTILINE):
        modules += [tok for tok in m.group(1).split() if not tok.startswith("-")]
    roots = "|".join(re.escape(r) for r in READ_ROOTS)
    paths = re.findall(rf"(?<![\w-])((?:{roots})/[^\s'\"()|;<>,=]+)", script)
    seen: set[str] = set()
    uniq_paths = []
    for p in paths:
        p = p.rstrip(".,:")
        if p not in seen:
            seen.add(p)
            uniq_paths.append(p)
    return modules, uniq_paths[:8]


def _detect_parallelism(script: str) -> list[str]:
    """Detect the parallelization mechanisms referenced in an sbatch script
    (MPI, OpenMP, dask, GNU parallel, multiprocessing, srun...).
    Returns the list of mechanisms found (empty if none detected)."""
    if not script:
        return []
    found = []
    # (label, regex) — case-insensitive, multiline search
    patterns = [
        ("MPI (mpirun/mpiexec)", r"\b(mpirun|mpiexec)\b"),
        ("MPI (srun)", r"\bsrun\b"),
        ("OpenMP", r"\bOMP_NUM_THREADS\b"),
        ("dask", r"\bdask(-scheduler|-worker|\.distributed)?\b|\bLocalCluster\b|\bSLURMCluster\b"),
        ("GNU parallel", r"\bparallel\b\s+(-|:::|--jobs)"),
        ("xargs -P", r"\bxargs\b[^\n]*-P"),
        ("Python multiprocessing", r"\bmultiprocessing\b|\bconcurrent\.futures\b|\bProcessPoolExecutor\b|\bjoblib\b"),
        ("Ray", r"\bray\b\s+(start|up)|\bray\.init\b"),
        ("Spark", r"\bspark-submit\b"),
        ("CUDA/GPU", r"\bCUDA_VISIBLE_DEVICES\b|\bnvidia-smi\b"),
        ("threads (&/wait)", r"&\s*\n[^\n]*\bwait\b"),
    ]
    for label, rx in patterns:
        if re.search(rx, script, re.IGNORECASE):
            found.append(label)
    return found


def _analyze_parallelism(script: str, alloc_cpus: int, n_nodes: int, n_tasks: int = 0) -> str | None:
    """Confront the detected parallelization mechanisms with the allocated
    resources. Returns one analysis line, or None if there is nothing relevant
    to say (single-core job with no parallelism = normal, stay silent)."""
    mechanisms = _detect_parallelism(script)
    multi_cpu = alloc_cpus > 1 or n_nodes > 1 or n_tasks > 1

    if mechanisms:
        base = f"Parallelization detected: {', '.join(mechanisms)}"
        # Multi-node without MPI/srun/distributed dask = suspicious
        distributed = any(m.startswith(("MPI", "dask", "Ray", "Spark")) for m in mechanisms)
        if n_nodes > 1 and not distributed:
            return (base + f". ⚠ {n_nodes} nodes allocated but no distributed "
                    "mechanism (MPI/dask/Ray) detected — inter-node "
                    "parallelization may not be effective.")
        return base + "."

    # No mechanism detected BUT multiple resources requested -> likely waste
    if multi_cpu:
        return (f"⚠ No parallelization mechanism detected in the script, "
                f"but {alloc_cpus} CPUs / {n_nodes} node(s) allocated. The job may "
                "be sequential: check that it actually uses the requested "
                "resources (otherwise reduce the allocation).")
    # Single-core with no parallelism = normal, nothing to report.
    return None


def _resolve_slurm_pattern(path: str, job_id: str, job_name: str, user: str) -> str:
    """Resolve SLURM filename patterns (%j %x %u %A %a %%) in a path.
    SLURM sometimes stores the raw pattern (e.g. .../%x-%j/output.log) instead
    of the resolved path, which breaks the open() unless resolved here.
    """
    base_id = job_id.split("_")[0].split(".")[0]
    repl = {
        "%j": base_id, "%J": base_id, "%A": base_id,
        "%x": job_name or "job", "%u": user or "", "%a": job_id.split("_")[-1] if "_" in job_id else "0",
        "%%": "%",
    }
    for pat, val in repl.items():
        path = path.replace(pat, val)
    return path


def _sacct_rows(job_id: str) -> list[dict] | str:
    out = _run_slurm(
        [
            "sacct", "-j", job_id, "--parsable2", "--format",
            "JobID,JobName,User,State,Elapsed,ElapsedRaw,TotalCPU,AllocCPUS,NNodes,ReqMem,MaxRSS,"
            "Timelimit,TimelimitRaw,NodeList,Start,End,ExitCode,WorkDir",
        ],
        max_lines=2000,
    )
    if out.startswith("ERROR"):
        return out
    rows = _parsable_to_dicts(out)
    if not rows:
        return f"ERROR: no sacct record for job {job_id}"
    return rows


# ---------------------------------------------------------------------------
# Native GPFS (mm* commands, read-only — requires root)
# ---------------------------------------------------------------------------


def gpfs_filesets_list() -> str:
    """List all GPFS filesets of the device (mmlsfileset)."""
    return _run([_mmcmd("mmlsfileset"), GPFS_DEVICE])


def gpfs_fileset_quota(fileset: str) -> str:
    """Quota of a GPFS fileset: space and inodes (mmlsquota -j).

    Args:
        fileset: fileset name (e.g. scratch)
    """
    if not _valid_name(fileset):
        return "ERROR: invalid fileset name"
    return _run([_mmcmd("mmlsquota"), "-j", fileset, "--block-size", "auto", GPFS_DEVICE])


def gpfs_all_quotas() -> str:
    """Quotas of every fileset of the device (mmrepquota -j). Large output."""
    return _run([_mmcmd("mmrepquota"), "-j", "--block-size", "auto", GPFS_DEVICE])


def gpfs_df() -> str:
    """Capacity of the GPFS pools of the device (mmdf)."""
    return _run([_mmcmd("mmdf"), GPFS_DEVICE, "--block-size", "auto"])


def filesystem_usage(path: str) -> str:
    """Usage of the FILESYSTEM containing a path (df -h) — NOT the directory size.

    Always returns the global usage of the mount point: /work,
    /work/scratch/data and /work/scratch/data/<login> give the SAME result
    (e.g. /work = 150T with 128T used, 85%). Calling this tool on several
    subdirectories is pointless, the answer will not change.

    Use for: "is the filesystem full?", "how much is left on /work?".
    Do NOT use for the size of a directory or of a user — use instead:
      - gpfs_fileset_quota(fileset): usage vs quota of a fileset (blocks + inodes)
      - list_dir(path, sort="size"): largest FILES of a directory

    Args:
        path: absolute path used to IDENTIFY the filesystem (e.g. /work).
            The exact subpath has no effect on the result.
    """
    if not os.path.isabs(path) or ".." in path:
        return "ERROR: absolute path required, without '..'"
    return _run(["df", "-h", path])


def stat_file(path: str) -> str:
    """Metadata of a file or directory: existence, size, owner, dates.
    Useful to check that a job input exists or to diagnose a 'file not
    found'. Confined to the allowed roots (default: /work).

    Args:
        path: absolute path
    """
    real = _contained_path(path)
    if real is None:
        return f"ERROR: path refused (outside allowed roots {','.join(READ_ROOTS)}): {path}"
    if not os.path.exists(real):
        parent = os.path.dirname(real)
        parent_state = "exists" if os.path.isdir(parent) else "does not exist either"
        return f"ABSENT: {real} (the parent directory {parent_state})"
    try:
        st = os.stat(real)
        try:
            import pwd, grp
            owner = pwd.getpwuid(st.st_uid).pw_name
            group = grp.getgrgid(st.st_gid).gr_name
        except KeyError:
            owner, group = str(st.st_uid), str(st.st_gid)
        kind = "directory" if os.path.isdir(real) else "file"
        return (
            f"{kind}: {real}\n"
            f"  Size: {_fmt_bytes(st.st_size)}\n"
            f"  Owner: {owner}:{group} (mode {oct(st.st_mode)[-4:]})\n"
            f"  Modified: {datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')}"
        )
    except OSError as e:
        return f"ERROR stat {real}: {e}"


def gpfs_health() -> str:
    """Health state of the GPFS cluster (mmhealth): nodes, disks, filesystem,
    quorum. The first reflex when 'the cluster feels slow'."""
    out = _run([_mmcmd("mmhealth"), "cluster", "show"])
    if out.startswith("ERROR"):
        # fallback: local node view
        return _run([_mmcmd("mmhealth"), "node", "show"])
    return out


def gpfs_fileset_path(fileset: str) -> str:
    """Junction path of a GPFS fileset (mmlsfileset -L): links a fileset name
    to its actual directory on disk, to then explore with list_dir or
    stat_file.

    Args:
        fileset: fileset name (e.g. scratch)
    """
    if not _valid_name(fileset):
        return "ERROR: invalid fileset name"
    return _run([_mmcmd("mmlsfileset"), GPFS_DEVICE, fileset, "-L"])


def list_dir(path: str, sort: str = "mtime", top: int = 50) -> str:
    """List a directory: type, size, modification date, owner.
    To explore a job's workspace, check what it produced, or find the real
    name of a log file. Confined to the allowed roots.

    CAUTION: the size of SUBDIRECTORIES is not computed (shown as '-',
    sort=size only ranks files). Use gpfs_fileset_quota for fileset-level
    capacity questions.

    Args:
        path: absolute path of the directory
        sort: ordering — mtime (default, most recent first), size, name
        top: number of entries returned (default 50, max 200)
    """
    real = _contained_path(path)
    if real is None:
        return f"ERROR: path refused (outside allowed roots {','.join(READ_ROOTS)}): {path}"
    if not os.path.isdir(real):
        return f"ERROR: not a directory: {real}"
    if sort not in ("mtime", "size", "name"):
        return "ERROR: sort must be mtime, size or name"
    entries = []
    truncated = False
    try:
        with os.scandir(real) as it:
            for i, e in enumerate(it):
                if i >= 5000:  # guard against giant directories
                    truncated = True
                    break
                try:
                    st = e.stat(follow_symlinks=False)
                    entries.append((e.name, e.is_dir(follow_symlinks=False),
                                    st.st_size, st.st_mtime, st.st_uid))
                except OSError:
                    continue
    except OSError as e:
        return f"ERROR reading {real}: {e}"
    if not entries:
        return f"(empty directory) {real}"
    key = {"mtime": lambda x: -x[3], "size": lambda x: -x[2], "name": lambda x: x[0]}[sort]
    entries.sort(key=key)
    top = max(1, min(top, 200))
    try:
        import pwd
        _uname = lambda uid: pwd.getpwuid(uid).pw_name
    except ImportError:
        _uname = str
    lines = [f"{real} ({len(entries)}{'+' if truncated else ''} entries, sorted by {sort}):"]
    for name, is_dir, size, mtime, uid in entries[:top]:
        try:
            owner = _uname(uid)
        except KeyError:
            owner = str(uid)
        kind = "d" if is_dir else "f"
        lines.append(
            f"  {kind} {_fmt_bytes(size) if not is_dir else '-':>12} "
            f"{datetime.fromtimestamp(mtime).isoformat(timespec='minutes')} "
            f"{owner:<14} {name}"
        )
    if len(entries) > top:
        lines.append(f"  ... ({len(entries) - top} more entries)")
    return "\n".join(lines)


def grep_file(path: str, pattern: str, max_matches: int = 50, context: int = 0,
              ignore_case: bool = True) -> str:
    """Search a pattern (regex) in a file and return the matching lines with
    their line numbers. The log-investigation tool: Error, Traceback, OOM,
    Killed... where tail_file only shows the end. Confined to the allowed
    roots.

    Args:
        path: absolute path of the file
        pattern: Python regular expression (e.g. 'error|traceback|killed')
        max_matches: max number of returned lines (default 50, max 200)
        context: context lines before/after each match (0 to 3)
        ignore_case: case-insensitive (default True)
    """
    real = _contained_path(path)
    if real is None:
        return f"ERROR: path refused (outside allowed roots {','.join(READ_ROOTS)}): {path}"
    if not os.path.isfile(real):
        return f"ERROR: file not found: {real}"
    if len(pattern) > 200:
        return "ERROR: pattern too long (max 200 characters)"
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return f"ERROR: invalid regex: {e}"
    max_matches = max(1, min(max_matches, 200))
    context = max(0, min(context, 3))
    out: list[str] = []
    n_matches = 0
    before: deque = deque(maxlen=context)
    after_left = 0
    try:
        with open(real, "r", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                line = line.rstrip("\n")[:2000]  # bound pathological lines
                if rx.search(line):
                    n_matches += 1
                    if context:
                        out.extend(f"  {no:>8}- {txt}" for no, txt in before)
                        before.clear()
                    out.append(f"  {lineno:>8}: {line}")
                    after_left = context
                    if n_matches >= max_matches:
                        out.append(f"  ... (stopped after {max_matches} matches, line {lineno})")
                        break
                elif after_left > 0:
                    out.append(f"  {lineno:>8}- {line}")
                    after_left -= 1
                elif context:
                    before.append((lineno, line))
    except OSError as e:
        return f"ERROR reading {real}: {e}"
    if not out:
        return f"No match for /{pattern}/ in {real}."
    header = f"{n_matches} match(es) for /{pattern}/ in {real}:"
    result = header + "\n" + "\n".join(out)
    if len(result) > 14000:
        result = result[:14000] + "\n... (output truncated)"
    return result


def tail_file(path: str, lines: int = 100) -> str:
    """Last lines of a text file (job log, script output...).
    Read-only, confined to the allowed roots (default: /work).

    Args:
        path: absolute path of the file
        lines: number of lines (default 100, max 500)
    """
    return _tail(path, lines)


# ---------------------------------------------------------------------------
# Elasticsearch — generic, read-only exploration (indices, search, logs)
# Anti-false-empty everywhere: an empty result always states WHAT was empty
# (no docs in the window vs unknown index vs unknown field), never a bare
# nothing the model could misread as "there is no problem".
# ---------------------------------------------------------------------------


def _es_auth() -> tuple[str, str] | None:
    user, password = os.environ.get("HPC_MCP_ES_USER"), os.environ.get("HPC_MCP_ES_PASS")
    if user and password:
        return (user, password)
    return None


def _index_allowed(index: str) -> bool:
    """Check an index name/pattern against HPC_MCP_ES_ALLOWED_INDICES
    (fnmatch patterns). Empty allowlist = everything allowed."""
    if not ES_ALLOWED_INDICES:
        return True
    return any(fnmatch.fnmatch(part, allowed)
               for part in index.split(",")
               for allowed in ES_ALLOWED_INDICES)


def _check_index(index: str) -> str | None:
    """Validate an index argument. Returns an error string, or None if OK."""
    if not index or not _INDEX_RE.match(index):
        return "ERROR: invalid index name (lowercase letters, digits, '_-.*,' only)"
    if not _index_allowed(index):
        return (f"ERROR: index '{index}' is not in the allowlist "
                f"(HPC_MCP_ES_ALLOWED_INDICES={','.join(ES_ALLOWED_INDICES)}).")
    return None


def _es_request(method: str, path: str, body: dict | None = None) -> dict | list | str:
    try:
        r = requests.request(
            method,
            f"{ES_URL}/{path}",
            json=body,
            auth=_es_auth(),
            verify=ES_VERIFY,
            timeout=CMD_TIMEOUT,
        )
        if r.status_code == 404:
            return (f"ERROR: Elasticsearch returned 404 for '{path.split('/')[0]}' — "
                    "the index does not exist (or the account lacks access). "
                    "This is NOT 'no data': list available indices with es_indices.")
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return f"ERROR Elasticsearch ({ES_URL}): {e}"


def _es_search(body: dict, index: str) -> dict | str:
    body.setdefault("track_total_hits", True)
    out = _es_request("POST", f"{index}/_search", body)
    if isinstance(out, list):
        return f"ERROR: unexpected Elasticsearch response shape for {index}/_search"
    return out


def _doc_line(src: dict, max_len: int = 300) -> str:
    """Compact one-line rendering of a document: prefer a message-like field,
    fall back to truncated JSON."""
    for field in ("message", "msg", "log", "event", "error"):
        if isinstance(src.get(field), str) and src[field].strip():
            return src[field].strip()[:max_len]
    return json.dumps(src, ensure_ascii=False, default=str)[:max_len]


def es_indices(pattern: str = "*") -> str:
    """List Elasticsearch indices matching a pattern, with doc count and size.
    The entry point of any ES exploration: find WHICH index holds the data
    before searching it.

    Args:
        pattern: index pattern (default '*'; e.g. 'logs-*', '*slurm*')
    """
    err = _check_index(pattern)
    if err:
        return err
    out = _es_request("GET", f"_cat/indices/{pattern}?format=json&h=index,docs.count,store.size,health&s=index")
    if isinstance(out, str):
        return out
    if not isinstance(out, list) or not out:
        return (f"No index matches '{pattern}'. This means the pattern matched "
                "nothing — try es_indices('*') to see everything available.")
    visible = [row for row in out if _index_allowed(row.get("index", ""))]
    hidden = len(out) - len(visible)
    lines = [f"{len(visible)} index(es) matching '{pattern}':"]
    for row in visible[:200]:
        lines.append(f"  {row.get('index', '?'):<48} {row.get('docs.count', '?'):>12} docs "
                     f"{row.get('store.size', '?'):>10}  {row.get('health', '?')}")
    if hidden:
        lines.append(f"  ({hidden} more index(es) hidden by the allowlist)")
    return "\n".join(lines)


def es_fields(index: str) -> str:
    """Field names and types of an index (its mapping). Call this before
    es_search/es_aggregate when a query returns nothing: a wrong field name is
    the most common cause of a false empty.

    Args:
        index: index name or pattern (e.g. 'logs-2026.08')
    """
    err = _check_index(index)
    if err:
        return err
    out = _es_request("GET", f"{index}/_mapping")
    if isinstance(out, str):
        return out

    fields: dict[str, str] = {}

    def _walk(props: dict, prefix: str) -> None:
        for name, spec in props.items():
            full = f"{prefix}{name}"
            if "properties" in spec:
                _walk(spec["properties"], full + ".")
            else:
                fields[full] = spec.get("type", "?")
                for sub, sub_spec in (spec.get("fields") or {}).items():
                    fields[f"{full}.{sub}"] = sub_spec.get("type", "?")

    for idx_body in out.values():
        _walk(idx_body.get("mappings", {}).get("properties", {}), "")
    if not fields:
        return f"Index '{index}' exists but has no mapped fields (empty index?)."
    lines = [f"{len(fields)} field(s) in {index}:"]
    for name in sorted(fields)[:200]:
        lines.append(f"  {name:<50} {fields[name]}")
    if len(fields) > 200:
        lines.append(f"  ... ({len(fields) - 200} more fields)")
    return "\n".join(lines)


def es_search(index: str, query: str = "*", top: int = 10, sort_field: str = "") -> str:
    """Search an index with a Lucene query string and return matching
    documents. The generic exploration tool — for time-ordered log tailing
    prefer es_tail_logs.

    The EXACT total is given in the header: it covers ALL matches even if the
    listing is truncated — never recount the displayed lines.

    Args:
        index: index name or pattern (e.g. 'logs-*')
        query: Lucene query string (default '*'; e.g. 'level:ERROR AND host:node042')
        top: number of documents returned (default 10, max 50)
        sort_field: optional field to sort on, descending (e.g. '@timestamp')
    """
    err = _check_index(index)
    if err:
        return err
    body: dict = {
        "size": max(1, min(top, 50)),
        "query": {"query_string": {"query": query or "*", "lenient": True}},
    }
    if sort_field:
        body["sort"] = [{sort_field: {"order": "desc", "unmapped_type": "date"}}]
    data = _es_search(body, index)
    if isinstance(data, str):
        return data
    total = data.get("hits", {}).get("total", {}).get("value", 0)
    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        return (f"0 documents match '{query}' in {index}. The index exists and "
                "responded: either nothing matches, or a field name in the query "
                "is wrong — check with es_fields('" + index + "').")
    lines = [f"{total} document(s) match '{query}' in {index} "
             f"(EXACT total; showing {len(hits)}):"]
    for h in hits:
        lines.append(f"  [{h.get('_index', '?')}] {_doc_line(h.get('_source', {}))}")
    return "\n".join(lines)


def es_aggregate(index: str, field: str, query: str = "*", top: int = 20) -> str:
    """Break down matching documents by the values of a field (terms
    aggregation): top hosts, top error codes, top users... The EXACT overall
    total is computed server-side and given in the header — never sum the
    displayed buckets to reinvent it.

    Args:
        index: index name or pattern
        field: field to group by (a keyword field; '.keyword' is tried
            automatically if the raw field is not aggregatable)
        query: Lucene query string to filter first (default '*')
        top: number of buckets (default 20, max 100)
    """
    err = _check_index(index)
    if err:
        return err
    if not re.match(r"^[\w.@-]+$", field or ""):
        return "ERROR: invalid field name"

    def _agg(f: str) -> dict | str:
        body = {
            "size": 0,
            "query": {"query_string": {"query": query or "*", "lenient": True}},
            "aggs": {"buckets": {"terms": {"field": f, "size": max(1, min(top, 100))}}},
        }
        return _es_search(body, index)

    data = _agg(field)
    if isinstance(data, str):
        return data
    # Text fields are not aggregatable -> retry on the .keyword sub-field.
    if "aggregations" not in data and not field.endswith(".keyword"):
        retry = _agg(field + ".keyword")
        if not isinstance(retry, str) and "aggregations" in retry:
            data, field = retry, field + ".keyword"
    total = data.get("hits", {}).get("total", {}).get("value", 0)
    buckets = data.get("aggregations", {}).get("buckets", {}).get("buckets", [])
    if not total:
        return (f"0 documents match '{query}' in {index} — nothing to aggregate. "
                f"Check field names with es_fields('{index}').")
    if not buckets:
        return (f"{total} document(s) match but the field '{field}' produced no "
                f"buckets: it is probably not aggregatable or absent from these "
                f"documents — check with es_fields('{index}').")
    other = data.get("aggregations", {}).get("buckets", {}).get("sum_other_doc_count", 0)
    lines = [f"TOTAL (exact): {total} document(s) matching '{query}' in {index}, "
             f"grouped by {field}:"]
    for b in buckets:
        lines.append(f"  {str(b.get('key')):<40} {b.get('doc_count', 0):>10,}")
    if other:
        lines.append(f"  (other values)                          {other:>10,}")
    return "\n".join(lines)


def es_tail_logs(index: str, query: str = "*", minutes: int = 60, top: int = 50,
                 time_field: str = "@timestamp") -> str:
    """Most recent documents of a time-based index (logs), newest first.
    The 'tail -f of the last hour' for any log index: service logs, syslog,
    application errors...

    Args:
        index: index name or pattern (e.g. 'logs-*')
        query: Lucene query string filter (default '*'; e.g. 'level:ERROR')
        minutes: time window looking back from now (default 60, max 7 days)
        top: number of documents (default 50, max 200)
        time_field: timestamp field of the index (default '@timestamp')
    """
    err = _check_index(index)
    if err:
        return err
    if not re.match(r"^[\w.@-]+$", time_field or ""):
        return "ERROR: invalid time_field name"
    minutes = max(1, min(minutes, 7 * 24 * 60))
    body = {
        "size": max(1, min(top, 200)),
        "query": {"bool": {
            "must": [{"query_string": {"query": query or "*", "lenient": True}}],
            "filter": [{"range": {time_field: {"gte": f"now-{minutes}m"}}}],
        }},
        "sort": [{time_field: {"order": "desc", "unmapped_type": "date"}}],
    }
    data = _es_search(body, index)
    if isinstance(data, str):
        return data
    total = data.get("hits", {}).get("total", {}).get("value", 0)
    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        return (f"0 documents in {index} over the last {minutes} min matching "
                f"'{query}'. The index responded: either the window is quiet, or "
                f"'{time_field}' is not its timestamp field — check with "
                f"es_fields('{index}') and pass time_field=... if needed.")
    lines = [f"{total} document(s) in the last {minutes} min matching '{query}' "
             f"in {index} (EXACT total; showing the {len(hits)} most recent):"]
    for h in hits:
        src = h.get("_source", {})
        ts = str(src.get(time_field, "?"))[:19]
        lines.append(f"  {ts}  {_doc_line(src, max_len=260)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------


def squeue_jobs(user: str = "", account: str = "", state: str = "", partition: str = "") -> str:
    """List SLURM jobs, queued or running.

    Args:
        user: filter by user (login)
        account: filter by SLURM account (billing group, e.g. physics)
        state: filter by state (PENDING, RUNNING, ...)
        partition: filter by partition
    """
    if user:
        if not _valid_name(user):
            return "ERROR: invalid login"
        if not _user_resolves(user):
            return _UNRESOLVED_MSG.format(u=user)
    cmd = [
        "squeue",
        "--Format",
        "JobID,UserName,Account,Partition,Name:30,State,TimeUsed,TimeLimit,NumNodes,tres-per-node,Reason:30",
    ]
    if user:
        cmd += ["-u", user]
    if account:
        if not _valid_name(account):
            return "ERROR: invalid account"
        cmd += ["-A", account]
    if state:
        cmd += ["-t", state]
    if partition:
        cmd += ["-p", partition]
    return _run_slurm(cmd)


def job_details(job_id: str) -> str:
    """Full details of a job (scontrol + sacct accounting).

    Args:
        job_id: SLURM job identifier
    """
    if not job_id.replace("_", "").replace(".", "").isalnum():
        return "ERROR: invalid job_id"
    scontrol = _run_slurm(["scontrol", "show", "job", job_id])
    sacct = _run_slurm(
        [
            "sacct", "-j", job_id, "--parsable2", "--format",
            "JobID,JobName,State,Elapsed,AllocTRES,ReqMem,MaxRSS,NodeList,ExitCode,Reason",
        ]
    )
    return f"=== scontrol ===\n{scontrol}\n\n=== sacct ===\n{sacct}"


def why_pending(job_id: str) -> str:
    """Diagnose why a job is waiting: SLURM reason + partition state.

    Args:
        job_id: SLURM job identifier
    """
    if not job_id.replace("_", "").replace(".", "").isalnum():
        return "ERROR: invalid job_id"
    scontrol = _run_slurm(["scontrol", "show", "job", job_id])
    if scontrol.startswith("ERROR"):
        if "invalid job id" in scontrol.lower():
            return (
                f"Job {job_id} is unknown to the controller: it is no longer in "
                "the queue (already started, finished or purged) — it is NOT a "
                "pending job. To find out what happened to it: "
                "diagnose_job(<jobid>) (full diagnosis) or "
                "sacct_history(user=..., days=...). Do not ask the user to run "
                "sacct themselves."
            )
        return scontrol

    info = {}
    for token in scontrol.replace("\n", " ").split():
        if "=" in token:
            k, _, v = token.partition("=")
            info[k] = v

    partition = info.get("Partition", "")
    reason = info.get("Reason", "unknown")
    priority = info.get("Priority", "?")

    sinfo = (
        _run_slurm(["sinfo", "-p", partition, "--Format", "Partition,Available,NodeAIOT,StateCompact,Gres:30"])
        if partition
        else "unknown partition"
    )
    ahead = _run_slurm(["squeue", "-p", partition, "-t", "PENDING", "--noheader", "-o", "%i"]) if partition else ""
    n_ahead = len(ahead.splitlines()) if ahead and not ahead.startswith("ERROR") else "?"

    return (
        f"SLURM reason: {reason}\n"
        f"Priority: {priority}\n"
        f"Partition: {partition} ({n_ahead} PENDING jobs in total)\n\n"
        f"=== Partition state ===\n{sinfo}"
    )


def sacct_history(user: str = "", account: str = "", days: int = 7,
                  state: str = "", top: int = 200) -> str:
    """History of finished jobs through sacct (-X: 1 line = 1 job, no steps),
    filterable by user OR by SLURM account. Do not confuse them: an account is
    a billing group containing several users.

    The TOTAL is computed and given in the header: it is exact even when the
    listing is truncated — NEVER recount the displayed lines or derive a
    volume from them. This tool is for SEEING jobs.

    Args:
        user: filter by user (login)
        account: filter by SLURM account (all users of the account)
        days: history depth in days (default 7)
        state: filter by final state (COMPLETED, FAILED, TIMEOUT, ...)
        top: number of jobs listed, the most RECENT ones (default 200, max 500).
            Only affects the display: the header total covers all jobs.
    """
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    if user and not account:
        if not _valid_name(user):
            return "ERROR: invalid login"
        if not _user_resolves(user):
            return _UNRESOLVED_MSG.format(u=user)
    cmd = [
        "sacct", "--parsable2", "-X", "-S", start, "--format",
        "JobID,User,Account,JobName,Partition,State,Elapsed,AllocTRES,NodeList,End",
    ]
    if account:
        if not _valid_name(account):
            return "ERROR: invalid account"
        cmd += ["-A", account, "--allusers"]
        if user:
            cmd += ["-u", user]
    else:
        cmd += ["-u", user or os.environ.get("USER", "")]
    if state:
        cmd += ["-s", state]

    # Very high max_lines: fetch everything, truncation is handled here (with
    # an exact total). Sentinel so the _run banner never triggers, which would
    # shift the header (a big account over 180d = several 100k lines).
    out = _run(cmd, max_lines=10**9)
    if out.startswith("ERROR"):
        if "timeout" in out:
            return (out + f"\n\nsacct did not answer in time on this window "
                    f"({days} days): expected on big accounts / long periods "
                    f"(slurmdbd). Do NOT conclude that there are no jobs. "
                    f"Reduce days or filter by state to narrow the request.")
        return out
    lines = out.splitlines()
    header, rows = (lines[0], lines[1:]) if lines else ("", [])
    n = len(rows)
    filt_str = "".join(
        f", {lbl}={v}" for lbl, v in (("account", account), ("user", user), ("state", state)) if v
    )
    if n == 0:
        return f"No finished job over the last {days} days{filt_str}."

    # Sort on End (last column), not on JobID: job ids may have been reset in
    # the past, in which case ID order does not follow time.
    rows.sort(key=lambda r: r.rsplit("|", 1)[-1])
    top = max(1, min(top, 500))
    shown = rows[-top:]

    out_lines = [f"{n} finished job(s) over {days} days{filt_str} "
                 f"(sacct -X: 1 line = 1 job, steps excluded)"]
    if n > top:
        out_lines.append(
            f"⚠ TRUNCATED LISTING: {top} jobs shown out of {n} (most recent). "
            f"The total of {n} above is EXACT — do not recount the lines."
        )
    out_lines += ["", header] + shown
    out_lines.append(
        "\n(For a specific job: job_script(<JobID>) gives the submission script, "
        "job_logs(<JobID>) the logs, diagnose_job(<JobID>) the full diagnosis.)"
    )
    return "\n".join(out_lines)


def gpu_usage_by_user(days: int = 7) -> str:
    """Aggregate the GPU-hours consumed per user over N days (through sacct).

    Args:
        days: analysis window in days (default 7)
    """
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    out = _run_slurm(
        ["sacct", "--allusers", "--parsable2", "-X", "-S", start, "--format",
         "User,AllocTRES,ElapsedRaw,State"],
        max_lines=100000,
    )
    if out.startswith("ERROR"):
        return out

    usage: dict[str, float] = {}
    for row in _parsable_to_dicts(out):
        tres = row.get("AllocTRES", "")
        if "gres/gpu=" not in tres:
            continue
        try:
            n_gpu = int(tres.split("gres/gpu=")[1].split(",")[0])
            elapsed_h = int(row.get("ElapsedRaw", "0")) / 3600
        except (ValueError, IndexError):
            continue
        user = row.get("User", "?") or "?"
        usage[user] = usage.get(user, 0.0) + n_gpu * elapsed_h

    if not usage:
        return f"No GPU consumption found over the last {days} days."
    ranking = sorted(usage.items(), key=lambda kv: kv[1], reverse=True)
    lines = [f"GPU-hours per user (last {days} days):"]
    lines += [f"  {u:<20} {h:>10.1f} GPU·h" for u, h in ranking]
    return "\n".join(lines)


def cluster_status() -> str:
    """Cluster overview: partitions, available nodes, down nodes."""
    sinfo = _run_slurm(["sinfo", "--Format", "Partition,Available,NodeAIOT,StateCompact,Gres:30,GresUsed:30"])
    down = _run_slurm(["sinfo", "-R", "--Format", "NodeList,StateCompact,Reason:60"])
    return f"=== Partitions ===\n{sinfo}\n\n=== Down/drained nodes ===\n{down}"


def fairshare(user: str = "", account: str = "") -> str:
    """SLURM fairshare (sshare): historical usage vs allocated share. This is
    the variable behind most 'my job won't start even though the queue is
    empty' — an exhausted fairshare sinks the priority.

    Args:
        user: login to inspect (empty = all root accounts)
        account: SLURM account to inspect
    """
    cmd = ["sshare", "-l", "--parsable2"]
    if user:
        if not _valid_name(user):
            return "ERROR: invalid login"
        if not _user_resolves(user):
            return _UNRESOLVED_MSG.format(u=user)
        cmd += ["-u", user]
    if account:
        if not _valid_name(account):
            return "ERROR: invalid account"
        cmd += ["-A", account]
    # View flags depend on the COMBINATION (historical bug: -U and -a together
    # on a user+account combo returned an empty result):
    #   user alone       -> -U (that user's associations only)
    #   account alone    -> -a (all users of the account)
    #   user + account   -> neither -U nor -a: the precise user x account association
    if user and not account:
        cmd += ["-U"]
    elif account and not user:
        cmd += ["-a"]
    out = _run_slurm(cmd)
    if out.startswith("ERROR"):
        return out
    rows = _parsable_to_dicts(out)
    if user and account and not any(r.get("User") for r in rows):
        return (f"No association found for user={user} in account={account}. "
                "Either the user is not a member of this account (check with "
                "user_overview), or one of the two names is wrong. An empty "
                "result here does NOT mean a null fairshare.")
    if not rows:
        return out  # unexpected format, raw passthrough
    lines = ["Fairshare — ONLY LevelFS is authoritative (LevelFS < 1 = above its "
             "share; the lower the worse). Do NOT compare share and usage with "
             "each other (their ratio is not LevelFS) and do NOT convert LevelFS "
             "into a factor ('X times more'):"]
    for r in rows[:40]:
        lines.append(
            f"  {r.get('Account', '?'):<25} {r.get('User', ''):<12} "
            f"share={r.get('NormShares', '?'):<9} usage={r.get('NormUsage', '?'):<10} "
            f"LevelFS={r.get('LevelFS', r.get('FairShare', '?'))}"
        )
    return "\n".join(lines)


def job_priority(job_id: str = "", partition: str = "") -> str:
    """Priority breakdown of PENDING jobs (sprio): age, fairshare, size,
    partition, QOS. Shows why one job goes before another.

    Args:
        job_id: a specific job (empty = all pending)
        partition: filter by partition
    """
    cmd = ["sprio", "-l"]
    if job_id:
        if not job_id.replace("_", "").replace(".", "").isalnum():
            return "ERROR: invalid job_id"
        cmd += ["-j", job_id]
    if partition:
        if not _valid_name(partition):
            return "ERROR: invalid partition"
        cmd += ["-p", partition]
    return _run_slurm(cmd, max_lines=60)


def job_logs(job_id: str, lines: int = 100) -> str:
    """Last lines of a job's StdOut/StdErr files.
    Only works while the job is still known to scontrol (running or recently
    finished). For a purged job, use tail_file with the log path directly.

    Args:
        job_id: SLURM job identifier
        lines: number of lines per file (default 100, max 500)
    """
    if not job_id.replace("_", "").replace(".", "").isalnum():
        return "ERROR: invalid job_id"

    stdout = stderr = workdir = job_name = user = ""

    scontrol = _run_slurm(["scontrol", "show", "job", job_id])
    if scontrol.startswith("ERROR"):
        return (
            f"Job {job_id}: unknown to scontrol (finished job, purged from the "
            "controller). Its log paths can no longer be recovered from SLURM — "
            "use tail_file directly with the log path (often in the submission "
            "directory), or list_dir on the working directory to find it."
        )
    info = {}
    for token in scontrol.replace("\n", " ").split():
        if "=" in token:
            k, _, v = token.partition("=")
            info[k] = v
    stdout, stderr = info.get("StdOut", ""), info.get("StdErr", "")
    workdir = info.get("WorkDir", "")
    job_name = info.get("JobName", "")
    user = info.get("UserId", "").split("(")[0]

    if not stdout and not stderr:
        hint = f" (WorkDir={workdir})" if workdir else ""
        return (
            f"No StdOut/StdErr path recorded for job {job_id}{hint}. "
            "The log probably sits in WorkDir under a custom name — "
            "use tail_file with the full path."
        )

    parts = []
    seen = set()
    for label, raw in (("StdErr", stderr), ("StdOut", stdout)):
        if not raw or raw == "/dev/null":
            continue
        # Resolve SLURM patterns (%x-%j etc.) that scontrol sometimes stores raw.
        path = _resolve_slurm_pattern(raw, job_id, job_name, user)
        # Relative path -> prefix with the submission WorkDir.
        if not os.path.isabs(path) and workdir:
            path = os.path.join(workdir, path)
        if path in seen:
            continue
        seen.add(path)
        note = f" (raw pattern: {raw})" if path != raw else ""
        parts.append(f"=== {label}: {path}{note} ===\n{_tail(path, lines)}")
    return "\n\n".join(parts) if parts else "No usable log file."


def _scontrol_info(job_id: str) -> dict | None:
    """Parse `scontrol show job` into a dict. None if the job is unknown to
    the controller (finished/purged). Useful fields: JobState, BatchFlag,
    NodeList, Partition, JobName. BatchFlag=0 => interactive job (salloc/srun,
    visualization session): no batch script."""
    out = _run_slurm(["scontrol", "show", "job", job_id])
    if out.startswith("ERROR"):
        return None
    info = {}
    for token in out.replace("\n", " ").split():
        if "=" in token:
            k, _, v = token.partition("=")
            info[k] = v
    return info


def _is_interactive(info: dict) -> bool:
    """A job without a batch script: BatchFlag=0 (salloc/srun, visualization)."""
    return info.get("BatchFlag", "1") == "0"


def job_script(job_id: str) -> str:
    """Submission (sbatch) script of a job, while it is known to the
    controller. Precious for diagnosis (see the requested #SBATCH directives,
    the || true that hide errors, etc.).
    An interactive job (salloc/srun, visualization session) HAS NO batch
    script: the tool says so explicitly instead of returning an error.

    Args:
        job_id: SLURM job identifier
    """
    if not job_id.replace("_", "").replace(".", "").isalnum():
        return "ERROR: invalid job_id"
    # Live interactive job -> no script, that's normal, say it.
    info = _scontrol_info(job_id)
    if info is not None and _is_interactive(info):
        return (
            f"Job {job_id} ({info.get('JobName', '?')}): INTERACTIVE job "
            f"(BatchFlag=0, salloc/srun allocation or visualization session) — "
            f"there is no batch submission script. "
            f"State: {info.get('JobState', '?')}, node: {info.get('NodeList', '?')}."
        )
    out = _run_slurm(["scontrol", "write", "batch_script", job_id, "-"])
    script = ""
    if not out.startswith("ERROR") and out.strip():
        script = out
    if not script:
        return (
            f"Script unavailable for job {job_id}: the controller no longer "
            "knows it (finished/purged job). scontrol only keeps scripts of "
            "live or recently finished jobs."
        )
    modules, paths = _extract_script_info(script)
    hints = []
    if modules:
        hints.append(f"Loaded modules detected: {', '.join(modules)}")
    if paths:
        hints.append(
            f"Referenced paths: {', '.join(paths[:4])} "
            "(checkable with the stat_file tool)"
        )
    result = script[:8000]
    if hints:
        result += "\n\n--- Analysis ---\n" + "\n".join(hints)
    return result


def job_efficiency(job_id: str) -> str:
    """Efficiency of a finished or running job (seff-style): CPU, memory,
    walltime actually used vs requested, with sizing verdicts.

    Args:
        job_id: SLURM job identifier
    """
    if not job_id.replace("_", "").replace(".", "").isalnum():
        return "ERROR: invalid job_id"
    rows = _sacct_rows(job_id)
    if isinstance(rows, str):
        return rows
    main = next((r for r in rows if "." not in r.get("JobID", ".")), rows[0])
    steps = [r for r in rows if "." in r.get("JobID", "")]

    state = main.get("State", "?")
    elapsed_raw = int(main.get("ElapsedRaw") or 0)
    alloc_cpus = int(main.get("AllocCPUS") or 0)
    n_nodes = int(main.get("NNodes") or 1)

    # Context: an interactive job (salloc/srun, visualization) or a job on an
    # interactive partition is idle BY DESIGN (waiting for the user). seff-style
    # CPU/walltime under-utilization verdicts make no sense there and would push
    # toward an unjustified scancel (oversubscribed partitions).
    info = _scontrol_info(job_id)
    partition = (info or {}).get("Partition", "") or main.get("Partition", "")
    is_service = (info is not None and _is_interactive(info)) or partition in INTERACTIVE_PARTITIONS

    lines = [
        f"Job {job_id} — {state} (exit {main.get('ExitCode', '?')})",
        f"Nodes: {main.get('NodeList', '?')} | Allocated CPUs: {alloc_cpus} | "
        f"Elapsed: {main.get('Elapsed', '?')} / Timelimit: {main.get('Timelimit', '?')}",
    ]
    if is_service:
        lines.append(
            "Type: interactive / visualization session — idle by design "
            "(waiting for the user). Interactive partitions are oversubscribed: "
            "low CPU utilization is NORMAL and expected, it is not waste. "
            "Do not recommend reducing CPUs or a scancel."
        )
    verdicts = []

    cpu_used = _parse_slurm_cpu(main.get("TotalCPU", ""))
    if cpu_used is not None and elapsed_raw and alloc_cpus:
        cpu_eff = cpu_used / (elapsed_raw * alloc_cpus)
        lines.append(f"CPU efficiency: {cpu_eff * 100:.1f}% (TotalCPU {main.get('TotalCPU')})")
        # CPU verdict only for actual compute jobs.
        if not is_service and cpu_eff < 0.5 and elapsed_raw > 300:
            verdicts.append(
                f"Under-used CPU ({cpu_eff * 100:.0f}%): reduce the requested CPUs or check the parallelization."
            )

    rss_values = [
        _parse_slurm_mem(r.get("MaxRSS", "")) for r in (steps or [main])
    ]
    rss_bytes = [v[0] for v in rss_values if v]
    req = _parse_slurm_mem(main.get("ReqMem", ""))
    if rss_bytes and req:
        max_rss = max(rss_bytes)
        req_bytes, mode = req
        if mode == "c":
            req_bytes *= max(alloc_cpus, 1)
        elif mode == "n":
            req_bytes *= max(n_nodes, 1)
        mem_eff = max_rss / req_bytes if req_bytes else 0
        lines.append(
            f"Memory efficiency: {mem_eff * 100:.1f}% "
            f"(MaxRSS {_fmt_bytes(max_rss)} / requested {_fmt_bytes(req_bytes)})"
        )
        if not is_service and mem_eff < 0.4 and elapsed_raw > 300:
            verdicts.append(
                f"Over-provisioned memory ({mem_eff * 100:.0f}%): reducing --mem would improve scheduling."
            )
        elif mem_eff > 0.9:
            verdicts.append("Memory close to the limit: OOM risk, increase --mem.")

    timelimit_raw = main.get("TimelimitRaw", "")
    if timelimit_raw.isdigit() and elapsed_raw:
        time_eff = elapsed_raw / (int(timelimit_raw) * 60)
        lines.append(f"Walltime used: {time_eff * 100:.1f}% of the timelimit")
        if "TIMEOUT" in state:
            verdicts.append("Job killed by TIMEOUT: increase --time or checkpoint.")
        elif not is_service and time_eff < 0.2 and elapsed_raw > 600:
            verdicts.append("Heavily over-estimated walltime: a more realistic --time improves backfill priority.")

    if "OUT_OF_MEMORY" in state:
        verdicts.append("Job killed by OOM: increase --mem or reduce the memory footprint.")

    if verdicts:
        lines.append("Verdicts:")
        lines += [f"  - {v}" for v in verdicts]
    elif not is_service:
        lines.append("Sizing looks globally correct.")
    return "\n".join(lines)


def _count_similar_short_jobs(username: str, job_name: str, max_elapsed_s: int = 300,
                              days: int = 7) -> dict | None:
    """Count, through sacct, how many SHORT jobs (elapsed < max_elapsed_s) the
    same user submitted recently, and how many share the same job name. Used
    to detect a 'many short jobs' pattern that would be a candidate for
    grouping (job array). Returns a stats dict or None."""
    if not username:
        return None
    rows = _sacct_window(days, user=username)
    if isinstance(rows, str) or not rows:
        return None
    short = [r for r in rows if 0 < int(r.get("ElapsedRaw") or 0) < max_elapsed_s]
    if not short:
        return None
    same = sum(1 for r in short if job_name and r.get("JobName") == job_name)
    elapsed_sorted = sorted(int(r.get("ElapsedRaw") or 0) for r in short)
    med = elapsed_sorted[len(elapsed_sorted) // 2]
    return {"total_short": len(short), "same_name": same,
            "median_elapsed": med, "days": days, "threshold": max_elapsed_s}


def _short_job_note(elapsed_raw: int, state: str, username: str, job_name: str,
                    is_service: bool) -> str | None:
    """When a job is short, bring the CONTEXT (how many similar short jobs
    recently) instead of judging the job in isolation. One short job is not an
    error; MANY short jobs = grouping candidate. Returns a note or None."""
    # "Short" threshold: 5 min. Service jobs (visualization/interactive) excluded.
    if is_service or elapsed_raw <= 0 or elapsed_raw >= 300:
        return None
    stats = _count_similar_short_jobs(username, job_name, max_elapsed_s=300, days=7)
    if not stats:
        return f"Short job ({elapsed_raw}s). Isolated over the last 7 days, nothing to report."
    n = stats["total_short"]
    same = stats["same_name"]
    if n < 20:
        return (f"Short job ({elapsed_raw}s). {username} submitted {n} short jobs "
                f"(<5min) over 7 days — moderate volume, nothing unusual.")
    # High volume -> grouping candidate
    med = f", median duration ~{stats['median_elapsed']:.0f}s" if stats.get("median_elapsed") else ""
    detail = f" including {same} under the same name '{job_name}'" if job_name and same else ""
    return (f"⚠ Short job ({elapsed_raw}s) AND a volume pattern: {username} submitted "
            f"{n} short jobs (<5min) over 7 days{detail}{med}. "
            "Many short jobs saturate the scheduler for little useful compute — "
            "a candidate for GROUPING (job array, or batching several tasks per job). "
            "Not an error, but a possible efficiency gain for the user and the cluster.")


def diagnose_job(job_id: str, log_lines: int = 30) -> str:
    """Full diagnosis of a job: state/reason, CPU/memory/walltime efficiency,
    last log lines, and node exporter metrics of the nodes during execution
    (if Prometheus is enabled). The all-in-one support tool.

    Args:
        job_id: SLURM job identifier
        log_lines: log lines to fetch per file (default 30)
    """
    if not job_id.replace("_", "").replace(".", "").isalnum():
        return "ERROR: invalid job_id"

    info = _scontrol_info(job_id)
    if info is not None and info.get("JobState") == "PENDING":
        return f"JOB DIAGNOSIS {job_id} — PENDING\n\n" + why_pending(job_id)

    # Strong header: anchors the report on THIS job (prevents a model from
    # recycling another job's data from a previous turn when the result is thin).
    if info is not None:
        header = (
            f"JOB DIAGNOSIS {job_id} — {info.get('JobState', '?')} — "
            f"{info.get('JobName', '?')} — node {info.get('NodeList', '?')} — "
            f"user {info.get('UserId', '?').split('(')[0]}"
        )
    else:
        header = f"JOB DIAGNOSIS {job_id} (finished/purged from the controller)"
    interactive = info is not None and _is_interactive(info)
    if interactive:
        header += "\nType: INTERACTIVE job (salloc/srun/visualization) — no batch script, logs often on a pty."

    sections = [header, "=== Efficiency ===\n" + job_efficiency(job_id)]

    script = job_script(job_id)
    # An interactive job returns an explicit message (not a script): include
    # it but do NOT attempt file analysis / chaining.
    if interactive:
        sections.append("=== Script ===\n" + script)
    elif not script.startswith(("ERROR", "Script unavailable")):
        sections.append("=== Submission script ===\n" + script[:3000])
        # Deterministic chaining: check the existence of the files/directories
        # the script references — a missing input is the #1 cause of jobs that
        # finish within seconds.
        raw_script = script.split("\n\n--- Analysis ---")[0]
        _, paths = _extract_script_info(raw_script)
        checks = []
        for p in paths[:5]:
            real = _contained_path(p)
            if real is None:
                continue
            if os.path.exists(real):
                try:
                    st = os.stat(real)
                    kind = "dir" if os.path.isdir(real) else _fmt_bytes(st.st_size)
                    checks.append(f"  OK      {p} ({kind})")
                except OSError as e:
                    checks.append(f"  ERROR   {p} ({e.strerror})")
            else:
                checks.append(f"  ABSENT  {p}")
        if checks:
            sections.append("=== Files referenced by the script ===\n" + "\n".join(checks))

        # Parallelization analysis: conditional (silent for non-parallel single-core).
        rows_p = _sacct_rows(job_id)
        if not isinstance(rows_p, str):
            m = next((r for r in rows_p if "." not in r.get("JobID", ".")), rows_p[0])
            para = _analyze_parallelism(
                raw_script,
                int(m.get("AllocCPUS") or 0),
                int(m.get("NNodes") or 1),
            )
            if para:
                sections.append("=== Parallelization ===\n" + para)

    logs = job_logs(job_id, log_lines)
    sections.append("=== Logs ===\n" + logs)

    if ENABLE_PROM:
        rows = _sacct_rows(job_id)
        if not isinstance(rows, str):
            main = next((r for r in rows if "." not in r.get("JobID", ".")), rows[0])
            nodelist = main.get("NodeList", "")
            start_s, end_s = main.get("Start", ""), main.get("End", "")
            nodes: list[str] = []
            if nodelist and nodelist not in ("None assigned", "Unknown"):
                if "[" in nodelist:
                    expanded = _run_slurm(["scontrol", "show", "hostnames", nodelist])
                    if not expanded.startswith("ERROR"):
                        nodes = expanded.splitlines()
                else:
                    nodes = [nodelist]
            try:
                start_ts = datetime.fromisoformat(start_s).timestamp()
                end_ts = (
                    datetime.now().timestamp()
                    if end_s in ("Unknown", "", "None")
                    else datetime.fromisoformat(end_s).timestamp()
                )
            except ValueError:
                start_ts = end_ts = 0
            if nodes and start_ts and end_ts > start_ts:
                inst_re = "(" + "|".join(f"{n}:9100" for n in nodes[:4]) + ")"
                metric_lines = []
                for label, q in (
                    ("load5", f'node_load5{{instance=~"{inst_re}"}}'),
                    (
                        "mem_used_pct",
                        f'(node_memory_MemTotal_bytes{{instance=~"{inst_re}"}} '
                        f'- node_memory_MemFree_bytes{{instance=~"{inst_re}"}} '
                        f'- node_memory_Cached_bytes{{instance=~"{inst_re}"}} '
                        f'- node_memory_Buffers_bytes{{instance=~"{inst_re}"}}) '
                        f'/ node_memory_MemTotal_bytes{{instance=~"{inst_re}"}} * 100',
                    ),
                ):
                    data = _prom_get(
                        "query_range",
                        {"query": q, "start": start_ts, "end": end_ts,
                         "step": max(int((end_ts - start_ts) / 100), 60)},
                    )
                    if isinstance(data, str):
                        metric_lines.append(f"{label}: {data}")
                        continue
                    for serie in data["data"]["result"]:
                        inst = serie["metric"].get("instance", "?")
                        floats = [float(v[1]) for v in serie.get("values", []) if v[1] not in ("NaN", None)]
                        if floats:
                            metric_lines.append(
                                f"{label} {inst}: min={min(floats):.1f} max={max(floats):.1f} "
                                f"avg={sum(floats) / len(floats):.1f}"
                            )
                if metric_lines:
                    sections.append(
                        "=== Node metrics during the job ===\n" + "\n".join(metric_lines)
                    )

    # "Short job" note: volume context (grouping candidate), not a judgement
    # of the isolated job. Conditional: only if the job is short and not a
    # service job.
    if not interactive:
        rows_s = _sacct_rows(job_id)
        if not isinstance(rows_s, str):
            m = next((r for r in rows_s if "." not in r.get("JobID", ".")), rows_s[0])
            partition = (info or {}).get("Partition", "") or m.get("Partition", "")
            note = _short_job_note(
                int(m.get("ElapsedRaw") or 0),
                m.get("State", ""),
                m.get("User", "") or (info or {}).get("UserId", "").split("(")[0],
                m.get("JobName", ""),
                is_service=(interactive or partition in INTERACTIVE_PARTITIONS),
            )
            if note:
                sections.append("=== Short jobs / grouping ===\n" + note)

    return "\n\n".join(sections)


def node_jobs(node: str) -> str:
    """SLURM jobs currently running on a given node. The complement of
    node_health: 'node101 is loaded, what is running on it?'.

    Args:
        node: node name (e.g. node101)
    """
    if not _valid_name(node):
        return "ERROR: invalid node name"
    out = _run_slurm([
        "squeue", "-w", node, "--Format",
        "JobID,UserName,Account,Name:30,State,TimeUsed,NumCPUs,MinMemory,tres-per-node",
    ])
    if not out.startswith("ERROR") and len(out.splitlines()) <= 1:
        return f"No job currently running on {node}."
    return out


def reservation_list() -> str:
    """Active SLURM reservations: reserved nodes, allowed accounts/users, time
    windows. Explains the 'why can't I access these nodes' and the jobs stuck
    on ReqNodeNotAvail."""
    out = _run_slurm(["scontrol", "show", "reservation"])
    if "No reservations" in out:
        return "No active reservation."
    return out


def qos_info(qos: str = "", user: str = "") -> str:
    """SLURM QOS limits and associations (sacctmgr). Without argument: every
    QOS with its limits. With qos: that QOS only. With user: the user's
    associations (accounts, allowed QOS, limits). Explains blocks like
    QOSMaxJobsPerUserLimit / AssocGrpTRES.

    Args:
        qos: a QOS name
        user: login — shows their account/QOS associations
    """
    if qos and not _valid_name(qos):
        return "ERROR: invalid QOS name"
    if user and not _valid_name(user):
        return "ERROR: invalid login"
    if user:
        if not _user_resolves(user):
            return _UNRESOLVED_MSG.format(u=user)
        return _run_slurm([
            "sacctmgr", "show", "assoc", f"user={user}", "--parsable2",
            "format=Account,User,Partition,QOS,MaxJobs,MaxSubmit,GrpTRES,MaxTRESPerJob",
        ])
    cmd = ["sacctmgr", "show", "qos"]
    if qos:
        cmd.append(qos)
    cmd += [
        "--parsable2",
        "format=Name,Priority,MaxWall,MaxTRESPerJob,MaxJobsPerUser,MaxSubmitJobsPerUser,MaxTRESPerUser,GrpTRES",
    ]
    return _run_slurm(cmd)


# Fatal SLURM reasons: the job will NEVER start without intervention.
_FATAL_REASONS = {
    "DependencyNeverSatisfied": (
        "dead dependency (the parent job failed or was cancelled) — the job "
        "will never start, scancel required"
    ),
    "JobHeldAdmin": "job HELD by an admin (scontrol release <jobid> to free it)",
    "JobHeldUser": "job HELD by the user (scontrol release <jobid>)",
    "launch_failed_requeued_held": "launch failure, requeued on hold — inspect the node then release",
    "PartitionNodeLimit": "requests more nodes than the partition offers — will never start as is",
    "PartitionTimeLimit": "requested walltime > partition limit — will never start as is",
    "PartitionConfig": "request incompatible with the partition configuration",
    "InvalidQOS": "requested QOS invalid or not allowed",
    "InvalidAccount": "invalid SLURM account",
    "AccountNotAllowed": "account not allowed on this partition",
    "QOSUsageThreshold": "QOS usage threshold reached",
}

# Suspect reasons: may resolve on their own, but worth a check.
_SUSPECT_REASONS = {
    "ReqNodeNotAvail": (
        "required node(s) unavailable (down, drained or reserved) — cross-check "
        "with cluster_status and reservation_list"
    ),
    "BeginTime": "scheduled start time in the future (--begin) — normal if intentional",
    "Licenses": "waiting for licenses",
    "BadConstraints": "constraints (--constraint) currently unsatisfiable",
}


def _array_task_count(job_id: str) -> int:
    """Number of tasks of a PENDING job array shown as '123_[1-100%4]' -> 100.
    Returns 1 for a simple job."""
    m = re.search(r"_\[([^\]]+)\]", job_id)
    if not m:
        return 1
    total = 0
    for part in m.group(1).split("%")[0].split(","):
        part = part.strip()
        if "-" in part:
            try:
                a, b = part.split("-")
                total += int(b) - int(a) + 1
            except ValueError:
                total += 1
        elif part.isdigit():
            total += 1
    return max(total, 1)


def stuck_jobs(partition: str = "", top_users: int = 10) -> str:
    """Detect PENDING jobs stuck in the queue: classify waiting reasons into
    FATAL (the job will NEVER start without intervention:
    DependencyNeverSatisfied, admin/user hold, request impossible for the
    partition, invalid QOS/account), SUSPECT (worth checking: ReqNodeNotAvail,
    licenses, begin time) and NORMAL waiting (Priority/Resources/Dependency —
    simple counter). The fleet-wide counterpart of why_pending: 'are there
    zombie jobs in the queue?'. Job arrays are expanded into task counts.

    Args:
        partition: restrict to a partition (empty = all)
        top_users: number of users listed per reason (default 10)
    """
    cmd = ["squeue", "-t", "PENDING", "--noheader", "-o", "%i|%u|%a|%P|%V|%r"]
    if partition:
        if not _valid_name(partition):
            return "ERROR: invalid partition"
        cmd += ["-p", partition]
    out = _run_slurm(cmd, max_lines=50000)
    if out.startswith("ERROR"):
        return out
    if not out.strip():
        return "No PENDING job" + (f" on {partition}." if partition else ".")

    now = datetime.now()
    fatal: dict[str, dict] = {}    # reason -> user -> stats
    suspect: dict[str, dict] = {}
    normal_counts: dict[str, int] = {}
    n_total = n_fatal = n_suspect = 0

    for line in out.splitlines():
        parts = line.split("|", 5)
        if len(parts) < 6:
            continue
        jobid, user, _account, _part, submit, reason = (p.strip() for p in parts)
        tasks = _array_task_count(jobid)
        n_total += tasks
        tokens = reason.split(",")[0].split()
        token = tokens[0] if tokens else "None"
        if token in _FATAL_REASONS:
            bucket, n_fatal = fatal, n_fatal + tasks
        elif token in _SUSPECT_REASONS:
            bucket, n_suspect = suspect, n_suspect + tasks
        else:
            normal_counts[token] = normal_counts.get(token, 0) + tasks
            continue
        age_h = None
        try:
            age_h = (now - datetime.fromisoformat(submit)).total_seconds() / 3600
        except ValueError:
            pass
        by_user = bucket.setdefault(token, {})
        u = by_user.setdefault(user, {"jobs": 0, "oldest_h": 0.0, "samples": []})
        u["jobs"] += tasks
        if age_h is not None and age_h > u["oldest_h"]:
            u["oldest_h"] = age_h
        if len(u["samples"]) < 4:
            u["samples"].append(jobid)

    lines = [
        f"PENDING queue{f' ({partition})' if partition else ''}: {n_total} task(s) — "
        f"{n_fatal} FATAL, {n_suspect} SUSPECT, {n_total - n_fatal - n_suspect} waiting normally"
    ]

    def _render(bucket: dict, docs: dict, icon: str) -> None:
        for token, by_user in sorted(
            bucket.items(), key=lambda kv: -sum(u["jobs"] for u in kv[1].values())
        ):
            total = sum(u["jobs"] for u in by_user.values())
            lines.append(f"\n{icon} {token} — {total} task(s), {len(by_user)} user(s)")
            lines.append(f"   {docs[token]}")
            ranked = sorted(by_user.items(), key=lambda kv: -kv[1]["jobs"])
            for user, st in ranked[: max(1, min(top_users, 50))]:
                oldest = f", oldest {st['oldest_h']:.0f}h" if st["oldest_h"] else ""
                lines.append(
                    f"     {user:<16} {st['jobs']:>6} task(s){oldest}  "
                    f"e.g.: {', '.join(st['samples'])}"
                )

    if fatal:
        _render(fatal, _FATAL_REASONS, "🔴")
    if suspect:
        _render(suspect, _SUSPECT_REASONS, "⚠")
    if not fatal and not suspect:
        lines.append("No stuck job — the queue is healthy.")
    if normal_counts:
        top_normal = sorted(normal_counts.items(), key=lambda kv: -kv[1])[:6]
        lines.append("\nNormal waiting: " + ", ".join(f"{k}={v}" for k, v in top_normal))
    if fatal:
        lines.append(
            "\n(FATAL = action required: scancel for DependencyNeverSatisfied, "
            "scontrol release for holds, otherwise fix the request. "
            "Detail of a specific job: why_pending(<jobid>).)"
        )
    return "\n".join(lines)


def _mem_ratio_overrides() -> dict[str, float]:
    """Parse HPC_MCP_MEM_RATIOS='cpu:7.8,gpu:4.7' -> {'cpu': 7.8, ...}."""
    out: dict[str, float] = {}
    for pair in os.environ.get("HPC_MCP_MEM_RATIOS", "").split(","):
        if ":" in pair:
            part, _, val = pair.partition(":")
            try:
                out[part.strip()] = float(val)
            except ValueError:
                continue
    return out


def _partition_mem_ratio(partition: str) -> float | str:
    """Hardware GB-of-RAM-per-CPU ratio of a partition. Overridable through
    HPC_MCP_MEM_RATIOS, otherwise computed from sinfo (RealMemory of
    slurm.conf / CPUs per node). Heterogeneous configs: take the highest
    ratio (lenient reading)."""
    overrides = _mem_ratio_overrides()
    if partition in overrides:
        return overrides[partition]
    out = _run_slurm(["sinfo", "-p", partition, "--noheader", "-e", "-o", "%c %m"])
    if out.startswith("ERROR"):
        return out
    ratios = []
    for line in out.splitlines():
        try:
            cpus_s, mem_s = line.split()
            cpus = int(re.sub(r"\D", "", cpus_s) or 0)
            mem_mb = float(re.sub(r"\D", "", mem_s) or 0)
        except ValueError:
            continue
        if cpus and mem_mb:
            ratios.append(mem_mb / 1024 / cpus)
    if not ratios:
        return f"ERROR: partition '{partition}' not found in sinfo"
    return max(ratios)


def memory_misuse_scan(partition: str, factor: float = 2.0, top: int = 15) -> str:
    """Fleet scan of the RUNNING jobs of a partition: spots per-CPU memory
    reservations clearly above the hardware ratio of the partition. Memory is
    THE co-scheduling resource: over-reserving it sterilizes CPUs for other
    jobs. Deliberately lenient (threshold = 2x the hardware ratio by default):
    only the exaggerated is flagged, not the comfortably sized.
    Groups jobs by (user, mem/CPU).

    Args:
        partition: partition to scan (typically the main compute partition)
        factor: flagging threshold = factor x hardware GB/CPU ratio (default 2.0)
        top: number of (user, mem/CPU) groups listed (default 15)
    """
    if not _valid_name(partition):
        return "ERROR: invalid partition"
    ratio = _partition_mem_ratio(partition)
    if isinstance(ratio, str):
        return ratio
    factor = max(1.0, min(factor, 20.0))
    threshold = ratio * factor

    out = _run_slurm(
        ["squeue", "-t", "RUNNING", "-p", partition, "--noheader",
         "--Format", "JobID:16,UserName:16,tres-alloc:90,Name:30"],
        max_lines=50000,
    )
    if out.startswith("ERROR"):
        return out
    if not out.strip():
        return f"No RUNNING job on {partition}."

    mult = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0, "": 1 / 1024}
    groups: dict[tuple, dict] = {}  # (user, round(mem/cpu)) -> stats
    n_jobs = n_flagged = 0
    excess_total = 0.0

    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        jobid, user, tres = parts[0], parts[1], parts[2]
        m_cpu = re.search(r"cpu=(\d+)", tres)
        m_mem = re.search(r"mem=([\d.]+)([KMGT]?)", tres)
        if not m_cpu or not m_mem:
            continue
        n_jobs += 1
        cpus = int(m_cpu.group(1))
        if not cpus:
            continue
        mem_gb = float(m_mem.group(1)) * mult[m_mem.group(2)]
        mem_cpu = mem_gb / cpus
        if mem_cpu <= threshold:
            continue
        n_flagged += 1
        excess = mem_gb - ratio * cpus
        excess_total += excess
        g = groups.setdefault(
            (user, round(mem_cpu)), {"jobs": 0, "excess": 0.0, "samples": []}
        )
        g["jobs"] += 1
        g["excess"] += excess
        if len(g["samples"]) < 3:
            g["samples"].append(jobid)

    lines = [
        f"Partition {partition} — hardware ratio ≈ {ratio:.1f} GB/CPU, "
        f"flagging threshold {threshold:.1f} GB/CPU (x{factor:g})",
        f"{n_jobs} RUNNING job(s) analyzed, {n_flagged} above the threshold",
    ]
    if not groups:
        lines.append("Nothing exaggerated — memory reservations look reasonable.")
        return "\n".join(lines)
    lines.append(
        f"Over-reserved memory ≈ {_fmt_bytes(excess_total * 1024**3)} "
        f"(equivalent to ~{excess_total / ratio:.0f} sterilized CPUs)"
    )
    lines.append("")
    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["excess"])
    for (user, mem_cpu), g in ranked[: max(1, min(top, 50))]:
        lines.append(
            f"  {g['jobs']:>4} job(s)  {user:<16} ~{mem_cpu:>3} GB/CPU reserved "
            f"(x{mem_cpu / ratio:.1f})  excess {_fmt_bytes(g['excess'] * 1024**3):>11}  "
            f"e.g.: {', '.join(g['samples'])}"
        )
    lines.append(
        "\n(A high mem/CPU is not necessarily an error — some codes have a real "
        "memory need. Confirm actual waste with job_efficiency(<jobid>) "
        "(MaxRSS vs requested) before contacting the user.)"
    )
    return "\n".join(lines)


def latency_probes() -> str:
    """Micro-benchmarks of cluster responsiveness: response time of squeue
    (SLURM controller), of 'module avail' (filesystem metadata), of an ls on a
    data root, and of a DNS resolution. Strictly READ-ONLY — write probes
    (touch) are deliberately excluded from this server. The reflex when 'the
    cluster feels slow': locates whether the slowness comes from SLURM, the
    filesystem or the network; complement with gpfs_health."""
    import time as _time

    ls_path = PROBE_LS_PATH or READ_ROOTS[0]
    probes = [
        ("squeue (SLURM controller)", ["squeue", "--noheader", "-t", "RUNNING"], 1.0),
        ("module avail (FS metadata)", ["bash", "-lc", "module avail"], 5.0),
        (f"ls {ls_path} (FS read)", ["ls", ls_path], 1.0),
    ]
    if PROBE_DNS_HOST:
        probes.append((f"nslookup {PROBE_DNS_HOST} (DNS)", ["nslookup", PROBE_DNS_HOST], 0.05))
    lines = ["Access times:"]
    global_ok = True
    for label, cmd, threshold in probes:
        t0 = _time.perf_counter()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=CMD_TIMEOUT)
        except FileNotFoundError:
            lines.append(f"  {label:<36} SKIP (command not found)")
            continue
        except subprocess.TimeoutExpired:
            lines.append(f"  {label:<36} 🔴 KO — timeout > {CMD_TIMEOUT}s (threshold {threshold:g} s)")
            global_ok = False
            continue
        elapsed = _time.perf_counter() - t0
        stderr = (r.stderr or "").lower()
        if r.returncode != 0 and "not found" in stderr:
            # e.g. the module function is not initialized for this user — not an infra failure
            lines.append(f"  {label:<36} SKIP (unavailable on this node)")
            continue
        ok = elapsed < threshold and r.returncode == 0
        if not ok:
            global_ok = False
        note = ""
        if r.returncode != 0:
            err = (r.stderr or r.stdout).strip().splitlines()
            note = f"  (rc={r.returncode}: {err[0][:60] if err else '?'})"
        lines.append(
            f"  {label:<36} {elapsed:>7.3f} s  (threshold {threshold:g} s)  "
            f"{'OK' if ok else '🔴 KO'}{note}"
        )
    lines.append(
        "Overall: " + ("OK" if global_ok
                       else "🔴 at least one probe KO — dig with gpfs_health / cluster_status")
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prometheus (optional basic auth and TLS verification)
# Standard exporters only: node exporter (:9100) and NVIDIA DCGM exporter
# (DCGM_FI_DEV_* metrics). Nothing site-specific.
# ---------------------------------------------------------------------------


def _prom_auth() -> tuple[str, str] | None:
    user, password = os.environ.get("HPC_MCP_PROM_USER"), os.environ.get("HPC_MCP_PROM_PASS")
    if user and password:
        return (user, password)
    try:
        with open(PROM_AUTH_FILE, "r", encoding="utf-8") as f:
            user, _, password = f.readline().strip().partition(":")
        if user and password:
            return (user, password)
    except FileNotFoundError:
        pass
    return None


def _prom_get(endpoint: str, params: dict) -> dict | str:
    try:
        r = requests.get(
            f"{PROM_URL}/api/v1/{endpoint}",
            params=params,
            auth=_prom_auth(),
            verify=PROM_VERIFY,
            timeout=CMD_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return f"ERROR Prometheus ({PROM_URL}): {e}"
    if data.get("status") != "success":
        return f"ERROR PromQL: {json.dumps(data, ensure_ascii=False)}"
    return data


def _prom_vector(promql: str) -> list[dict] | str:
    """Instant query, returns the list of series or an error string."""
    data = _prom_get("query", {"query": promql})
    if isinstance(data, str):
        return data
    return data["data"]["result"]


def _prom_scalar(promql: str) -> float | None:
    data = _prom_get("query", {"query": promql})
    if isinstance(data, str):
        return None
    res = data["data"]["result"]
    if not res:
        return None
    try:
        return float(res[0]["value"][1])
    except (KeyError, ValueError, IndexError):
        return None


def prometheus_query(promql: str) -> str:
    """Run an instant PromQL query against the cluster's Prometheus.

    Tip: for slowly scraped metrics, wrapping in last_over_time(...[6h])
    avoids gaps. Nodes appear in the node exporter as instance="<node>:9100".

    Args:
        promql: the PromQL query
    """
    data = _prom_get("query", {"query": promql})
    if isinstance(data, str):
        return data
    res = data["data"]["result"]
    if not res:
        return "No result (empty series). Check the metric name or wrap in last_over_time(...[6h])."
    lines = []
    for serie in res[:50]:
        metric = serie.get("metric", {})
        value = serie.get("value", [None, "?"])[1]
        label_str = ",".join(f'{k}="{v}"' for k, v in metric.items() if k != "__name__")
        name = metric.get("__name__", "")
        lines.append(f"{name}{{{label_str}}} = {value}")
    if len(res) > 50:
        lines.append(f"... ({len(res) - 50} more series truncated)")
    return "\n".join(lines)


def prometheus_range(promql: str, hours: int = 24, step: str = "30m") -> str:
    """PromQL query over a time range (query_range), compact output
    (first/min/max/avg/last per series). If the result is empty, retries
    automatically with step=5m.

    Args:
        promql: the PromQL query
        hours: history depth in hours (default 24)
        step: resolution (default "30m")
    """
    end = datetime.now()
    start = end - timedelta(hours=hours)
    params = {
        "query": promql,
        "start": start.timestamp(),
        "end": end.timestamp(),
        "step": step,
    }
    data = _prom_get("query_range", params)
    if isinstance(data, str):
        return data
    results = data["data"]["result"]

    if not results and step != "5m":
        params["step"] = "5m"
        data = _prom_get("query_range", params)
        if isinstance(data, str):
            return data
        results = data["data"]["result"]

    if not results:
        return "No result over the requested range (including with step=5m)."
    lines = []
    for serie in results[:10]:
        metric = serie.get("metric", {})
        label_str = ",".join(f'{k}="{v}"' for k, v in metric.items() if k != "__name__")
        name = metric.get("__name__", "")
        floats = [float(v[1]) for v in serie.get("values", []) if v[1] not in ("NaN", None)]
        if not floats:
            continue
        lines.append(
            f"{name}{{{label_str}}}: first={floats[0]:.3g} min={min(floats):.3g} "
            f"max={max(floats):.3g} avg={sum(floats)/len(floats):.3g} last={floats[-1]:.3g} "
            f"(n={len(floats)})"
        )
    if len(results) > 10:
        lines.append(f"... ({len(results) - 10} more series truncated)")
    return "\n".join(lines)


def node_health(node: str) -> str:
    """Instant health of a node through node exporter: load, CPU, memory,
    network, uptime. Works for every node type (compute, visualization, login,
    GPU...).

    Args:
        node: node name (e.g. node066, visu01, gpu08)
    """
    if not _valid_name(node):
        return "ERROR: invalid node name"
    inst = f"{node}:9100"
    load1 = _prom_scalar(f'node_load1{{instance="{inst}"}}')
    if load1 is None:
        # Not found in the exporter ≠ down: exporter stopped, node outside
        # monitoring, or wrong name. Cross-check with SLURM to lift the doubt
        # instead of letting the void be read as a state.
        msg = (f"Node {node} not found in the node exporter (instance {inst}). "
               f"CAUTION: this does NOT mean the node is down — the exporter "
               f"may be stopped, the node outside monitoring, or the name wrong.")
        if ENABLE_SLURM:
            sl = _run_slurm(["sinfo", "-h", "-n", node, "-o", "%N|%T|%E"])
            if sl.startswith("ERROR") or not sl.strip():
                return (msg + f" On the SLURM side, '{node}' is also unknown to "
                        f"sinfo — the node probably does not exist under this "
                        f"name (check the spelling). No failure state can be "
                        f"asserted about a nonexistent node.")
            return (msg + f" SLURM however does know it "
                    f"(NodeList|State|Reason):\n  {sl.strip()}\n"
                    f"Metrics unavailable but SLURM state above: rely on it "
                    f"(and cluster_status for the overall view).")
        return msg + " Check the name, and the SLURM state through cluster_status."
    load5 = _prom_scalar(f'node_load5{{instance="{inst}"}}')
    cpu_pct = _prom_scalar(
        f'100 - (avg(irate(node_cpu_seconds_total{{mode="idle",instance="{inst}"}}[5m])) * 100)'
    )
    mem_total = _prom_scalar(f'node_memory_MemTotal_bytes{{instance="{inst}"}}')
    mem_used = _prom_scalar(
        f'node_memory_MemTotal_bytes{{instance="{inst}"}} '
        f'- node_memory_MemFree_bytes{{instance="{inst}"}} '
        f'- (node_memory_Cached_bytes{{instance="{inst}"}} + node_memory_Buffers_bytes{{instance="{inst}"}})'
    )
    net_rx = _prom_scalar(f'sum(irate(node_network_receive_bytes_total{{instance="{inst}"}}[5m])) * 8')
    net_tx = _prom_scalar(f'sum(irate(node_network_transmit_bytes_total{{instance="{inst}"}}[5m])) * 8')
    uptime = _prom_scalar(f'time() - node_boot_time_seconds{{instance="{inst}"}}')

    def _f(v, fmt="{:.1f}", fallback="?"):
        return fmt.format(v) if v is not None else fallback

    mem_str = "?"
    if mem_used is not None and mem_total:
        mem_str = f"{_fmt_bytes(mem_used)} / {_fmt_bytes(mem_total)} ({mem_used / mem_total * 100:.0f}%)"
    return (
        f"Node {node}:\n"
        f"  Load 1m/5m: {_f(load1)} / {_f(load5)}\n"
        f"  CPU: {_f(cpu_pct)}%\n"
        f"  Memory: {mem_str}\n"
        f"  Network: rx {_f((net_rx or 0) / 1e6)} Mb/s, tx {_f((net_tx or 0) / 1e6)} Mb/s\n"
        f"  Uptime: {_f((uptime or 0) / 86400)} days"
    )


def top_loaded_nodes(pattern: str = ".*", top: int = 10) -> str:
    """The most loaded nodes (node_load5) among those matching a pattern.

    Args:
        pattern: regex on the node exporter instance (e.g. node.*, visu.*, gpu.*)
        top: number of nodes (default 10)
    """
    if not _valid_pattern(pattern):
        return "ERROR: invalid pattern"
    q = f'topk({max(1, min(top, 50))}, node_load5{{instance=~"{pattern}(:9100)?"}})'
    data = _prom_get("query", {"query": q})
    if isinstance(data, str):
        return data
    res = data["data"]["result"]
    if not res:
        return f"No node matches '{pattern}' in the node exporter."
    lines = [f"Most loaded nodes (load5, pattern {pattern}):"]
    for serie in sorted(res, key=lambda s: -float(s["value"][1])):
        inst = serie["metric"].get("instance", "?").removesuffix(":9100")
        lines.append(f"  {inst:<20} {float(serie['value'][1]):>8.2f}")
    return "\n".join(lines)


def _dcgm_host_of(metric: dict) -> str:
    """Best-effort host label of a DCGM series (dcgm-exporter sets Hostname;
    fall back to the scrape instance)."""
    return (metric.get("Hostname")
            or metric.get("hostname")
            or metric.get("instance", "?").split(":")[0])


def gpu_status(node: str = "", top: int = 10) -> str:
    """GPU state through the standard NVIDIA DCGM exporter (DCGM_FI_DEV_*
    metrics). Without argument: cluster-wide view (GPU count, utilization
    distribution, most and least loaded hosts, recent XID errors). With node:
    per-GPU detail of that host (utilization, framebuffer memory, temperature,
    power, model).

    Args:
        node: host name for the detailed view (empty = cluster overview)
        top: number of hosts listed in the overview (default 10)
    """
    if node and not _valid_name(node):
        return "ERROR: invalid node name"

    if node:
        sel = f'Hostname=~"{node}(\\\\..*)?"'
        util = _prom_vector(f"DCGM_FI_DEV_GPU_UTIL{{{sel}}}")
        if isinstance(util, str):
            return util
        if not util:
            # Not found ≠ no GPU: exporter absent from that host, or wrong name.
            return (f"No DCGM metric for host '{node}'. CAUTION: this does NOT "
                    f"prove the host has no GPUs — the DCGM exporter may not run "
                    f"there, or the name may be wrong. Cross-check with "
                    f"node_health('{node}') and gpu_status() for the hosts that "
                    f"do expose GPU metrics.")

        def _per_gpu(metric_name: str) -> dict[str, float]:
            res = _prom_vector(f"{metric_name}{{{sel}}}")
            out: dict[str, float] = {}
            if isinstance(res, list):
                for s in res:
                    try:
                        out[s["metric"].get("gpu", "?")] = float(s["value"][1])
                    except (KeyError, ValueError):
                        continue
            return out

        fb_used = _per_gpu("DCGM_FI_DEV_FB_USED")
        fb_free = _per_gpu("DCGM_FI_DEV_FB_FREE")
        temp = _per_gpu("DCGM_FI_DEV_GPU_TEMP")
        power = _per_gpu("DCGM_FI_DEV_POWER_USAGE")
        lines = [f"GPUs on {node} ({len(util)} GPU(s)):"]
        for s in sorted(util, key=lambda s: s["metric"].get("gpu", "")):
            gpu = s["metric"].get("gpu", "?")
            model = s["metric"].get("modelName", "")
            u = float(s["value"][1])
            used = fb_used.get(gpu)
            total = (fb_used.get(gpu, 0) or 0) + (fb_free.get(gpu, 0) or 0)
            mem = (f"{used / 1024:.1f}/{total / 1024:.1f} GiB"
                   if used is not None and total else "?")
            t = f"{temp[gpu]:.0f}°C" if gpu in temp else "?"
            p = f"{power[gpu]:.0f}W" if gpu in power else "?"
            lines.append(f"  GPU {gpu}: util {u:>3.0f}%  FB {mem:>18}  {t:>6}  {p:>6}  {model}")
        return "\n".join(lines)

    # Cluster overview
    util = _prom_vector("DCGM_FI_DEV_GPU_UTIL")
    if isinstance(util, str):
        return util
    if not util:
        return ("No DCGM_FI_DEV_GPU_UTIL metric in Prometheus. Either the "
                "cluster has no DCGM exporter deployed, or it is down — this "
                "says nothing about the GPUs themselves.")
    values = []
    by_host: dict[str, list[float]] = {}
    for s in util:
        try:
            v = float(s["value"][1])
        except (KeyError, ValueError):
            continue
        values.append(v)
        by_host.setdefault(_dcgm_host_of(s["metric"]), []).append(v)
    n = len(values)
    idle = sum(1 for v in values if v < 5)
    busy = sum(1 for v in values if v > 80)
    avg = sum(values) / n if n else 0
    lines = [
        f"GPU overview (DCGM): {n} GPU(s) on {len(by_host)} host(s)",
        f"  Average utilization: {avg:.0f}% — {busy} GPU(s) >80%, {idle} idle (<5%)",
    ]
    ranked = sorted(by_host.items(), key=lambda kv: -(sum(kv[1]) / len(kv[1])))
    top = max(1, min(top, 50))
    lines.append("  Busiest hosts (avg util):")
    for host, vals in ranked[:top]:
        lines.append(f"    {host:<24} {sum(vals) / len(vals):>5.0f}%  ({len(vals)} GPU)")
    if idle:
        lines.append("  Idle GPUs by host:")
        for host, vals in sorted(by_host.items()):
            n_idle = sum(1 for v in vals if v < 5)
            if n_idle:
                lines.append(f"    {host:<24} {n_idle} idle GPU(s)")
    xid = _prom_vector("increase(DCGM_FI_DEV_XID_ERRORS[24h]) > 0")
    if isinstance(xid, list) and xid:
        lines.append("  ⚠ XID errors over 24h (hardware/driver incidents):")
        for s in xid[:10]:
            lines.append(f"    {_dcgm_host_of(s['metric']):<24} GPU {s['metric'].get('gpu', '?')}: "
                         f"{float(s['value'][1]):.0f} error(s)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Composites — user / account overviews
# Chain the tools in the right order with the anti-false-empty guards,
# detect CLUSTERS of similar failing jobs (normalized name x state) and go
# read the log of one representative job per cluster. Bounded output.
# ---------------------------------------------------------------------------

_DIGITS_RE = re.compile(r"\d+")
_BAD_STATES = ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")


def _norm_job_name(name: str) -> str:
    """Normalize a job name for clustering: digit runs become '#'
    ('VA259_02_mission_123' -> 'VA#_#_mission_#')."""
    return _DIGITS_RE.sub("#", (name or "?"))[:48]


def _sacct_window(days: int, user: str = "", account: str = "") -> list[dict] | str:
    """Finished jobs (sacct -X) over a window, as a list of dicts."""
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    cmd = ["sacct", "--parsable2", "-X", "-S", start, "--format",
           "JobID,User,JobName,Partition,State,ElapsedRaw,AllocTRES,End,ExitCode"]
    if account:
        cmd += ["-A", account, "--allusers"]
        if user:
            cmd += ["-u", user]
    elif user:
        cmd += ["-u", user]
    out = _run_slurm(cmd, max_lines=100000)
    if out.startswith("ERROR"):
        return out
    return _parsable_to_dicts(out)


def _row_cpuh(r: dict) -> float:
    """CPU-hours of one sacct row (cpu= of AllocTRES x ElapsedRaw)."""
    m = re.search(r"(?:^|,)cpu=(\d+)", r.get("AllocTRES", "") or "")
    try:
        return int(m.group(1)) * int(r.get("ElapsedRaw") or 0) / 3600 if m else 0.0
    except ValueError:
        return 0.0


def _state_of(r: dict) -> str:
    """First token of State ('CANCELLED by 123' -> 'CANCELLED'), robust to
    empty or blank values: returns '?' instead of raising IndexError
    (historical account_overview/user_overview bug on sacct rows with an
    empty State)."""
    return ((r.get("State") or "").split() or ["?"])[0]


def _job_clusters(rows: list[dict], min_size: int = 5, top: int = 3) -> list[dict]:
    """Clusters of SIMILAR failing jobs: grouped by (normalized name,
    FAILED/TIMEOUT/OOM/NODE_FAIL state). Returns the biggest clusters, each
    with its most recent job as a sample whose log can be read."""
    groups: dict[tuple, dict] = {}
    for r in rows:
        state = _state_of(r)
        key_state = next((b for b in _BAD_STATES if state.startswith(b)), None)
        if key_state is None:
            continue
        key = (_norm_job_name(r.get("JobName", "")), key_state)
        g = groups.setdefault(key, {"n": 0, "users": set(), "exit": {},
                                    "sample": "", "sample_end": ""})
        g["n"] += 1
        g["users"].add(r.get("User") or "?")
        ec = r.get("ExitCode") or "?"
        g["exit"][ec] = g["exit"].get(ec, 0) + 1
        if (r.get("End") or "") >= g["sample_end"]:
            g["sample_end"] = r.get("End") or ""
            g["sample"] = r.get("JobID") or ""
    clusters = [{"pattern": k[0], "state": k[1], **v}
                for k, v in groups.items() if v["n"] >= min_size]
    clusters.sort(key=lambda c: -c["n"])
    return clusters[:top]


def _clusters_section(clusters: list[dict], with_logs: bool,
                      show_users: bool, log_lines: int = 12) -> list[str]:
    """Render the clusters section + log excerpts of the representative jobs
    (2 clusters max to bound the output)."""
    if not clusters:
        return []
    out = ["=== Failure clusters (similar jobs) ==="]
    for c in clusters:
        exits = ", ".join(f"{k}x{v}" for k, v in
                          sorted(c["exit"].items(), key=lambda kv: -kv[1])[:3])
        who = f"  users: {', '.join(sorted(c['users'])[:5])}" if show_users else ""
        out.append(f"  {c['n']:>4} x {c['state']:<13} '{c['pattern']}'  "
                   f"exit: {exits}{who}  e.g.: {c['sample']}")
    if with_logs:
        for c in clusters[:2]:
            if not c["sample"]:
                continue
            log = job_logs(c["sample"], lines=log_lines)
            out.append(f"\n--- Log of job {c['sample']} (representative of "
                       f"'{c['pattern']}', {c['n']} jobs in {c['state']}) ---\n"
                       + log[:2200])
    return out


def user_overview(login: str, days: int = 7, with_logs: bool = True) -> str:
    """Full overview of a user: running jobs, history summary, clusters of
    similar failing jobs WITH a log excerpt of one representative job per
    cluster, and fairshare. THE entry point for 'give me the state of
    <login>' — chains the tools in the right order with the anti-false-empty
    guards (prior NSS resolution: the empty sections of this report are
    RELIABLE).

    Args:
        login: user login
        days: history window in days (default 7)
        with_logs: log excerpt per failure cluster (default True)
    """
    if not _valid_name(login):
        return "ERROR: invalid login"
    if not _user_resolves(login):
        return _UNRESOLVED_MSG.format(u=login)

    sections = [f"OVERVIEW — user {login} (last {days} days) "
                f"[login resolved through NSS: empty sections are reliable]"]

    # Running jobs
    sq = _run_slurm(["squeue", "-u", login, "--noheader",
                     "-o", "%i|%P|%j|%T|%M|%C|%m"])
    if sq.startswith("ERROR"):
        sections.append("=== Running jobs ===\n" + sq)
    elif not sq.strip():
        sections.append("=== Running jobs ===\nNone.")
    else:
        rows_sq = [l.split("|") for l in sq.splitlines() if l.strip()]
        body = [f"  {p[0]:<12} {p[1]:<14} {p[2][:26]:<26} {p[3]:<9} {p[4]:>11}  {p[5]}c {p[6]}"
                for p in rows_sq[:12] if len(p) >= 7]
        extra = f"\n  ... ({len(rows_sq) - 12} more)" if len(rows_sq) > 12 else ""
        sections.append(f"=== Running jobs ({len(rows_sq)}) ===\n" + "\n".join(body) + extra)

    # History
    rows = _sacct_window(days, user=login)
    clusters: list[dict] = []
    if isinstance(rows, str):
        sections.append(f"=== History {days}d ===\n" + rows)
    elif not rows:
        sections.append(f"=== History {days}d ===\nNo job over the window.")
    else:
        states: dict[str, int] = {}
        parts: dict[str, int] = {}
        cpuh = 0.0
        for r in rows:
            s = _state_of(r)
            states[s] = states.get(s, 0) + 1
            p = r.get("Partition") or "?"
            parts[p] = parts.get(p, 0) + 1
            cpuh += _row_cpuh(r)
        st = ", ".join(f"{k}={v}" for k, v in sorted(states.items(), key=lambda kv: -kv[1]))
        pt = ", ".join(f"{k}={v}" for k, v in sorted(parts.items(), key=lambda kv: -kv[1])[:6])
        sections.append(f"=== History {days}d ===\n{len(rows)} jobs, "
                        f"{cpuh:,.0f} CPU·h — {st}\nPartitions: {pt}")
        clusters = _job_clusters(rows)

    sections += _clusters_section(clusters, with_logs, show_users=False)

    # Fairshare (compact)
    fs = fairshare(user=login)
    sections.append("=== Fairshare ===\n" + "\n".join(fs.splitlines()[:8]))

    out = "\n\n".join(sections)
    return out[:14000] + ("\n... (truncated)" if len(out) > 14000 else "")


def account_overview(account: str, days: int = 7, with_logs: bool = True) -> str:
    """Overview of a SLURM account: members, running jobs, top members by
    CPU-hours over the window, and clusters of similar failing jobs WITH a
    log excerpt of one representative job. THE entry point for 'state of
    account X'. Anti-false-empty guard: an account unknown to sacctmgr
    returns an explicit error.

    Args:
        account: SLURM account (e.g. physics, support)
        days: history window in days (default 7)
        with_logs: log excerpt per failure cluster (default True)
    """
    if not _valid_name(account):
        return "ERROR: invalid account"
    assoc = _run_slurm(["sacctmgr", "-n", "show", "assoc", f"account={account}",
                        "--parsable2", "format=User"])
    if assoc.startswith("ERROR"):
        return assoc
    if not assoc.strip():
        return (f"ERROR: account '{account}' unknown to sacctmgr (no "
                "association) — check the name. An empty result on a "
                "nonexistent account is not 'no activity'.")
    members = sorted({l.strip().rstrip("|") for l in assoc.splitlines()} - {""})

    sections = [f"OVERVIEW — account {account} (last {days} days) — "
                f"{len(members)} member(s): {', '.join(members[:15])}"
                + (" ..." if len(members) > 15 else "")]

    # Running jobs
    sq = _run_slurm(["squeue", "-A", account, "--noheader", "-o", "%i|%u|%P|%T"])
    if sq.startswith("ERROR"):
        sections.append("=== Running jobs ===\n" + sq)
    elif not sq.strip():
        sections.append("=== Running jobs ===\nNone.")
    else:
        st_c: dict[str, int] = {}
        u_c: dict[str, int] = {}
        n = 0
        for l in sq.splitlines():
            p = l.split("|")
            if len(p) < 4:
                continue
            n += 1
            st_c[p[3]] = st_c.get(p[3], 0) + 1
            u_c[p[1]] = u_c.get(p[1], 0) + 1
        st = ", ".join(f"{k}={v}" for k, v in sorted(st_c.items(), key=lambda kv: -kv[1]))
        top_u = ", ".join(f"{k}({v})" for k, v in sorted(u_c.items(), key=lambda kv: -kv[1])[:8])
        sections.append(f"=== Running jobs ({n}) ===\nStates: {st}\nBy user: {top_u}")

    # History + top members by CPU-hours
    rows = _sacct_window(days, account=account)
    clusters: list[dict] = []
    if isinstance(rows, str):
        sections.append(f"=== History {days}d ===\n" + rows)
    elif not rows:
        sections.append(f"=== History {days}d ===\nNo job over the window.")
    else:
        states: dict[str, int] = {}
        by_user: dict[str, dict] = {}
        for r in rows:
            s = _state_of(r)
            states[s] = states.get(s, 0) + 1
            u = r.get("User") or "?"
            d = by_user.setdefault(u, {"jobs": 0, "cpuh": 0.0})
            d["jobs"] += 1
            d["cpuh"] += _row_cpuh(r)
        st = ", ".join(f"{k}={v}" for k, v in sorted(states.items(), key=lambda kv: -kv[1]))
        top_m = sorted(by_user.items(), key=lambda kv: -kv[1]["cpuh"])[:8]
        u_lines = [f"    {u:<16} {d['cpuh']:>9,.0f} CPU·h  {d['jobs']:>6,} jobs"
                   for u, d in top_m]
        sections.append(f"=== History {days}d ===\n{len(rows)} jobs — {st}\n"
                        "  Top members (CPU·h):\n" + "\n".join(u_lines))
        clusters = _job_clusters(rows)

    sections += _clusters_section(clusters, with_logs, show_users=True)

    out = "\n\n".join(sections)
    return out[:14000] + ("\n... (truncated)" if len(out) > 14000 else "")


_GROUPS = {
    "gpfs": (ENABLE_GPFS, [gpfs_filesets_list, gpfs_fileset_quota, gpfs_all_quotas, gpfs_df,
                           gpfs_health, gpfs_fileset_path, filesystem_usage, tail_file, stat_file,
                           list_dir, grep_file]),
    "es": (ENABLE_ES, [es_indices, es_fields, es_search, es_aggregate, es_tail_logs]),
    "slurm": (ENABLE_SLURM, [squeue_jobs, job_details, why_pending, sacct_history,
                             gpu_usage_by_user, cluster_status, node_jobs, reservation_list,
                             qos_info, fairshare, job_priority, latency_probes,
                             job_logs, stuck_jobs, memory_misuse_scan, job_script, job_efficiency, diagnose_job,
                             user_overview, account_overview]),
    "prom": (ENABLE_PROM, [prometheus_query, prometheus_range, node_health,
                           top_loaded_nodes, gpu_status]),
}

for group, (enabled, funcs) in _GROUPS.items():
    if enabled:
        for func in funcs:
            mcp.tool()(func)


def main() -> None:
    """Console entry point (`hpc-mcp`)."""
    if os.environ.get("HPC_MCP_TRANSPORT", "http").lower() == "stdio":
        mcp.run(transport="stdio")
        return

    enabled = [g for g, (on, _) in _GROUPS.items() if on]
    n_tools = sum(len(funcs) for on, funcs in _GROUPS.values() if on)
    auth = "Bearer token ACTIVE" if MCP_AUTH_TOKEN else "auth DISABLED (HPC_MCP_AUTH_TOKEN empty)"
    print(f"hpc-mcp -> http://{HOST}:{PORT}/mcp | groups: {', '.join(enabled)} ({n_tools} tools) | {auth}")

    import uvicorn

    app = _AuthMiddleware(mcp.streamable_http_app())
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
