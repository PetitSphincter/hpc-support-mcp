"""Smoke tests: pure helpers + tool registration (no cluster access)."""
import asyncio

from hpc_mcp import server


def test_tools_registered():
    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert len(tools) == 41
    for expected in ("diagnose_job", "user_overview", "account_overview",
                     "stuck_jobs", "latency_probes", "gpfs_health", "node_health",
                     "es_indices", "es_fields", "es_search", "es_aggregate",
                     "es_tail_logs", "gpu_status"):
        assert expected in names


def test_parse_slurm_cpu():
    assert server._parse_slurm_cpu("01:00:00") == 3600.0
    assert server._parse_slurm_cpu("1-01:00:00") == 90000.0
    assert server._parse_slurm_cpu("00:00:30") == 30.0
    assert server._parse_slurm_cpu("") is None


def test_parse_slurm_mem():
    val = server._parse_slurm_mem("4000Mc")
    assert val is not None and val[1] == "c"
    assert server._parse_slurm_mem("") is None


def test_fmt_bytes():
    assert "GiB" in server._fmt_bytes(1024**3)


def test_norm_job_name():
    assert server._norm_job_name("VA259_02_mission_123") == "VA#_#_mission_#"
    assert server._norm_job_name("") == "?"


def test_valid_name():
    assert server._valid_name("node066")
    assert server._valid_name("l2p_dev1")
    assert not server._valid_name("a;rm -rf")
    assert not server._valid_name("")


def test_pattern_accepts_underscore():
    assert server._valid_pattern("l2p_dev1")
    assert server._valid_pattern("node.*")
    assert not server._valid_pattern('a"b')


def test_index_validation():
    assert server._check_index("logs-*") is None
    assert server._check_index("Bad Index") is not None
    assert server._check_index("") is not None


def test_index_allowlist(monkeypatch):
    monkeypatch.setattr(server, "ES_ALLOWED_INDICES", ["logs-*", "app"])
    assert server._index_allowed("logs-2026.08")
    assert server._index_allowed("app")
    assert not server._index_allowed("secrets")
    monkeypatch.setattr(server, "ES_ALLOWED_INDICES", [])
    assert server._index_allowed("anything")


def test_extract_script_info():
    script = "#!/bin/bash\nmodule load python/3.11 gcc\nsrun ./run.sh\n"
    modules, _paths = server._extract_script_info(script)
    assert "python/3.11" in modules and "gcc" in modules


def test_array_task_count():
    assert server._array_task_count("123_[1-100%4]") == 100
    assert server._array_task_count("123_[1,3,5]") == 3
    assert server._array_task_count("456") == 1


def test_contained_path_confinement():
    root = server.READ_ROOTS[0]
    assert server._contained_path(f"{root}/sub/file.log") is not None
    assert server._contained_path("/etc/passwd") is None
    assert server._contained_path(f"{root}/../etc/passwd") is None


def test_error_messages_are_english():
    assert server._run(["definitely-not-a-command-xyz"]).startswith("ERROR")
    assert "ERROR" in server._UNRESOLVED_MSG

