import json
import os
import stat

import pytest

from archx.architecture import create_architecture_dict
from archx.interface import query_interface
from archx.interface.gem5.gem5 import generate_config, run_gem5
from archx.workload import create_workload_dict

design_dir = os.path.join(
    os.path.dirname(__file__), "..", "zoo", "agraph", "designs", "riscv_mcpat"
)


def load_design():
    architecture_dict = create_architecture_dict(
        os.path.join(design_dir, "xeon_tulsa.architecture.yaml")
    )
    workload_dict = create_workload_dict(
        os.path.join(design_dir, "xeon_tulsa.workload.yaml")
    )
    return architecture_dict, workload_dict


def test_generate_config_xeon_tulsa():
    text = generate_config(*load_design())
    compile(text, "config.py", "exec")

    # values from Xeon.xml, sized through the cacti7/mcpat query keys
    for expected in [
        'CPUTypes("o3")',
        'ISA("x86")',
        "num_cores=2",
        'clk_freq="3400MHz"',
        'clk_freq="850MHz"',
        'l1i_size="131072B"',
        "l1i_assoc=8",
        'l1d_size="16384B"',
        "l1d_assoc=4",
        'l2_size="1048576B"',
        "l2_assoc=8",
        'l3_size="1048576B"',
        "l3_assoc=16",
        "num_l3_banks=16",
        "cpu.numROBEntries = 128",
        "cpu.LQEntries = 48",
        "cpu.SQEntries = 96",
        "cpu.instQueues[0].numEntries = 64",
        "board.cache_line_size = 64",
        "SingleChannelDDR3_1600(",
        "arguments=['64', '64', '64']",
    ]:
        assert expected in text, expected


def make_fake_gem5(root, exit_code=0):
    """A stand-in gem5 binary that writes stats.txt into --outdir and logs each call."""
    binary = root / "build" / "ALL" / "gem5.opt"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        "#!/bin/sh\n"
        'outdir="${1#--outdir=}"\n'
        'mkdir -p "$outdir"\n'
        'echo stats > "$outdir/stats.txt"\n'
        f'echo call >> "{root / "calls.txt"}"\n'
        f"exit {exit_code}\n"
    )
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return root / "calls.txt"


def setup_fake_run(monkeypatch, tmp_path, exit_code=0):
    calls = make_fake_gem5(tmp_path, exit_code)
    monkeypatch.setenv("GEM5_PATH", str(tmp_path))
    monkeypatch.setenv("ARCHX_GEM5_CACHE_DIR", str(tmp_path / "gem5_cache"))
    monkeypatch.setenv("ARCHX_INTERFACE_CACHE_DIR", str(tmp_path / "interface_cache"))
    guest = tmp_path / "guest"
    guest.write_text("binary")
    return calls, guest


def test_run_gem5_cache(monkeypatch, tmp_path):
    calls, guest = setup_fake_run(monkeypatch, tmp_path)

    run_dir = run_gem5("print(1)\n", str(guest))
    assert open(os.path.join(run_dir, "config.py")).read() == "print(1)\n"
    assert os.path.isfile(os.path.join(run_dir, "m5out", "stats.txt"))
    assert "wall_seconds" in json.load(open(os.path.join(run_dir, "done.json")))

    # same config and binaries: cache hit, gem5 not called again
    assert run_gem5("print(1)\n", str(guest)) == run_dir
    assert calls.read_text().count("call") == 1

    # a different config is a different run
    assert run_gem5("print(2)\n", str(guest)) != run_dir
    assert calls.read_text().count("call") == 2


def test_run_gem5_failure_not_cached(monkeypatch, tmp_path):
    calls, guest = setup_fake_run(monkeypatch, tmp_path, exit_code=1)

    for _ in range(2):
        with pytest.raises(RuntimeError):
            run_gem5("print(1)\n", str(guest))
    assert calls.read_text().count("call") == 2


def test_query_interface_gem5(monkeypatch, tmp_path):
    calls, guest = setup_fake_run(monkeypatch, tmp_path)
    architecture_dict, workload_dict = load_design()
    workload_dict["configuration"]["binary"] = str(guest)

    output = query_interface(
        "xeon_tulsa",
        {
            "interface": "gem5",
            "architecture": architecture_dict,
            "workload": workload_dict,
        },
    )
    assert os.path.isfile(output["stats_file"])
    assert output["stats_file"].startswith(output["run_dir"])
    assert calls.read_text().count("call") == 1
