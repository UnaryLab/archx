from loguru import logger

from archx.interface import query_interface


def test_query_interface_mcpat_l2(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCHX_INTERFACE_CACHE_DIR", str(tmp_path / "cache"))
    module = "l2"
    query = {
        "class": "l2",
        "interface": "mcpat",
        "technology": 45,
        "frequency": 1000,
        "width": 512,
        "depth": 1024,
        "bank": 8,
    }
    output = query_interface(module, query, output_dir=str(tmp_path))
    logger.info(f"test_query_interface_mcpat_l2: {output}")

    assert output["dynamic_energy"]["read"]["value"] > 0
    assert output["dynamic_energy"]["write"]["value"] > 0
    assert output["leakage_power"]["value"] > 0
    assert output["area"]["value"] > 0
