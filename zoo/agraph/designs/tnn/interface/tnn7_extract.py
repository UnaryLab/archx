import os, re, csv, glob

from loguru import logger

from archx.utils import read_yaml, create_dir
from archx.interface.csv_cmos.csv_cmos_extract import (
    tech_map,
    quadratic_interpolation_keywords,
)


TECHNOLOGY = "TNN7"
STAGE = "postRouteOpt"
# only for modules whose *_CRITICAL_PATH.rpt has no constrained timing path (no clocked path to read)
DEFAULT_FREQUENCY_MHZ = 100.0
# Report Area, not Standard Cell Area: postRouteOpt Standard Cell Area jumps with no change in Report
# Area (fsm_weight_update 22.861 -> 39.191 um^2), likely physical-only cells (fillers, taps) added in
# postRouteOpt; it reaches or exceeds Core Area for some modules. Report Area / 16 matches the
# TNN7 paper's Table II cell areas. Areas stay as reported: the ASAP7 LEF is 4x scaled, so they are
# ~16x physical; the reader of the results applies that, not this interface.
AREA_COLUMN = "Report Area (um^2)"


def find_report(module_dir, suffix):
    # file names do not always match the folder (tnn_adder/adder_*.rpt), so match on the suffix only
    files = glob.glob(os.path.join(module_dir, f"*_{suffix}.rpt"))
    assert len(files) == 1, logger.error(
        f"Expect one *_{suffix}.rpt in <{module_dir}>, found {files}."
    )
    return files[0]


def read_frequency_mhz(module, rpt):
    with open(rpt) as f:
        text = f.read()
    # Phase Shift is the launch-to-capture clock edge distance in ps (ASAP7 time unit; DETAILS.rpt WS(ns)
    # agrees); each path here is captured on one clock and launched on it or on an unclocked input ('@'),
    # so it equals that clock's period
    match = re.search(r"^\+ Phase Shift\s+([\d.]+)", text, re.MULTILINE)
    if match:
        return 1e6 / float(match.group(1))
    assert "No constrained timing paths found" in text, logger.error(
        f"No Phase Shift line in <{rpt}>."
    )
    logger.warning(
        f"<{module}>: no constrained timing path in <{rpt}>; use default {DEFAULT_FREQUENCY_MHZ} MHz."
    )
    return DEFAULT_FREQUENCY_MHZ


def read_stage_row(module, rpt):
    with open(rpt) as f:
        rows = [row for row in csv.DictReader(f, skipinitialspace=True)]
    for row in rows:
        if row["Physical Design Stage"] == STAGE:
            return row
    logger.warning(
        f'<{module}>: no {STAGE} row in <{rpt}>; use last stage <{rows[-1]["Physical Design Stage"]}>.'
    )
    return rows[-1]


def extract():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rpt_dir = os.path.join(current_dir, TECHNOLOGY)
    csv_dir = os.path.join(current_dir, "include", "csv")
    create_dir(csv_dir)
    param_dict = read_yaml(os.path.join(current_dir, "param.yaml"))
    modules = sorted(os.listdir(rpt_dir))
    assert set(param_dict) <= set(modules), logger.error(
        f"param.yaml modules without reports: {set(param_dict) - set(modules)}."
    )

    for module in modules:
        module_dir = os.path.join(rpt_dir, module)
        row = read_stage_row(module, find_report(module_dir, "DETAILS"))
        frequency = read_frequency_mhz(module, find_report(module_dir, "CRITICAL_PATH"))
        dynamic = float(row["Dynamic Power (mW)"])
        leakage = float(row["Leakage Power (mW)"])
        area = float(row[AREA_COLUMN]) / 10**6  # mm^2
        assert dynamic != 0, logger.error(f"dynamic power={dynamic} for {module} is 0")
        assert leakage != 0, logger.error(f"leakage power={leakage} for {module} is 0")
        assert area != 0, logger.error(f"area={area} for {module} is 0")

        interpolation = "linear"
        for keyword in quadratic_interpolation_keywords:
            if keyword in module:
                interpolation = "quadratic"

        params = param_dict.get(module) or {}
        header = [
            "technology",
            "frequency_mhz",
            "dynamic_mw",
            "leakage_mw",
            "area_mm2",
            "interpolation",
        ] + [key.lower() for key in params]
        content = [
            tech_map[TECHNOLOGY],
            frequency,
            dynamic,
            leakage,
            area,
            interpolation,
        ] + list(params.values())
        module_csv = os.path.join(csv_dir, module + ".csv")
        with open(module_csv, "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(header)
            csvwriter.writerow(content)
        logger.success(f"Write <{module_csv}> at {frequency:g} MHz.")


if __name__ == "__main__":
    extract()
