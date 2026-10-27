import csv, os, copy

from collections import OrderedDict
from loguru import logger

from archx.utils import interpolate_oneD_linear, interpolate_oneD_quadratic


CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include", "csv")
interpolate = {
    "linear": interpolate_oneD_linear,
    "quadratic": interpolate_oneD_quadratic,
}
# every other csv column is a size parameter from param.yaml
skip_list = [
    "technology",
    "frequency_mhz",
    "dynamic_mw",
    "leakage_mw",
    "area_mm2",
    "interpolation",
]


# one csv row per module, written by tnn7_extract.py at the module's own PnR clock (frequency_mhz)
def query(
    name: str, interface: str, query: OrderedDict, input_dir=None, output_dir=None
):
    query = copy.deepcopy(query)
    query_class = query.pop("class").lower()
    query_technology = query.pop("technology")
    query_frequency = query.pop("frequency")

    csv_file = os.path.join(CSV_DIR, query_class + ".csv")
    assert os.path.isfile(csv_file), logger.error(
        f"Missing csv <{csv_file}>; run tnn7_extract.py on the PnR reports of <{query_class}>."
    )
    with open(csv_file) as f:
        (row,) = csv.DictReader(f)
    assert str(query_technology) == row["technology"], logger.error(
        f"Invalid technology: <{query_technology}>. <{csv_file}> has <{row['technology']}>."
    )
    interpolation = row["interpolation"].lower()
    assert interpolation in interpolate, logger.error(
        f"Invalid interpolation <{interpolation}> in <{csv_file}>. Valid options are <{list(interpolate)}>."
    )
    for attr_key in query:
        assert attr_key in row, logger.error(f"Invalid query attribute <{attr_key}>.")

    dynamic_power_mW = (
        float(row["dynamic_mw"]) * query_frequency / float(row["frequency_mhz"])
    )
    leakage_power_mW = float(row["leakage_mw"])
    area_mm2 = float(row["area_mm2"])
    # an unqueried size parameter counts as 1, as in csv_cmos
    for attr_key in row:
        if attr_key not in skip_list:
            x = query.get(attr_key, 1)
            x_row = float(row[attr_key])
            dynamic_power_mW = interpolate[interpolation](
                x, [{"x": 0, "y": 0}, {"x": x_row, "y": dynamic_power_mW}]
            )
            leakage_power_mW = interpolate[interpolation](
                x, [{"x": 0, "y": 0}, {"x": x_row, "y": leakage_power_mW}]
            )
            area_mm2 = interpolate[interpolation](
                x, [{"x": 0, "y": 0}, {"x": x_row, "y": area_mm2}]
            )

    output_dict = OrderedDict()
    output_dict["technology"] = OrderedDict({"value": query_technology, "unit": "nm"})
    output_dict["frequency"] = OrderedDict({"value": query_frequency, "unit": "MHz"})
    output_dict["dynamic_energy"] = OrderedDict(
        {"value": dynamic_power_mW / query_frequency, "unit": "nJ"}
    )
    output_dict["leakage_power"] = OrderedDict(
        {"value": leakage_power_mW, "unit": "mW"}
    )
    output_dict["area"] = OrderedDict({"value": area_mm2, "unit": "mm^2"})
    return output_dict
