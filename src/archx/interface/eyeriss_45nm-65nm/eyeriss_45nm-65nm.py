import csv, os, copy

from collections import OrderedDict
from loguru import logger

from archx.utils import interpolate_oneD_linear, interpolate_oneD_quadratic
from archx.interface.csv_cmos.csv_cmos import skip_list, interpolation_list


CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include', 'csv')
interpolate = {'linear': interpolate_oneD_linear, 'quadratic': interpolate_oneD_quadratic}
# 45 nm values relative to 65 nm, DeepScaleTool (Sarangi & Baas, ISCAS 2021)
SCALE_45_TO_65 = {'area': 0.49, 'energy': 0.63, 'power': 0.78}


# pe 'mac' and 'gated' share this single dynamic_energy: one PnR power report holds one switching activity
def query(name: str, interface: str, query: OrderedDict, input_dir=None, output_dir=None):
    query = copy.deepcopy(query)
    query_class = query.pop('class').lower()
    query_technology = query.pop('technology')
    query_frequency = query.pop('frequency')
    assert query_technology in (45, 65), logger.error(f'Invalid technology: <{query_technology}>. Valid options are 45 (raw report) and 65 (scaled from 45).')

    csv_file = os.path.join(CSV_DIR, query_class + '.csv')
    assert os.path.isfile(csv_file), logger.error(f'Missing csv <{csv_file}>; run eyeriss_45nm-65nm_extract.py on the syn/PnR reports of <{query_class}>.')
    with open(csv_file) as f:
        rows = [row for row in csv.DictReader(f) if row['technology'] == '45']
    assert rows, logger.error(f'No 45 nm row in <{csv_file}>.')
    row = rows[-1]

    interpolation = row['interpolation'].lower()
    assert interpolation in interpolation_list, logger.error(f'Invalid interpolation <{interpolation}> in <{csv_file}>. Valid options are <{str(interpolation_list)}>.')
    for attr_key in query:
        assert attr_key in row, logger.error(f'Invalid query attribute <{attr_key}>.')

    # csv power columns are in mW, despite the _uw names inherited from csv_cmos
    dynamic_power_mW = float(row['dynamic_uw']) * query_frequency / float(row['frequency'])
    leakage_power_mW = float(row['leakage_uw'])
    area_mm2 = float(row['area_mm2'])
    if query_technology == 65:
        dynamic_power_mW /= SCALE_45_TO_65['energy']
        leakage_power_mW /= SCALE_45_TO_65['power']
        area_mm2 /= SCALE_45_TO_65['area']
    for attr_key in row:
        if attr_key not in skip_list:
            x = query.get(attr_key, 1)
            x_row = float(row[attr_key])
            dynamic_power_mW = interpolate[interpolation](x, [{'x': 0, 'y': 0}, {'x': x_row, 'y': dynamic_power_mW}])
            leakage_power_mW = interpolate[interpolation](x, [{'x': 0, 'y': 0}, {'x': x_row, 'y': leakage_power_mW}])
            area_mm2 = interpolate[interpolation](x, [{'x': 0, 'y': 0}, {'x': x_row, 'y': area_mm2}])

    output_dict = OrderedDict()
    output_dict['technology'] = OrderedDict({'value': query_technology, 'unit': 'nm'})
    output_dict['frequency'] = OrderedDict({'value': query_frequency, 'unit': 'MHz'})
    output_dict['dynamic_energy'] = OrderedDict({'value': dynamic_power_mW / query_frequency, 'unit': 'nJ'})
    output_dict['leakage_power'] = OrderedDict({'value': leakage_power_mW, 'unit': 'mW'})
    output_dict['area'] = OrderedDict({'value': area_mm2, 'unit': 'mm^2'})
    return output_dict
