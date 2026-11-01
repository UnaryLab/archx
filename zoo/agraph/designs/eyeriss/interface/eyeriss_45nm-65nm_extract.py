import os, csv

from loguru import logger

from archx.utils import strip_list, read_yaml, create_dir
from archx.interface.csv_cmos.csv_cmos_extract import tech_map, quadratic_interpolation_keywords


def read_rpt(rpt_dir, module):
    area, leakage, dynamic = 0, 0, 0
    with open(f'{rpt_dir}/{module}/{module}_DETAILS.rpt', 'r') as file:
        for entry in file:
            elems = strip_list(entry.strip().split(','))
            if len(elems) > 0 and str(elems[0]) == 'postRouteOpt':
                area = float(elems[2]) / 10**6 # mm^2
                leakage = float(elems[6]) # mW, kept in the _uw column as csv_cmos does
                dynamic = float(elems[7]) # mW, kept in the _uw column as csv_cmos does

    assert dynamic != 0, logger.error(f'dynamic power={dynamic} for {module} is 0')
    assert leakage != 0, logger.error(f'leakage power={leakage} for {module} is 0')
    assert area != 0, logger.error(f'area={area} for {module} is 0')
    return area, leakage, dynamic


def extract(technology, frequency):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rpt_dir = f'{current_dir}/syn_pnr_rpt/{technology}'
    csv_dir = f'{current_dir}/include/csv/'
    create_dir(csv_dir)
    param_dict = read_yaml(f'{current_dir}/param.yaml')

    # dif_module: <rpt module> in an entry subtracts that module's area, leakage and dynamic power from every row of its csv,
    # dif_scale times (default 1), e.g. 2 to take out a register twice the rpt module's width
    difs = {}
    for name, params in param_dict.items():
        if params and 'dif_module' in params:
            difs[name] = (params.pop('dif_module'), params.pop('dif_scale', 1))

    # csv name -> [(rpt module, params)]. A param.yaml entry whose values are all dicts names the csv,
    # and each nested key is the rpt module of one action row (e.g. pe: {pe: {action: mac}, pe_zero_gated: {action: zero_gated}})
    # Rpt modules used as nested rows get no standalone csv.
    csvs = {module: [(module, param_dict.get(module) or {})] for module in sorted(os.listdir(rpt_dir))}
    for name, params in param_dict.items():
        if params and all(isinstance(sub, dict) for sub in params.values()):
            assert all('action' in sub for sub in params.values()), logger.error(f'every module of <{name}> needs an action')
            assert len({tuple(sub) for sub in params.values()}) == 1, logger.error(f'modules of <{name}> have different param keys or key order')
            for module in params:
                csvs.pop(module, None)
            csvs[name] = list(params.items())

    for name, modules in sorted(csvs.items()):
        interpolation = 'linear'
        for keyword in quadratic_interpolation_keywords:
            if keyword in name:
                interpolation = 'quadratic'

        # leakage and area come from the first module; the others only add their own dynamic power
        rpts = [read_rpt(rpt_dir, module) for module, _ in modules]
        area, leakage, _ = rpts[0]
        if name in difs:
            dif_module, dif_scale = difs[name]
            dif_area, dif_leakage, dif_dynamic = (dif_scale * value for value in read_rpt(rpt_dir, dif_module))
            area -= dif_area
            leakage -= dif_leakage
            rpts = [(rpt_area, rpt_leakage, dynamic - dif_dynamic) for rpt_area, rpt_leakage, dynamic in rpts]
            assert area > 0 and leakage > 0 and all(dynamic > 0 for _, _, dynamic in rpts), logger.error(f'{dif_scale} x <{dif_module}> is not smaller than <{name}>')

        header = ['technology', 'frequency', 'dynamic_uw', 'leakage_uw', 'area_mm2', 'num_instances', 'interpolation'] + [key.lower() for key in modules[0][1]]
        module_csv = csv_dir + name + '.csv'
        with open(module_csv, 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(header)
            for (_, params), (_, _, dynamic) in zip(modules, rpts):
                csvwriter.writerow([tech_map[technology], frequency, dynamic, leakage, area, 1, interpolation] + list(params.values()))
        logger.success(f'Write <{module_csv}>.')


if __name__ == '__main__':
    technology = 'NanGate45'
    frequency = 200 # 2.5 ns clock in every *_CRITICAL_PATH.rpt
    extract(technology, frequency)
