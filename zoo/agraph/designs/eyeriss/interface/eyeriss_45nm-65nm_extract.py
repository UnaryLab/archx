import os, csv

from loguru import logger

from archx.utils import strip_list, read_yaml, create_dir
from archx.interface.csv_cmos.csv_cmos_extract import tech_map, quadratic_interpolation_keywords


def extract(technology, frequency):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rpt_dir = f'{current_dir}/syn_pnr_rpt/{technology}'
    csv_dir = f'{current_dir}/include/csv/'
    create_dir(csv_dir)
    param_dict = read_yaml(f'{current_dir}/param.yaml')

    for module in sorted(os.listdir(rpt_dir)):
        interpolation = 'linear'
        for keyword in quadratic_interpolation_keywords:
            if keyword in module:
                interpolation = 'quadratic'

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

        params = param_dict.get(module) or {}
        header = ['technology', 'frequency', 'dynamic_uw', 'leakage_uw', 'area_mm2', 'num_instances', 'interpolation'] + [key.lower() for key in params]
        content = [tech_map[technology], frequency, dynamic, leakage, area, 1, interpolation] + list(params.values())
        module_csv = csv_dir + module + '.csv'
        with open(module_csv, 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(header)
            csvwriter.writerow(content)
        logger.success(f'Write <{module_csv}>.')


if __name__ == '__main__':
    technology = 'NanGate45'
    frequency = 200 # 2.5 ns clock in every *_CRITICAL_PATH.rpt
    extract(technology, frequency)
