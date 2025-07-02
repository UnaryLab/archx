import os, csv, shutil, glob

from loguru import logger

from archx.utils import strip_list, read_yaml, create_dir


linear_interpolation_keywords = ['acc', 'add', 'sub', 'reg', 'rng', 'shifter']
quadratic_interpolation_keywords = ['multiplier']


def extract_power_report(
    report=None
):
    """
    all outputs have the unit of mW
    """
    file = open(report, 'r')
    for entry in file:
        elems = entry.strip().split(' ')
        elems = strip_list(elems)
        if len(elems) >= 6:
            if elems[0] == 'Total' and elems[1] == 'Dynamic' and elems[2] == 'Power' and elems[3] == '=':
                dynamic = float(elems[4])
                unit = str(elems[5])
                if unit == 'nW':
                    dynamic /= 1000000.0
                elif unit == 'uW':
                    dynamic /= 1000.0
                elif unit == 'mW':
                    dynamic *= 1.0
                else:
                    logger.warning(f'Unknown unit for dynamic power: {unit}.')

            if elems[0] == 'Cell' and elems[1] == 'Leakage' and elems[2] == 'Power' and elems[3] == '=':
                leakage = float(elems[4])
                unit = str(elems[5])
                if unit == 'nW':
                    leakage /= 1000000.0
                elif unit == 'uW':
                    leakage /= 1000.0
                elif unit == 'mW':
                    leakage *= 1.0
                else:
                    logger.warning('Unknown unit for leakage power: {unit}.')
    file.close()
    return dynamic, leakage


def extract_area_report(
    report=None
):
    """
    output has the unit of mm^2
    """
    file = open(report, 'r')
    for entry in file:
        elems = entry.strip().split(' ')
        elems = strip_list(elems)
        if len(elems) >= 3:
            if str(elems[0]) == 'Total' and str(elems[1]) == 'cell' and str(elems[2]) == 'area:':
                area = float(elems[3])

            if str(elems[0]) == 'Total' and str(elems[1]) == 'area:':
                if str(elems[2]) != 'undefined':
                    if area < float(elems[2]):
                        area = float(elems[2])
                    
    area /= 1000000.0
    file.close()
    return area


def extract_syn_report(technology=32, frequency=400):
    """
    technology in nm, frequency in MHz
    """
    file_root = os.path.dirname(os.path.abspath(__file__))
    rpt_root = file_root + '/include/rpt'
    csv_root = file_root + '/include/csv'
    syn_root = file_root + '/include/syn'
    rtl_root = file_root + '/include/rtl'
    param_yml = file_root + '/include/rtl/param.yaml'
    param_dict = read_yaml(param_yml)

    # move rpt from rtl dir to rpt dir
    for module in os.listdir(syn_root):
        syn_dir = syn_root + '/' + module
        rpt_dir = rpt_root + '/' + module
        logger.info(f'Process synthesis results of module <{module}>.')
        create_dir(rpt_dir)
        for syn_result in glob.glob(syn_dir + '/' + module + '_*.syn.txt'):
            shutil.copy2(syn_result, rpt_dir)
        shutil.copy2(syn_dir + '/' + module + '.syn.rpt', rpt_dir)
        logger.info(f'  Collect synthesis results from <{syn_dir}> to <{rpt_dir}>.')

        # define interpolation method
        interpolation = 'linear'
        for keyword in linear_interpolation_keywords:
            if keyword in module:
                interpolation = 'linear'
        for keyword in quadratic_interpolation_keywords:
            if keyword in module:
                interpolation = 'quadratic'
        
        # extract synthesis rpt
        area_rpt     = rpt_dir + '/' + module + '_area.syn.txt'
        power_rpt    = rpt_dir + '/' + module + '_power.syn.txt'
        assert os.path.exists(area_rpt), logger.error(f'Area report for module <{module}> does not exist.')
        assert os.path.exists(power_rpt), logger.error(f'Power report for module <{module}> does not exist.')
        dynamic, leakage = extract_power_report(power_rpt)
        area = extract_area_report(area_rpt)
        logger.info(f'  Extract synthesis results.')

        # parsing parameter dict
        param_name = []
        param_value = []
        if module in param_dict and param_dict[module] != None:
            for param in param_dict[module].items():
                param_name.append(param[0].lower())
                param_value.append(param[1])

        # generate csv table without parameters
        module_csv = csv_root + '/' + module + '.csv'
        with open(module_csv, 'w') as f:
            csvwriter = csv.writer(f)
            header = ['technology', 'frequency', 'dynamic', 'leakage', 'area', 'num_instances', 'interpolation'] + param_name
            csvwriter.writerow(header)
            content = [technology, frequency, dynamic, leakage, area, 1, interpolation] + param_value
            csvwriter.writerow(content)
        logger.info(f'  Generate query csv file <{module_csv}>.')


if __name__ == '__main__':
    extract_syn_report(32, 400)
