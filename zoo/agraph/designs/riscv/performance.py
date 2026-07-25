from collections import OrderedDict
from archx.utils import get_prod
from bs4 import BeautifulSoup

def decode_instruction(instruction):
    # instruction is a hex string, e.g., '0x00000013'
    inst = int(instruction, 16)

    if inst == 0x13:
        return 'addi_0'

    opcode = inst & 0x7f
    funct3 = (inst >> 12) & 0x7
    funct7 = (inst >> 25) & 0x7f

    # R-type
    if opcode == 0x33:
        if funct3 == 0x0 and funct7 == 0x00:
            return 'add'
        # add more R-type as needed

    # I-type
    elif opcode == 0x13:
        if funct3 == 0x0:
            return 'addi'
        elif funct3 == 0x7:
            return 'andi'
        elif funct3 == 0x1:
            return 'slli'
        elif funct3 == 0x5:
            return 'srli'
        # add more I-type as needed

    # Load
    elif opcode == 0x03:
        if funct3 == 0x2:
            return 'lw'

    # Store
    elif opcode == 0x23:
        if funct3 == 0x2:
            return 'sw'

    # Branch
    elif opcode == 0x63:
        if funct3 == 0x0:
            return 'beq'
        elif funct3 == 0x4:
            return 'blt'

    # JAL
    elif opcode == 0x6f:
        return 'jal'

    # JALR
    elif opcode == 0x67:
        return 'jalr'

    # LUI
    elif opcode == 0x37:
        return 'lui'

    return 'add'

def gemm(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    with open('zoo/agraph/designs/riscv/if_id_trace.html', 'r', encoding='utf-8') as f:
        trace = BeautifulSoup(f, 'html.parser')

    rows = trace.find_all('tr')[1:]  # Skip header row


    pc = None
    prev_pc = None
    instruction = None
    instr_dict = OrderedDict()
    # Local memory for last 5 instructions: list of dicts with keys 'pc', 'prev_pc', 'instr_type'
    last5 = []

    for row in rows:
        cols = row.find_all('td')
        cycle = int(cols[0].text)
        if pc is not None:
            prev_pc = pc
        pc = cols[1].text.split('=')[1]
        instruction = cols[2].text.split('=')[1]
        instr_type = decode_instruction(instruction)

        if instr_type not in instr_dict:
            instr_dict[instr_type] = 0
        instr_dict[instr_type] += 1

    pipeline_count = OrderedDict({'count': cycle})
    add_count = OrderedDict({'count': instr_dict.get('add', 0)})
    addi_count = OrderedDict({'count': instr_dict.get('addi', 0)})
    addi_0_count = OrderedDict({'count': instr_dict.get('addi_0', 0)})
    andi_count = OrderedDict({'count': instr_dict.get('andi', 0)})
    beq_count = OrderedDict({'count': instr_dict.get('beq', 0)})
    blt_count = OrderedDict({'count': instr_dict.get('blt', 0)})
    jal_count = OrderedDict({'count': instr_dict.get('jal', 0)})
    jalr_count = OrderedDict({'count': instr_dict.get('jalr', 0)})
    lui_count = OrderedDict({'count': instr_dict.get('lui', 0)})
    lw_count = OrderedDict({'count': instr_dict.get('lw', 0)})
    slli_count = OrderedDict({'count': instr_dict.get('slli', 0)})
    srli_count = OrderedDict({'count': instr_dict.get('srli', 0)})
    sw_count = OrderedDict({'count': instr_dict.get('sw', 0)})

    performance_dict['subevent'] = OrderedDict({
        'pipeline': pipeline_count,
        'add': add_count,
        'addi_0': addi_0_count,
        'addi': addi_count,
        'andi': andi_count,
        'beq': beq_count,
        'blt': blt_count,
        'jal': jal_count,
        'jalr': jalr_count,
        'lui': lui_count,
        'lw': lw_count,
        'slli': slli_count,
        'srli': srli_count,
        'sw': sw_count
    })

    return performance_dict

def pipeline(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    if_count = OrderedDict({'count': get_prod(architecture_dict['instr_fetch']['instance'])})
    id_count = OrderedDict({'count': get_prod(architecture_dict['decode']['instance'])})
    ex_count = OrderedDict({'count': get_prod(architecture_dict['execute']['instance'])})
    mem_count = OrderedDict({'count': get_prod(architecture_dict['mem']['instance'])})
    wb_count = OrderedDict({'count': get_prod(architecture_dict['wb']['instance'])})
    hd_count = OrderedDict({'count': get_prod(architecture_dict['hd']['instance'])})
    f_count = OrderedDict({'count': get_prod(architecture_dict['forward']['instance'])})

    frequency = architecture_dict['instr_fetch']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})

    performance_dict['subevent'] = OrderedDict({'instr_fetch': if_count,
                                                'decode': id_count,
                                                'execute': ex_count,
                                                'mem': mem_count,
                                                'wb': wb_count,
                                                'hd': hd_count,
                                                'forward': f_count})
    return performance_dict

def add(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    frequency = architecture_dict['compute']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    
    compute_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'add'})})


    performance_dict['subevent'] = OrderedDict({'compute': compute_dict})
    return performance_dict

def addi(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    frequency = architecture_dict['compute']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    
    compute_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'addi'})})


    performance_dict['subevent'] = OrderedDict({'compute': compute_dict})
    return performance_dict

def addi_0(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    frequency = architecture_dict['compute']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    
    compute_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'addi_0'})})


    performance_dict['subevent'] = OrderedDict({'compute': compute_dict})
    return performance_dict

def andi(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    frequency = architecture_dict['compute']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    
    compute_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'andi'})})


    performance_dict['subevent'] = OrderedDict({'compute': compute_dict})
    return performance_dict

def beq(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    frequency = architecture_dict['compute']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    
    compute_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'beq'})})


    performance_dict['subevent'] = OrderedDict({'compute': compute_dict})
    return performance_dict

def blt(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    frequency = architecture_dict['compute']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    
    compute_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'blt'})})


    performance_dict['subevent'] = OrderedDict({'compute': compute_dict})
    return performance_dict

def jal(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    frequency = architecture_dict['compute']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    
    compute_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'jal'})})


    performance_dict['subevent'] = OrderedDict({'compute': compute_dict})
    return performance_dict

def jalr(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    frequency = architecture_dict['compute']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    
    compute_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'jalr'})})


    performance_dict['subevent'] = OrderedDict({'compute': compute_dict})
    return performance_dict

def lui(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    frequency = architecture_dict['compute']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    
    compute_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'lui'})})


    performance_dict['subevent'] = OrderedDict({'compute': compute_dict})
    return performance_dict

def lw(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    frequency = architecture_dict['compute']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    
    compute_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'lw'})})


    performance_dict['subevent'] = OrderedDict({'compute': compute_dict})
    return performance_dict

def slli(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    frequency = architecture_dict['compute']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    
    compute_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'slli'})})


    performance_dict['subevent'] = OrderedDict({'compute': compute_dict})
    return performance_dict

def srli(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    frequency = architecture_dict['compute']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    
    compute_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'srli'})})


    performance_dict['subevent'] = OrderedDict({'compute': compute_dict})
    return performance_dict

def sw(architecture_dict: OrderedDict, workload_dict: OrderedDict=None)->OrderedDict:
    performance_dict = OrderedDict()

    frequency = architecture_dict['compute']['query']['frequency']
    performance_dict['cycle_count'] = OrderedDict({'value': 1, 'unit': 'cycle'})
    performance_dict['runtime'] = OrderedDict({'value': 1 / 1000 / frequency, 'unit': 'ms'})
    
    compute_dict = OrderedDict({'operation': OrderedDict({'dynamic_energy': 'sw'})})


    performance_dict['subevent'] = OrderedDict({'compute': compute_dict})
    return performance_dict