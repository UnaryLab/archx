from collections import OrderedDict
from archx.utils import get_prod

# archx looks up each event's function by name in this file, so the shared event functions are imported here
from zoo.agraph.designs.eyeriss.common.performance_shared import (
    alexnet,
    conv_subevents,
    dram_read,
    dram_write,
    filt_glb_read,
    filter_noc_transfer,
    ifmap_noc_transfer,
    ifmap_psum_glb_read,
    ifmap_psum_glb_write,
    ofmap_relu,
    psum_gin_transfer,
    psum_gon_transfer,
    rlc_decode,
    rlc_encode,
    spad_read,
    spad_write,
)

PE_MODULE = "pe"


def _conv(architecture_dict, workload_dict, index):
    performance_dict = OrderedDict()
    performance_dict["subevent"] = conv_subevents(
        architecture_dict, workload_dict, index, PE_MODULE
    )[0]
    return performance_dict


def conv1(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _conv(architecture_dict, workload_dict, 0)


def conv2(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _conv(architecture_dict, workload_dict, 1)


def conv3(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _conv(architecture_dict, workload_dict, 2)


def conv4(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _conv(architecture_dict, workload_dict, 3)


def conv5(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _conv(architecture_dict, workload_dict, 4)


def _pe_cycle(architecture_dict, operation):
    """One PE-array cycle: every PE runs `operation` once."""
    performance_dict = OrderedDict()
    frequency = architecture_dict[PE_MODULE]["query"]["frequency"]
    pe_total = get_prod(architecture_dict[PE_MODULE]["instance"])

    performance_dict["cycle_count"] = OrderedDict({"value": 1, "unit": "cycle"})
    performance_dict["runtime"] = OrderedDict(
        {"value": 1 / 1000 / frequency, "unit": "ms"}
    )
    performance_dict["subevent"] = OrderedDict(
        {
            PE_MODULE: OrderedDict(
                {
                    "count": pe_total,
                    "operation": OrderedDict({"dynamic_energy": operation}),
                }
            )
        }
    )

    return performance_dict


def pe_mac(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _pe_cycle(architecture_dict, "mac")


def pe_gated(
    architecture_dict: OrderedDict, workload_dict: OrderedDict = None
) -> OrderedDict:
    return _pe_cycle(architecture_dict, "zero_gated")
