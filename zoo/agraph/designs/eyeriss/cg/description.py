from archx.programming.graph.agraph import AGraph

from zoo.agraph.designs.eyeriss.common.description_shared import (
    BITWIDTH,
    DESIGN_PATH,
    PE_ARRAY,
    add_attributes,
    add_metrics,
    add_shared_events,
    add_shared_modules,
    add_workload,
)

PERFORMANCE = f"{DESIGN_PATH}/cg/performance.py"


def description(path):
    agraph = AGraph(path=path)
    architecture = agraph.architecture
    event = agraph.event
    metric = agraph.metric
    workload = agraph.workload

    add_attributes(architecture)

    # fmt: off
    # pe.v includes the ifmap/psum spads and four 4-entry fifos
    architecture.add_module(name="pe", instance=PE_ARRAY, tag=["pe", "onchip"], query={"class": "pe", "width": BITWIDTH})
    # fmt: on
    add_shared_modules(architecture, ["pe", "memory", "onchip"])

    add_shared_events(event, PERFORMANCE)
    event.add_event(name="pe_mac", subevent=["pe"], performance=PERFORMANCE)
    # a zero ifmap cycle, charged from the pe's zero_gated power report
    event.add_event(name="pe_gated", subevent=["pe"], performance=PERFORMANCE)

    add_metrics(metric)
    add_workload(workload)

    return agraph.generate()
