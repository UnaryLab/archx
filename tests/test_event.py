# following two lines are used in testing
import sys, os, shutil

import graph_tool.all as gt

from loguru import logger

from archx.event import create_event_graph, save_event_graph, load_event_graph
from archx.utils import get_path, create_dir


def test_create_event_graph():
    event_file = 'examples/mac_1_cycle/input/event/example.event.yaml'
    event_graph = create_event_graph(event_file)

    create_dir('tests/test_event/')
    
    gt.graph_draw(event_graph, 
            vertex_text = event_graph.vp.event, 
            vertex_font_size=10, 
            edge_pen_width = event_graph.ep.count,
            output_size=(800, 800),
            output='tests/test_event/weighted_graph.pdf')

    ckpt_file = 'tests/test_event/example.event.gt'
    save_event_graph(event_graph, ckpt_file)
    event_graph_ckpt = load_event_graph(ckpt_file)

    gt.graph_draw(event_graph, 
            vertex_text = event_graph.vp.event, 
            vertex_font_size=10, 
            edge_pen_width = event_graph.ep.count,
            output_size=(800, 800),
            output='tests/test_event/weighted_graph_ckpt.pdf')

    # check whether two graphs are isomorphic
    assert gt.isomorphism(event_graph, event_graph_ckpt)


def test_cleanup():
    path = get_path('tests/test_event/')
    shutil.rmtree(path)


if __name__ == "__main__":
    test_create_event_graph()
    test_cleanup()

