use std::collections::HashSet;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::algo::toposort;

use crate::graph::{NodeData, EdgeData};

/// All simple paths from src to dst via iterative DFS.
/// Replaces graph_tool's gt.all_paths().
pub fn all_paths(
    graph: &DiGraph<NodeData, EdgeData>,
    src: NodeIndex,
    dst: NodeIndex,
) -> Vec<Vec<NodeIndex>> {
    let mut results = Vec::new();
    let mut stack: Vec<(NodeIndex, Vec<NodeIndex>, HashSet<NodeIndex>)> = vec![
        (src, vec![src], {
            let mut s = HashSet::new();
            s.insert(src);
            s
        }),
    ];

    while let Some((current, path, visited)) = stack.pop() {
        if current == dst {
            results.push(path);
            continue;
        }
        for neighbor in graph.neighbors_directed(current, Direction::Outgoing) {
            if !visited.contains(&neighbor) {
                let mut new_path = path.clone();
                new_path.push(neighbor);
                let mut new_visited = visited.clone();
                new_visited.insert(neighbor);
                stack.push((neighbor, new_path, new_visited));
            }
        }
    }

    results
}

/// Returns nodes reachable from `start` in bottom-up order (leaves first, start last).
/// Replaces topological_sort_reverse() from the Python code.
pub fn topological_sort_subgraph(
    graph: &DiGraph<NodeData, EdgeData>,
    start: NodeIndex,
) -> Vec<NodeIndex> {
    // Forward BFS to find all nodes reachable from start
    let reachable = reachable_from(graph, start);

    // Full topological sort of the graph
    let topo = match toposort(graph, None) {
        Ok(order) => order,
        Err(_) => panic!("Graph contains a cycle; expected a DAG"),
    };

    // Filter to reachable only, then reverse (bottom-up)
    let mut filtered: Vec<NodeIndex> = topo.into_iter()
        .filter(|n| reachable.contains(n))
        .collect();
    filtered.reverse();
    filtered
}

/// BFS forward from `start` — returns set of all reachable nodes (including start).
pub fn reachable_from(
    graph: &DiGraph<NodeData, EdgeData>,
    start: NodeIndex,
) -> HashSet<NodeIndex> {
    let mut visited = HashSet::new();
    let mut queue = vec![start];
    while let Some(node) = queue.pop() {
        if visited.insert(node) {
            for neighbor in graph.neighbors_directed(node, Direction::Outgoing) {
                queue.push(neighbor);
            }
        }
    }
    visited
}
