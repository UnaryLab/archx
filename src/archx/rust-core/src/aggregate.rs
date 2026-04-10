use std::collections::HashSet;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;

use crate::graph::{NodeData, EdgeData};
use crate::metric::MetricValue;
use crate::paths::{all_paths, topological_sort_subgraph};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_leaf(graph: &DiGraph<NodeData, EdgeData>, idx: NodeIndex) -> bool {
    graph.neighbors_directed(idx, Direction::Outgoing).next().is_none()
}

/// Get the metric value from a target node, resolving multi-op via the edge's operation map.
fn get_metric_value(
    graph: &DiGraph<NodeData, EdgeData>,
    edge_src: NodeIndex,
    edge_tgt: NodeIndex,
    metric: &str,
) -> f64 {
    let node = &graph[edge_tgt];
    match node.metrics.get(metric) {
        Some(MetricValue::Single(s)) => s.value,
        Some(MetricValue::MultiOp(_)) => {
            // look up which operation this edge specifies
            let edge_idx = graph.find_edge(edge_src, edge_tgt)
                .expect("edge must exist");
            let op = graph[edge_idx].operation.get(metric)
                .expect("multi-op metric requires operation on edge");
            node.metrics[metric].get_op_value(op)
                .unwrap_or_else(|| panic!("operation '{}' not found in metric '{}'", op, metric))
        }
        None => 0.0,
    }
}

/// Compute total path count * factor from workload_idx to event_idx.
pub fn compute_path_count(
    graph: &DiGraph<NodeData, EdgeData>,
    workload_idx: NodeIndex,
    event_idx: NodeIndex,
    metric: &str,
) -> f64 {
    let paths = all_paths(graph, workload_idx, event_idx);
    let mut total = 0.0;
    for path in &paths {
        let mut path_count = 1.0;
        let mut path_factor = 1.0;
        for w in path.windows(2) {
            let edge_idx = graph.find_edge(w[0], w[1]).expect("edge must exist");
            path_count *= graph[edge_idx].count;
            if let Some(&f) = graph[edge_idx].factor.get(metric) {
                path_factor *= f;
            }
        }
        total += path_count * path_factor;
    }
    total
}

// ---------------------------------------------------------------------------
// Mode: module
// ---------------------------------------------------------------------------

/// Recursively sum all leaf nodes reachable from `current`, multiplied by instance count.
/// Already-visited leaves are skipped to avoid double-counting shared modules.
pub fn aggregate_module(
    graph: &DiGraph<NodeData, EdgeData>,
    current: NodeIndex,
    metric: &str,
    evaluated: &mut HashSet<NodeIndex>,
) -> f64 {
    if is_leaf(graph, current) {
        if !evaluated.insert(current) {
            return 0.0; // already counted
        }
        let node = &graph[current];
        let val = match node.metrics.get(metric) {
            Some(MetricValue::Single(s)) => s.value,
            _ => panic!("module '{}' metric '{}' must be Single for 'module' aggregation",
                        node.name, metric),
        };
        val * node.instance
    } else {
        graph.neighbors_directed(current, Direction::Outgoing)
            .map(|child| aggregate_module(graph, child, metric, evaluated))
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Mode: summation
// ---------------------------------------------------------------------------

/// Bottom-up topological aggregation.
/// For each node: parent.metric += count * child_value * factor  for each in-edge.
pub fn aggregate_summation(
    graph: &mut DiGraph<NodeData, EdgeData>,
    start: NodeIndex,
    metric: &str,
) {
    let topo = topological_sort_subgraph(graph, start); // leaves first
    let topo_set: HashSet<NodeIndex> = topo.iter().copied().collect();

    for &v in &topo {
        // Collect contributions this node makes to its parents (in-edges).
        // We must collect first because we cannot borrow graph mutably and immutably at once.
        let contributions: Vec<(NodeIndex, f64)> = graph
            .edges_directed(v, Direction::Incoming)
            .filter(|e| topo_set.contains(&e.source()))
            .map(|e| {
                let src = e.source();
                let count = e.weight().count;
                let factor = e.weight().factor.get(metric).copied().unwrap_or(1.0);
                let child_val = get_metric_value(graph, src, v, metric);
                (src, count * child_val * factor)
            })
            .collect();

        for (src, contrib) in contributions {
            if let Some(m) = graph[src].metrics.get_mut(metric) {
                m.as_single_mut().value += contrib;
            }
        }
    }
}

/// Special case: multi-op leaf with workload specified.
/// Enumerate all paths from workload → event, determine operation per path,
/// accumulate per-op count, then compute total = Σ op_value * op_count.
pub fn aggregate_summation_multiop(
    graph: &DiGraph<NodeData, EdgeData>,
    workload_idx: NodeIndex,
    event_idx: NodeIndex,
    metric: &str,
) -> f64 {
    let paths = all_paths(graph, workload_idx, event_idx);

    // op → (metric_value, accumulated_count)
    let mut op_metric: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut op_count: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    for path in &paths {
        let mut path_count = 1.0;
        let mut path_factor = 1.0;

        for w in path.windows(2) {
            let edge_idx = graph.find_edge(w[0], w[1]).expect("edge must exist");
            path_count *= graph[edge_idx].count;
            if let Some(&f) = graph[edge_idx].factor.get(metric) {
                path_factor *= f;
            }
        }

        // Operation is on the last edge (parent → module)
        if let Some(w) = path.windows(2).last() {
            let edge_idx = graph.find_edge(w[0], w[1]).expect("edge must exist");
            let op = graph[edge_idx].operation.get(metric)
                .cloned()
                .unwrap_or_default()
                .to_lowercase();

            // Record metric value for this op
            let node = &graph[event_idx];
            let mval = node.metrics[metric].get_op_value(&op).unwrap_or(0.0);
            op_metric.insert(op.clone(), mval);
            *op_count.entry(op).or_insert(0.0) += path_count * path_factor;
        }
    }

    op_metric.iter()
        .map(|(op, &val)| val * op_count.get(op).copied().unwrap_or(0.0))
        .sum()
}

// ---------------------------------------------------------------------------
// aggregate_event_count — DFS path count using edge counts only (no factor)
// ---------------------------------------------------------------------------

/// Count how many times `event` executes under `workload` by summing the
/// product of edge counts over every path from workload → event.
pub fn aggregate_event_count(
    graph: &DiGraph<NodeData, EdgeData>,
    workload_idx: NodeIndex,
    event_idx: NodeIndex,
) -> f64 {
    let mut total = 0.0;
    // Stack: (current_node, accumulated_count)
    let mut stack: Vec<(NodeIndex, f64)> = vec![(workload_idx, 1.0)];
    while let Some((current, acc)) = stack.pop() {
        if current == event_idx {
            total += acc;
            continue;
        }
        for neighbor in graph.neighbors_directed(current, Direction::Outgoing) {
            let edge_idx = graph.find_edge(current, neighbor).expect("edge must exist");
            let count = graph[edge_idx].count;
            stack.push((neighbor, acc * count));
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Mode: specified
// ---------------------------------------------------------------------------

/// Parallel/sequential aggregation over non-leaf children only.
/// Leaf children are ignored (their values come from performance models via specified_metrics).
pub fn aggregate_specified(
    graph: &mut DiGraph<NodeData, EdgeData>,
    start: NodeIndex,
    metric: &str,
) {
    let topo = topological_sort_subgraph(graph, start); // leaves first

    for &v in &topo {
        if is_leaf(graph, v) {
            continue;
        }

        // Collect out-edge data before mutating
        let out_data: Vec<(NodeIndex, f64, f64, String, bool)> = graph
            .edges_directed(v, Direction::Outgoing)
            .map(|e| {
                let tgt = e.target();
                let count = e.weight().count;
                let factor = e.weight().factor.get(metric).copied().unwrap_or(1.0);
                let agg = e.weight().aggregation.clone();
                let tgt_is_leaf = is_leaf(graph, tgt);
                (tgt, count, factor, agg, tgt_is_leaf)
            })
            .collect();

        let mut parallel_max: f64 = 0.0;
        let mut sequential_acc: f64 = 0.0;
        let mut connect_leaf_only = true;
        let mut connect_leaf_any = false;

        for (tgt, count, factor, agg, tgt_is_leaf) in &out_data {
            if *tgt_is_leaf {
                connect_leaf_any = true;
            } else {
                connect_leaf_only = false;
                let child_val = match graph[*tgt].metrics.get(metric) {
                    Some(m) => m.as_single().value,
                    None => 0.0,
                };
                let contribution = child_val * count * factor;
                if agg == "parallel" {
                    if contribution > parallel_max {
                        parallel_max = contribution;
                    }
                } else {
                    sequential_acc += contribution;
                }
            }
        }

        let result = if connect_leaf_only {
            // Fall back to value injected by performance model
            graph[v].specified_metrics.get(metric)
                .map(|s| s.value)
                .unwrap_or(0.0)
        } else {
            sequential_acc + parallel_max
        };

        if let Some(m) = graph[v].metrics.get_mut(metric) {
            m.as_single_mut().value = result;
        }
    }
}
