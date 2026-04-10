use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// A single {value, unit} metric — for events and single-operation modules.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SingleMetric {
    pub value: f64,
    pub unit: String,
}

/// Metric storage for one (metric_name) slot on a node.
/// Single: normal event node or single-op module  → {value, unit}
/// MultiOp: module with named operations (e.g. sram read/write) → {op → {value, unit}}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MetricValue {
    Single(SingleMetric),
    MultiOp(IndexMap<String, SingleMetric>),
}

impl MetricValue {
    pub fn as_single(&self) -> &SingleMetric {
        match self {
            MetricValue::Single(s) => s,
            MetricValue::MultiOp(_) => panic!("called as_single on MultiOp metric"),
        }
    }

    pub fn as_single_mut(&mut self) -> &mut SingleMetric {
        match self {
            MetricValue::Single(s) => s,
            MetricValue::MultiOp(_) => panic!("called as_single_mut on MultiOp metric"),
        }
    }

    pub fn is_single(&self) -> bool {
        matches!(self, MetricValue::Single(_))
    }

    pub fn is_multi_op(&self) -> bool {
        matches!(self, MetricValue::MultiOp(_))
    }

    pub fn op_names(&self) -> Vec<String> {
        match self {
            MetricValue::MultiOp(m) => m.keys().cloned().collect(),
            MetricValue::Single(_) => vec![],
        }
    }

    pub fn get_op_value(&self, op: &str) -> Option<f64> {
        match self {
            MetricValue::MultiOp(m) => m.get(op).map(|s| s.value),
            MetricValue::Single(_) => None,
        }
    }
}

/// All metrics for one node: metric_name → MetricValue
pub type NodeMetrics = IndexMap<String, MetricValue>;
