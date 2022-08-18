use std::collections::HashMap;
use crate::loss_functions::{weighted_gini_impurity, LossFunction};
use crate::tree::tree_core::{NodeData, NodeType};

/// Given a vector of tree splits pick one that minimises loss
///
/// O(n) where n os the number of splits
pub(crate) async fn pick_best_split(
    all_splits: &Vec<TreeSplit>,
    features: &[Vec<f64>],
    targets: &[isize],
    loss_fn: &LossFunction,
    filter: Option<&[usize]>,
) -> Option<SplitResult> {
    let mut best_split: Option<SplitResult> = None;
    let mut min_loss = 1.0;
    for split in all_splits {
        let split_result = check_split(features, targets, &split, loss_fn, filter).await;
        if split_result.loss < min_loss {
            min_loss = split_result.loss;
            best_split = Some(split_result);
        }
    }
    best_split
}

#[derive(Clone)]
pub(crate) struct SplitResult {
    pub(crate) value: f64,
    pub(crate) column: usize,
    pub(crate) loss: f64,
    pub(crate) node_1_indices: Vec<usize>,
    pub(crate) node_2_indices: Vec<usize>,
    pub(crate) majority_class: isize
}

/// Given a tree split, check the loss of that split
async fn check_split(
    features: &[Vec<f64>],
    targets: &[isize],
    split: &TreeSplit,
    loss_fn: &LossFunction,
    filter: Option<&[usize]>,
) -> SplitResult {
    // dynamic dispatch of loss function, will allow more ot be implemented later
    let loss_function: fn(Vec<&isize>, Vec<&isize>) -> f64 = match loss_fn {
        LossFunction::Gini => weighted_gini_impurity,
    };

    // targets for each leaf
    let mut node_1_targets = vec![];
    let mut node_2_targets = vec![];

    let mut node_1_indices = vec![];
    let mut node_2_indices = vec![];

    for (i, (record, record_target)) in features.iter().zip(targets).enumerate() {
        if record[split.column] > split.value {
            if let Some(index_filter) = filter {
                if index_filter.contains(&i) {
                    node_1_targets.push(record_target);
                    node_1_indices.push(i)
                }
            } else {
                node_1_targets.push(record_target);
                node_1_indices.push(i)
            }
        } else {
            if let Some(index_filter) = filter {
                if index_filter.contains(&i) {
                    node_2_targets.push(record_target);
                    node_2_indices.push(i)
                }
            } else {
                node_2_targets.push(record_target);
                node_2_indices.push(i)
            }
        }
    }
    // todo find a better way to get majority class
    let mut m = node_1_targets.clone();
    let loss = loss_function(node_1_targets, node_2_targets.clone());

    m.extend(&node_2_targets);
    let mut count_map = HashMap::new();
    for el in m{
        *count_map.entry(el).or_insert(0) += 1;
    }

    let majority_class = count_map.into_iter()
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|(k, _v)| k);

    SplitResult {
        value: split.value,
        column: split.column,
        loss,
        node_1_indices,
        node_2_indices,
        majority_class: *majority_class.unwrap()
    }
}

// todo docs
#[derive(Debug)]
pub struct TreeSplit {
    value: f64,
    column: usize,
    loss: Option<f64>,
}
// todo docs
struct BestSplit {
    value: f64,
    column: usize,
    loss: Option<f64>,
    node_1_indices: Vec<usize>,
    node_2_indices: Vec<usize>,
}

/// Generate all possible branch splits for a given set of features
/// todo docs
/// todo this is O(col) + 2 * O(col * row), expensive!! Try and reduce this
pub async fn generate_splits(features: &[Vec<f64>]) -> Vec<TreeSplit> {
    let no_columns = features[0].len();

    let mut columns = vec![];

    for _ in 0..no_columns {
        columns.push(vec![]);
    }

    for row in features {
        for i in 0..no_columns - 1 {
            columns[i].push(row[i])
        }
    }

    let mut all_splits = vec![];

    for col in 0..no_columns - 1 {
        columns[col].sort_by(|a, b| a.partial_cmp(b).expect("NaN in vector"));
        columns[col].dedup();
        for val in &columns[col] {
            all_splits.push(TreeSplit {
                value: *val,
                column: col,
                loss: None,
            })
        }
    }

    all_splits
}

pub(crate) async fn create_node_data(split: &SplitResult) -> NodeData {
    let node_type = match split.loss {
        y if y == 0f64 => NodeType::Leaf,
        _ => NodeType::Branch,
    };

    NodeData {
        node_type,
        column: split.column,
        value: split.value,
        loss: split.loss,
        majority_class: split.majority_class
    }
}
