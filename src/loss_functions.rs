use std::collections::HashMap;
use std::hash::Hash;


#[derive(Default, Clone, Debug)]
pub enum LossFunction {
    #[default]
    Gini,
}

/// Gini impurity of a vector
///
/// # Arguemnts
/// * `data` - the vector to calculate gini impurity of
///
/// # Returns
/// the gini impurity
pub fn gini_impurity<I>(data: Vec<&I>) -> f64
where
    I: Eq,
    I: Hash,
{
    if data.is_empty() {
        return 1.0;
    }
    fn p_squared(count: usize, len: f64) -> f64 {
        let p = count as f64 / len;
        p * p
    }
    let len = data.len() as f64;
    let mut count = HashMap::new();
    for value in data {
        *count.entry(value).or_insert(0) += 1;
    }
    let sum: f64 = count
        .into_iter()
        .map(|(_, c)| c)
        .map(|x| p_squared(x, len))
        .sum();
    1.0f64 - sum
}

/// Calculate the weighted average of the gini impurity of two nodes for a given split
///
/// # Arguments
/// * `node_1_targets` - targets for node 1
/// * `node_2_targets` - targets for node 2
///
/// # Returns
/// the weighted gini impurity
pub fn weighted_gini_impurity(node_1_targets: Vec<&isize>, node_2_targets: Vec<&isize>) -> f64 {
    let node_1_len = node_1_targets.len();
    let node_2_len = node_2_targets.len();

    let node_1_weight = node_1_len as f64 / (node_1_len + node_2_len) as f64;
    let node_2_weight = node_2_len as f64 / (node_1_len + node_2_len) as f64;

    node_1_weight * gini_impurity(node_1_targets) + node_2_weight * gini_impurity(node_2_targets)
}
