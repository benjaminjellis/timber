use std::array::from_ref;
use crate::loss_functions::LossFunction;
use crate::tree::tree_core::{ChildType, Node, NodeType};
use crate::tree::tree_utils::TreeSplit;
use crate::tree::{
    tree_core::{Arena, NodeId},
    tree_utils::{create_node_data, generate_splits, pick_best_split},
};
use crate::Model;
use crate::metrics::{Metric, accuracy};

use async_trait::async_trait;
use async_recursion::async_recursion;
use derive_builder::Builder;
use futures::future::join_all;
use std::time::Instant;
use std::sync::Arc;

#[derive(Default, Builder, Debug, Clone)]
pub struct ClassificationTree {
    #[builder(default)]
    pub loss_fn: LossFunction,
    #[builder(default = "2")]
    pub max_depth: usize,
    #[builder(default = "10")]
    pub min_samples_per_node: usize,
    #[builder(default)]
    pub nodes: Arena,
}



impl ClassificationTree {
    fn new(loss_fn: LossFunction, max_depth: usize, min_samples_per_node: usize) -> Self {
        Self {
            loss_fn,
            max_depth,
            min_samples_per_node,
            nodes: Arena { nodes: vec![] },
        }
    }


    #[async_recursion]
    async fn build_child_nodes(
        &mut self,
        child_node_1_filter: Option<&'async_recursion [usize]>,
        child_node_2_filter: Option<&'async_recursion [usize]>,
        all_splits: &Vec<TreeSplit>,
        features: &[Vec<f64>],
        targets: &[isize],
        loss_fn: &LossFunction,
        parent_node: &Option<NodeId>,
    ) {
        // first child node
        let best_split_child_1 =
            pick_best_split(&all_splits, features, targets, loss_fn, child_node_1_filter).await;
        let new_node_data_child_1 = create_node_data(&best_split_child_1.clone().unwrap()).await;
        // add new node as a child of the root node
        let new_node_id_child_1 =
            self.nodes
                .add_new_node(new_node_data_child_1, parent_node, Some(ChildType::First));

        // recursive call
        match new_node_data_child_1.node_type {
            NodeType::Leaf => (),
            NodeType::Branch => self.build_child_nodes(
                Some(&best_split_child_1.as_ref().unwrap().node_1_indices),
                Some(&best_split_child_1.as_ref().unwrap().node_2_indices),
                &all_splits,
                features,
                targets,
                &self.loss_fn.clone(),
                &Some(new_node_id_child_1),
            ).await,
        }

        // second child none
        let best_split_child_2 = pick_best_split(
            &all_splits,
            features,
            targets,
            &self.loss_fn,
            child_node_2_filter,
        )
            .await;

        let new_node_data_child_2 = create_node_data(&best_split_child_2.clone().unwrap()).await;
        let new_node_id_child_2 = self.nodes.add_new_node(
            new_node_data_child_2,
            parent_node,
            Some(ChildType::Second),
        );

        // recursive call
        match new_node_data_child_2.node_type {
            NodeType::Leaf => (),
            NodeType::Branch => self.build_child_nodes(
                Some(&best_split_child_2.as_ref().unwrap().node_1_indices),
                Some(&best_split_child_2.as_ref().unwrap().node_2_indices),
                &all_splits,
                features,
                targets,
                &self.loss_fn.clone(),
                &Some(new_node_id_child_2),
            ).await,
        }
    }

    #[async_recursion]
    async fn navigate_tree(&self, record: &[f64], current_node: &Node) -> isize {
        let mut child_node_id: &Option<NodeId>;
        if record[current_node.data.column] > current_node.data.value {
            child_node_id = &current_node.first_child;
        } else {
            child_node_id = &current_node.second_child;
        }
        if let Some(child_node) = child_node_id {
            let next_node = &self.nodes.nodes[child_node.index];
            self.navigate_tree(record, next_node).await
        } else {
            current_node.data.majority_class
        }
    }
}

#[async_trait]
impl Model for ClassificationTree {
    async fn fit(&mut self, features: &[Vec<f64>], targets: &[isize]) {
        let mut current_depth = 0usize;
        // todo add data validation

        let mut parent_node: Option<NodeId> = None;
        // calculate all splits just once
        let all_splits = generate_splits(features).await;

        // find the root node of the tree
        let root_best_split =
            pick_best_split(&all_splits, features, targets, &self.loss_fn, None).await;
        if let Some(ref root_split) = root_best_split {
            let root_node_data = create_node_data(&root_split).await;
            let root_node = self.nodes.add_new_node(root_node_data, &parent_node, None);
            current_depth += 1;

            match root_node_data.node_type {
                NodeType::Leaf => (),
                NodeType::Branch => self.build_child_nodes(
                    Some(&root_split.node_1_indices),
                    Some(&root_split.node_2_indices),
                    &all_splits,
                    features,
                    targets,
                    &self.loss_fn.clone(),
                    &Some(root_node),
                ).await,
            }
        } else {
            panic!("Couldn't find a good split when searching, this occurred when trying to build the root node of the tress")
        }
    }

    async fn predict(&self, features: &[Vec<f64>]) -> Vec<isize> {
        let root_node = &self.nodes.nodes[0];
        let pred_futures = features
            .iter()
            .map(|record |async {
                self.navigate_tree(record, root_node).await
            }
            )
            .collect::<Vec<_>>();
        let preds = join_all(pred_futures).await;
        preds
    }


    async fn predict_proba(&self, features: &[Vec<f64>]) -> Vec<(f64, f64)>{
        unimplemented!()
    }

    async fn score(&self, features: &[Vec<f64>], targets: &[isize], metric: Metric) -> f64{
        let preds = &self.predict(features).await;

        let metirc_fn = match metric {
            Metric::Accuracy => accuracy,
            _ => panic!()
        };

        let metric_result = metirc_fn(preds, targets);
        metric_result
    }
}
