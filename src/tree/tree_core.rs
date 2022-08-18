#[derive(Clone, Debug)]
pub struct Arena {
    pub nodes: Vec<Node>,
}

impl Arena {
    pub fn add_new_node(
        &mut self,
        data: NodeData,
        parent: &Option<NodeId>,
        child_type: Option<ChildType>,
    ) -> NodeId {
        // Get the next free index
        let next_index = self.nodes.len();
        let new_node_id = NodeId { index: next_index };
        // Push the node into the arena
        self.nodes.push(Node {
            parent: parent.to_owned(),
            first_child: None,
            second_child: None,
            data,
        });
        if let Some(parent_node) = parent {
            match child_type {
                Some(ChildType::First) => {
                    self.nodes[parent_node.index].first_child = Some(new_node_id.clone())
                }
                Some(ChildType::Second) => {
                    self.nodes[parent_node.index].second_child = Some(new_node_id.clone())
                }
                _ => (),
            }
        }

        new_node_id
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub parent: Option<NodeId>,
    pub first_child: Option<NodeId>,
    pub second_child: Option<NodeId>,
    pub data: NodeData,
}

#[derive(Clone, Debug)]
pub struct NodeId {
    pub index: usize,
}

///
#[derive(Clone, Debug, Copy)]
pub enum NodeType {
    Branch,
    Leaf,
}

///
///
/// # Arguments
/// * `node_type`
/// * `column`
/// * `value`
/// * `loss`
#[derive(Clone, Copy, Debug)]
pub struct NodeData {
    pub node_type: NodeType,
    pub column: usize,
    pub value: f64,
    pub loss: f64,
    pub majority_class: isize
}

impl Default for Arena {
    fn default() -> Self {
        Arena { nodes: vec![] }
    }
}

pub enum ChildType {
    First,
    Second,
}
