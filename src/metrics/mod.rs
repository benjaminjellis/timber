mod accuracy;
pub use accuracy::*;

pub enum Metric{
    Accuracy,
    F1Score,
    Precision,
    Recall
}