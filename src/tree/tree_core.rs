use async_trait::async_trait;

#[async_trait]
pub trait Model {
    async fn fit(&mut self, features: &[Vec<f64>], targets: &[isize]) {}
}
