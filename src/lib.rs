pub mod loss_functions;
pub mod tree;
pub mod metrics;
mod utils;

use async_trait::async_trait;

#[async_trait]
pub trait Model {
    async fn fit(&mut self, features: &[Vec<f64>], targets: &[isize]);

    async fn predict(&self, features: &[Vec<f64>]) -> Vec<isize>;

    async fn predict_proba(&self, features: &[Vec<f64>]) -> Vec<(f64, f64)>;

    async fn score(&self, features: &[Vec<f64>], targets: &[isize], metric: metrics::Metric) -> f64;
}

#[cfg(test)]
mod tests {
    use crate::tree::ClassificationTreeBuilder;
    use crate::utils::load_milk_train_dataset;
    use crate::Model;
    use std::time::Instant;

    #[tokio::test]
    async fn test_api() {
        let (features, targets) = load_milk_train_dataset();
        let s = Instant::now();
        let mut model = ClassificationTreeBuilder::default().build().unwrap();

        model.fit(&features, &targets).await;
        let e = s.elapsed().as_secs_f32();

        let test_feature = vec![
            vec![6.6, 40.0, 1.0, 0.0, 1.0, 1.0, 255.0],
            vec![4.5, 60.0, 0.0, 1.0, 1.0, 1.0, 250.0],
            vec![9.0, 43.0, 1.0, 0.0, 1.0, 1.0, 250.0],
            vec![6.8, 45.0, 0.0, 0.0, 0.0, 1.0, 255.0],
            vec![6.6, 38.0, 0.0, 0.0, 0.0, 0.0, 255.0],
            vec![8.1, 66.0, 1.0, 0.0, 1.0, 1.0, 255.0],
            vec![9.5, 34.0, 1.0, 1.0, 0.0, 1.0, 255.0],
            vec![6.8, 45.0, 0.0, 1.0, 1.0, 1.0, 255.0],
            vec![9.0, 43.0, 1.0, 1.0, 1.0, 1.0, 248.0],
            vec![6.5, 38.0, 1.0, 0.0, 1.0, 0.0, 255.0],
            vec![3.0, 40.0, 1.0, 1.0, 1.0, 1.0, 255.0],
            vec![6.8, 40.0, 1.0, 0.0, 1.0, 0.0, 245.0]];

        let pred = model.predict(&test_feature).await;
        dbg!(pred);
    }
}
