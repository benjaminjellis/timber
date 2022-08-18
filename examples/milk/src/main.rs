use anyhow::Result;
use std::fs::read_to_string;
use std::time::Instant;
use timber::{tree::ClassificationTreeBuilder, Model};
use timber::metrics::Metric;

fn load_dataset() -> (Vec<Vec<f64>>, Vec<isize>) {
    let mut features = vec![];
    let mut targets = vec![];
    for _ in 0..1372 {
        features.push(vec![]);
    }
    let file = read_to_string("./data/data_banknote_authentication.txt")
        .expect("Coudln't read file to string");
    let rows = file.split("\r\n").collect::<Vec<&str>>();
    for (i, row) in rows.into_iter().enumerate() {
        let vals = row.split(",").collect::<Vec<&str>>();
        for (c, val) in vals.into_iter().enumerate() {
            if c < 4 {
                features[i].push(val.parse::<f64>().unwrap());
            } else {
                targets.push(val.parse::<isize>().unwrap())
            }
        }
    }
    (features, targets)
}

pub fn load_milk_train_dataset() -> (Vec<Vec<f64>>, Vec<isize>) {
    let features_file =
        read_to_string("./data/train_features.csv").expect("Coudln't read file to string");
    let rows = features_file.split("\n").collect::<Vec<&str>>();
    let n_rows = rows.len();
    let mut features = vec![];
    for _ in 0..n_rows {
        features.push(vec![]);
    }
    for (i, row) in rows.into_iter().enumerate() {
        let vals = row.split(",").collect::<Vec<&str>>();
        for val in vals {
            features[i].push(val.parse::<f64>().unwrap())
        }
    }

    let targets_file =
        read_to_string("./data/train_target.csv").expect("Coudln't read file to string");
    let rows = targets_file.split("\n").collect::<Vec<&str>>();
    let mut targets = vec![];
    for val in rows {
        targets.push(val.parse::<isize>().unwrap())
    }
    (features, targets)
}

#[tokio::main]
async fn main() -> Result<()> {
    let (features, targets) = load_milk_train_dataset();

    let mut model = ClassificationTreeBuilder::default().build().unwrap();

    let s = Instant::now();
    model.fit(&features, &targets).await;
    let train_time = (s.elapsed().as_secs_f64()) / 1e-6f64;
    println!("Train time: {train_time}");

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
        vec![6.8, 40.0, 1.0, 0.0, 1.0, 0.0, 245.0],
    ];

    let test_targets = vec![2isize, 0, 0, 1, 1, 0, 0, 2, 0, 1, 0, 1];

    let s = Instant::now();

    let score = model.score(&test_feature, &test_targets, Metric::Accuracy).await * 100f64;

    let score_time = (s.elapsed().as_secs_f64()) / 1e-6f64 ;
    println!("Score time: {score_time} μs");
    println!("Accuracy of model: {score}");
    Ok(())
}
