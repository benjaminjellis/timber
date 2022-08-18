use std::fs::read_to_string;

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
