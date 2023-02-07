extern crate gbdt;

use gbdt::config::Config;
use gbdt::decision_tree::{DataVec, PredVec};
use gbdt::gradient_boost::GBDT;
use gbdt::input::{load, InputFormat};
use rayon::prelude::*;
use std::fs;


fn main() {
    let num_feat = 5;
    // load data
    let train_file = "PATH_TO/all_c125_latest_train.csv";
    let test_file = "PATH_TO/all_c125_latest_test.csv";

    let mut input_format = InputFormat::csv_format();
    input_format.set_feature_size(num_feat);
    input_format.set_label_index(num_feat);
    let is = 0..12;

    is.into_par_iter().for_each(|i| {
        let js = 0..3;
        for j in js {
            let ks = 0..6;
            for k in ks {
                let mut train_dv: DataVec =
                    load(train_file, input_format).expect("failed to load training data");
                let test_dv: DataVec =
                    load(test_file, input_format).expect("failed to load test data");

                let mut cfg = Config::new();
                cfg.set_feature_size(num_feat);
                cfg.set_max_depth(2 + j);
                cfg.set_iterations(85 + i * 10);
                cfg.set_shrinkage(0.03 + (k as f32) * 0.015);
//                cfg.set_loss("SquaredError");
                cfg.set_loss("LAD");
                //                    cfg.set_debug(true);
                cfg.set_training_optimization_level(2);

                // train and save the model
                let mut gbdt = GBDT::new(&cfg);
                gbdt.fit(&mut train_dv);
                //let model: GBDT = serde_json::from_str(model::MODEL).unwrap();

                // load the model and do inference
                let model = gbdt;
                //            dbg!(&test_dv);
                let predicted: PredVec = model.predict(&test_dv);
                let mut l1 = 0.;
                let mut l2 = 0.;
                let p = predicted.len() as f32;
                let mut l1_base = 0.;
                let mut l2_base = 0.;
                for i in 0..predicted.len() {
                    let d = predicted[i] - test_dv[i].label;
                    let b_d = test_dv[i].feature[0] - test_dv[i].label;
                    l1 += f32::abs(d) / p;
                    l2 += d * d / p;
                    l1_base += f32::abs(b_d) / p;
                    l2_base += b_d * b_d / p;
                }
                l2_base = l2_base.sqrt();
                l2 = l2.sqrt();
                println!("{},{},{},{} {},{},{}", l1, l1_base, l2, l2_base, i, j, k);
                fs::create_dir("models");
                model
                    .save_model(&format!(
                        "models/{}-{}-{}-{}.model",
                        l1,
                        2 + j,
                        85 + i * 10,
                        (0.03 + (k as f32) * 0.015)
                    ))
                    .expect("failed to save the model");
            }
        }
    });
}
