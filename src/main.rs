extern crate gbdt;

use gbdt::config::Config;
use gbdt::decision_tree::{DataVec, PredVec};
use gbdt::gradient_boost::GBDT;
use gbdt::input::{load, InputFormat};
use rayon::prelude::*;

fn main() {
    // load data
    let train_file = "/home/jshaw/scratch/2023_skani_training/c125_latest_train.csv";
    let test_file = "/home/jshaw/scratch/2023_skani_training/c125_latest_test.csv";

    let mut input_format = InputFormat::csv_format();
    input_format.set_feature_size(14);
    input_format.set_label_index(14);
    let train = true;
    let is = 0..8;

    if train {
        is.into_par_iter().for_each(|i| {
            let js = 0..3;
            for j in js {
                let ks = 0..3;
                for k in ks {
                    let mut train_dv: DataVec =
                        load(train_file, input_format).expect("failed to load training data");
                    let test_dv: DataVec =
                        load(test_file, input_format).expect("failed to load test data");

                    let mut cfg = Config::new();
                    cfg.set_feature_size(14);
                    cfg.set_max_depth(3 + j);
                    cfg.set_iterations(100  + i * 10);
                    cfg.set_shrinkage(0.05 + (k as f32) * 0.03);
                    //                    cfg.set_loss("SquaredError");
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
                    model
                        .save_model(&format!(
                            "models/{}-{}-{}-{}.model",
                            l2,
                            3 + j,
                            100 + i * 10,
                            (0.04 + (k as f32) * 0.03)
                        ))
                        .expect("failed to save the model");
                }
            }
        });
    } else {
        let mut train_dv: DataVec =
            load(train_file, input_format).expect("failed to load training data");
        let test_dv: DataVec = load(test_file, input_format).expect("failed to load test data");

        let model = GBDT::from_xgoost_dump(
            "/home/jshaw/scratch/2023_skani_training/test.model",
            "reg:squarederror",
        )
        .unwrap();
        //        let model = GBDT::load_model("gbdt.model").expect("failed to load the model");
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
            l2 += f32::sqrt(d * d) / p;
            l1_base += f32::abs(b_d) / p;
            l2_base += f32::sqrt(b_d * b_d) / p;
            //println!("base {}, label {}, predicted{}",test_dv[i].feature[0], test_dv[i].label, predicted[i]);
        }
        println!("{},{}", l1, l1_base);
    }
}
