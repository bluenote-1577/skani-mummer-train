extern crate gbdt;

use skani_training_serialize::model;
use gbdt::config::Config;
use gbdt::decision_tree::{DataVec, PredVec};
use gbdt::fitness::almost_equal_thrs;
use gbdt::gradient_boost::GBDT;
use gbdt::input::{load, InputFormat};

use serde::{Deserialize, Serialize};
use serde_json::Result;
fn main() {
    // load data
    let train_file = "/home/jshaw/scratch/2023_skani_training/latest_train.csv";
    let test_file = "/home/jshaw/scratch/2023_skani_training/latest_train.csv";

    let mut input_format = InputFormat::csv_format();
    input_format.set_feature_size(12);
    input_format.set_label_index(12);

    let mut train_dv: DataVec =
        load(train_file, input_format).expect("failed to load training data");
    let test_dv: DataVec = load(test_file, input_format).expect("failed to load test data");

    let train = true;
    if train {
        let mut cfg = Config::new();
        cfg.set_feature_size(12);
        cfg.set_max_depth(4);
        cfg.set_iterations(200);
        cfg.set_shrinkage(0.1);
        cfg.set_loss("MeanSquared");
        cfg.set_debug(true);
        cfg.set_training_optimization_level(2);

        
        // train and save the model
        let mut gbdt = GBDT::new(&cfg);
        gbdt.fit(&mut train_dv);
        gbdt.save_model("gbdt.model")
            .expect("failed to save the model");
    }
    //let model: GBDT = serde_json::from_str(model::MODEL).unwrap();


    // load the model and do inference
//    let model = GBDT::load_model("gbdt.model").expect("failed to load the model");
//    dbg!(&test_dv);
//    let predicted: PredVec = model.predict(&test_dv);
//    for i in 0..100 {
//        println!("pred {},true {}, base {}", predicted[i], test_dv[i].label, &test_dv[i].feature[0]);
//    }
}
