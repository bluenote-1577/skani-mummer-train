# skani-mummer-train
Repository for training skani on mummer data (ANIm) to improve accuracy.

## Introduction

This repo is a submodule for [skani](https://github.com/bluenote-1577/skani/) used for training a gradient boosted tree for debiasing ANI calculations. 
This repo is likely not pertinent to the average user, but you may find it useful if you want to improve the learned ANI debiasing algorithm or figure out how 
it was trained. 

## How to use

### Inputs to generated training/test data

The `anim-nayfach.txt` and `c125_nayfach_skani.txt` files show the desired ANIm and skani inputs. 

- The ANIm file can be any sort of gold ANI calculation in generality,
all that we require is the reference name, query name, and predicted ANI are the first three columns (the last two columns are not used). 

- The skani file is generated by running skani the with `--detailed` command to give extra output information that is used for training. **Make sure to 
use the --no-learned-ani option in skani for generating this file to avoid training on trained data**. 

### Generating testing/training data

Run the `skani_anim_to_csv.py` script. Modify the arguments accordingly -- replace the "nayfach" files with your training set, and the "elgg" files with your testing set.

This produces two csv files: `c125_latest_test.csv, c125_latest_train.csv`. Of course, c125 is labelled this way because skani uses c = 125 as 
the default parameter, but feel free to rename however you want depending on how the skani output was generated. 

### Training/Testing

To train, use `cargo run`. The only dependency is that rust is installed. This will start training a gradient boosted tree model and do a grid search across
a set of parameters. The output models will be serialized into a directory called `models`. The models will be of the form `0.18635842-7-230-0.08.model`, where the
first numbers indicate the L1 loss, tree depth, # of trees, and learning rate. 

### Deploying model to skani

To make deployment minimally painful for users, I have decided to simply serialize the model into a json string into code. skani deserializes the json string at runtime,
but because the string is simply a variable in memory, we can statically compile rust. 

Look at the `model_to_src.sh` script in the skani repository for
how to turn integrate an output model into skani.
