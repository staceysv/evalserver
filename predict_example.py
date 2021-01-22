# predict_example.py
#----------------------------
# The model builder (benchmark participant) runs this script.
# Given
# - a type=test_dataset Artifact containing the test images to be labeled and
# - a type=model Artifact containing a trained model (for convenience, assuming the
#   participant has already logged the corresponding training run to wandb)
# evaluate the model on the test data and upload
# - a type=predictions Artifact containing the model's predictions on the test data.
#
# Running model inference is factored out into a separate module
# for convenience, and inference.py implements "test_model", which takes in
# - the test_dataset Artifact
# - a table with the expected format of predictions (image id, visualized prediction mask,
#   and raw prediction mask) to be filled out using the model's predictions

import numpy as np
import os
from PIL import Image
import util
import wandb

import inference
# fastai gymnastics: this needs to be defined in the main namespace
# to properly load a saved model
from inference import iou

# intialize a test run to the Submit project for the benchmark
run = wandb.init(project=util.SUBMIT_PROJECT, job_type="test")

# get the latest version of the test data from the Demo project
TEST_DATA_AT = "{}/test_data:latest".format(util.DEMO_PROJECT)
test_data_artifact = run.use_artifact(TEST_DATA_AT)

# create a submission table with the expected fields:
# - test image id
# - prediction image as a wandb mask for best visualization
#   (this is optional but much easier to read than the raw mask)
# - raw mask (no wandb mask, just the predicted labels) 
test_table = wandb.Table(columns=["id", "prediction", "raw_mask"])

# fill table with predictions
result_table = inference.test_model(run, test_data_artifact, test_table)

# log filled result table to artifact
test_res_at = wandb.Artifact("test_predictions", type="predictions")
test_res_at.add(result_table, "test_results")
run.log_artifact(test_res_at)
run.finish()
