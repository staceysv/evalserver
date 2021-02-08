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
import argparse
import numpy as np
import os
from PIL import Image
import util
import wandb

import inference
# fastai gymnastics: this needs to be defined in the main namespace
# to properly load a saved model
from inference import iou

def log_model_predictions(args):
  # intialize a candidate model entry run to the Entry project for the benchmark
  run = wandb.init(project=util.ENTRY_PROJECT, job_type="model_entry")
  run.config.team_name = args.team_name
  run.config.entry_name = args.model_name

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
  ENTRY_NAME = "entry_" + args.team_name
  test_res_at = wandb.Artifact(ENTRY_NAME, type="entry_predictions")
  test_res_at.add(result_table, "test_results")
  run.log_artifact(test_res_at)
  run.finish()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-t",
    "--team_name",
    type=str,
    default="",
    help="Team or participant name for this benchmark entry")
  parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    default="",
    help="Model name for this benchmark entry (optional)")
  args = parser.parse_args()
  log_model_predictions(args) 
