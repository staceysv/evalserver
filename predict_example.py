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
  run = wandb.init(project=args.entry_project, job_type="model_entry")
  run.config.team_name = args.team_name
  run.config.entry_name = args.model_name

  # get the latest version of the test data from the Demo project
  TEST_DATA_AT = "{}/{}:latest".format(args.demo_project, args.test_data)
  test_data_artifact = run.use_artifact(TEST_DATA_AT)

  # create a submission table with the expected fields:
  # - test image id
  # - prediction image as a wandb mask for best visualization
  #   (this is optional but much easier to read than the raw mask)
  # - raw mask (no wandb mask, just the predicted labels) 
  test_table = wandb.Table(columns=["id", "prediction", "raw_mask"])

  # fill table with predictions
  result_table = inference.test_model(run, test_data_artifact, test_table, \
                 args.train_project, args.model_name, args.entity, args.model_version)

  # log filled result table to artifact
  ENTRY_NAME = "entry_" + args.team_name
  test_res_at = wandb.Artifact(ENTRY_NAME, type="entry_predictions")
  test_res_at.add(result_table, "test_results")
  run.log_artifact(test_res_at)
  run.finish()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    help="Model name for this benchmark entry")
  parser.add_argument(
    "-d",
    "--demo_project",
    type=str,
    default=util.DEMO_PROJECT,
    help="demo project name: where participants will find the test data")
  parser.add_argument(
    "--test_data",
    type=str,
    default="test_data",
    help="name of test data artifact in demo project")
  parser.add_argument(
    "--entity",
    type=str,
    default="stacey",
    help="entity (team username) which trained the candidate model")
  parser.add_argument(
    "-e",
    "--entry_project",
    type=str,
    default=util.ENTRY_PROJECT,
    help="entry project name: where participants upload test predictions")
  parser.add_argument(
    "-s",
    "--train_project",
    type=str,
    default="",
    help="Team project where candidate model was saved")
  parser.add_argument(
    "-v",
    "--model_version",
    type=str,
    default="latest",
    help="Model version for this benchmark entry")

  args = parser.parse_args()
  log_model_predictions(args) 
