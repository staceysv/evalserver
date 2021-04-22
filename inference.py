# inference.py
#-------------------
# A benchmark participant creates a test_model function for each candidate model.
# Given
# - the test data as an Artifact
# - the submission table with the expected columns
# this function loads the candidate model, runs inference on each item, and
# fills out the table with predictions
 
import numpy as np
import os
from PIL import Image
import util
import wandb

from pathlib import Path
from fastai.vision import *

# locations specific to model builder/participant
# (may change for different submissions)
# curently this mapping is manual--in an actual benchmark, this code would be written
# by each participating team
# 1. Find a run by team_name in the training project "segment_dsviz" or "dsviz-demo",
# e.g. "worldly-galaxy-8"
# 2. Look up the name and version of the Artifact of type=model created by this run,
# e.g. "resnet34:v1"
# 3. Plug in the training project, model name, and model version below
TRAIN_PROJECT= "segment_dsviz"
MODEL_NAME = "resnet34"
MODEL_VERSION = "latest"

# IOU loss function used for training model
SMOOTH = 1e-6
def iou(input, target):
  target = target.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
  intersection = (input.argmax(dim=1) & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
  union = (input.argmax(dim=1) | target).float().sum((1, 2))         # Will be zero if both are 0
  iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our division to avoid 0/0
  return iou.mean()

def test_model(run, test_data_artifact, test_table, train_project, model_name, model_version, entity):
  # download the test data to a local directory
  test_data_dir = test_data_artifact.download()

  # load a model previously trained and saved via Artifacts
  MODEL_AT = "{}/{}/{}:{}".format(entity, train_project, model_name, model_version)
  model_at = run.use_artifact(MODEL_AT)
  model_path = model_at.get_path(model_name).download()

  # a bit of Path gymnastics for fastai
  model_file = model_path.split("/")[-1]
  model_load_path = "/".join(model_path.split("/")[:-1])
  # load model via fastai
  unet_model = load_learner(Path(model_load_path), model_file)

  # load test images using fastai
  test_images_path =  Path(test_data_dir + "/images/")
  test_data = ImageList.from_folder(test_images_path)
  unet_model.data.add_test(test_data, tfms=None, tfm_y=False)

  test_batch = unet_model.data.test_ds
  test_ids = unet_model.data.test_ds.items

  # loop over all images in the test batch
  for i, img in enumerate(test_batch):
    # get the original image as an array
    orig_image = img[0]
    bg_image = image2np(orig_image.data*255).astype(np.uint8)

    # predict the labels using our model
    prediction = unet_model.predict(orig_image)[0]
    # cret
    prediction_mask = image2np(prediction.data).astype(np.uint8)
    # extract test image id (more fastai gymnastics)
    test_id = str(test_ids[i]).split("/")[-1].split(".")[0]

    # create prediction mask and log to table
    row = [str(test_id), util.wb_mask(bg_image, pred_mask=prediction_mask), wandb.Image(prediction_mask)]
    test_table.add_data(*row)
  return test_table
