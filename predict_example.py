# run by the model buuilder
# given a test_dataset artifact, and a model artifact
# evaluate model and upload predictions as joined table, type = predictions
# - easily plug in own model evaluation code
# steps:
# - download data
# - iterate
# - construct prredictions table

import numpy as np
import os
from PIL import Image
import wandb

from pathlib import Path
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.callback import Callback
import json
from wandb.fastai import WandbCallback
from functools import partialmethod


DEMO_PROJECT = "evalserve"
PREDICT_PROJECT = "evalserve_predict"
TRAINING_PROJECT = "dsviz-segment"
MODEL_NAME = "resnet18"

run = wandb.init(project=PREDICT_PROJECT, job_type="test")

# get test data
TEST_DATA_AT = "{}/test_data:latest".format(DEMO_PROJECT)
test_data_at = run.use_artifact(TEST_DATA_AT) 
test_data_local = test_data_at.download()

# get model...? one we already trained
MODEL_AT = "{}/{}:latest".format(TRAINING_PROJECT, MODEL_NAME)
model_at = run.use_artifact(MODEL_AT)
model_path = model_at.get_path(MODEL_NAME).download()

# evaluate model
# a bit of path gymnastics for fastai
model_file = model_path.split("/")[-1]
model_load_path = "/".join(model_path.split("/")[:-1])
# load model via fastai
unet_model = load_learner(Path(model_load_path), model_file)

print("WORKING!")
#-----------------------

# download test images so they are available locally
test_dir = test_artifact.download(LOCAL_MODEL_DIR)
test_images_path =  Path(test_dir + "/images/")
print("TEST: ", test_images_path)
# create test dataset in fastai
test_data = ImageList.from_folder(test_images_path)
unet_model.data.add_test(test_data, tfms=None, tfm_y=False)

test_batch = unet_model.data.test_ds
test_ids = unet_model.data.test_ds.items

test_res_at = wandb.Artifact("test_pred_" + wandb.run.id, "test_preds")
test_table = wandb.Table(columns=["id", "prediction"])

# store predictions across all resnet model variants as one artifact
model_test_at = wandb.Artifact("resnet_results", "model_test")
model_test_table = wandb.Table(columns=["id", "prediction"])

for i, img in enumerate(test_batch):
   # log raw image as array
   orig_image = img[0]
   bg_image = image2np(orig_image.data*255).astype(np.uint8)

   # our prediction
   prediction = unet_model.predict(orig_image)[0]
   prediction_mask = image2np(prediction.data).astype(np.uint8)
   test_id = str(test_ids[i]).split("/")[-1].split(".")[0]

   # create prediction mask and log to table
   row = [str(test_id), wb_mask(bg_image, pred_mask=prediction_mask)]
   test_table.add_data(*row)
   model_test_table.add_data(*row)

print("Saving data to WandB...")
test_res_at.add(test_table, "test_results")
run.log_artifact(test_res_at)
model_test_at.add(model_test_table, "model_test_results")
run.log_artifact(model_test_at)
