# upload_test_data.py
# -----------------------------
# The dataset owner runs this script to create two Artifacts:
#   - type=test_dataset: the ids & images, no labels
#   - type=labeled_test_dataset, contains the correct labels
# These are stored in different projects (Demo project and Answers
# project, respectively) so the test answers can be held secret.

import numpy as np
import os
from PIL import Image
import util
import wandb

# path to local dataset
SRC_IMAGES = "../../../BigData/bdd100K/bdd100k/seg/images/train/"
SRC_LABELS = "../../../BigData/bdd100K/bdd100k/seg/labels/train/"

# create test data artifact and table (ids and images only) 
demo_at = wandb.Artifact("test_data", type="test_dataset")
columns = ["id", "raw_image"]
demo_table = wandb.Table(columns=columns)

# create the answers (labeled) artifact 
answers_at = wandb.Artifact("answer_key", type="labeled_test_dataset")
answer_cols = ["id", "raw_image", "labeled_image", "raw_label"]
answer_table = wandb.Table(columns=answer_cols)

# upload images 
images = [f for f in os.listdir(SRC_IMAGES)][:util.NUM_EXAMPLES]
for idx, image in enumerate(images):
  train_id = image.split(".")[0]
  image_file = os.path.join(SRC_IMAGES, image)
  raw_image = wandb.Image(image_file)
  
  label_file = os.path.join(SRC_LABELS, train_id + "_train_id.png")
 
  # visualize the labels with full-featured semantic segmentation
  annotated = wandb.Image(image_file, classes=util.class_set,
                        masks={"ground_truth" : {"mask_data": np.array(Image.open(label_file))}})

  # add images only to the Demo visualization table
  demo_table.add_data(train_id, raw_image)
  # add files to artifact (optional, needed here to explicitly set the path for download) 
  demo_at.add_file(image_file, os.path.join("images", image))

  # add images and labels to the Answer visualization table and artifact
  answer_table.add_data(train_id, raw_image, annotated, wandb.Image(np.array(Image.open(label_file))))
  answers_at.add_file(image_file, os.path.join("images", image))
  answers_at.add_file(label_file, os.path.join("labels", train_id + "_train_id.png"))

# add tables to artifacts
demo_at.add(demo_table, "test_data")
answers_at.add(answer_table, "answer_key")

# upload the unlabeled test data to the Demo project
run = wandb.init(project=util.DEMO_PROJECT, job_type="upload_test_data")
run.log_artifact(demo_at)
run.finish()

# upload the labeled test data to the Answer project
run = wandb.init(project=util.ANSWER_PROJECT, job_type="upload_test_data")
run.log_artifact(answers_at)
run.finish()



