# - run by the dataset owner
# - creates two artifacts:
#   - type=test_dataset, no lables, dsviz of just images & ids
#   - type=labeled_test_dataset, has labels joined
# - these are in different projects so labels can be held secret
import numpy as np
import os
from PIL import Image
import util
import wandb

SRC_IMAGES = "../../../BigData/bdd100K/bdd100k/seg/images/train/"
SRC_LABELS = "../../../BigData/bdd100K/bdd100k/seg/labels/train/"

# upload 
columns = ["id", "raw_image"]
demo_table = wandb.Table(columns=columns)

# answers table
answer_cols = ["id", "raw_image", "labeled_image", "raw_label"]
answer_table = wandb.Table(columns=answer_cols) #columns)

demo_at = wandb.Artifact("test_data", type="test_dataset")
answers_at = wandb.Artifact("answer_key", type="labeled_test_dataset")

images = [f for f in os.listdir(SRC_IMAGES)][:util.NUM_EXAMPLES]
for idx, image in enumerate(images):
  print("image: ", image)
  image_file = os.path.join(SRC_IMAGES, image)
  train_id = image.split(".")[0]
  label_file = os.path.join(SRC_LABELS, train_id + "_train_id.png")
  raw_image = wandb.Image(image_file)
  
  # get ground truth
  annotated = wandb.Image(image_file, classes=util.class_set,
                        masks={"ground_truth" : {"mask_data": np.array(Image.open(label_file))}})

  demo_at.add_file(image_file, os.path.join("images", image))
  demo_table.add_data(train_id, raw_image)

  answers_at.add_file(image_file, os.path.join("images", image))
  answers_at.add_file(label_file, os.path.join("labels", train_id + "_train_id.png"))
  answer_table.add_data(train_id, raw_image, annotated, wandb.Image(np.array(Image.open(label_file))))

demo_at.add(demo_table, "train_data")
answers_at.add(answer_table, "answer_key")

# upload demo questions
run = wandb.init(project=util.DEMO_PROJECT, job_type="upload")
run.log_artifact(demo_at)
run.finish()

# upload answer key
run = wandb.init(project=util.ANSWER_PROJECT, job_type="upload")
run.log_artifact(answers_at)
run.finish()



