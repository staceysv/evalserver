# - run by the dataset owner
# - creates two artifacts:
#   - type=test_dataset, no lables, dsviz of just images & ids
#   - type=labeled_test_dataset, has labels joined
# - these are in different projects so labels can be held secret
import numpy as np
import os
from PIL import Image
import wandb

# same entity for now, two projects
# evalserve_answers"
# evalserve

DEMO_PROJECT = "evalserve"
ANSWER_KEY_PROJECT = "answers_evalserve"
SRC_IMAGES = "../../../BigData/bdd100K/bdd100k/seg/images/train/"
SRC_LABELS = "../../../BigData/bdd100K/bdd100k/seg/labels/train/"
NUM_EXAMPLES = 50

# upload 
columns = ["id", "raw_image"]
demo_table = wandb.Table(columns=columns)
# answers table
answer_cols = ["id", "raw_image", "labeled_image", "raw_label"]
answer_table = wandb.Table(columns=answer_cols) #columns)

# classes
BDD_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
]
BDD_IDS = list(range(len(BDD_CLASSES) - 1)) + [255]
class_set = wandb.Classes([{'name': name, 'id': id} 
                           for name, id in zip(BDD_CLASSES, BDD_IDS)])


demo_at = wandb.Artifact("test_data", type="test_dataset")
answers_at = wandb.Artifact("answer_key", type="labeled_test_dataset")

images = [f for f in os.listdir(SRC_IMAGES)[:NUM_EXAMPLES]]
for idx, image in enumerate(images):
  image_file = os.path.join(SRC_IMAGES, image)
  train_id = image.split(".")[0]
  label_file = os.path.join(SRC_LABELS, train_id + "_train_id.png")
  raw_image = wandb.Image(image_file)
  
  # get ground truth
  annotated = wandb.Image(image_file, classes=class_set,
                        masks={"ground_truth" : {"mask_data": np.array(Image.open(label_file))}})

  demo_at.add_file(image_file, os.path.join("images", image_file))
  demo_table.add_data(train_id, raw_image)

  answers_at.add_file(image_file, os.path.join("images", image_file))
  answers_at.add_file(label_file, os.path.join("labels", label_file))
  answer_table.add_data(train_id, raw_image, annotated, wandb.Image(label_file))

demo_at.add(demo_table, "train_data")
answers_at.add(answer_table, "answer_key")

# upload demo questions
run = wandb.init(project=DEMO_PROJECT, job_type="upload")
run.log_artifact(demo_at)
run.finish()

# upload answer key
run = wandb.init(project=ANSWER_KEY_PROJECT, job_type="upload")
run.log_artifact(answers_at)
run.finish()



