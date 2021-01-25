# utils and shared variables
import wandb
import numpy as np

# wandb project names used to separate ground truth labels
# and participant submissions. Participants view the Demo
# project and write their predictions to the Submit project.
# Benchmark owners evaluate and organize all submissions in
# the Answer project.

DEMO_PROJECT = "evalserve"
ANSWER_PROJECT = "answers_evalserve"
SUBMIT_PROJECT = "evalserve_predict"

# total images to log (for demo purposes)
NUM_EXAMPLES = 50

# total number of pixels in each training image
TOTAL_PIXELS = float(720*1280)

# classes from the Berkeley Deep Drive 100K dataset
# https://bdd-data.berkeley.edu/
BDD_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
]
BDD_IDS = list(range(len(BDD_CLASSES) - 1)) + [255]

# convenience wrapper for wandb semantic segmentation visualization
class_set = wandb.Classes([{'name': name, 'id': id}
                           for name, id in zip(BDD_CLASSES, BDD_IDS)])

# Cityscapes category definitons (which classes are in which category)
CITYSCAPES_CATEGORIES = {
  "flat" : ["road", "sidewalk"],
  "human" : ["person", "rider"],
  "vehicle" : ["car", "bus", "truck", "motorcycle", "bicycle", "train"],
  "construction" : ["building", "wall", "fence"],
  "object" : ["pole", "traffic light", "traffic sign"],
  "nature" : ["vegetation", "terrain"],
  "sky" : ["sky"]
}

CITYSCAPE_IDS = {
  "flat" : [0, 1],
  "human" : [11, 12], 
  "vehicle" : [13, 14, 15, 16, 17, 18],
  "construction" : [2, 3, 4], 
  "object" : [5, 6, 7],
  "nature" : [8, 9],
  "sky" : [10] 
}

# wrapper for logging masks to W&B
def wb_mask(bg_img, pred_mask=[], true_mask=[]):
  masks = {}
  if len(pred_mask) > 0:
    masks["prediction"] = {"mask_data" : pred_mask}
  if len(true_mask) > 0:
    masks["ground truth"] = {"mask_data" : true_mask}
  return wandb.Image(bg_img, classes=class_set, masks=masks)

# smooth fractions to avoid division by zero and always return the same type
def smooth(num, denom):
  if np.isclose(denom, 0):
    return np.nan_to_num(0.0, 0.0, 0.0, 0.0)
  else:
    return np.nan_to_num(num / denom, 0.0, 0.0, 0.0)

# generic IOU for prediction masks and a given class id
def iou_2D(mask_guess, mask_b, class_id):
    # 4x upsample the guessed mask because of the size reduction in training
    # mask_a = mask_guess.repeat(2, axis=0).repeat(2, axis=1)
    intersection = np.nan_to_num(((mask_a == class_id) & (mask_b == class_id)).sum(axis=(0,1)), 0, 0, 0)
    union = np.nan_to_num(((mask_a == class_id) | (mask_b == class_id)).sum(axis=(0,1)), 0, 0, 0)
    if np.isclose(union, 0):
      return np.nan_to_num(0.0, 0.0, 0.0, 0.0)
    else:
      return np.nan_to_num(intersection / union, 0.0, 0.0, 0.0)
