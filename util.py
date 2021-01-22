# utils and shared variables
import wandb

# projects

DEMO_PROJECT = "evalserve"
ANSWER_PROJECT = "answers_evalserve"
SUBMIT_PROJECT = "evalserve_predict"
NUM_EXAMPLES = 50

# total number of pixels in each training image
TOTAL_PIXELS = float(720*1280)

# classes
BDD_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
]
BDD_IDS = list(range(len(BDD_CLASSES) - 1)) + [255]
class_set = wandb.Classes([{'name': name, 'id': id}
                           for name, id in zip(BDD_CLASSES, BDD_IDS)])

CS_CATEGORIES = {
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


