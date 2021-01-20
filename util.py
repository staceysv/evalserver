# utils and shared variables
import wandb

# projects

DEMO_PROJECT = "evalserve"
ANSWER_PROJECT = "answers_evalserve"
SUBMIT_PROJECT = "evalserve_predict"
NUM_EXAMPLES = 50


# classes
BDD_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'
]
BDD_IDS = list(range(len(BDD_CLASSES) - 1)) + [255]
class_set = wandb.Classes([{'name': name, 'id': id}
                           for name, id in zip(BDD_CLASSES, BDD_IDS)])

# wrapper for logging masks to W&B
def wb_mask(bg_img, pred_mask=[], true_mask=[]):
  masks = {}
  if len(pred_mask) > 0:
    masks["prediction"] = {"mask_data" : pred_mask}
  if len(true_mask) > 0:
    masks["ground truth"] = {"mask_data" : true_mask}
  return wandb.Image(bg_img, classes=class_set, masks=masks)


