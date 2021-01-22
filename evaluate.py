# - run by dataset owner
# - needs a type=predictions artifact and a type=test_dataset_with_label artifact
# - log per-image metrics, esp total false positive and false negative pixels
# - run summary metrics, could be set on run or in metadata
# - evaluate everything for which we don't find a job_type = evaluate )

import numpy as np
import os
import pandas as pd
from PIL import Image
import util
import wandb

def per_category_metrics(ious, fps, fns):
  metrics = {}
  for metric, category in zip(["iou", "fps", "fns"], [ious, fps, fns]):
    for category_name, ids in util.CITYSCAPE_IDS.items():
      category_metric = np.mean([category[class_id] for class_id in ids])
      metrics["cat_"+metric + "_" + category_name] = category_metric

  # average all category
  category_average = np.mean([v for v in metrics.values()])
  metrics["mean_category_iou"] = category_average
  return metrics

# ious is util.NUM_EXAMPLES rows of 19 columns
def mean_metrics(ious, fps, fns):
  # mean per class ious
  ious_np = np.mean(ious, axis=0)
 
  # per-class mean iou
  per_class_mean_iou = np.mean(ious_np)

  fps_floats = np.array(fps)
  norm_fps = fps_floats / util.TOTAL_PIXELS
  # mean normalized per class fps
  mean_norm_fps = np.mean(norm_fps, axis=0)
  fns_floats = np.array(fns)
  norm_fns = fns_floats / util.TOTAL_PIXELS
  mean_norm_fns = np.mean(norm_fns, axis=0)

  return per_class_mean_iou, ious_np, mean_norm_fps, mean_norm_fns
  

# 2D version
# return false pos and false neg per class
def iou_flat(mask_guess, mask_b, class_id):
    # 4x upsample mask_a
    mask_a = mask_guess.repeat(2, axis=0).repeat(2, axis=1)
    # cross your fingers
    intersection = np.nan_to_num(((mask_a == class_id) & (mask_b == class_id)).sum(axis=(0,1)), 0, 0, 0)
    union = np.nan_to_num(((mask_a == class_id) | (mask_b == class_id)).sum(axis=(0,1)), 0, 0, 0)
    if np.isclose(union, 0):
      return np.nan_to_num(0.0, 0.0, 0.0, 0.0)
    else:
      return np.nan_to_num(intersection / union, 0.0, 0.0, 0.0)

def smooth(num, denom):
  if np.isclose(denom, 0):
    return np.nan_to_num(0.0, 0.0, 0.0, 0.0)
  else:
    return np.nan_to_num(num / denom, 0.0, 0.0, 0.0)

# for every class_id
def pixel_count(guess, truth):
  iou_scores = {}
  net_fp = 0
  net_fn = 0
  net_tp = 0
  # NOTE: void class doesn't contribute
  for class_id in range(19):
    fp = ((guess == class_id) & (truth != class_id)).sum(axis=(0,1))
    fn = ((guess != class_id) & (truth == class_id)).sum(axis=(0,1))
    tp = ((guess == class_id) & (truth == class_id)).sum(axis=(0,1))
    iou = smooth(float(tp), float(tp + fp + fn)) #, 0.0, 0.0, 0.0)
    iou_scores[class_id] = {"fp": fp, "fn" : fn, "tp" : tp, "iou" : iou}
    net_fp += fp
    net_fn += fn
    net_tp += tp
  net_iou = smooth(float(net_tp), float(net_tp + net_fp + net_fn))
  iou_scores["net_tp"] = net_tp
  iou_scores["net_fp"] = net_fp
  iou_scores["net_fn"] = net_fn
  iou_scores["net_iou"] = net_iou
  return iou_scores 

def score_masks(mask_guess, mask_truth):
  guess = mask_guess.repeat(2, axis=0).repeat(2, axis=1)
  return pixel_count(guess, mask_truth)


def iou_3D(mask_a, mask_b, class_id):
    return np.nan_to_num(((mask_a == class_id) & (mask_b == class_id)).sum(axis=(1,2)) / ((mask_a == class_id) | (mask_b == class_id)).sum(axis=(1,2)), 0, 0, 0)

run = wandb.init(project=util.ANSWER_PROJECT, job_type="evaluate")

# get predictions
predictions_at = run.use_artifact("{}/test_predictions:latest".format(util.SUBMIT_PROJECT))
guess_table = predictions_at.get("test_results")

guess_ids = np.array([guess_table.data[i][0] for i in range(util.NUM_EXAMPLES)])
guess_images = np.array([np.array(guess_table.data[i][2]._image) for i in range(util.NUM_EXAMPLES)])
guess_pretty = [guess_table.data[i][1] for i in range(util.NUM_EXAMPLES)]

guess = pd.DataFrame({'id' : guess_ids, "guess": [i for i in range(len(guess_images))]}) 

# get ground truth
answers_at = run.use_artifact("{}/answer_key:latest".format(util.ANSWER_PROJECT))
true_table = answers_at.get("answer_key")

true_ids = np.array([true_table.data[i][0] for i in range(util.NUM_EXAMPLES)])
true_images = np.array([np.array(true_table.data[i][3]._image) for i in range(util.NUM_EXAMPLES)])
true_pretty = [true_table.data[i][2] for i in range(util.NUM_EXAMPLES)]

truth = pd.DataFrame({'id' : true_ids, "truth" : [i for i in range(len(true_images))]})

results = guess.join(truth, lsuffix="_guess", rsuffix="_truth")

# now log a final table
columns=["id", "prediction", "ground_truth"]
columns.extend(["overall IOU", "false positive", "false negative"])

ious = []
fps = []
fns = []
# make a full score table
columns.extend(["iou_" + s for s in util.BDD_CLASSES[:-1]])
columns.extend(["fn_" + s for s in util.BDD_CLASSES[:-1]])
columns.extend(["fp_" + s for s in util.BDD_CLASSES[:-1]])
full_num_table = wandb.Table(columns=columns)
for index, row in results.iterrows():
  #s = iou_flat(guess_images[row["guess"]], true_images[row["truth"]], 0)
  scores = score_masks(guess_images[row["guess"]], true_images[row["truth"]])
  # all the scores per class
  r = [row["id_truth"], guess_pretty[row["guess"]], true_pretty[row["truth"]], scores["net_iou"], scores["net_fp"], scores["net_fn"]]
  class_ious = [scores[class_id]["iou"] for class_id in range(19)]
  ious.append(class_ious)
  r.extend(class_ious)
  class_fns = [scores[class_id]["fn"] for class_id in range(19)]
  fns.append(class_fns)
  r.extend(class_fns)
  class_fps = [scores[class_id]["fp"] for class_id in range(19)]
  fps.append(class_fps)
  r.extend(class_fps)
  
  full_num_table.add_data(*r)
  #wandb.log({"net_fp" : scores["net_fp"], "net_fn" : scores["net_fn"], "net_iou" : scores["net_iou"]})

# what if we make this a dataframe
# now what.
# - report mean across all ious
# - report mean across all fps and fns (TODO: normalize?)
# - report per-category mean iou
per_class_mean_iou, mean_ious, mean_fps, mean_fns = mean_metrics(ious, fps, fns)
iou_d = { "iou_" + util.BDD_CLASSES[i] : m  for i, m in enumerate(mean_ious)}
fps_d = { "fps_" + util.BDD_CLASSES[i] : m  for i, m in enumerate(mean_fps)}
fns_d = { "fns_" + util.BDD_CLASSES[i] : m  for i, m in enumerate(mean_fns)}
wandb.log(iou_d)
wandb.log(fps_d)
wandb.log(fns_d)
wandb.log({"mean_class_iou" : per_class_mean_iou})

per_category = per_category_metrics(mean_ious, mean_fps, mean_fns)
wandb.log(per_category)

results_at = wandb.Artifact("eval_results", type="results")
results_at.add(full_num_table, "full_num_results")
run.log_artifact(results_at)
run.finish()

