# evaluate.py
# --------------
# The dataset owner runs this script to evaluate a candidate model's performance
# by comparing the model's predictions to the ground truth labels.
# Given
# - a type=predictions Artifact from a participant's model inference ("test" type job)
# - a type=labeled_test_dataset Artifact from the Answers project (containing the correct
#   labels and only visible to the benchmark/dataset owner)
# compute and log several metrics at several levels of granularity.
# The metrics include:
# - iou: intersection over union, or TP / TP + FP + FN summed over all semantic classes except for "void"  
# - fps: false positive pixels where the model identified an incorrect class
# - fns: false negative pixels where the model failed to identify a correct class
#   Note: sometimes the pixels are normalized for faster relative comparison.
# 
# These metrics are logged on several levels:
# - per individual image in the test dataset
# - per individual semantic class ("road", "car", "person", "building", etc)
# - per category ("vehicle" = "car" or "bus" or "truck" or "bicycle"...)
# - average across all images, all classes, or all categories.

# The per-image metrics are logged to a dsviz table inside of artifacts, and the
# aggregate metrics are logged to the wandb run summary.
# 
# Note: this script could be croned to execute for any "predictions" artifact in the Submit project
# which is not input to a further run of job_type=evaluate.

import numpy as np
import os
import pandas as pd
from PIL import Image
import util
import wandb

# Given the per-class metrics, compute the per-category metrics
# (aggregate over the component classes, e.g. category human = class person and class rider)
def per_category_metrics(ious, fps, fns):
  metrics = {}
  for metric_name, metric_type in zip(["iou", "fps", "fns"], [ious, fps, fns]):
    for category_name, ids in util.CITYSCAPE_IDS.items():
      category_metric = np.mean([metric_type[class_id] for class_id in ids])
      metrics["cat_" + metric_name + "_" + category_name] = category_metric

  # average across categories, per metric
  for metric in ["cat_iou", "cat_fps", "cat_fns"]:
    metric_vals = [v for k, v in metrics.items() if k.startswith(metric)]
    metrics["mean_" + metric] = np.mean(metric_vals)

  return metrics

# ious is util.NUM_EXAMPLES rows of len(util.BDD_CLASSES) - 1 columns
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
  for class_id in range(len(util.BDD_CLASSES) - 1):
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
  
  scores = score_masks(guess_images[row["guess"]], true_images[row["truth"]])
  # all the scores per class
  r = [row["id_truth"], guess_pretty[row["guess"]], true_pretty[row["truth"]], scores["net_iou"], scores["net_fp"], scores["net_fn"]]
  class_ious = [scores[class_id]["iou"] for class_id in range(len(util.BDD_CLASSES) - 1)]
  ious.append(class_ious)
  r.extend(class_ious)
  class_fns = [scores[class_id]["fn"] for class_id in range(len(util.BDD_CLASSES) - 1)]
  fns.append(class_fns)
  r.extend(class_fns)
  class_fps = [scores[class_id]["fp"] for class_id in range(len(util.BDD_CLASSES) - 1)]
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
wandb.log({"iou_class" : iou_d})
wandb.log({"fps_class" : fps_d})
wandb.log({"fns_class" : fns_d})
wandb.log({"mean_class_iou" : per_class_mean_iou})

per_category = per_category_metrics(mean_ious, mean_fps, mean_fns)
wandb.log({"category" : per_category})

results_at = wandb.Artifact("eval_results", type="results")
results_at.add(full_num_table, "full_num_results")
run.log_artifact(results_at)
run.finish()

