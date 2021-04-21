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
import argparse
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
      metrics["category/" + metric_name + "_" + category_name] = category_metric
  # average across categories, per metric
  for metric in ["category/iou", "category/fps", "category/fns"]:
    metric_vals = [v for k, v in metrics.items() if k.startswith(metric)]
    metrics["mean_category/" + metric.split("/")[1]] = np.mean(metric_vals)
  return metrics

# compute the mean across each metric type, within and across classes,
# where each type is of the shape util.NUM_EXAMPLES rows of len(util.BDD_CLASSES) - 1 columns
def mean_metrics(ious, fps, fns):
  # per-class mean iou and overall mean iou across classes
  ious_np = np.mean(ious, axis=0)
  per_class_mean_iou = np.mean(ious_np)

  # mean normalized per class false positive pizels
  fps_floats = np.array(fps)
  norm_fps = fps_floats / util.TOTAL_PIXELS
  mean_norm_fps = np.mean(norm_fps, axis=0)

  # mean normalized per class false negative pixels
  fns_floats = np.array(fns)
  norm_fns = fns_floats / util.TOTAL_PIXELS
  mean_norm_fns = np.mean(norm_fns, axis=0)

  return per_class_mean_iou, ious_np, mean_norm_fps, mean_norm_fns

# for each pair of image masks, count the pixels falling into the
# three buckets of true positives, false positives, and false negatives
# for each semantic class, then compute IOU and these three per-pixel metric 
# types both per class and across all classes (prefixed with "net")
def pixel_count(guess, truth):
  iou_scores = {}
  net_fp = 0
  net_fn = 0
  net_tp = 0
  # compute for all semantic class ids
  # NOTE: void class doesn't contribute to these metrics
  for class_id in range(len(util.BDD_CLASSES) - 1):
    fp = ((guess == class_id) & (truth != class_id)).sum(axis=(0,1))
    fn = ((guess != class_id) & (truth == class_id)).sum(axis=(0,1))
    tp = ((guess == class_id) & (truth == class_id)).sum(axis=(0,1))
    iou = util.smooth(float(tp), float(tp + fp + fn))
    # record metrics for each class
    iou_scores[class_id] = {"fp": fp, "fn" : fn, "tp" : tp, "iou" : iou}
    net_fp += fp
    net_fn += fn
    net_tp += tp
  for category_name, category_ids in util.CITYSCAPE_IDS.items():
    sum_fp = 0
    sum_fn = 0
    sum_tp = 0
    for cat_id in category_ids:
      sum_fp += iou_scores[cat_id]["fp"]
      sum_fn += iou_scores[cat_id]["fn"]
      sum_tp += iou_scores[cat_id]["tp"]
    cat_iou = util.smooth(float(sum_tp), float(sum_tp + sum_fp + sum_fn))
    iou_scores[category_name] = {"fp" : sum_fp, "fn" : sum_fn, "iou" : cat_iou}
 
  # record "net" metrics across all classes for this image
  net_iou = util.smooth(float(net_tp), float(net_tp + net_fp + net_fn))
  iou_scores["net_iou"] = net_iou
  iou_scores["net_tp"] = net_tp
  iou_scores["net_fp"] = net_fp
  iou_scores["net_fn"] = net_fn
  return iou_scores 

# wrapper to score two masks (model's guess vs ground truth)
def score_masks(mask_guess, mask_truth):
  # scale up the mask from the model prediction as the model is fed
  # images at 25% the size of the actual ground truth label file
  guess = mask_guess.repeat(2, axis=0).repeat(2, axis=1)
  return pixel_count(guess, mask_truth)

def evaluate_model(args):
  # by default, only log labeled test images (the benchmark ground truth/
  # the corrrect answers) to the private Answer project, which is only
  # visible to the benchmark owners
  # optionally set log_project = Demo project (which all participants can see) to
  # reveal the correct labels on test images in the public Demo project
  LOG_PROJECT = args.log_project
  
  # create a new evalution run in the Answer project (default, invisible to participants)
  # or the Demo project (optional, would make all submissions and correct answers visble to everyone)
  run = wandb.init(project=LOG_PROJECT, job_type="evaluate")

  # Guess table / Predictions Artifact
  # fetch the participant's predictions from their Entry project
  run.config.team_name = args.team_name
  predictions_at = run.use_artifact("{}/entry_{}:latest".format(args.entry_project, args.team_name))
  guess_table = predictions_at.get("test_results")
  # extract relevant columns
  guess_ids = np.array([guess_table.data[i][0] for i in range(util.NUM_EXAMPLES)])
  guess_images = np.array([np.array(guess_table.data[i][2]._image) for i in range(util.NUM_EXAMPLES)])
  guess_wb_masks = [guess_table.data[i][1] for i in range(util.NUM_EXAMPLES)]

  # join ids to image index
  guess = pd.DataFrame({'id' : guess_ids, "guess": [i for i in range(len(guess_images))]}) 

  # Answer table / Labeled test dataset Artifact
  # fetch the latest version of the ground truth labels
  answers_at = run.use_artifact("{}/answer_key:latest".format(args.answer_project))
  true_table = answers_at.get("answer_key")

  # extract relevant columns
  true_ids = np.array([true_table.data[i][0] for i in range(util.NUM_EXAMPLES)])
  true_images = np.array([np.array(true_table.data[i][3]._image) for i in range(util.NUM_EXAMPLES)])
  true_wb_masks = [true_table.data[i][2] for i in range(util.NUM_EXAMPLES)]

  truth = pd.DataFrame({'id' : true_ids, "truth" : [i for i in range(len(true_images))]})

  # join guess and truth using Pandas
  results = guess.join(truth, lsuffix="_guess", rsuffix="_truth")

  # fields to log to benchmark evaluation dashboard
  columns=["id", "prediction", "ground truth", "overall IOU", "false positive", "false negative"]
  # add columns for all metrics to the evaluation table,
  # tracking for each semantic class:
  # - IOU
  # - false negative pixels (FNS)
  # - false positive pixels (FPS)
  columns.extend(["iou_" + s for s in util.BDD_CLASSES[:-1]])
  columns.extend(["fn_" + s for s in util.BDD_CLASSES[:-1]])
  columns.extend(["fp_" + s for s in util.BDD_CLASSES[:-1]])
  # add semantic categories
  columns.extend(["iou_" + s for s in util.CITYSCAPE_IDS.keys()])
  columns.extend(["fn_" + s for s in util.CITYSCAPE_IDS.keys()])
  columns.extend(["fp_" + s for s in util.CITYSCAPE_IDS.keys()])
  eval_table = wandb.Table(columns=columns)

  ious = []
  fns = []
  fps = []
  for index, row in results.iterrows():
    # compute scores
    scores = score_masks(guess_images[row["guess"]], true_images[row["truth"]])
    # log the net metrics and visualizations for each image
    r = [row["id_truth"], guess_wb_masks[row["guess"]], true_wb_masks[row["truth"]], scores["net_iou"], scores["net_fp"], scores["net_fn"]]
    # append per-class metrics to table
    for metric_name, per_class_list in zip(["iou", "fn", "fp"], [ious, fns, fps]):
      per_class_scores = [scores[class_id][metric_name] for class_id in range(len(util.BDD_CLASSES) - 1)]
      per_class_list.append(per_class_scores)
      r.extend(per_class_scores)
    # append per-category metrics to table
    for metric_name in ["iou", "fn", "fp"]:
      r.extend([scores[category_name][metric_name] for category_name in util.CITYSCAPE_IDS.keys()])
    
    eval_table.add_data(*r)

  # compute mean metrics for each semantic class and across all classes
  per_class_mean_iou, mean_ious, mean_fps, mean_fns = mean_metrics(ious, fps, fns)

  # log to W&B for easy comparison across benchmark submissions 
  wandb.log({ "iou_class/" + util.BDD_CLASSES[i] : m  for i, m in enumerate(mean_ious)})
  wandb.log({ "fps_class/" + util.BDD_CLASSES[i] : m  for i, m in enumerate(mean_fps)})
  wandb.log({ "fns_class/" + util.BDD_CLASSES[i] : m  for i, m in enumerate(mean_fns)})

  wandb.log({"mean_class_iou" : per_class_mean_iou})
  wandb.log({"mean_class_fps" : np.mean(mean_fps)})
  wandb.log({"mean_class_fns" : np.mean(mean_fns)})

  # compute and log per category metrics
  per_category = per_category_metrics(mean_ious, mean_fps, mean_fns)
  wandb.log(per_category)

  # write to W&B
  results_at = wandb.Artifact("eval_" + args.team_name, type="results")
  results_at.add(eval_table, "eval_results")
  run.log_artifact(results_at)
  run.finish()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-t",
    "--team_name",
    type=str,
    default="",
    help="Team or participant name to evaluate")
  parser.add_argument(
    "-a",
    "--answer_project",
    type=str,
    default=util.ANSWER_PROJECT,
    help="answer project: where the ground truth for the test data is stored")
  parser.add_argument(
    "-e",
    "--entry_project",
    type=str,
    default=util.ENTRY_PROJECT,
    help="project which holds the team's benchmark entry")
  parser.add_argument(
    "-l",
    "--log_project",
    type=str,
    default=util.ANSWER_PROJECT,
    help="log project: where to log the evaluation results for this entry (defaultsto the Answer project to hide true labels from benchmark participants)")
  args = parser.parse_args()
  evaluate_model(args)
 
    
