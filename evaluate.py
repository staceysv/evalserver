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

# 2D version
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

def iou_3D(mask_a, mask_b, class_id):
    return np.nan_to_num(((mask_a == class_id) & (mask_b == class_id)).sum(axis=(1,2)) / ((mask_a == class_id) | (mask_b == class_id)).sum(axis=(1,2)), 0, 0, 0)

run = wandb.init(project=util.ANSWER_PROJECT, job_type="evaluate")

# get predictions
predictions_at = run.use_artifact("{}/test_predictions:latest".format(util.SUBMIT_PROJECT))
guess_table = predictions_at.get("test_results")

guess_ids = np.array([guess_table.data[i][0] for i in range(50)])
guess_images = np.array([np.array(guess_table.data[i][2]._image) for i in range(50)])
guess_pretty = [guess_table.data[i][1] for i in range(50)]

guess = pd.DataFrame({'id' : guess_ids, "guess": [i for i in range(len(guess_images))]}) 

# get ground truth
answers_at = run.use_artifact("{}/answer_key:latest".format(util.ANSWER_PROJECT))
true_table = answers_at.get("answer_key")

true_ids = np.array([true_table.data[i][0] for i in range(50)])
true_images = np.array([np.array(true_table.data[i][3]._image) for i in range(50)])
true_pretty = [true_table.data[i][2] for i in range(50)]

truth = pd.DataFrame({'id' : true_ids, "truth" : [i for i in range(len(true_images))]})

results = guess.join(truth, lsuffix="_guess", rsuffix="_truth")

# now log a final table
eval_table = wandb.Table(columns=["id", "prediction", "ground_truth", "iou"])

# OK now they are joined
for index, row in results.iterrows():
  s = iou_flat(guess_images[row["guess"]], true_images[row["truth"]], 0)
  eval_table.add_data(row["id_truth"], guess_pretty[row["guess"]], true_pretty[row["truth"]], s)

results_at = wandb.Artifact("eval_results", type="results")
results_at.add(eval_table, "eval_results")
run.log_artifact(results_at)


# join guess_images and true_images by id
# score each pair
# log new 

#print(scores)
## 
  

# eval table
#eval_table = wandb.JoinedTable(guess_table, truth_table, "id")

# compute eval_table
#print("EXAMPLE")
#print(eval_table.data[0])

