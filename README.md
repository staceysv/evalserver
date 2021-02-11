# Benchmark Evaluation Server

This repo is an example of a benchmark evaluation server in W&B.
At a high level, the workflow is:
- the benchmark owner sets up [an evaluation dataset](https://wandb.ai/stacey/evalserver_test/artifacts/test_dataset/test_data/63753dbf2199578d8fcb/files/test_data.table.json) with correct labels ([answers](https://wandb.ai/stacey/evalserver_answers/artifacts/labeled_test_dataset/answer_key/565a82d0b381f02ec032/files/answer_key.table.json)) invisible to benchmark participants
- participants test their models on the test dataset and submit their model's predictions
- the owner evaluates these predictions against the correct answers and saves performance metrics (following [Cityscapes](https://www.cityscapes-dataset.com/benchmarks/#scene-labeling-task)) across the submissions

# Project layout

## Answer project

The Answer project is the core project viewable only to the benchmark owners. This contains the ground truth labels (answer key) for the test dataset, stores evaluation metrics for all the participant submissions, and contains dashboards for comparing the candidate models.
* [example evaluation key](https://wandb.ai/stacey/evalserver_answers/artifacts/labeled_test_dataset/answer_key/565a82d0b381f02ec032/files/answer_key.table.json)

## Demo project

The Demo project is the open project where any participant can view the test dataset (without labels) and any benchmark details. The evaluation results can optionally be shared here if/when the benchmark owners would like to reveal the answers and each submitted model's performance publicly.
* [test dataset ids and images](https://wandb.ai/stacey/evalserver_test/artifacts/test_dataset/test_data/63753dbf2199578d8fcb/files/test_data.table.json)
* [demo project workspace](https://wandb.ai/stacey/evalserver_test/) - for reference, by default it is empty

## Entry project

The Entry project is where participants submit their candidate models and predictions. To keep entries private and not visible to all the other participants, a new Entry project can be created for every benchmark participant team. For simplicity in the current setup, all participants share one Entry project, and each entry is idenitified by a "team_name" and optional "entry_name" (to better differentiate or more easily refer to their model varitants). We have example submissions from Ada, Bob, Charlie, etc. The submissions are versioned by team name, so every submission attempt by the same team creates a new version of the "entry_predictions" artifact, corresponding to that team's latest/presumably best predictions. To prevent different teams from seeing each others' entries and attempts, this evalserver could be set up with one separte W&B project per participating team.
* [example entry with strong predictions](https://wandb.ai/stacey/evalserver_entry/artifacts/entry_predictions/entry_Daenerys/7328219b584e0361a99a/files/test_results.table.json)
* [example entry with two attempts and weaker predictions](https://wandb.ai/stacey/evalserver_entry/artifacts/entry_predictions/entry_Ada/764c8a1319ba8922557a/files/test_results.table.json)
* [entry project workspace](https://wandb.ai/stacey/evalserve_entry/)

## Participants' training projects

Participants can optionally log their training to W&B. Here is [one example with simulated teams](https://wandb.ai/stacey/segment_dsviz) and [another with a more realistic single team](https://wandb.ai/stacey/dsviz-demo)

# Scripts

## upload_test_data.py

Benchmark owners run this to create two artifacts: the [unlabeled test data](https://wandb.ai/stacey/evalserver_test/artifacts/test_dataset/test_data/63753dbf2199578d8fcb/files/test_data.table.json)  in a Demo project and the [correctly labeled test data](https://wandb.ai/stacey/evalserver_answers/artifacts/labeled_test_dataset/answer_key/565a82d0b381f02ec032/files/answer_key.table.json)  with correct labels in an Answer project (only visible to the owner and not to participants).

## predict_example.py

Participants run this to load in the Demo project test data and a candidate model to test:
```
predict_example.py --team_name Stacey --model_name resnet34_baseline
```
The model loading part is factored out as `test_model()` in `inference.py` to enable switching easily between many trained models (and simulated teams, in my case: Ada, Bob, Charlie...). This script logs the candidate model's predictions on the test set to a [separate Entry project](https://wandb.ai/stacey/evalserver_entry). A new Entry project could be created for each participating team, accessible to the owner and the specific team only, so that different teams do not see each other's submissions. In this example, I use a single shared Entry project with simulated teams. Each subsequent entry for a given team name creates a new version of the team's predictions, assuming this is the latest/best attempt for that team. 

## evaluate.py

Benchmark owners run this to evaluate team's model predictions as submitted to the Entry project and to compute final performance metrics:
```
evaluate.py --team_name Stacey [--log_answer_images]
```

### Automatically detecting and evaluating new entries
This evaluation script can easily be automated to periodically evaluate any recently submitted models. For example, a cron job could run every N hours to check if there are any new submissions: any new "entry_[TEAM_NAME]" Artifact versions of Artifact type "entry_preditions" in an Entry project which do not yet have a matching version in the Answer project (no "eval_[TEAM_NAME]" Artifact version of Artifact type "results"). This job could then evaluate any new "entry_prediction" versions and log the corresponding evaluation "results" Artifact to the Answer project.

### Optionally releasing the ground truth labels 

These evaluation metrics are easily compared across entries within the Answer project, where all entries are only visible to the owners/administrators of thebbenchmark (and not the participants). If/when the benchmark owners are ready, the correct answers can be shared with all participants via the Demo project. Another option is for each team's performance metrics to be visible to that team in their Entry project without including the ground truth labels. Note that with this dataset, model performance on driving scenes is fairly easy for humans to approximate by looking at region predictions on _any_ image, even if the numerical ground truth is not returned (e.g. if the participants visualize their own models' predictions in W&B, they'll be able to tell at a glance where the model is confusing trucks and buses, or pedestrians and cyclists, etc).

# Metrics

We compute and log several metric types at several levels of granularity.
The metric types include:
* **iou**: intersection over union, or TP / (TP + FP + FN) summed over all semantic classes except for "void"  
* **fps**: false positive pixels, where the model identified an incorrect class
* **fns**: false negative pixels, where the model failed to identify a correct class
Note: beyond the per-image level, the pixel counts are normalized (by the total pixel count for the image) for faster relative comparison when looking at tables of numerical results.

The levels of granularity for each metric include:
* **per image**: computed for each individual image in the test dataset; logged to the "eval_results" table in the Answer project: [example](https://wandb.ai/stacey/evalserver_answers/artifacts/results/eval_Daenerys/5efc7ec72c533def2f81/files/eval_results.table.json) 
* **per semantic class** ("road", "car", "person", etc): computed across all images in the test dataset; logged to each [evaluation run summary](https://wandb.ai/stacey/evalserver_answers/runs/22ojbylg/overview) and [run page](https://wandb.ai/stacey/evalserver_answers/runs/22ojbylg?workspace=user-stacey), nested unbby metric type
* **per semantic category** ("vehicle" = "car" or "bus" or "truck" or "bicycle"...): logged to each [evaluation run summary](https://wandb.ai/stacey/evalserver_answers/runs/22ojbylg/overview) and [run page](https://wandb.ai/stacey/evalserver_answers/runs/22ojbylg), nested under "category".
* **mean across all classes or all categories**: logged to [evaluation run summary](https://wandb.ai/stacey/evalserver_answers/runs/22ojbylg/overview) and [run page](https://wandb.ai/stacey/evalserver_answers/runs/22ojbylg) with "mean_" prefix

The per-image metrics are logged to a dataset visualization table inside of the artifacts view in the Answer project, and the aggregate metrics are logged to each evaluation run's summary and aggregated in the [main workspace of the Answer project](https://wandb.ai/stacey/evalserver_answers).

# Visualization links

* [Main workspace to compare candidate model performance](https://wandb.ai/stacey/evalserver_answers)
* [Best model predictions so far](https://wandb.ai/stacey/evalserver_answers/artifacts/results/eval_Daenerys/5efc7ec72c533def2f81/files/eval_results.table.json)
* [Benchmark participant entries](https://wandb.ai/stacey/evalserver_entry)
* [Benchmark test data for participants](https://wandb.ai/stacey/evalserver_test/artifacts/test_dataset/test_data/63753dbf2199578d8fcb/files/test_data.table.json)
