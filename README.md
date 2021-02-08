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
* [example entry with two attempts and weak predictions](https://wandb.ai/stacey/evalserver_entry/artifacts/entry_predictions/entry_Ada/764c8a1319ba8922557a/files/test_results.table.json)
* [entry project workspace](https://wandb.ai/stacey/evalserve_entry/)

## Participants' training projects

Participants can optionally log their training to W&B. Here is [one example with simulated teams](https://wandb.ai/stacey/segment_dsviz) and [another with a more realistic single team](https://wandb.ai/stacey/dsviz-demo)

# Scripts

## upload_test_data.py

Benchmark owners run this to create two artifacts: the [unlabeled test data](https://wandb.ai/stacey/evalserver_test/artifacts/test_dataset/test_data/63753dbf2199578d8fcb/files/test_data.table.json)  in a Demo project and the [correctly labeled test data](https://wandb.ai/stacey/evalserver_answers/artifacts/labeled_test_dataset/answer_key/565a82d0b381f02ec032/files/answer_key.table.json)  with correct labels in an Answer project (only visible to the owner and not to participants).

## predict_example.py

Participants run this to load in the Demo project test data and a candidate model to test (factored out as `test_model()` in `inference.py` to enable switching easily between many trained models). This logs predictions to a [separate Entry project](https://wandb.ai/stacey/segment_dsviz), which is shared across teams for simplicity in this example. In general, a new Entry project could be created for each participating team, accessible to the owner and the specific team only, so that different teams do not see each others' submissions.

## evaluate.py

Benchmark owners run this to evaluate model predictions in the Entry project and compute final performance metrics, stored in the Answer project by default. These evaluation metrics are easily compared within the Answer project and only visible to benchmark/dataset owners. If/when the benchmark owners are ready, these can be shared with all participants via the Demo project. Another option is for each team's performance to be visible to that team in their Entry project without including the ground truth labels. Note: in this dataset and example, it's very easy for humans to see the correct answer/approximate model performance by looking at any predicted regions on _any_ image, since the predictions and ground truth are generally easy to interpret for humans (e.g. clearly this object should be a car and not a bus, this pedestrian was missed, etc). 

# Metrics

We compute and log several metric types at several levels of granularity.
The metric types include:
* iou: intersection over union, or TP / TP + FP + FN summed over all semantic classes except for "void"  
* fps: false positive pixels where the model identified an incorrect class
* fns: false negative pixels where the model failed to identify a correct class
Note: beyond the per-image level, the pixels are normalized for faster relative comparison when looking at tables of results.

The levels of granularity include:
* per individual image in the test dataset: logged to dataset visualization table in the evaluation artifact of the Answer project
* per individual semantic class ("road", "car", "person", "building", etc): logged to evaluation run summary, nested under metric type
* per category ("vehicle" = "car" or "bus" or "truck" or "bicycle"...): logged to evaluation run summary, nested under "category"
* average across all images, all classes, or all categories: logged to evaluation run summary with "mean_" prefix

The per-image metrics are logged to a dataset visualization table inside of the artifacts view in the Answer project, and the aggregate metrics are logged to the wandb run summary.

# Visualization links

* [Prediction project](https://wandb.ai/stacey/evalserve_predict)
* [Answer project to compare models](https://wandb.ai/stacey/answers_evalserve)
* [Best model predictions so far](https://wandb.ai/stacey/answers_evalserve/artifacts/results/eval_results/8c1729d783f95e3d037a)
