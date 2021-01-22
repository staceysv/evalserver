# Benchmark Evaluation Server

This repo is an example of a benchmark evaluation server in W&B.
At a high level, the workflow is:
- the benchmark owner sets up an evaluation dataset with correct labels (answers) invisible to benchmark participants
- participants test their models on the test dataset and submit their model's predictions
- the owner evaluates these predictions against the correct answers and saves performance metrics (following [Cityscapes](https://www.cityscapes-dataset.com/benchmarks/#scene-labeling-task)) across the submissions

# Scripts

## upload_test_data.py

Benchmark owners run this to create two artifacts: the unlabeled test data in a Demo project and the test data with correct labels in an Answer project (only visible to the owner and not to participants).

## predict_example.py

Participants run this to load in the Demo project test data and a candidate model to test (factored out as `test_model()` in `inference.py` to enable switching easily between many trained models). This logs predictions to a separate Submit project, such that participants do not see each others' submissions (e.g. a new Submit project could be created for each participant, accesible to the owner and the specific participant only).

## evaluate.py

Benchmark owners run this to evaluate model predictions in the Submit project and compute final metrics, stored in the Answer project. These evaluation metrics are easily compared within the Answer project and only visible to benchmark/dataset owners. 

# Metrics

We compute and log several metric types at several levels of granularity.
The metric types include:
* iou: intersection over union, or TP / TP + FP + FN summed over all semantic classes except for "void"  
* fps: false positive pixels where the model identified an incorrect class
* fns: false negative pixels where the model failed to identify a correct class
Note: beyond the per-image level, the pixels are normalized for faster relative comparison.

The levels of granularity include:
* per individual image in the test dataset: logged to dataset visualization table in evaluation artifact
* per individual semantic class ("road", "car", "person", "building", etc): logged to evaluation run summary, nested under metric type
* per category ("vehicle" = "car" or "bus" or "truck" or "bicycle"...): logged to evaluation run summary, nested under "category"
* average across all images, all classes, or all categories: logged to evaluation run summary with "mean_" prefix

The per-image metrics are logged to a dsviz table inside of artifacts, and the aggregate metrics are logged to the wandb run summary.

# Visualization links

* [Prediction project](https://wandb.ai/stacey/evalserve_predict)
* [Answer project to compare models](https://wandb.ai/stacey/answers_evalserve)
* [Best model predictions so far](https://wandb.ai/stacey/answers_evalserve/artifacts/results/eval_results/8c1729d783f95e3d037a)
