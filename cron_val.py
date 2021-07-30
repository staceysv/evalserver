import wandb
import util

api = wandb.Api({"project" : "evalserver_entry"})
api_eval = wandb.Api({"project" : "evalserver_answers"})

run_entry = wandb.init(project="evalserver_entry", job_type="check_eval")
run_eval = wandb.init(project="evalserver_answers", job_type="check_eval")

preds = api.artifact_type('entry_predictions')
for p in preds.collections():
  artifact_name = p.name
  print(artifact_name)
  all_a = api.artifact_versions('entry_predictions', artifact_name, per_page=1000)
  print([a._version_index for a in all_a])
#latest._version_index)
print("and for the evals")
evals = api_eval.artifact_type('results')
for p in evals.collections():
  artifact_name = p.name
  print(artifact_name)
  all_a = api_eval.artifact_versions('results', artifact_name, per_page=1000)
  print([a._version_index for a in all_a])


run_entry.finish()
run_eval.finish()
