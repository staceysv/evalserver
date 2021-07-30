
def test_model(run, test_data_artifact):
  # download the test data to a local directory
  test_data_dir = test_data_artifact.download()

  # load a model previously trained and saved via Artifacts
  MODEL_AT = "{}/{}:{}".format(TRAINING_PROJECT, MODEL_NAME, MODEL_VERSION)
  model_at = run.use_artifact(MODEL_AT)
  model_path = model_at.get_path(MODEL_NAME).download()

