.PHONY: mlflow_server_local mlflow_run_local mlflow_ui

mlflow_server_local:
	mlflow server --host 127.0.0.1 --port 8080

mlflow_run_local:
	mlflow run . --env-manager local --experiment-name model-demo

mlflow_ui:
	mlflow ui
