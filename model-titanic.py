from sklearn.model_selection import train_test_split
import pandas as pd 
from pathlib import Path
import yaml

from functools import partial

from sklearn.ensemble import RandomForestClassifier
import typer

from transfo import preporcessing
from setting import setter
import mlflow
from sklearn.metrics import f1_score
from mlflow.models.signature import infer_signature

app = typer.Typer()

PATH_TO_TITANIC_DATA = Path(__file__).resolve().parents[1] / "dataset_training" / "titanic_train.csv"
titanic = pd.read_csv(PATH_TO_TITANIC_DATA)

titanic.rename(columns={'Survived': 'class'}, inplace=True)

columns_preditors = [column for column in titanic.columns if column != "class"]
column_target = "class"

X = titanic[columns_preditors]
y = titanic[column_target]

SEX_MAPPER = {'male':0,'female':1}
EMBARKED_MAPPER = {'S':0,'C':1,'Q':2}


def update_versionning_model():
    path_versioning_model = Path(__file__).resolve().parent / "version.yml"

    with path_versioning_model.open("r") as yaml_data_versionning:
        versionng_yaml = yaml.safe_load(yaml_data_versionning)

    versionng_yaml["version"] += 1

    with path_versioning_model.open("w") as yaml_data_versionning:
        versionng_yaml = yaml.safe_dump(
            versionng_yaml, yaml_data_versionning, indent=4, default_flow_style=False
        )

@app.command()
def main(
      bootstrap: bool, 
      criterion: str, 
      max_features: float,
      min_samples_leaf: int,
      min_samples_split: int,
      n_estimators: int
    ):
    partial_drop_columns = partial(preporcessing.drop_columns, columns=['Name','Ticket','Cabin'])
    partial_sex_mapper = partial(preporcessing.mapper, mapper=SEX_MAPPER, columns=["Sex"])
    partial_embarked_mapper = partial(preporcessing.mapper, mapper=EMBARKED_MAPPER, columns=["Embarked"])

    pipeline = preporcessing.composite_function(
        partial_drop_columns,
        partial_sex_mapper,
        partial_embarked_mapper,
        preporcessing.fillna_pipe,
    )
   
    X_preprocessed = pipeline(X)

    training_features, testing_features, training_target, testing_target = \
                train_test_split(X_preprocessed, y, random_state=42)


    model = RandomForestClassifier(
                            bootstrap=bootstrap, 
                            criterion=criterion,
                            max_features=max_features, 
                            min_samples_leaf=min_samples_leaf, 
                            min_samples_split=min_samples_split, 
                            n_estimators=n_estimators
                        )

    model.fit(training_features, training_target)
    results = model.predict(testing_features)
    f1_score_ = f1_score(testing_target, results)

    mlflow.set_tracking_uri(setter.MlflowMetaParameters.TRACKING_URI)
    mlflow.set_experiment(experiment_name=setter.MlflowMetaParameters.EXPERIMENT_NAME)
    
    with mlflow.start_run():
            mlflow.log_metric("f1_score", f1_score_)

            mlflow.log_param("bootstrap", bootstrap)
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("max_features", max_features)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)
            mlflow.log_param("min_samples_split", min_samples_split)
            mlflow.log_param("n_estimators", n_estimators)

            mlflow.log_metric("f1_score", f1_score_)

            signature = infer_signature(testing_features, model.predict(testing_features))
            mlflow.sklearn.log_model(
                  model, 
                  "model", 
                  signature=signature,
                  registered_model_name=setter.MlflowMetaParameters.REGISTRY_NAME
            )

    update_versionning_model()

if __name__ == "__main__":
      app()