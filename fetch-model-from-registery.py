from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
import yaml
from setting import setter
from functools import partial
from transfo import preporcessing

from loguru import logger

mlflow.set_tracking_uri(setter.MlflowMetaParameters.TRACKING_URI)
mlflow.set_experiment(setter.MlflowMetaParameters.EXPERIMENT_NAME)

model_name = setter.MlflowMetaParameters.REGISTRY_NAME

path_versioning_model = Path(__file__).resolve().parent / "version.yml"

with path_versioning_model.open("r") as yaml_data_versionning:
    versionng_yaml = yaml.safe_load(yaml_data_versionning)


PATH_TO_TITANIC_DATA = Path(__file__).resolve().parents[1] / "dataset_training" / "titanic_train.csv"
titanic = pd.read_csv(PATH_TO_TITANIC_DATA)

titanic.rename(columns={'Survived': 'class'}, inplace=True)

columns_preditors = [column for column in titanic.columns if column != "class"]
column_target = "class"

X = titanic[columns_preditors]
y = titanic[column_target]

SEX_MAPPER = {'male':0,'female':1}
EMBARKED_MAPPER = {'S':0,'C':1,'Q':2}

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

model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{versionng_yaml["version"]}")

predictions = model.predict(X_preprocessed)

logger.info(predictions)
