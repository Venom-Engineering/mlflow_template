from enum import StrEnum

class MlflowMetaParameters(StrEnum):
    EXPERIMENT_NAME = "model-demo"
    TRACKING_URI = "http://127.0.0.1:8080"
    REGISTRY_NAME = "titanic-registry" 