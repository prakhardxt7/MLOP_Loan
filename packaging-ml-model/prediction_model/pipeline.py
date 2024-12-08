from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import sys

#This ensures that you can import modules from the package's root directory regardless of where the script is being executed. It’s particularly useful for projects with a complex directory structure
PACKAGE_ROOT = path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
import prediction_model.processing.preprocessings as pp
from sklearn.ensemble import RandomForestClassifier
import numpy as np

classification_pipeline = Pipeline(
    [
        ('DomainProcessing', pp.DomainProcessing(variable_to_add=config.FEATURE_TO_ADD)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('LabelEncoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('LogTransform', pp.LogTransforms(variables=config.LOG_FEATURES)),
        ('RandomForestClassifier', RandomForestClassifier(random_state=0))  # Corrected line
    ]
)
