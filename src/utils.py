import sys
import os
from src.exceptions import CustomException
from src.logger import logging


import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params) -> dict:
    try:
        report = {}
        for i in range(len(list(models))):
            current_model = list(models.values())[i]
            current_model_params = list(params.values())[i]

            logging.info('Grid search activated ...')
            gs = GridSearchCV(current_model, cv=5,
                              param_grid=current_model_params)
            gs.fit(X_train, y_train)

            logging.info(f"Best Parameter for {
                         list(models.keys())[i]} is {gs.best_params_}")
            current_model.set_params(**gs.best_params_)
            current_model.fit(X_train, y_train)

            y_train_pred = current_model.predict(X_train)
            y_test_pred = current_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
