import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

from src.exceptions import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            logging.info("Split training & test input data")
            models = {
                "Linear Regression": LinearRegression(),
                "K Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGB Regressor": XGBRegressor(),
                "Ada Boost Regressor": AdaBoostRegressor(),
                "Cat Boost Regressor": CatBoostRegressor()
            }

            model_report: dict = evaluate_model(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            logging.info("All Models are trained & evaluated")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")

            logging.info(f"Best model found is {
                         best_model_name} & being saved ...")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_skore = r2_score(y_test, predicted)

            return r2_skore

        except Exception as e:
            raise CustomException(e, sys)
