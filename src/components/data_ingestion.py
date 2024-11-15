# Data ingestion is the process of collecting and preparing data for use in analysis or machine learning. It involves getting data from various sources, cleaning it up, and organizing it into a usable format.

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd
import os
import sys
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


@dataclass
class DataIngestConfig:
    train_data_pth: str = os.path.join('artifacts', "train.csv")
    test_data_pth: str = os.path.join('artifacts', "test.csv")
    raw_data_pth: str = os.path.join('artifacts', "raw.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method ...")
        try:
            df = pd.read_csv('notebooks/data/stud.csv')
            logging.info('Read the dataset as dataframe ✔️')

            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_pth), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_pth,
                      index=False, header=True)

            logging.info('Train Test Split Initialized')

            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_pth,
                             index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_pth,
                            index=False, header=True)

            logging.info("Ingestion Complete ✔️")

            return (
                self.ingestion_config.train_data_pth,
                self.ingestion_config.test_data_pth,
                # self.ingestion_config.raw_data_pth,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data)
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
