import numpy as np
import pandas as pd
import sys
import os
from dataclasses import dataclass
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
import pickle

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        return rmse, mae, r2

    def initiate_model_trainer(self,train_array, test_array):
        try:
            logging.info('split training and test input data')

            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info('split completed')



            # using rfc = RandomForestClassifier(random_state=42) only
            xgb = XGBClassifier(learning_rate=  0.1, max_depth= 3, n_estimators= 100, subsample = 0.8)
            logging.info("Training Random Forest Classifier")
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Random Forest Classifier Accuracy: {accuracy:.4f}")
            
            

            logging.info(f"Best Model: {xgb} with score: {accuracy}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=xgb
            )

            return accuracy
        
        except Exception as e:
            raise CustomException(e, sys)

