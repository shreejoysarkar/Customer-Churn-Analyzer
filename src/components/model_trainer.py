import numpy as np
import pandas as pd
import sys
import os
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
import pickle

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

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

            models = {
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "XGBoost": XGBClassifier(random_state=42)
                }
            '''params = {
                "Decision Tree" :{
                    'criterion':['gini', 'entropy', 'log_loss'],
                    # 'splitter' :['best','random'],
                    #  'max_features' :['sqrt','log2'],

                },
                "Random Forest":{
                    #'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    #'max_features':['sqrt','log2'],
                    'n_estimators':[8,16,32,64,128,256]
                },


                "XGBoost":{
                    'n_estimators':[8,16,32,64,128,256],
                    'learning_rate':[0.01,0.1,0.2],
                    'max_depth':[3,4,5,6]
                }
            }'''
            # cv score for each model
            cv_scores = {}
            for model_name, model in models.items():
                logging.info(f"Performing 5-fold CV for {model_name}")
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
                cv_scores[model_name] = scores
                logging.info(f"{model_name} CV Mean Accuracy: {np.mean(scores):.4f}")
            logging.info(f"cv_score: {print(cv_scores)}")

            # using rfc = RandomForestClassifier(random_state=42) only
            rfc = RandomForestClassifier(random_state=42)
            logging.info("Training Random Forest Classifier")
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Random Forest Classifier Accuracy: {accuracy:.4f}")
            
            
            '''logging.info("Training models")
            model_report : dict = evaluate_models(X_train = X_train,y_train = y_train,X_test = X_test,y_test = y_test, models = models)
            logging.info(f"Model Report: {model_report}")
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            
            if best_model_score < 0.4:
                raise CustomException("No best model found with sufficient accuracy",sys)
            '''
            logging.info(f"Best Model: {rfc} with score: {accuracy}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=rfc
            )
            predicted = rfc.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            print(f"Model Accuracy: {accuracy:.4f}")
            return accuracy
        
        except Exception as e:
            raise CustomException(e, sys)

