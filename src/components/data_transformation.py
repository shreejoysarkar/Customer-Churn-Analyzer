import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from sklearn.compose import ColumnTransformer
from src.utils import save_object




@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = "artifacts/preprocessor.pkl"

class DataTransformation:
    def __init__(self):
        self.DataTransformationConfig = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        This function creates a data transformation object that includes preprocessing steps
        such as label encoding for categorical features and handling numerical features.
        It returns a preprocessor object and the transformed DataFrame.
        '''
        try:
            
            logging.info("Creating data transformation object")
            #df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
            Categorical_Features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
            numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            #target_feature = 'Churn'

            

            cat_pipeline = Pipeline(steps=[
            ('ordinal_encoder', OrdinalEncoder())
            ])
            
        
            
            logging.info("Label encoding successfully")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', numerical_features),
                    ('cat', cat_pipeline, Categorical_Features)
                    #('target', target_pipeline, target_feature)
                ])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("training and test data loaded successfully")

            logging.info("Obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformation_object()
            target_coloumn = 'Churn'
            #numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

            input_features_train = train_df.drop(columns=[target_coloumn], axis=1)
            target_feature_train = train_df[target_coloumn]
            input_features_test = test_df.drop(columns=[target_coloumn], axis=1)
            target_feature_test = test_df[target_coloumn]

            logging.info("Applying preprocessing object on training and testing dataframes")

            save_object(
                file_path=self.DataTransformationConfig.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            input_features_train = preprocessor_obj.fit_transform(input_features_train)
            input_features_test = preprocessor_obj.transform(input_features_test)

            train_arr = np.c_[input_features_train, target_feature_train.values]
            test_arr = np.c_[input_features_test, target_feature_test.values]
            logging.info("Data transformation completed successfully")

            return (
                train_arr,
                test_arr,
                self.DataTransformationConfig.preprocessor_obj_file_path,
            )


            

        except Exception as e:  
            raise CustomException(e, sys)