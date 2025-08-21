import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd


from sklearn.preprocessing import  StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
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
    #using smote for handling imbalanced data
    smote_obj_file_path: str = "artifacts/smote.pkl"

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

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
            ('ordinal_encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ])
            
        
            
            logging.info("Label encoding successfully")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_features),
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
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)

            

            logging.info("training and test data loaded successfully")

            numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            for col in numerical_features:
                train_df[col] = pd.to_numeric(train_df[col].replace(' ', np.nan), errors='coerce')
                test_df[col] = pd.to_numeric(test_df[col].replace(' ', np.nan), errors='coerce')            

            logging.info("Obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformation_object()
            target_coloumn = 'Churn'
            #numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

            input_features_train = train_df.drop(columns=[target_coloumn], axis=1)
            target_feature_train = train_df[target_coloumn].map({'Yes': 1, 'No': 0}).astype(int)
            input_features_test = test_df.drop(columns=[target_coloumn], axis=1)
            target_feature_test  = test_df[target_coloumn].map({'Yes': 1, 'No': 0}).astype(int)

            logging.info("Applying preprocessing object on training and testing dataframes")

            save_object(
                file_path=self.DataTransformationConfig.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            input_features_train = preprocessor_obj.fit_transform(input_features_train)
            input_features_test = preprocessor_obj.transform(input_features_test)
           
            logging.info("Applying SMOTE for handling imbalanced data")
            smote = SMOTE(random_state=42)
            input_features_train, target_feature_train = smote.fit_resample(input_features_train, target_feature_train) 
            logging.info("SMOTE applied successfully")

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