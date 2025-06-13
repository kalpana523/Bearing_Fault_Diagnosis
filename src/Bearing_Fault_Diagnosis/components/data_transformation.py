import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from Bearing_Fault_Diagnosis.utils import save_object
from Bearing_Fault_Diagnosis.exception import CustomException
from Bearing_Fault_Diagnosis.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns):
        '''
        This function creates a preprocessing pipeline: imputation -> scaling -> PCA
        '''
        try:
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='mean')),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95))
            ])

            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Successfully loaded train and test datasets")

            target_column_name = "fault"
            numerical_columns = [col for col in train_df.columns if col != target_column_name]

            preprocessing_obj = self.get_data_transformer_object(numerical_columns)
            ## divide the train dataset to independent and dependent features
            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]
            ## divide the test dataset to independent and dependent features
            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            logging.info("Applying preprocessing pipeline")

            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            logging.info("Saving preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
