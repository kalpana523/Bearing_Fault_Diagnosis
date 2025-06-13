import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from Bearing_Fault_Diagnosis.logger import logging
from Bearing_Fault_Diagnosis.exception import CustomException
from Bearing_Fault_Diagnosis.components.data_ingestion import DataIngestion
from Bearing_Fault_Diagnosis.components.data_ingestion import DataIngestionConfig
from Bearing_Fault_Diagnosis.components.data_transformation import DataTransformationConfig, DataTransformation

if __name__ == "__main__":
    logging.info("The execution has started")
    
    try:
       # data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        
       # data_transformation_config=DataIngestionConfig()
        data_transformation=DataTransformation()
        data_transformation.initiate_data_transformation( train_data_path,test_data_path)
        
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
