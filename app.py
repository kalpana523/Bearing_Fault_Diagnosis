import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from Bearing_Fault_Diagnosis.logger import logging
from Bearing_Fault_Diagnosis.exception import CustomException
from Bearing_Fault_Diagnosis.components.data_ingestion import DataIngestion
from Bearing_Fault_Diagnosis.components.data_transformation import DataTransformation
from Bearing_Fault_Diagnosis.components.model_trainer import ModelTrainer
from Bearing_Fault_Diagnosis.components.model_evaluation import plot_model_performance  # ✅ Optional: if visualizing

if __name__ == "__main__":
    logging.info("The execution has started")
    
    try:
        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        # Step 3: Model Training
        model_trainer = ModelTrainer()
        r2_score, model_report = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        print(f"\nR2 Score of the best model: {r2_score}")
        
        # Step 4 (Optional): Visualize all model performances
        plot_model_performance(model_report)

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
