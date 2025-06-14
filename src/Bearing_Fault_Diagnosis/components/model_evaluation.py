import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from Bearing_Fault_Diagnosis.exception import CustomException
from Bearing_Fault_Diagnosis.logger import logging

def plot_model_performance(report_dict, save_path="artifacts/model_scores.png"):
    try:
        if not report_dict:
            raise ValueError("Empty model report provided for visualization.")

        model_names = list(report_dict.keys())
        train_scores = [report_dict[m]["train_r2"] for m in model_names]
        test_scores = [report_dict[m]["test_r2"] for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        plt.figure(figsize=(14, 6))
        plt.bar(x - width/2, train_scores, width, label='Train R²', color='skyblue')
        plt.bar(x + width/2, test_scores, width, label='Test R²', color='salmon')

        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.title('Model Train vs Test Performance')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

        logging.info(f"Model performance comparison plot saved at: {save_path}")

    except Exception as e:
        raise CustomException(e, sys)
