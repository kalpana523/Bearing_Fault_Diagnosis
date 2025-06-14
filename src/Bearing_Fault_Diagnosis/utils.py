import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from Bearing_Fault_Diagnosis.exception import CustomException
from Bearing_Fault_Diagnosis.logger import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
import pymysql
import pickle
import numpy as np
from sklearn.metrics import r2_score

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def read_sql_data():
    logging.info("Reading SQL database has started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established")
        df = pd.read_sql_query('SELECT * FROM fault_analysis', mydb)
        print(df.head())
        return df

    except Exception as ex:
        raise CustomException(ex, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        train_scores = {}
        test_scores = {}
        best_model_name = None
        best_model_score = float('-inf')
        best_model = None

        for model_name, model in models.items():
            logging.info(f"Training {model_name}")
            gs = GridSearchCV(model, param[model_name], cv=3)
            gs.fit(X_train, y_train)

            best_params = gs.best_params_
            model.set_params(**best_params)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            train_scores[model_name] = train_r2
            test_scores[model_name] = test_r2
            report[model_name] = {"train_r2": train_r2, "test_r2": test_r2}

            if test_r2 > best_model_score:
                best_model_score = test_r2
                best_model_name = model_name
                best_model = model

        return report, best_model_name, best_model_score, best_model

    except Exception as e:
        raise CustomException(e, sys)

