import requests
import joblib
import os
import mlflow
import mlflow.sklearn
import random
    
def get_model():
    uri_artifacts = "https://dagshub.com/rahmantaufik27/mlflow-sml-rtaufik27.mlflow"
    mlflow.set_tracking_uri(uri_artifacts)
    model_uri = "models:/rf_model_tuning/6"
    model = mlflow.sklearn.load_model(model_uri)
    joblib.dump(model, "model/rf_model_tuning_v5.pkl")
    return model

def generate_random_features():
    return {
        "MonthlyIncome": random.randint(1000, 20000),
        "Age": random.randint(18, 60),
        "TotalWorkingYears": random.randint(0, 40),
        "OverTime": random.choice([0, 1]),
        "MonthlyRate": random.randint(1000, 20000),
        "DailyRate": random.randint(100, 2000),
        "DistanceFromHome": random.randint(1, 50),
        "HourlyRate": random.randint(10, 200),
        "NumCompaniesWorked": random.randint(0, 10)
    }

def load_model():
    MODEL_PATH = "model/rf_model_tuning_v5.pkl"
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None
    
def set_features():
    return [
        "MonthlyIncome", "Age", "TotalWorkingYears", "OverTime",
        "MonthlyRate", "DailyRate", "DistanceFromHome",
        "HourlyRate", "NumCompaniesWorked"
    ]