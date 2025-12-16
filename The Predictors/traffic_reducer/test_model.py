
import pickle
import sys
import os
import pandas as pd
import numpy as np
import joblib

print(f"Python version: {sys.version}")

# Adjust path to find the model
dataset_path = r"c:\Users\zadkiel\Downloads\traffic_reducer_dataset-20251211T152732Z-3-001\traffic_reducer_dataset\modelo_entrenado\modelo_semaforo_ia.pkl"

def try_load(path):
    print(f"Attempting to load from {path}")
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Pickle load failed: {e}")
    
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Joblib load failed: {e}")
    return None

model = try_load(dataset_path)

if model:
    print("Model loaded successfully!")
    # ... (rest of prediction logic)
    test_data = [[10, 50, 10, 10]] 
    try:
        prediction = model.predict(test_data)
        print(f"Prediction for {test_data}: {prediction}")
    except Exception as e:
         print(f"Prediction failed (might need DataFrame): {e}")
         try:
             df = pd.DataFrame(test_data, columns=['Norte', 'Sur', 'Este', 'Oeste'])
             prediction = model.predict(df)
             print(f"Prediction with DataFrame: {prediction}")
         except Exception as e2:
             print(f"Prediction with DataFrame also failed: {e2}")
else:
    print("FAILED to load model.")
