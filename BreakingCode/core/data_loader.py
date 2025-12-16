import pandas as pd
import os

DATASET = "datasets"

def load_csv(name):
    path = os.path.join(DATASET, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Falta {name}")
    return pd.read_csv(path)

def load_base_data():
    return {
        "est": load_csv("estudiantes.csv"),
        "rend": load_csv("rendimiento.csv"),
        "obs": load_csv("observaciones.csv"),
        "areas": load_csv("areas_estudio.csv"),
    }
