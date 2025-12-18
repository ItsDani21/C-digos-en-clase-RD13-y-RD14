import pandas as pd

def load_data(path="data/juegos_limpios.csv"):
    df = pd.read_csv(path)
    return df
