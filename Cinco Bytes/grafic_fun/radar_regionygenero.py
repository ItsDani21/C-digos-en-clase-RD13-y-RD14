import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def grafic_radar_region_genero(df):
    #Radar de Ventas por Region y Genero
    platforms = ['na_sales', 'jp_sales', 'pal_sales', 'other_sales']
    genre_platform = df.groupby('genre')[platforms].sum()

    max_val = genre_platform.values.max()
    angles = np.linspace(0,2 * np.pi, len(platforms), endpoint=False).tolist()
    angles += angles[:1] #Aqui cerramos el circulo

    plt.figure(figsize=(8,8))
    for genre in genre_platform.index:
        values = genre_platform.loc[genre].tolist()
        values += values[:1]
        plt.polar(angles,values,marker='o',label=genre)
    plt.xticks(angles[:-1], platforms)
    plt.title("Radar de Ventas por Region y Genero")
    plt.legend(loc='upper right')
    plt.show()