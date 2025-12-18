import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def grafic_VentasvsCritica(df):

    genre_summary = df.groupby('genre').agg({
        'total_sales': "sum",
        "critic_score": 'mean'
    }).reset_index()
    genre_summary["participation"] = genre_summary["total_sales"] / genre_summary['total_sales'].sum() * 100

    plt.figure(figsize=(8,6))
    plt.scatter(
        genre_summary['total_sales'],
        genre_summary['critic_score'],
        s=genre_summary['participation']*10,
        alpha=0.6,
        color='red'
    )
    for i, row in genre_summary.iterrows():
        plt.text(row['total_sales'], row['critic_score'], row['genre'], fontsize=9, ha='center', va='center')
    plt.xlabel("ventas Totales (M)")
    plt.ylabel("Critica promedio")
    plt.title("Burbujas de Genero: Ventas vs Critica (Tamaño = Participacion)")
    plt.grid(True)
    plt.show()