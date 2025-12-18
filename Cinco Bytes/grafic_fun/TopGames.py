import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def top_games_available(df):
    top_puntuacion = df.sort_values(by=["critic_score","total_sales", "console"], ascending=False)

    top_10 = top_puntuacion.head(10)[["title", "console", "critic_score", "total_sales"]]

    #Enseñar tabla
    print("Top 10 juegos mejor puntuados:")
    print(top_10)

#   Configuracion de estilo
    sns.set(style="darkgrid")

    # Gráfica de puntuación
    plt.figure(figsize=(12,6))
    sns.barplot(
        data=top_10,
        y="title",
        x="critic_score",
        palette="Blues_r"
    )
    
    plt.title("Top 10 Juegos Mejor Puntuados (Critic Score)")
    plt.xlabel("Puntuación de Críticos")
    plt.ylabel("Juego")
    plt.show()