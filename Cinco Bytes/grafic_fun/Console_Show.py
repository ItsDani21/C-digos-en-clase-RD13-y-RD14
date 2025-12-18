import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def Available_Games_Console(df):
    consolas_mas_juegos = (
    df.groupby("console")["title"]
    .count()  # contar títulos
    .reset_index()
    .rename(columns={"title": "num_juegos"})
    .sort_values(by="num_juegos", ascending=False)
)

# Tomar el top 10
    top_consolas_juegos = consolas_mas_juegos.head(10)

    print("Top 10 consolas con más juegos:")
    print(top_consolas_juegos)

#Grafico de columnas vertical
    plt.figure(figsize=(12,6))
    sns.barplot(
        data=top_consolas_juegos,
        x="console",
        y="num_juegos",
        palette="mako"
    )
    plt.title("Top 10 Consolas con Más Juegos")
    plt.xlabel("Consola")
    plt.ylabel("Número de Juegos")
    plt.xticks(rotation=45)

#Mostrar numero exacto encima de cada barra
    for index, row in top_consolas_juegos.iterrows():
        plt.text(index, row.num_juegos + 0.5, row.num_juegos, ha='center')

plt.show()