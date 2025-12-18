import matplotlib.pyplot as plt
import seaborn as sns

def grafico_horas_vs_promedio(dataframe):
    plt.figure(figsize=(10, 6))
    datos = dataframe.groupby('rango_horas')['escala_promedio'].mean()
    datos.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('Promedio Académico según Horas de Estudio', fontsize=16, weight='bold')
    plt.xlabel('Rango de Horas', fontsize=12)
    plt.ylabel('Promedio Académico', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def grafico_metodos_estudio(dataframe):
    plt.figure(figsize=(8, 8))
    datos = dataframe['metodo_estudio'].value_counts()
    colores = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    datos.plot(kind='pie', autopct='%1.1f%%', colors=colores, startangle=90)
    plt.title('Distribución de Métodos de Estudio', fontsize=16, weight='bold')
    plt.ylabel('')
    plt.tight_layout()
    return plt.gcf()

def grafico_distractores_impacto(dataframe):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataframe, x='principales_distractores', y='escala_promedio', palette='Set2')
    plt.title('Impacto de Distractores en el Rendimiento Académico', fontsize=16, weight='bold')
    plt.xlabel('Tipo de Distractor', fontsize=12)
    plt.ylabel('Promedio Académico', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return plt.gcf()