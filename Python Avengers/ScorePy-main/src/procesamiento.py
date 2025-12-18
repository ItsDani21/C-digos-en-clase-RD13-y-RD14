import pandas as pd
import numpy as np

def explorar_datos(dataframe):
    print(dataframe.info())
    print(dataframe.describe())
    valores_nulos = dataframe.isnull().sum()
    if valores_nulos.sum() == 0:
        print("No hay valores nulos")
    else:
        print(valores_nulos[valores_nulos > 0])

def limpiar_datos(dataframe):
    df_limpio = dataframe.copy()
    if 'Marca temporal' in df_limpio.columns:
        df_limpio = df_limpio.drop('Marca temporal', axis=1)

    nuevos_nombres = [
        'edad_alumno', 'curso_alumno', 'horas_estudiadas',
        'horario_fijo', 'frecuencia_repaso', 'metodo_estudio',
        'escala_promedio', 'promedio_reflejo', 'principales_distractores',
        'recursos_estudios', 'sentimientos_respecto_estudio', 'motivacion_estudio'
    ]
    if len(df_limpio.columns) != len(nuevos_nombres):
        print(f"Advertencia: CSV con {len(df_limpio.columns)} columnas (esperado {len(nuevos_nombres)})")
        return df_limpio

    df_limpio.columns = nuevos_nombres

    columnas_texto = [
        'curso_alumno', 'horario_fijo', 'frecuencia_repaso', 'metodo_estudio',
        'promedio_reflejo', 'principales_distractores',
        'recursos_estudios', 'sentimientos_respecto_estudio',
        'motivacion_estudio'
    ]
    for col in columnas_texto:
        if col in df_limpio.columns:
            df_limpio[col] = df_limpio[col].astype(str).str.strip().str.upper()

    df_limpio['edad_alumno'] = pd.to_numeric(df_limpio['edad_alumno'], errors='coerce')
    df_limpio['horas_estudiadas'] = pd.to_numeric(df_limpio['horas_estudiadas'], errors='coerce')
    df_limpio['escala_promedio'] = pd.to_numeric(df_limpio['escala_promedio'], errors='coerce')

    bins = [0, 2, 4, np.inf]
    labels = ['1-2 horas', '3-4 horas', '5+ horas']
    df_limpio['rango_horas'] = pd.cut(df_limpio['horas_estudiadas'], bins=bins, labels=labels)

    valores_nulos = df_limpio.isnull().sum()
    if valores_nulos.sum() > 0:
        print(valores_nulos[valores_nulos > 0])

    return df_limpio