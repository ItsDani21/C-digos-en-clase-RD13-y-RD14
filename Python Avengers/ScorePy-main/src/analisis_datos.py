import pandas as _pd
import numpy as _np


def _cols_existen(df, cols):
    return all(c in df.columns for c in cols)


def analisis_promedio_por_metodo(dataframe):
    """Promedio, cantidad y desviación estándar por método de estudio."""
    necesarias = ['metodo_estudio', 'escala_promedio']
    if not _cols_existen(dataframe, necesarias):
        return _pd.DataFrame()
    resultado = (
        dataframe.groupby('metodo_estudio')['escala_promedio']
        .agg([
            ('promedio', 'mean'),
            ('estudiantes', 'count'),
            ('desviacion', 'std'),
        ])
        .sort_values('promedio', ascending=False)
    )
    return resultado.round(2)


def analisis_horas_vs_promedio(dataframe):
    """Correlación simple entre horas estudiadas y promedio académico."""
    necesarias = ['horas_estudiadas', 'escala_promedio']
    if not _cols_existen(dataframe, necesarias):
        return _pd.DataFrame()
    correlacion = dataframe[necesarias].corr()
    print("\n📊 CORRELACIÓN: Horas de estudio vs. Promedio académico")
    print("=" * 60)
    valor = correlacion.iloc[0, 1]
    print(f"Coeficiente de correlación: {valor:.3f}")

    if valor > 0.5:
        print("✅ Correlación POSITIVA FUERTE: Más horas = Mejor promedio")
    elif valor > 0.3:
        print("✅ Correlación POSITIVA MODERADA")
    elif valor > 0:
        print("⚠️  Correlación POSITIVA DÉBIL")
    else:
        print("❌ Correlación NEGATIVA o NULA")
    return correlacion


def analisis_distractores(dataframe):
    """Promedio y cantidad de estudiantes por distractor principal."""
    necesarias = ['principales_distractores', 'escala_promedio']
    if not _cols_existen(dataframe, necesarias):
        return _pd.DataFrame()
    resultado = (
        dataframe.groupby('principales_distractores')['escala_promedio']
        .agg([
            ('promedio', 'mean'),
            ('estudiantes', 'count'),
        ])
        .sort_values('promedio', ascending=False)
    )
    return resultado.round(2)


def analisis_promedio_por_rango_horas(dataframe):
    """Promedio académico por rango de horas (usa columna 'rango_horas')."""
    if not _cols_existen(dataframe, ['rango_horas', 'escala_promedio']):
        return _pd.DataFrame()
    res = (
        dataframe.groupby('rango_horas')['escala_promedio']
        .agg([('promedio', 'mean'), ('estudiantes', 'count')])
        .sort_index()
    )
    return res.round(2)


def analisis_correlacion_edad_promedio(dataframe):
    """Correlación entre edad del alumno y promedio (float)."""
    if not _cols_existen(dataframe, ['edad_alumno', 'escala_promedio']):
        return _np.nan
    return dataframe[['edad_alumno', 'escala_promedio']].corr().iloc[0, 1]


def analisis_motivacion_vs_promedio(dataframe):
    """Promedio por nivel de motivación declarado."""
    if not _cols_existen(dataframe, ['motivacion_estudio', 'escala_promedio']):
        return _pd.DataFrame()
    res = (
        dataframe.groupby('motivacion_estudio')['escala_promedio']
        .agg([('promedio', 'mean'), ('estudiantes', 'count')])
        .sort_values('promedio', ascending=False)
    )
    return res.round(2)


def analisis_recursos_estudio_popularidad(dataframe):
    """Cantidad de estudiantes por tipo de recurso de estudio (frecuencia)."""
    if 'recursos_estudios' not in dataframe.columns:
        return _pd.DataFrame()
    conteo = dataframe['recursos_estudios'].value_counts().rename('estudiantes')
    return conteo.to_frame()


def analisis_promedio_por_frecuencia_repaso(dataframe):
    """Promedio académico según la frecuencia de repaso declarada."""
    if not _cols_existen(dataframe, ['frecuencia_repaso', 'escala_promedio']):
        return _pd.DataFrame()
    res = (
        dataframe.groupby('frecuencia_repaso')['escala_promedio']
        .agg([('promedio', 'mean'), ('estudiantes', 'count')])
        .sort_values('promedio', ascending=False)
    )
    return res.round(2)


def analisis_impacto_horario_fijo(dataframe):
    """Compara promedio entre quienes tienen horario fijo y quienes no."""
    if not _cols_existen(dataframe, ['horario_fijo', 'escala_promedio']):
        return {}
    resumen = (
        dataframe.groupby('horario_fijo')['escala_promedio']
        .agg(['mean', 'count', 'std'])
        .rename(columns={'mean': 'promedio', 'count': 'estudiantes', 'std': 'desviacion'})
    )
    resumen = resumen.round(2)
    diff = _np.nan
    try:
        valores = resumen['promedio'].tolist()
        if len(valores) == 2:
            diff = round(valores[0] - valores[1], 2)
    except Exception:
        pass
    return {
        'detalle': resumen,
        'diferencia_promedio': diff,
    }


def analisis_sentimientos_vs_promedio(dataframe):
    """Promedio académico según sentimiento respecto al estudio."""
    if not _cols_existen(dataframe, ['sentimientos_respecto_estudio', 'escala_promedio']):
        return _pd.DataFrame()
    res = (
        dataframe.groupby('sentimientos_respecto_estudio')['escala_promedio']
        .agg([('promedio', 'mean'), ('estudiantes', 'count')])
        .sort_values('promedio', ascending=False)
    )
    return res.round(2)