import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from core.nlp import (
    puntaje_ambiente,
    limpiar_texto,
    quitar_acentos,
    score_keywords
)
from core.config import (
    PALABRAS_CIENCIA,
    PALABRAS_NUMERO,
    PALABRAS_SOCIAL,
    MODIFICADORES,
    RIESGO_EXPR
)


def asignar_area_reglas(row):
    n, a, f = row["nota_promedio"], row["asistencia"], row["F"]
    sc, sn, ss = row["score_ciencia"], row["score_num"], row["score_social"]

    base = 0.5*n + 0.3*a + 0.2*f*100
    scores = {}

    # --- ciencias ---
    if sc > sn and sc > ss:
        scores[3] = base * 1.2
        scores[5] = base * 1.1

    # --- numerico ---
    if sn >= sc:
        scores[9] = base * 1.1
        scores[17] = base * 1.1

    # --- social ---
    if ss > sn:
        scores[8] = base * 1.1
        scores[7] = base * 1.1

    # fallback
    if not scores:
        scores[25] = base

    return max(scores, key=scores.get)


def preparar_dataset(est, rend, obs, modelo_nlp):
    periodos = ["P1","P2","P3","P4"]
    rend["nota"] = rend["CF"].fillna(rend[periodos].mean(axis=1))

    notas = rend.groupby("id_estudiante")["nota"].mean().reset_index(name="nota_promedio")
    asist = rend.groupby("id_estudiante")["asistencia"].mean().reset_index(name="asistencia")
    obs_g = obs.groupby("id_estudiante")["observacion"].apply(list).reset_index()

    df = est.merge(notas, on="id_estudiante", how="left") \
            .merge(asist, on="id_estudiante", how="left") \
            .merge(obs_g, on="id_estudiante", how="left")

    df["F"] = df["observacion"].apply(
        lambda x: puntaje_ambiente(" ".join(x), modelo_nlp)
        if isinstance(x, list) else 0.5
    )

    df["var_F"] = df["observacion"].apply(
        lambda x: np.std([puntaje_ambiente(o, modelo_nlp) for o in x])
        if isinstance(x, list) and len(x) > 1 else 0
    )

    df["num_obs"] = df["observacion"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    df[["score_ciencia", "score_num", "score_social"]] = df["observacion"].apply(
    lambda x: pd.Series(calc_scores(x)))
    
    df["id_area"] = df.apply(asignar_area_reglas, axis=1)

    return df


def entrenar_modelo(df):
    features = [
        "nota_promedio","asistencia","F","var_F","edad","num_obs",
        "score_ciencia","score_num","score_social"
    ]

    X, y = df[features], df["id_area"]

    pre = ColumnTransformer([("num", "passthrough", features)])
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X, y)

    return pipe

def calc_scores(obs_list):
    if not isinstance(obs_list, list):
        return 0, 0, 0

    texto = limpiar_texto(" ".join(obs_list))
    texto = limpiar_texto(texto)
    texto = quitar_acentos(texto)
    
    sc = score_keywords(texto, PALABRAS_CIENCIA, MODIFICADORES)
    sn = score_keywords(texto, PALABRAS_NUMERO, MODIFICADORES)
    ss = score_keywords(texto, PALABRAS_SOCIAL, MODIFICADORES)

    riesgo = score_keywords(texto, RIESGO_EXPR)
    
    sc -= riesgo * 0.2
    sn -= riesgo * 0.2
    ss -= riesgo * 0.2
    
    return max(0, sc), max(0, sn), max(0, ss)
