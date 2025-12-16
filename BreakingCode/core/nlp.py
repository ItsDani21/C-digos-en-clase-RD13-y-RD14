import re
import pickle
import os
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from core.config import PALABRAS_POS, PALABRAS_NEG
from nltk.corpus import stopwords
import nltk
import unicodedata


MODEL_PATH = "modelo_riesgo.pkl"

def limpiar_texto(t):
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"[^a-záéíóúñü ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def quitar_acentos(texto):
    if not isinstance(texto, str):
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )


def extraer_features(texto):
    texto = texto.lower()
    feats = {}
    for p in PALABRAS_POS:
        feats[f"pos_{p}"] = texto.count(p)
    for n in PALABRAS_NEG:
        feats[f"neg_{n}"] = texto.count(n)
    feats["longitud"] = len(texto)
    feats["num_palabras"] = len(texto.split())
    feats["ratio_neg"] = sum(feats[f"neg_{n}"] for n in PALABRAS_NEG) / (feats["num_palabras"] + 1)
    return feats


def cargar_modelo_nlp():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    nltk.download("stopwords", quiet=True)
    STOPWORDS_ES = stopwords.words("spanish")

    df_train = pd.read_csv("datasets/nlp_observaciones_entrenamiento.csv")
    df_train["texto"] = df_train["texto"].astype(str).apply(limpiar_texto)
    textos = df_train["texto"].tolist()
    y = df_train["label"].values

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2), stop_words=STOPWORDS_ES, min_df=2)
    X_tfidf = tfidf.fit_transform(textos)

    # Features manuales
    manual = [extraer_features(t) for t in textos]
    vec = DictVectorizer()
    X_manual = vec.fit_transform(manual)

    # Unión
    X = hstack([X_tfidf, X_manual])

    # Entrenar
    base = LinearSVC()
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(X, y)

    model_data = (clf, tfidf, vec)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    return model_data


def puntaje_ambiente(texto, modelo_nlp=None):
    if not texto or pd.isna(texto) or modelo_nlp is None:
        return 0.5

    texto = limpiar_texto(texto)
    clf, tfidf, vec = modelo_nlp

    X_tfidf = tfidf.transform([texto])
    X_manual = vec.transform([extraer_features(texto)])
    X = hstack([X_tfidf, X_manual])

    return float(clf.predict_proba(X)[0][1])

def score_keywords(texto, palabras, modificadores=None):

    texto = quitar_acentos(texto.lower())
    palabras = [quitar_acentos(p.lower()) for p in palabras]
    
    total = 0
    for p in palabras:
        # Buscar palabra exacta
        matches = re.finditer(r'\b' + re.escape(p) + r'\b', texto)
        for m in matches:
            total += 1
            if modificadores:
                # Chequear si antes de la palabra hay un modificador
                start = max(0, m.start()-15)
                contexto = texto[start:m.start()]
                if any(mod in contexto for mod in modificadores):
                    total += 0.5  # ponderación extra
    return total


def score_riesgo(texto, riesgo_expr):
    texto = quitar_acentos(texto.lower())
    riesgo_expr = [quitar_acentos(r.lower()) for r in riesgo_expr]
    return sum(1 for r in riesgo_expr if r in texto)