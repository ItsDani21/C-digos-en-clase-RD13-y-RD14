
# 🧠 Sistema Inteligente de Evaluación y Predicción Estudiantil  
**Con IA, NLP y Visualización Interactiva**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)  
![IA](https://img.shields.io/badge/IA-NLP%20%2B%20ML-orange.svg)  
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)  
![Status](https://img.shields.io/badge/Status-Activo-success.svg)

---

## 🌟 ¿Qué es ahora?
**De terminal → web interactiva.**  
Sistema que **predice riesgo de deserción**, **recomienda áreas académicas/profesiones** y **visualiza talento** usando **IA aplicada a datos escolares**.

---

## 🚀 Tecnologías que usamos ahora

| Tecnología | Uso |
|------------|-----|
| **Python 3.8+** | Backend y ML |
| **Streamlit** | UI web interactiva |
| **Pandas / NumPy** | Análisis y limpieza de datos |
| **Scikit-learn** | Modelos de IA (RandomForest, LogisticRegression, SVM, TF-IDF) |
| **NLTK** | Procesamiento de lenguaje natural (NLP) |
| **Matplotlib** | Gráficos y visualizaciones |
| **Imbalanced-learn** | Balanceo de clases (SMOTE) |
| **SciPy** | Operaciones con matrices dispersas |

---

## 📦 Librerías que necesitas instalar

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn nltk imbalanced-learn scipy
```

---

## 🧠 ¿Qué hace el sistema con IA?

| Función | IA Usada | Descripción |
|---------|----------|-------------|
| **Riesgo de deserción** | `LinearSVC + TF-IDF + CalibratedClassifierCV` | Analiza **observaciones docentes** y predice **probabilidad de abandono** |
| **Recomendación de áreas/carreras** | `RandomForestClassifier` | Sugiere **áreas académicas o profesiones** según **nota, asistencia y ambiente** |
| **Score de ambiente (F)** | `TF-IDF + palabras clave + expresiones de riesgo` | Devuelve **probabilidad 0-1** de ambiente emocional positivo |
| **Radar de habilidades** | `Matplotlib polar` | Visualiza **competencias por área** con gráficos interactivos |
| **Dashboard personalizado** | `Streamlit + Matplotlib` | Muestra **KPIs, gráficos y tablas** por estudiante |

---

## 🧪 Modelos y datasets

| Dataset | Uso | Modelo |
|---------|-----|--------|
| `estudiantes.csv` | Datos del alumno | Entrada de features |
| `asignaturas.csv` | Datos de la materia y profesor | Entrada de features |
| `rendimiento.csv` | Notas y asistencia | Features numéricas |
| `observaciones.csv` | Textos de docentes | Entrada del **NLP** |
| `nlp_observaciones_entrenamiento.csv` | Entrenamiento del **score de ambiente** | `TF-IDF + SVM + Calibración` |

---

## 🖥️ Interfaz web con Streamlit

| Página | Función |
|--------|---------|
| **Estudiantes** | gráficos individuales y generales |
| **Riesgo** | **Predicción de abandono** con NLP |
| **Área Académica** | **Recomendación de áreas** con RandomForest |
| **Dashboard** | **KPIs, histogramas, tabla resumen** |

---

## 🎯 Ejemplo de uso

```bash
streamlit run app.py
```

1. **Selecciona un estudiante**
2. **Ve su riesgo, área recomendada y radar de habilidades**
3. **Explora observaciones, gráficos y probabilidades**

---

## 👥 Autores

- **Rushaner Minaya** - [RushanerM](https://github.com/RushanerM)  
- **Cristian Beltre** - [p0lquer](https://github.com/p0lquer)  
- **Francis Céspedes** - [Francis-Manuel374](https://github.com/Francis-Manuel374)  
- **Anderson Frias** - [anderj14](https://github.com/anderj14)  
- **Wilnel Pérez** - [Wilnel-Pérez](https://github.com/Wilnel-Pérez)

---

## 🙏 Agradecimientos

**Samsung Innovation Campus** - Por la formación y recursos  
**Docentes y mentores** - Por su guía  
**Compañeros** - Por la colaboración
