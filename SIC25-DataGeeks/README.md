# 🌊 HydroSense  
**Sistema de evaluación y alerta temprana de riesgo de inundación mediante aprendizaje automático**

## 📌 ¿Qué es HydroSense?

**HydroSense** es un proyecto de análisis y predicción de **riesgo de inundación** que utiliza técnicas de **aprendizaje automático** para transformar datos ambientales en **información accionable**.

En lugar de limitarse a decir *“habrá o no inundación”*, HydroSense busca algo más útil en contextos reales:

> **Estimar niveles de riesgo (Bajo / Medio / Alto)** y comunicar probabilidades claras que permitan **tomar decisiones preventivas**.

El proyecto combina:
- análisis de datos,
- modelado estadístico,
- interpretación de resultados,
- y una **interfaz interactiva** pensada para usuarios técnicos y no técnicos.

---

## 🎯 Problema que aborda

Las inundaciones no suelen ser eventos “todo o nada”.  
En la práctica, la pregunta importante suele ser:

- ¿Qué tan probable es que ocurra una inundación?
- ¿Estamos en un nivel de riesgo aceptable o crítico?
- ¿Cuándo conviene activar una alerta?

HydroSense parte de esta realidad y propone una **clasificación gradual del riesgo**, en lugar de una predicción binaria.

---

## 🧠 Enfoque conceptual

HydroSense no intenta simular físicamente una inundación (hidrodinámica, topografía, etc.).  
En su lugar:

- **Aprende patrones** a partir de datos históricos.
- **Relaciona múltiples variables** ambientales.
- Produce una **estimación probabilística de riesgo**.

Este enfoque es especialmente útil cuando:
- no se dispone de modelos físicos completos,
- los datos son heterogéneos,
- se necesita rapidez y flexibilidad.

---

## 🧪 Cómo funciona el modelo (explicado sin código)

1. **Entrada de datos**
   - El sistema recibe un conjunto de variables ambientales.
   - Entre ellas existe una métrica continua llamada `FloodProbability`.

2. **Construcción del riesgo**
   - En vez de usar esa probabilidad directamente, se transforma en **tres niveles de riesgo**:
     - **Bajo**
     - **Medio**
     - **Alto**
   - Esto se hace usando **cuartiles estadísticos**, lo que permite:
     - adaptar el riesgo a la distribución real de los datos,
     - evitar umbrales arbitrarios.

3. **Entrenamiento**
   - Se entrena un modelo de tipo **Random Forest**, conocido por:
     - manejar relaciones no lineales,
     - ser robusto al ruido,
     - ofrecer interpretabilidad mediante importancias de variables.

4. **Evaluación**
   - El modelo se evalúa con métricas estándar:
     - accuracy,
     - balanced accuracy,
     - matriz de confusión.
   - Se analiza **qué tipos de error comete** (por ejemplo, confundir riesgo alto con medio).

5. **Predicción interactiva**
   - El usuario puede simular una “zona” ajustando valores de entrada.
   - El sistema devuelve:
     - la clase de riesgo estimada,
     - la probabilidad de cada nivel,
     - un **aviso de alerta configurable**.

---

## 🚨 Sistema de alerta

Uno de los puntos clave de HydroSense es que **no impone una decisión rígida**.

El usuario puede definir:
- a partir de qué probabilidad de *Riesgo Alto* se considera crítica la situación.

Ejemplo:
- Umbral = 45%
- Si `P(Alto) ≥ 45%` → se emite alerta.

Esto permite adaptar el sistema según:
- el coste de una falsa alarma,
- el riesgo de no actuar a tiempo,
- el contexto operativo (prevención vs emergencia).

---

## 🖥️ Interfaz interactiva

HydroSense incluye un **dashboard en Streamlit** que permite:

- Cargar datasets propios.
- Entrenar y re-entrenar el modelo.
- Visualizar:
  - distribución de clases,
  - métricas de evaluación,
  - matriz de confusión normalizada,
  - importancia de variables.
- Simular escenarios y recibir avisos claros.

El objetivo es que **los resultados no se queden en un notebook**, sino que puedan ser **explorados, explicados y discutidos**.

---

## 🔍 Interpretabilidad

Además de predecir, HydroSense ayuda a responder:

- ¿Qué variables influyen más en el riesgo?
- ¿Por qué el modelo considera una zona como riesgosa?
- ¿Qué cambia si alteramos ciertos factores?

Esto es clave para:
- confianza en el sistema,
- toma de decisiones informada,
- comunicación con actores no técnicos.

---

## ⚠️ Limitaciones y uso responsable

HydroSense:
- **no reemplaza modelos físicos ni estudios técnicos oficiales**,
- **no predice inundaciones reales de forma determinista**,
- **no debe usarse como única fuente para decisiones críticas**.

Su propósito es:
> servir como **herramienta de apoyo**, exploración y alerta temprana basada en datos.

---

## 🌱 Posibles extensiones futuras

- Integración de datos en tiempo real (sensores, APIs).
- Ajuste dinámico de umbrales según contexto.
- Comparación con otros modelos (XGBoost, LightGBM).
- Versiones regionales adaptadas a distintos climas.
- Módulos de explicación avanzada (SHAP).

---

## 🧩 Conclusión

HydroSense es un proyecto que demuestra cómo el aprendizaje automático puede:

- traducir datos complejos en señales comprensibles,
- apoyar la prevención de riesgos,
- ofrecer información probabilística útil,
- y hacerlo de forma transparente e interactiva.

No busca ser un modelo “mágico”, sino una **herramienta honesta, explicable y adaptable** para entender mejor el riesgo de inundación.
