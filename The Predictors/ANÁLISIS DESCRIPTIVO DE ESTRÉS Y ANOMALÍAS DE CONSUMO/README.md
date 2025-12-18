# ⚡ Proyecto Final: ANÁLISIS DESCRIPTIVO DE ESTRÉS Y ANOMALÍAS DE CONSUMO: Indicadores de Riesgo de Fallas en la Red Eléctrica.

## 👥 Autores y Equipo
| Nombre del Grupo | **The Predictors** |
| :--- | :--- |
| **Integrantes** | Gabriel Nuñez Paulino, Xander Cruz de la Rosa, Eudy Yunior Lorenzo Ramirez, Luis Zadkiel Duran |

---

## 🎯 Resumen del Proyecto

Este proyecto desarrolla y valida el **Indicador de Estrés Sistémico (IES)**, una métrica clave diseñada para transformar la gestión de fallas de la red eléctrica de EDEEESTE de un modelo reactivo a una **planificación preventiva**.

El IES cuantifica la vulnerabilidad de la red al medir cuántos apagones ocurren en proporción a la energía que debe entregar. El análisis identifica los **patrones estacionales de riesgo** más críticos, proporcionando la base analítica para optimizar la asignación de recursos de mantenimiento.

---

## Estructura del Proyecto

El proyecto se desarrolla en un entorno de Jupyter Notebook o Google Colab y consta de las siguientes secciones principales:

1.  **Carga y Limpieza de Datos**: Carga los datasets proporcionados, identifica las columnas relevantes (especialmente las de mes y año) y realiza la limpieza necesaria, incluyendo la conversión de nombres de meses a su representación numérica.

2.  **Indicador de Estrés Sistémico (IES)**: Calcula el IES como la relación entre el número de apagones y el consumo de energía (normalizado por millón de kWh).

3.  **Identificación de Meses Críticos (Patrón Estacional)**: Agrupa los datos por mes y calcula el IES promedio mensual para identificar los meses con mayor riesgo histórico.

4.  **Análisis de Correlación entre Consumo y Apagones**: Calcula la correlación entre el consumo y los apagones a nivel global y anual para entender la relación entre la carga del sistema y las fallas.

5.  **Análisis de Variabilidad (Estabilidad del Sistema)**: Calcula el promedio y la desviación estándar anual del IES para evaluar la estabilidad del riesgo a lo largo del tiempo.

6.  **Tendencia Temporal del IES Descriptiva**: Analiza la tendencia anual del IES para determinar si el sistema está mejorando o empeorando.

7.  **Visualizaciones**: Genera gráficos relevantes como mapas de calor de riesgo y gráficos de doble eje para visualizar la relación entre consumo y apagones.
-----

## 💡 Planteamiento y Objetivos

### Problema Operacional
La red de distribución enfrenta fallas recurrentes sin un indicador que mida el estrés operativo real. Esta deficiencia resulta en una gestión reactiva, elevando los costos y comprometiendo la calidad del servicio al cliente.

### Objetivos Clave
1.  **Creación del IES:** Diseñar y validar la fórmula $\text{IES} = \frac{\text{Apagones}}{\text{Consumo}}$ para medir el riesgo de falla del sistema.
2.  **Detección de Estacionalidad:** Identificar los **meses críticos** (picos de riesgo) que demandan intervención preventiva.
3.  **Cuantificación de Sensibilidad:** Determinar la **alta correlación** entre el incremento de carga (Consumo) y el aumento de fallas (Apagones).

---

## 🛠️ Herramientas y Bases de Datos

El análisis se ejecuta en **Google Colaboratory (Python)**, facilitando la carga de múltiples archivos mediante el *stack* de datos:

* **Lenguaje:** Python 3
* **Gestión de Datos:** `pandas`
* **Cálculo y Estadística:** `numpy` y `scipy.stats`
* **Visualización:** `matplotlib` y `seaborn`

### Bases de Datos Requeridas
El proyecto utiliza los siguientes datasets:
*   `Estadisticas-de-Energia-Entregada-y-Perdida-por-Provincia-Agosto-2025.ods`
*   `Estadisticas-de-Averias-y-Emergencias-Atendidas-por-Provincia-agosto-2025 (1).xlsx`
*   `Estadisticas-de-Bajas-o-cancelaciones-por-provincias-agosto-2025.xlsx`
*   `Estadisticas-de-Facturacion-Por-Tarifa-y-Rango-De-Consumo-Agosto-2025.ods`
*   `samsung_dataset (1).csv` (Este dataset parece ser el principal utilizado para los cálculos y visualizaciones del IES).

---

## 💻 Guía de Ejecución del Código (`ProyectoSIC.ipynb`)

Para ejecutar el análisis, cargue los archivos mencionados en una sesión de Google Colaboratory. El código está organizado en secciones lógicas.

### I. Carga y Preparación de Datos (Celdas 1-8)
**Funcionalidad clave:** Limpieza y estandarización de la columna temporal.

1.  **Carga Inicial:** La primera celda utiliza `files.upload()` para cargar los cuatro *datasets* del disco local al entorno de Colab.
2.  **Identificación y Mapeo:** El código identifica las columnas de mes y aplica un diccionario de mapeo para convertir los nombres de los meses (`enero`, `febrero`, etc.) a sus **valores numéricos** (1, 2, etc.).
3.  **Consolidación:** Las variables clave (`apagones`, `consumo`, `Mes`, `Año`) son extraídas y consolidadas en el **DataFrame principal (`df`)** para los cálculos posteriores.

### II. Generación de Indicadores Clave (Celdas 9-13)
**Funcionalidad clave:** Cálculo y validación del riesgo IES.

| Celda | Nombre de la Funcionalidad | Output / Propósito |
| :--- | :--- | :--- |
| **9** | **Cálculo del IES** | **Crea la columna `IES_MM_kWh`** (Apagones por millón de kWh), el KPI central del proyecto. |
| **10** | **Riesgo Estacional** | Calcula el **IES promedio mensual** y muestra el *ranking* de los meses más críticos (Patrón Estacional). |
| **11** | **Análisis de Correlación** | Calcula la correlación entre `apagones` y `consumo` global y anual para validar la **sensibilidad de la red a la carga**. |
| **12** | **Variabilidad Anual** | Mide la **desviación estándar** del IES por año, indicando la estabilidad o volatilidad del riesgo. |
| **13** | **Tendencia Temporal** | Mide el **cambio anual** del IES promedio para determinar si el estrés del sistema está mejorando o empeorando estructuralmente. |

### III. Visualización de Resultados (Celdas 14-16)
**Funcionalidad clave:** Generación de evidencia visual para la toma de decisiones.

| Celda | Nombre de la Funcionalidad | Descripción del Gráfico Generado |
| :--- | :--- | :--- |
| **14** | **Mapa de Calor de Riesgo** | Matriz de calor (Año vs. Mes) mostrando la intensidad del IES, útil para ver la **recurrencia histórica**. |
| **15** | **Gráfico de Barras Estacional** | Visualización principal que destaca los **meses críticos de alto riesgo** (Septiembre y Enero) en el IES promedio. |
| **16** | **Gráfico de Doble Eje** | Serie de tiempo que superpone la evolución de `Consumo` y `Apagones`, validando visualmente la **alta correlación directa** entre carga y falla. |


## 🧩 Impacto y Aplicaciones Futuras
- Integración del IES en tableros de control corporativos (Power BI o Grafana).  
- Extensión del modelo a otras distribuidoras nacionales.  
- Desarrollo de un sistema de **alertas tempranas automáticas** basado en los valores críticos del IES.

---

## 🪪 Licencia y Créditos
Proyecto académico desarrollado en el marco del **Programa Samsung Innovation Campus (SIC) 2025**  
