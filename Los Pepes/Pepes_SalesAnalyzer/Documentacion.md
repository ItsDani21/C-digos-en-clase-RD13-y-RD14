# *Pepes_SalesAnalyzer: Proyecto de Análisis de Ventas Farmacéuticas*

---

## 🧩 `Planteamiento del proyecto`

El proyecto tiene como finalidad automatizar el proceso de análisis de ventas de una empresa farmacéutica, permitiendo integrar, limpiar y analizar grandes volúmenes de información.  

A través de este flujo ETL (Extracción, Transformación y Carga), se busca obtener una visión integral del comportamiento de los clientes, identificar los productos más vendidos, detectar las temporadas de mayor y menor demanda, y segmentar la base de clientes según sus hábitos de compra.  

Finalmente, se desarrolla un dashboard interactivo que facilita la exploración y toma de decisiones estratégicas a partir de los datos procesados.

---

## 🎯 `Objetivos`

### **Objetivo general**
Implementar un sistema de análisis de ventas que integre el proceso ETL, análisis exploratorio, segmentación de clientes y visualización interactiva mediante Python y sus librerías analíticas.

### **Objetivos específicos**
1. **Automatizar la carga y limpieza** de un archivo CSV que maneja la base de datos con un volumen alto de información.  
2. **Validar y transformar los datos** asegurando coherencia en tipos de datos y formatos.  
3. **Analizar patrones de venta**: productos más vendidos, clientes frecuentes y comportamiento mensual de las ventas.  
4. **Aplicar el modelo RFM (Recency, Frequency, Monetary)** para segmentar los clientes en grupos estratégicos (Champions, Leales, En riesgo, etc.).  
5. **Diseñar un dashboard interactivo** que permita visualizar de manera dinámica los resultados del análisis y segmentación.

---

## 🛠️ `Herramientas utilizadas`

| Categoría | Herramientas | Función principal |
|------------|---------------|------------------|
| **Manipulación de datos** | `pandas`, `numpy` | Carga, limpieza, transformación y análisis estadístico de los datos. |
| **Visualización estática** | `matplotlib` | Creación de gráficos básicos de barras y líneas para explorar tendencias. |
| **Visualización interactiva** | `plotly.express` | Gráficos dinámicos para analizar clientes, productos y ventas por segmento. |
| **Dashboard interactivo** | `dash` | Construcción de una interfaz web para filtrar y visualizar los resultados del análisis. |
| **ETL y automatización** | `os` | Lectura automática de un archivo CSV desde una carpeta específica. |
| **Google Colab** | Entorno de ejecución | Permite integrar Google Drive y ejecutar el proyecto en la nube. |

---



## 📊 `Breve explicación del resultado`

El resultado del proyecto es un sistema analítico completo que:
- Unifica automáticamente los datos de ventas de un archivo que contiene gran cantidad de información en un solo conjunto limpio y estructurado.  
- Identifica los clientes más activos, los productos más vendidos y las temporadas de mayor y menor venta.  
- Clasifica a los clientes según su comportamiento de compra mediante la metodología RFM, generando grupos estratégicos como *Champions*, *Leales*, *En riesgo*, etc.  
- Presenta toda la información en un dashboard interactivo, donde el usuario puede filtrar por segmento y visualizar patrones de gasto, frecuencia de compra, mapas de calor y evolución mensual de las ventas.  

En conclusión, el proyecto convierte datos de ventas en información visual, útil y estratégica para la toma de decisiones comerciales.
