# Traffic Reducer

##  Problema

Muchos Paises enfrentan una problemática en la gestión del tráfico: los semáforos operan en temporización fija que no reflejan las condiciones reales del flujo vehicular. Este desfase entre la demanda de tránsito y la lógica de control genera:

- **Largas filas en algunos carriles** mientras otros permanecen vacíos
- **Aumento del tiempo de espera** y pérdida de tiempo productivo
- **Congestión innecesaria** en intersecciones críticas
- **Mayor consumo de combustible** y estrés para los conductores
- **Riesgo para vehículos de emergencia** (ambulancias, bomberos, patrullas) cuando un tapón en un semáforo les impide avanzar rápidamente

##  Solución

Traffic Reducer es una solución basada en **inteligencia artificial** orientada a mejorar la gestión del tráfico vehicular en intersecciones urbanas. El sistema:

- **Analiza visualmente** el volumen de vehículos por dirección mediante cámaras
- **Utiliza IA** para procesar la información en tiempo real
- **Prioriza dinámicamente** las vías con mayor carga vehicular
- **Automatiza la toma de decisiones** sobre el control semafórico
- **Se adapta** a las condiciones reales del tránsito momento a momento


##  Objetivos del Proyecto

### Objetivo Principal
**Disminuir el tiempo promedio de espera** de los vehículos en intersecciones, automatizando la decisión de qué semáforo debe ponerse en verde mediante un modelo de IA.

### Métricas de Impacto
- Reducción del tiempo promedio de espera en intersecciones
- Disminución de la longitud de las filas vehiculares
- Reducción del número de detenciones innecesarias
- Mejora en la fluidez general del tráfico

### Valor Generado

**Social:**
- Mejora la experiencia de desplazamiento diario
- Reduce niveles de estrés asociados al tránsito
- Facilita el paso de servicios esenciales (ambulancias, bomberos, transporte público)

**Ambiental:**
- Disminuye el tiempo que los vehículos permanecen detenidos
- Reduce emisiones contaminantes
- Disminuye el consumo de combustibles fósiles
- Reduce la contaminación acústica

**Económico:**
- Solución de bajo costo vs. sistemas tradicionales
- Reduce pérdida de tiempo productivo
- Optimiza el consumo de combustible
- Viable para ciudades con restricciones presupuestarias

---

*Traffic Reducer: Inteligencia artificial al servicio de la movilidad urbana*






## MANUAL DE INSTALACIÓN Y USO - SYSTEMA DE TRÁFICO IA

Este archivo detalla los pasos para instalar y ejecutar el sistema de control de tráfico con visión artificial.

### 1. REQUISITOS PREVIOS
Asegúrate de tener instalado Python (preferiblemente versión 3.10 o 3.11).
Puedes verificarlo abriendo la terminal y escribiendo:
   python --version

### 2. INSTALACIÓN DE DEPENDENCIAS (LIBRERÍAS)
El sistema necesita varias librerías para funcionar (IA, Visión, Web, etc.).
Todas están listadas en el archivo "requirements.txt".

Para instalarlas automáticamente, abre una terminal en esta carpeta y ejecuta:

   pip install -r requirements.txt

Esto instalará:
- Flask (Servidor Web)  
- OpenCV (Visión Artificial)
- Ultralytics (Detección de Autos YOLOv8)
- YT-DLP (Conexión con YouTube)
- Pandas/Numpy/Sklearn (Procesamiento de Datos)

### 3. CÓMO ARRANCAR EL PROGRAMA
Una vez instaladas las dependencias, sigue estos pasos para iniciar el sistema:

1. Abre la terminal en la carpeta del proyecto.
2. Ejecuta el siguiente comando:

   py traffic_app/app.py

   (Nota: Si "py" no funciona, prueba con "python traffic_app/app.py" o "python3 traffic_app/app.py")

3. Verás mensajes en la terminal indicando que el modelo se cargó y la cámara se está iniciando.
4. Cuando veas "Stream Connected" y "Running on http://127.0.0.1:5000", el sistema está listo.

### 4. USO DEL SISTEMA
1. Abre tu navegador web (Chrome, Edge, etc.).
2. Entra a la dirección: http://127.0.0.1:5000
3. Verás la simulación de tráfico.
4. Para activar la Visión Artificial en vivo, haz clic en el interruptor:
   "LIVE SIMULATION MODE"
5. El sistema empezará a detectar autos en el video de YouTube y ajustará los semáforos automáticamente según la "Ley de la Mayoría" (Carril con más autos = Luz Verde).

### SOLUCIÓN DE PROBLEMAS
- Si ves errores rojos en la terminal sobre "TLS" o "Socket": Ignóralos, el sistema tiene un sistema de reconexión automática que los gestiona.
- Si el video no carga: Verifica tu conexión a internet, ya que depende de YouTube en vivo.
