Perfecto. Acá va el plan completo, en fases, pensado para ejecutarse en Colab con GPU gratuita. Cada fase es un notebook o sección que podemos ir armando cuando me digas "dale".

**Fase 0 — Setup del entorno (Colab)**

* Runtime con GPU (T4 alcanza para esto).  
* Instalar: `mediapipe`, `opencv-python`, `pandas`, `numpy`, `tensorflow` (o `torch` si preferís), `kaggle`.  
* Montar Google Drive para persistir features extraídos (los frames/videos NO los vas a guardar en Drive por espacio, pero los landmarks sí — son livianos).

**Fase 1 — Obtención del dataset** DAiSEE tiene dos caminos:

* **Oficial**: formulario en people.iith.ac.in/vineethnb/resources/daisee → te mandan un link (puede tardar días en aprobarse).  
* **Mirror en Kaggle** (`olgaparfenova/daisee`): descarga directa vía Kaggle API, ideal para arrancar ya con el prototipo mientras esperás la aprobación oficial (en la tesis podés citar la fuente oficial igual, ya que el contenido es el mismo).

Te recomiendo arrancar por Kaggle para no perder tiempo, y validar contra el oficial cuando llegue.

**Fase 2 — Extracción de frames**

* Los videos vienen en carpetas por participante/clip, con labels en `TrainLabels.csv`, `ValidationLabels.csv`, `TestLabels.csv` (columnas: Boredom, Engagement, Confusion, Frustration, cada una 0-3).  
* Muestreo a tasa constante (ej. 1 frame cada N para no reventar el storage de Colab — con 2.7M frames totales no conviene extraer todo).  
* Guardar frames en `/content/frames/` (efímero) y procesar en el momento, sin persistir imágenes crudas (coherente con tu enfoque de privacidad del Cap. 2.1.12.3).

**Fase 3 — Extracción de landmarks con MediaPipe Face Mesh**

* Por cada frame: correr Face Mesh, extraer los puntos relevantes para:  
  * Apertura ocular (EAR \- Eye Aspect Ratio, landmarks del contorno de ojos)  
  * Orientación de cabeza (yaw/pitch/roll, vía landmarks de nariz/mentón/sienes o `solvePnP`)  
* Guardar como secuencias numéricas (no imágenes) → esto sí va a Drive, es liviano.  
* Manejo de frames sin rostro detectado: marcar como NaN.

**Fase 4 — Tratamiento de datos faltantes**

* Interpolación lineal para gaps cortos.  
* Descarte de secuencias con demasiados frames faltantes (definir umbral, ej. \>30%).

**Fase 5 — Armado de dataset secuencial**

* Ventanas de longitud fija (frames por clip) con las features (EAR, yaw, pitch, roll, etc.) ordenadas temporalmente.  
* Split train/val/test **por participante** (no por clip) para evitar leakage — DAiSEE ya viene particionado así, conviene respetarlo.  
* Decidir qué label usar como target: para "atención" probablemente mapees **Engagement** (y quizás combines con Boredom invertido) a tu clase binaria/multinivel de atención.

**Fase 6 — Modelo LSTM**

* Arquitectura simple: LSTM (1-2 capas) → Dense → salida (clasificación en niveles o regresión).  
* Baseline primero, sin ViBED-Net todavía (eso lo dejás para benchmark de comparación si te da el tiempo).

**Fase 7 — Entrenamiento y evaluación**

* Métricas: accuracy, F1 por clase (DAiSEE está desbalanceado, ojo con eso).  
* Comparar contra los benchmarks que citás en la tesis (64-67% LSTM, 73% ViBED-Net) para saber si tu implementación es razonable.

**Fase 8 — Export para producción**

* Convertir el modelo a un formato liviano (TF.js si la extensión corre en el navegador, o ONNX) para que corra client-side/Edge AI como definiste en el alcance.

