# Desarrollo del módulo de detección de desatención — Avance metodológico y resultados

*Nota: este documento está redactado para insertarse como sección/capítulo de la tesis (ajustar numeración según corresponda al índice final). Mantiene el estilo impersonal y las citas ISO 690-2010 (autor-fecha) ya utilizados en el resto del trabajo.*

## 1\. Obtención y preprocesamiento del conjunto de datos

Para el entrenamiento y la evaluación del módulo de detección de indicadores de desatención se utilizó el conjunto de datos DAiSEE (Gupta *et al.*, 2016), conforme a lo definido en el apartado de estrategia y gestión de conjuntos de datos (sección 2.1.12). El acceso al dataset se realizó a través de un mirror público en Kaggle, mientras se gestiona en paralelo la aprobación del acceso oficial solicitado a los autores.

La estructura obtenida coincide con la documentada por los autores originales: 9.068 clips de video distribuidos en las particiones Train, Validation y Test, organizados por participante, junto con archivos de etiquetas (`TrainLabels.csv`, `ValidationLabels.csv`, `TestLabels.csv`) que registran los niveles de Boredom, Engagement, Confusion y Frustration (escala ordinal de 0 a 3\) para cada clip. Cada clip tiene una duración de 10 segundos a 30 cuadros por segundo (300 cuadros), con una resolución de 640×480 píxeles.

De los 9.068 clips nominales, se identificaron 652 (7,2%) sin archivo de video físicamente disponible en el mirror utilizado, por lo que el conjunto de trabajo quedó constituido por 8.416 clips con video y etiqueta disponibles.

### 1.1. Extracción de landmarks faciales

Siguiendo la metodología descrita en la sección 2.1.12.2, se implementó un pipeline de extracción que, para cada clip:

1. Muestrea 60 cuadros distribuidos uniformemente a lo largo de los 300 cuadros originales del clip, valor seleccionado por su correspondencia directa con el benchmark de referencia utilizado por Gothwal *et al.* (2025) para la evaluación de secuencias con arquitecturas LSTM.  
2. Procesa cada cuadro mediante la API vigente de MediaPipe (`FaceLandmarker`, Tasks API), dado que la API "legacy" (`mp.solutions.face_mesh`) referenciada originalmente en Lugaresi *et al.* (2019) fue discontinuada por Google en 2023\.  
3. Extrae, por cuadro, cinco variables: apertura ocular promedio, izquierda y derecha (Eye Aspect Ratio, EAR) y orientación de cabeza (yaw, pitch), esta última obtenida directamente de la matriz de transformación facial que provee el detector (alineación contra un modelo facial canónico), en lugar de un cálculo manual mediante `solvePnP`.  
4. Descarta la imagen inmediatamente después de procesarla, sin almacenamiento de video ni de cuadros individuales, en línea con el criterio de procesamiento local/Edge AI y minimización de datos definido en la sección 2.1.12.3.

La Tabla 1 resume los resultados de la extracción por partición.

**Tabla 1\.** Resultados de la extracción de landmarks por partición del dataset.

| Partición | Clips con video | Clips procesados (`ok`) | Sin video físico | % NaN promedio | Clips con \>30% cuadros sin rostro detectado |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Train | 5.358 | 4.852 | 506 | 0,02% | 1 |
| Validation | 1.429 | 1.429 | 0 | 0,02% | 0 |
| Test | 1.784 | 1.638 | 146 | 0,07% | 0 |
| **Total** | **8.571** | **7.919** | **652** | **\~0,03%** | **1** |

La calidad de detección resultó consistentemente alta en las tres particiones, con un porcentaje de cuadros sin rostro detectado marginal (0,02%-0,07% en promedio por clip), lo cual valida la robustez del detector facial de MediaPipe frente a las condiciones de captura "in the wild" propias de DAiSEE (variaciones de iluminación, oclusiones parciales, ángulos de cámara).

### 1.2. Tratamiento de datos faltantes

Conforme a la metodología de saneamiento definida en la sección 2.1.12.2, se aplicó interpolación lineal por columna (feature) para completar los cuadros sin detección facial dentro de cada secuencia, y se descartaron los clips que superaran el umbral de 30% de cuadros faltantes. Dado el bajo porcentaje de valores faltantes reportado en la Tabla 1, este proceso resultó en una pérdida mínima adicional: únicamente 1 clip de Train fue descartado por superar el umbral, quedando un conjunto final de **7.918 clips** (4.851 Train / 1.429 Validation / 1.638 Test) con secuencias completas de dimensión (60 cuadros × 5 variables), listas para el modelado.

## 2\. Definición del target de atención/desatención

Antes de entrenar cualquier modelo, fue necesario decidir qué combinación de las cuatro etiquetas de DAiSEE (Boredom, Engagement, Confusion, Frustration) se utilizaría como variable objetivo para representar la desatención del estudiante, dado que el alcance del presente trabajo se circunscribe específicamente a este fenómeno.

### 2.1. Análisis de distribución de clases

El análisis de la distribución de las etiquetas Engagement y Boredom en el conjunto de entrenamiento (Tabla 2\) reveló un desbalance severo en Engagement, con el 95,2% de los clips concentrados en los niveles "alto" y "muy alto" (niveles 2 y 3), y apenas un 4,8% distribuido entre los niveles "muy bajo" y "bajo". Este patrón se repitió de forma consistente en las particiones de Validation y Test.

**Tabla 2\.** Distribución de clases de Engagement y Boredom por partición (valores absolutos y porcentaje).

| Nivel | Train — Engagement | Train — Boredom | Validation — Engagement | Validation — Boredom | Test — Engagement | Test — Boredom |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 0 (muy bajo/nada) | 33 (0,7%) | 2.181 (45,0%) | 23 (1,6%) | 446 (31,2%) | 4 (0,2%) | 747 (45,6%) |
| 1 (bajo) | 200 (4,1%) | 1.479 (30,5%) | 143 (10,0%) | 376 (26,3%) | 81 (4,9%) | 519 (31,7%) |
| 2 (alto) | 2.474 (51,0%) | 1.043 (21,5%) | 813 (56,9%) | 475 (33,2%) | 849 (51,8%) | 335 (20,4%) |
| 3 (muy alto) | 2.144 (44,2%) | 148 (3,0%) | 450 (31,5%) | 132 (9,2%) | 704 (43,0%) | 37 (2,3%) |

La etiqueta Boredom presentó una distribución considerablemente más equilibrada, con sus dos niveles extremos (0 y 3\) sumando entre el 33% y el 48% del total según la partición, y una masa de datos más repartida entre los cuatro niveles.

### 2.2. Relación entre Engagement y Boredom

Con el fin de determinar si ambas etiquetas aportan información redundante o complementaria, se calculó la correlación de Pearson entre Engagement y Boredom sobre el conjunto de entrenamiento, obteniéndose un valor de **\-0,419**. La Tabla 3 presenta la tabla de contingencia correspondiente.

**Tabla 3\.** Tabla de contingencia Engagement × Boredom (conjunto de Train, valores absolutos).

| Engagement \\ Boredom | 0 | 1 | 2 | 3 |
| :---- | :---- | :---- | :---- | :---- |
| 0 | 5 | 2 | 8 | 18 |
| 1 | 22 | 35 | 109 | 34 |
| 2 | 764 | 903 | 731 | 76 |
| 3 | 1.390 | 539 | 195 | 20 |

Una correlación de magnitud moderada (no cercana a \-1) indica que Engagement y Boredom no constituyen la misma señal invertida, sino que capturan aspectos parcialmente distintos del estado atencional del estudiante. En particular, se observa que dentro del nivel Engagement=2 (el más numeroso del dataset) los clips se reparten de forma casi pareja entre los tres primeros niveles de Boredom, lo que evidencia que Engagement por sí solo no permite discriminar adecuadamente distintos grados de desatención.

### 2.3. Decisión de diseño

En función de lo expuesto, se descartó el uso de Engagement como variable objetivo principal, dado su severo desbalance de clases (95% concentrado en 2 de 4 niveles), y se adoptó **Boredom** como base para la construcción del target de desatención, colapsando sus cuatro niveles ordinales en tres categorías para mitigar la escasa representación del nivel 3 (2,3%-9,2% según partición):

- **Nivel 0 — Atento:** Boredom original \= 0\.  
- **Nivel 1 — Desatención leve:** Boredom original \= 1\.  
- **Nivel 2 — Desatención marcada:** Boredom original \= 2 o 3\.

Esta decisión resultó en una distribución de clases considerablemente más balanceada respecto de la alternativa binaria basada en Engagement (Tabla 4), reduciendo el riesgo de que un clasificador colapse prediciendo siempre la clase mayoritaria.

**Tabla 4\.** Distribución del target de 3 niveles derivado de Boredom, por partición.

| Nivel | Train | Validation | Test |
| :---- | :---- | :---- | :---- |
| Atento | 2.181 (45,0%) | 446 (31,2%) | 747 (45,6%) |
| Desatención leve | 1.479 (30,5%) | 376 (26,3%) | 519 (31,7%) |
| Desatención marcada | 1.191 (24,6%) | 607 (42,5%) | 372 (22,7%) |

Se observa una distribución de clases divergente en la partición de Validation respecto de Train y Test (con una proporción mayor de "Desatención marcada"), atribuible a la composición demográfica de los participantes asignados a esta partición por los autores originales del dataset, dado que la división de DAiSEE se realiza por participante y no por muestreo aleatorio de clips.

## 3\. Modelo baseline (Fase 1\)

### 3.1. Arquitectura

Se implementó un modelo secuencial basado en redes LSTM (*Long Short-Term Memory*), consistente con el enfoque adoptado en los antecedentes citados en la sección 2.2.3 (Gothwal *et al.*, 2025). La arquitectura consta de dos capas LSTM (64 y 32 unidades respectivamente) con regularización por *dropout* (0,3), seguidas de una capa densa de 16 unidades con activación ReLU y una capa de salida *softmax* de 3 unidades, correspondientes a los niveles del target definido en la sección 2.3.

### 3.2. Normalización y balanceo de clases

Las variables de entrada (EAR y ángulos de orientación de cabeza) presentan escalas heterogéneas — EAR en un rango aproximado de 0,1 a 0,4, frente a yaw/pitch expresados en grados (aproximadamente \-90° a 90°) —, por lo que se aplicó estandarización (`StandardScaler`), ajustada exclusivamente sobre el conjunto de entrenamiento para evitar la fuga de información hacia las particiones de validación y prueba.

Dado el desbalance residual del target de 3 niveles (Tabla 4), se incorporó una estrategia de ponderación de clases (*class weighting*) durante el entrenamiento, calculada de forma inversamente proporcional a la frecuencia de cada clase, en lugar de recurrir en esta primera instancia a la incorporación de un conjunto de datos adicional para compensar el desbalance.

### 3.3. Resultados obtenidos

Para contextualizar el desempeño del modelo, se estableció como referencia un **clasificador trivial** que predice sistemáticamente la clase mayoritaria del conjunto de entrenamiento ("Atento"), obteniendo una exactitud (*accuracy*) del **45,6%** sobre el conjunto de Test — cualquier modelo propuesto debe superar claramente este valor para considerarse informativo.

Adicionalmente, con el objetivo de diagnosticar si la limitación observada correspondía a las variables utilizadas o a la arquitectura secuencial, se evaluaron dos modelos no temporales de referencia (Regresión Logística y Random Forest) entrenados sobre estadísticas resumen (media, desvío estándar, mínimo y máximo) de cada variable calculadas sobre la totalidad de los 60 cuadros de cada clip. La Tabla 5 resume los resultados comparativos sobre el conjunto de Test.

**Tabla 5\.** Comparación de desempeño en el conjunto de Test (Fase 1, 5 features base).

| Modelo | Accuracy | F1 macro | Observación |
| :---- | :---- | :---- | :---- |
| Clasificador trivial (clase mayoritaria) | 45,6% | — | Referencia base |
| Regresión Logística (features resumen) | 37,4% | 0,370 | No temporal |
| Random Forest (features resumen) | 39,6% | 0,376 | No temporal |
| LSTM (secuencia completa) | 36,9% | 0,347 | Modelo propuesto |

Los tres modelos evaluados obtuvieron una exactitud **inferior** a la del clasificador trivial, resultado que se sostiene tanto en el enfoque secuencial (LSTM) como en los enfoques no temporales (Regresión Logística, Random Forest). El análisis de importancia de variables del Random Forest (Tabla 6\) mostró una distribución de relevancia relativamente uniforme entre las cinco variables extraídas, sin que ninguna concentrara una capacidad discriminativa marcadamente superior — patrón consistente con una señal débil respecto de la tarea de clasificación planteada.

**Tabla 6\.** Importancia de variables (Random Forest, top 10 de 20 features resumen).

| Variable | Importancia |
| :---- | :---- |
| pitch (mínimo) | 0,0635 |
| pitch (media) | 0,0622 |
| pitch (máximo) | 0,0587 |
| EAR izquierdo (media) | 0,0564 |
| pitch (desvío estándar) | 0,0536 |
| yaw (máximo) | 0,0513 |
| EAR izquierdo (desvío estándar) | 0,0508 |
| EAR derecho (máximo) | 0,0507 |
| EAR izquierdo (mínimo) | 0,0496 |
| yaw (desvío estándar) | 0,0485 |

## 4\. Discusión de resultados (Fase 1\)

El hallazgo de que modelos entrenados exclusivamente sobre apertura ocular y orientación de cabeza no logran superar un clasificador trivial para la predicción de Boredom encuentra respaldo directo en el trabajo original de DAiSEE. Gupta *et al.* (2016) señalan explícitamente que las señales faciales asociadas al aburrimiento incluyen la **postura corporal superior** del estudiante (por ejemplo, una postura reclinada hacia atrás asociada a mayor aburrimiento, frente a una postura erguida asociada a mayor compromiso), información que no es capturada por un pipeline centrado exclusivamente en el rostro como el implementado en el presente trabajo.

Esta observación resulta coherente con estudios más recientes sobre el mismo problema: Das y Dev (2025) reportan, para la tarea de clasificación de compromiso (*engagement*) sobre el dataset DAiSEE, que la incorporación de *Action Units* faciales (micro-expresiones definidas según el Facial Action Coding System, capturando movimientos de cejas, párpados y boca no representados en la geometría de landmarks básica) mejora significativamente el desempeño de los modelos respecto de utilizar únicamente landmarks, gaze y pose de cabeza. En ese mismo estudio, los modelos evaluados sobre DAiSEE combinando características faciales enriquecidas con redes profundas (EfficientNet) alcanzan una exactitud de aproximadamente 47,2%, en un orden de magnitud similar al obtenido en el presente trabajo, lo cual sitúa los resultados alcanzados dentro de un rango consistente con la dificultad reportada en la literatura para esta clase de tareas sobre DAiSEE, más allá de la elección puntual de variables.

Se concluye, en consecuencia, que la limitación observada no corresponde a un error de implementación ni a una configuración inadecuada del modelo secuencial, sino a una **restricción de las variables de entrada** disponibles en la versión inicial del pipeline (apertura ocular y orientación de cabeza únicamente), insuficientes para capturar la totalidad de las señales conductuales asociadas al aburrimiento/desatención según la conceptualización utilizada por los anotadores originales de DAiSEE.

## 5\. Limitaciones identificadas al cierre de Fase 1

A partir del diagnóstico anterior, se identificaron las siguientes líneas de mejora para la fase de modelado posterior:

1. **Incorporación de nuevas variables faciales.** Extender el pipeline de extracción (sección 1.1) para incluir: (a) la relación de apertura bucal (*Mouth Aspect Ratio*, MAR), indicador directamente vinculado al bostezo y asociado en la literatura a la somnolencia y fatiga cognitiva, calculable a partir de los mismos landmarks ya extraídos sin costo computacional adicional significativo; y (b) *blendshapes* faciales provistos nativamente por la API de MediaPipe utilizada (`FaceLandmarker`, mediante el parámetro `output_face_blendshapes`), que proveen aproximadamente 52 puntuaciones de micro-expresión facial conceptualmente equivalentes a los *Action Units* referenciados por Das y Dev (2025).  
2. **Evaluación de un conjunto de datos complementario.** Se descartó, en esta etapa, la incorporación de un segundo dataset (por ejemplo, el utilizado en estudios de EmotiW/Kaur *et al.*, 2018\) para mitigar el desbalance de clases, priorizando en su lugar técnicas de ponderación dentro del propio conjunto DAiSEE. Esta alternativa se mantiene como línea de trabajo futuro.  
3. **Limitación estructural del dataset.** Independientemente de las mejoras de features propuestas, se documenta como limitación del presente trabajo que DAiSEE fue construido a partir de grabaciones centradas en el rostro del participante, por lo que no resulta posible incorporar variables de postura corporal —identificadas por los propios autores del dataset como relevantes para el aburrimiento— sin recurrir a una fuente de datos adicional con captura de plano corporal más amplio, lo cual excede el alcance definido para el presente trabajo (sección 1.2, "Alcance").

---

## 6\. Fase 2: enriquecimiento de features y arquitectura recurrente bidireccional

### 6.1. Extracción de un conjunto de features enriquecido

En respuesta a las limitaciones documentadas en la sección 5, se re-ejecutó el pipeline de extracción incorporando las variables propuestas, ampliando el vector de características por cuadro de 5 a **22 variables**:

**Tabla 7\.** Grupos de variables extraídas por cuadro en la versión enriquecida del pipeline (`features_v2`).

| Grupo | Variables | Cantidad |
| :---- | :---- | :---- |
| Apertura ocular | `ear_avg`, `ear_left`, `ear_right` | 3 |
| Orientación de cabeza | `yaw`, `pitch` | 2 |
| Boca | `mar` (Mouth Aspect Ratio) | 1 |
| Parpadeo (blendshapes) | `eyeBlinkLeft`, `eyeBlinkRight` | 2 |
| Entrecerrar ojos (blendshapes) | `eyeSquintLeft`, `eyeSquintRight` | 2 |
| Dirección de mirada — *gaze* (blendshapes) | `eyeLookIn/Out/Up/Down` × {Left, Right} | 8 |
| Cejas (blendshapes) | `browDownLeft`, `browDownRight`, `browInnerUp` | 3 |
| Mandíbula (blendshapes) | `jawOpen` | 1 |
| **Total** |  | **22** |

Sobre este conjunto de 22 variables se derivaron adicionalmente **10 features de delta** (variación cuadro a cuadro de un subconjunto curado de variables, capturando la dinámica temporal de la señal y no solo su nivel), llevando el vector de entrada final a **32 features por cuadro**, manteniéndose la secuencia de 60 cuadros por clip definida en la sección 1.1.

Esta ampliación responde directamente a la recomendación de la sección 5.1: los blendshapes de MediaPipe operan como equivalente funcional de los *Action Units* del Facial Action Coding System referenciados por Das y Dev (2025), y el MAR incorpora la señal de bostezo/apertura bucal ausente en la versión inicial del pipeline.

### 6.2. Re-extracción y consolidación de datos

Durante la ejecución de la Fase 2 se detectó que, en un primer momento, únicamente la partición Train había sido (parcialmente) reprocesada con el pipeline enriquecido, mientras que Validation y Test se encontraban vacías, producto de que la re-descarga completa del dataset DAiSEE (8.232 videos: 4.976 Train / 1.536 Validation / 1.720 Test) fue posterior al procesamiento inicial de Train. Se identificó asimismo una inconsistencia de log-hygiene en el archivo de checkpointing (`Train_processing_log.csv`), que acumulaba filas duplicadas por `ClipID` en reintentos sucesivos (10.130 filas de log frente a 5.358 clips reales), sin que esto afectara la integridad de los archivos `.npz` ya generados, que constituyen la fuente de verdad utilizada para el conteo real de clips procesados (`ok`) por partición.

Subsanadas estas inconsistencias, se completó la extracción y consolidación de las tres particiones, incluyendo: interpolación de valores faltantes (`.ffill().bfill()` por secuencia), cálculo de los 10 deltas curados, y ajuste de un `StandardScaler` exclusivamente sobre Train, aplicado luego a Validation y Test.

### 6.3. Arquitectura del modelo (BiGRU)

Se reemplazó la arquitectura LSTM unidireccional de Fase 1 (sección 3.1) por una red recurrente bidireccional de tipo GRU (*Gated Recurrent Unit*), con la siguiente composición: una capa `Masking` (para ignorar el padding de secuencias, si lo hubiera), una capa `Bidirectional(GRU(32))`, normalización por lotes (`BatchNormalization`), una capa densa de 16 unidades con activación ReLU y *dropout*, y una capa de salida *softmax* de 3 unidades. El modelo incorpora regularización L2 sobre los pesos recurrentes y *gradient clipping* (`clipnorm`) para estabilizar el entrenamiento, totalizando aproximadamente **14.019 parámetros entrenables**.

La elección de una arquitectura bidireccional (frente a la unidireccional de Fase 1\) permite que la representación de cada cuadro incorpore contexto tanto de los cuadros anteriores como de los posteriores dentro de la ventana de 60 cuadros, lo cual resulta razonable dado que el clip completo (10 segundos) está disponible de antemano al momento de la clasificación (no se trata de un escenario de inferencia estrictamente en tiempo real cuadro a cuadro).

### 6.4. Ponderación de clases y métrica de selección de modelo

Se generalizó el esquema de ponderación de clases de Fase 1 (sección 3.2) a una función de ponderación suavizada, parametrizada por un exponente **β** que interpola entre no aplicar ponderación (β=0, todas las clases con peso 1\) y aplicar la ponderación inversamente proporcional completa a la frecuencia de clase (β=1). Dado el desalineamiento de distribución de clases entre Validation y Test documentado en la sección 2.3 (Tabla 4), se adoptó **macro-F1 sobre Validation** como criterio de selección de modelo y de *early stopping*, reservando la exactitud sobre Test —acompañada de su intervalo de confianza por bootstrap— como métrica de reporte final, conforme a lo establecido en el plan de la fase.

## 7\. Resultados de Fase 2

### 7.1. Modelos baseline no temporales (features enriquecidas)

Replicando el diagnóstico metodológico de la sección 3.3 pero sobre el conjunto de 32 features enriquecidas (resumidas estadísticamente por clip), se obtuvieron los siguientes resultados sobre Test: Regresión Logística, exactitud 43,6%; Random Forest, exactitud 44,9%. Ambos modelos no temporales, sin capturar dinámica secuencial alguna, ya superan a los tres modelos de Fase 1 (Tabla 5\) y se ubican en un rango cercano —aunque todavía sin superar de forma robusta— al clasificador trivial (45,6%).

### 7.2. Barrido del parámetro de ponderación de clases (β)

Se entrenó el modelo BiGRU (sección 6.3) para cinco valores del exponente de ponderación β, evaluando en cada caso la exactitud y el F1 macro sobre Validation y Test. La Tabla 8 resume los resultados.

**Tabla 8\.** Barrido de β — BiGRU, features enriquecidas (32).

| β | Val. accuracy | Val. F1 macro | Test accuracy | Test F1 macro |
| :---- | :---- | :---- | :---- | :---- |
| 0,00 | 0,405 | 0,381 | 0,447 | 0,350 |
| 0,25 | 0,376 | 0,360 | 0,404 | 0,368 |
| 0,50 | 0,432 | 0,418 | 0,433 | 0,386 |
| 0,75 | 0,446 | 0,426 | 0,419 | 0,400 |
| 1,00 | 0,446 | 0,422 | 0,420 | 0,408 |

Conforme al criterio de selección definido en la sección 6.4 (macro-F1 en Validation), el valor β=0,75 resultó el mejor candidato en esta corrida exploratoria (reducción de épocas para agilizar la búsqueda). Al re-entrenar dicha configuración con el presupuesto completo de épocas, se obtuvo una exactitud de Test de **40,6%** (intervalo de confianza al 95% por bootstrap: \[0,383; 0,430\]), resultado que **no supera de forma robusta** el clasificador trivial (45,6%, fuera del intervalo pero en dirección desfavorable) y que además fue **inferior** al obtenido en la corrida exploratoria equivalente (41,9%). Esta discrepancia entre corridas con la misma configuración de hiperparámetros se interpreta como evidencia de una **variabilidad run-to-run no despreciable** a la escala de datos y de modelo utilizada, más que como un error de configuración, y se documenta como una limitación a tener en cuenta al reportar cualquier resultado puntual de este modelo.

### 7.3. Estudio de ablación de features

Con el fin de aislar el aporte de cada grupo de variables incorporado en la Fase 2, se entrenó el modelo BiGRU sobre cuatro configuraciones incrementales de features, manteniendo fijos el resto de los hiperparámetros. La Tabla 9 resume los resultados sobre Test.

**Tabla 9\.** Ablación de features — BiGRU, Test.

| Configuración | Features incluidas | n features | Accuracy | F1 macro |
| :---- | :---- | :---- | :---- | :---- |
| A | EAR, yaw, pitch (Fase 1\) | 3 | 0,357 | 0,337 |
| B | A \+ MAR | 4 | 0,418 | 0,311 |
| C | B \+ blendshapes | 22 | 0,440 | 0,361 |
| D | C \+ deltas curados | 32 | 0,435 | 0,363 |

Los resultados de la ablación permiten extraer tres conclusiones. Primero, los *blendshapes* constituyen el aporte individual más relevante del enriquecimiento de features: el salto de la configuración B a C (agregando 18 variables de blendshapes) produce la mejora más marcada tanto en exactitud (+0,022) como en F1 macro (+0,050). Segundo, la incorporación de MAR (configuración B) mejora la exactitud de forma notable respecto de A (+0,061) pero **empeora** el F1 macro (-0,026), lo cual sugiere que esta variable favorece la predicción de la clase mayoritaria sin necesariamente aportar capacidad discriminativa entre las tres clases del target. Tercero, los deltas curados (configuración D) no aportan una mejora clara respecto de C: la exactitud se mantiene prácticamente estable (ligero descenso de 0,005) mientras que el F1 macro mejora marginalmente (+0,002), por lo que su inclusión no está claramente justificada en términos de costo-beneficio y se identifica como candidata a remover en una futura simplificación del pipeline.

### 7.4. Convergencia entre modelos y techo de señal

La comparación conjunta de los resultados de las secciones 7.1 a 7.3 —Regresión Logística (43,6%), Random Forest (44,9%), BiGRU a través de los cinco valores de β (40,4%-44,7%) y la configuración C de ablación (44,0%)— muestra una convergencia consistente de todos los modelos evaluados hacia un rango de exactitud de aproximadamente **40%-45%** sobre Test, independientemente de la arquitectura (lineal, basada en árboles, o recurrente bidireccional) y de la estrategia de ponderación de clases utilizada. Esta convergencia entre familias de modelos sustancialmente distintas constituye evidencia en contra de que la limitación observada responda a un problema de ajuste de hiperparámetros de un modelo en particular, y en favor de la hipótesis de un **techo de señal** impuesto por el conjunto de features disponible (landmarks, blendshapes y dinámica derivada, todos ellos limitados al rostro) para la tarea de clasificación de Boredom en 3 niveles tal como fue definida en la sección 2.3.

### 7.5. Exportación del modelo (TFLite / Edge AI)

Conforme al objetivo de procesamiento local definido en la sección 2.1.12.3 y 1.1, se realizó la exportación del modelo BiGRU entrenado al formato TensorFlow Lite. La conversión directa del modelo con tamaño de lote (*batch*) dinámico resultó en un error de conversión (`TensorListReserve requires element_shape to be static`), atribuible a una limitación conocida de TFLite para operaciones bidireccionales (`Bidirectional(GRU)`) con forma de lote no estática. Se resolvió reconstruyendo una versión del modelo idéntica en arquitectura y pesos (transferidos mediante `set_weights()`) pero con tamaño de lote fijo (`batch_size=1`), apto para inferencia clip por clip en el dispositivo de destino. Esta variante convirtió exitosamente utilizando únicamente operaciones nativas de TFLite (`TFLITE_BUILTINS`, sin necesidad de operaciones `SELECT_TF_OPS`/Flex), con un tamaño final de **41.320 bytes**, considerablemente más liviano que la alternativa con soporte Flex habilitado (64.448 bytes) y sin la dependencia adicional que esta última implica en tiempo de inferencia.

## 8\. Discusión de resultados de Fase 2

Los resultados obtenidos en Fase 2 muestran una mejora consistente respecto de Fase 1 en la métrica de F1 macro (de 0,347 en el LSTM original a valores de hasta 0,408-0,426 en las mejores configuraciones de BiGRU), validando parcialmente la hipótesis planteada al cierre de Fase 1 (sección 5\) de que la incorporación de blendshapes y MAR aportaría señal adicional no capturada por el pipeline inicial. Sin embargo, en términos de exactitud (*accuracy*) —métrica sobre la cual se definió originalmente el criterio de éxito mínimo del proyecto (superar el 45,6% del clasificador trivial)— ningún modelo evaluado en Fase 2 logró superarlo de forma robusta y sostenida, con el mejor resultado individual (44,9%, Random Forest) todavía por debajo del umbral, aun cuando estadísticamente más próximo que los resultados de Fase 1\.

La convergencia entre arquitecturas documentada en la sección 7.4, sumada a la variabilidad run-to-run observada en la sección 7.2, sugiere que continuar optimizando hiperparámetros de la arquitectura BiGRU actual presenta rendimientos marginales decrecientes, y que una mejora sustancial adicional requeriría modificar la naturaleza de las variables de entrada (por ejemplo, incorporando información de apariencia/textura no reducible a landmarks o blendshapes) o reconsiderar la definición del target de clasificación, cuestión que se desarrolla en la sección siguiente.

## 9\. Contraste con el requisito de exactitud mínima (70%) y la literatura de referencia

Durante el desarrollo de la Fase 2 se estableció, como requisito de la cátedra, una exactitud mínima del 70% para la detección de desatención. Puesto en relación con los resultados obtenidos (secciones 7 y 8\) y con la propia revisión de estado del arte incluida en la entrega del 25% del presente trabajo (sección 2.2.3), esta cifra requiere un análisis cuidadoso antes de fijarse como objetivo de re-modelado, por dos motivos concretos.

### 9.1. Posible confusión entre las etiquetas Engagement y Boredom

Los trabajos citados en la sección 2.2.3 del presente informe —incluyendo la arquitectura ViBED-Net (Gothwal *et al.*, 2025), que reporta una exactitud del 73,43% sobre DAiSEE, así como los antecedentes de menor desempeño reportados en la misma sección (aproximadamente 51%-67% según la arquitectura)— refieren en su mayoría a la clasificación de la etiqueta **Engagement** del dataset, y no de **Boredom**, que es la etiqueta efectivamente utilizada como base del target del presente trabajo (sección 2.3). Esta distinción es sustancial: tal como se documentó en la sección 2.1 (Tabla 2), la distribución de Engagement en DAiSEE presenta un desbalance severo, con el 95,2% de los clips de Train concentrados en los dos niveles superiores ("alto" y "muy alto"). Un desbalance de esta magnitud implica que un clasificador trivial de Engagement (que prediga siempre la clase mayoritaria) ya alcanzaría, por construcción, una exactitud superior al 90%, y que modelos con capacidad discriminativa moderada pueden reportar exactitudes en el rango 60%-75% sin que ello implique necesariamente una capacidad de discriminación fina entre niveles de compromiso/atención. Boredom, en cambio, presenta una distribución sustancialmente más equilibrada (clasificador trivial: 45,6%), lo que la convierte en una tarea de clasificación intrínsecamente más difícil, pero también más informativa en términos de la variabilidad real que discrimina.

En consecuencia, se considera probable que la cifra de 70% comunicada por la cátedra tenga como referencia implícita a trabajos que reportan sobre Engagement (o sobre variables de atención derivadas de esta con distribución igualmente desbalanceada), y no sobre Boredom en la formulación de 3 niveles adoptada en la sección 2.3 del presente trabajo.

### 9.2. Diferencias arquitectónicas respecto de los benchmarks de referencia

Independientemente de la etiqueta utilizada, corresponde señalar que los modelos que reportan los desempeños más altos en la literatura citada (EfficientNet B7 \+ LSTM/BiLSTM/TCN, 64,67%-67,48%; ViBED-Net, 73,43%) operan sobre **cuadros de video completos** procesados mediante redes convolucionales profundas (EfficientNet, EfficientNetV2), en el caso de ViBED-Net combinando además dos flujos de información —uno centrado en el rostro y otro en el contexto de la escena— seguidos de un Transformer y una LSTM. Este enfoque difiere sustancialmente del pipeline implementado en el presente trabajo (secciones 1.1 y 6.1), que reduce cada cuadro a un vector de landmarks/blendshapes de baja dimensionalidad (22-32 variables) y descarta la imagen inmediatamente después de procesarla, sin retener información de apariencia, textura o contexto de escena, en línea con el criterio de procesamiento local y minimización de datos adoptado como parte del diseño Edge AI del proyecto (sección 2.1.12.3).

Esta diferencia de enfoque implica que alcanzar niveles de exactitud comparables a los de ViBED-Net o EfficientNet+LSTM probablemente requeriría incorporar información de apariencia/textura (por ejemplo, mediante una red convolucional liviana aplicada cuadro a cuadro), lo cual constituye un cambio de arquitectura sustancialmente mayor al ajuste de hiperparámetros realizado hasta el momento, y que introduce una tensión a resolver con el objetivo de diseño de privacidad y procesamiento local que distingue al presente proyecto de los antecedentes relevados (Tabla comparativa, sección 2.1). Cabe notar que dicha tensión no es necesariamente irresoluble —una red convolucional liviana podría, en principio, procesarse localmente cuadro a cuadro sin almacenar video—, pero representa una decisión de diseño pendiente de discusión con la dirección del trabajo.

### 9.3. Reformulación propuesta: atención binaria

Como alternativa a evaluar, se plantea la posibilidad de reformular el target como una variable binaria de atención/desatención (equivalente a "1 − atención"), en lugar de los tres niveles derivados de Boredom utilizados hasta el momento. Una reformulación binaria elevaría el valor del clasificador trivial (al reducirse el número de clases, la clase mayoritaria concentra necesariamente una proporción mayor del total), lo cual podría acercar superficialmente los resultados al 70% requerido. Sin embargo, en base a la evidencia de techo de señal documentada en la sección 7.4 —consistente entre arquitecturas y estrategias de ponderación—, no es esperable que una simple reducción del número de clases resuelva por sí sola la limitación de fondo relacionada con la información disponible en las variables de entrada; el efecto principal de esta reformulación sería previsiblemente un desplazamiento hacia arriba tanto del clasificador trivial como del desempeño de los modelos entrenados, sin que necesariamente cambie la brecha relativa entre ambos. Se recomienda validar esta hipótesis empíricamente antes de adoptar la reformulación binaria como solución al requisito de 70%.

## 10\. Conclusiones de Fase 2 y próximos pasos

El trabajo de Fase 2 permitió mejorar de forma consistente el F1 macro del modelo respecto de la Fase 1, validando el valor de incorporar blendshapes y MAR al pipeline de extracción, e identificó con precisión —mediante un estudio de ablación controlado— que los blendshapes constituyen el aporte más significativo del enriquecimiento de features, mientras que los deltas curados no están claramente justificados. No obstante, el criterio original de éxito mínimo (exactitud superior al 45,6%) no fue superado de forma robusta por ningún modelo evaluado, y la convergencia de resultados entre arquitecturas sustancialmente distintas (lineal, árboles, recurrente bidireccional) sugiere que la limitación no responde a un problema de optimización sino a un techo de señal impuesto por las variables de entrada disponibles.

Se identifica como línea de trabajo prioritaria e inmediata la aclaración, junto con la dirección del trabajo, del alcance exacto del requisito de 70% comunicado por la cátedra —en particular, si dicho valor refiere a la etiqueta Engagement (severamente desbalanceada) o a una formulación equivalente a Boredom/atención (más equilibrada y, según lo documentado en la sección 9, intrínsecamente más difícil)—, dado que de esta definición depende de forma directa la factibilidad del objetivo y la necesidad —o no— de incorporar un cambio arquitectónico mayor (información de apariencia/textura mediante redes convolucionales) que podría entrar en tensión con el objetivo de diseño de procesamiento local y privacidad que distingue al presente proyecto de los antecedentes relevados.

Como líneas de trabajo futuro adicionales se identifican: (a) la validación empírica de la reformulación binaria de atención/desatención propuesta en la sección 9.3; (b) una evaluación de estabilidad run-to-run del modelo BiGRU (múltiples semillas de inicialización) para cuantificar la variabilidad documentada en la sección 7.2 antes de reportar cualquier resultado puntual como definitivo; y (c) la evaluación de un re-split agrupado por participante entre Train y Validation, dado el desalineamiento de distribución de clases documentado en la sección 2.3, como posible mitigación adicional independiente del cambio de arquitectura.

## Referencias a incorporar en la bibliografía

*Formato ISO 690-2010, consistente con el resto de la bibliografía del trabajo.*

DAS, Riju y DEV, Soumyabrata, 2025\. Optimizing student engagement detection using facial and behavioral features. *Neural Computing and Applications* \[En línea\]. Vol. 37, n.º 23, págs. 19063-19085 \[consultado 2026-07-07\]. Disponible en: DOI: [10.1007/s00521-025-11317-z](https://doi.org/10.1007/s00521-025-11317-z).

GOTHWAL, Prateek; BANERJEE, Deeptimaan y BISWAS, Ashis Kumer, 2025\. ViBED-Net: dual-stream video engagement/boredom-detection network combining face-aware and scene-aware representations. \[Citación completa a verificar y completar contra la bibliografía ya incorporada en la entrega del 25%, sección 2.2.3/2.2.6.1, y la lista de referencias del trabajo\].  
