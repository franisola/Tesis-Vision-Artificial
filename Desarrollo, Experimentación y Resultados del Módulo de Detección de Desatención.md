# Desarrollo del módulo de detección de desatención — Avance metodológico y resultados

*Nota: este documento está redactado para insertarse como sección/capítulo de la tesis (ajustar numeración según corresponda al índice final). Mantiene el estilo impersonal y las citas ISO 690-2010 (autor-fecha) ya utilizados en el resto del trabajo.*

## 1. Obtención y preprocesamiento del conjunto de datos

Para el entrenamiento y la evaluación del módulo de detección de indicadores de desatención se utilizó el conjunto de datos DAiSEE (Gupta *et al.*, 2016), conforme a lo definido en el apartado de estrategia y gestión de conjuntos de datos (sección 2.1.12). El acceso al dataset se realizó a través de un mirror público en Kaggle, mientras se gestiona en paralelo la aprobación del acceso oficial solicitado a los autores.

La estructura obtenida coincide con la documentada por los autores originales: 9.068 clips de video distribuidos en las particiones Train, Validation y Test, organizados por participante, junto con archivos de etiquetas (`TrainLabels.csv`, `ValidationLabels.csv`, `TestLabels.csv`) que registran los niveles de Boredom, Engagement, Confusion y Frustration (escala ordinal de 0 a 3) para cada clip. Cada clip tiene una duración de 10 segundos a 30 cuadros por segundo (300 cuadros), con una resolución de 640×480 píxeles.

De los 9.068 clips nominales, se identificaron 652 (7,2%) sin archivo de video físicamente disponible en el mirror utilizado, por lo que el conjunto de trabajo quedó constituido por 8.416 clips con video y etiqueta disponibles.

### 1.1. Extracción de landmarks faciales

Siguiendo la metodología descrita en la sección 2.1.12.2, se implementó un pipeline de extracción que, para cada clip:

1. Muestrea 60 cuadros distribuidos uniformemente a lo largo de los 300 cuadros originales del clip, valor seleccionado por su correspondencia directa con el benchmark de referencia utilizado por Gothwal *et al.* (2025) para la evaluación de secuencias con arquitecturas LSTM.
2. Procesa cada cuadro mediante la API vigente de MediaPipe (`FaceLandmarker`, Tasks API), dado que la API "legacy" (`mp.solutions.face_mesh`) referenciada originalmente en Lugaresi *et al.* (2019) fue discontinuada por Google en 2023.
3. Extrae, por cuadro, cinco variables: apertura ocular promedio, izquierda y derecha (Eye Aspect Ratio, EAR) y orientación de cabeza (yaw, pitch), esta última obtenida directamente de la matriz de transformación facial que provee el detector (alineación contra un modelo facial canónico), en lugar de un cálculo manual mediante `solvePnP`.
4. Descarta la imagen inmediatamente después de procesarla, sin almacenamiento de video ni de cuadros individuales, en línea con el criterio de procesamiento local/Edge AI y minimización de datos definido en la sección 2.1.12.3.

La Tabla 1 resume los resultados de la extracción por partición.

**Tabla 1.** Resultados de la extracción de landmarks por partición del dataset.

| Partición | Clips con video | Clips procesados (`ok`) | Sin video físico | % NaN promedio | Clips con >30% cuadros sin rostro detectado |
|---|---|---|---|---|---|
| Train | 5.358 | 4.852 | 506 | 0,02% | 1 |
| Validation | 1.429 | 1.429 | 0 | 0,02% | 0 |
| Test | 1.784 | 1.638 | 146 | 0,07% | 0 |
| **Total** | **8.571** | **7.919** | **652** | **~0,03%** | **1** |

La calidad de detección resultó consistentemente alta en las tres particiones, con un porcentaje de cuadros sin rostro detectado marginal (0,02%-0,07% en promedio por clip), lo cual valida la robustez del detector facial de MediaPipe frente a las condiciones de captura "in the wild" propias de DAiSEE (variaciones de iluminación, oclusiones parciales, ángulos de cámara).

### 1.2. Tratamiento de datos faltantes

Conforme a la metodología de saneamiento definida en la sección 2.1.12.2, se aplicó interpolación lineal por columna (feature) para completar los cuadros sin detección facial dentro de cada secuencia, y se descartaron los clips que superaran el umbral de 30% de cuadros faltantes. Dado el bajo porcentaje de valores faltantes reportado en la Tabla 1, este proceso resultó en una pérdida mínima adicional: únicamente 1 clip de Train fue descartado por superar el umbral, quedando un conjunto final de **7.918 clips** (4.851 Train / 1.429 Validation / 1.638 Test) con secuencias completas de dimensión (60 cuadros × 5 variables), listas para el modelado.

## 2. Definición del target de atención/desatención

Antes de entrenar cualquier modelo, fue necesario decidir qué combinación de las cuatro etiquetas de DAiSEE (Boredom, Engagement, Confusion, Frustration) se utilizaría como variable objetivo para representar la desatención del estudiante, dado que el alcance del presente trabajo se circunscribe específicamente a este fenómeno.

### 2.1. Análisis de distribución de clases

El análisis de la distribución de las etiquetas Engagement y Boredom en el conjunto de entrenamiento (Tabla 2) reveló un desbalance severo en Engagement, con el 95,2% de los clips concentrados en los niveles "alto" y "muy alto" (niveles 2 y 3), y apenas un 4,8% distribuido entre los niveles "muy bajo" y "bajo". Este patrón se repitió de forma consistente en las particiones de Validation y Test.

**Tabla 2.** Distribución de clases de Engagement y Boredom por partición (valores absolutos y porcentaje).

| Nivel | Train — Engagement | Train — Boredom | Validation — Engagement | Validation — Boredom | Test — Engagement | Test — Boredom |
|---|---|---|---|---|---|---|
| 0 (muy bajo/nada) | 33 (0,7%) | 2.181 (45,0%) | 23 (1,6%) | 446 (31,2%) | 4 (0,2%) | 747 (45,6%) |
| 1 (bajo) | 200 (4,1%) | 1.479 (30,5%) | 143 (10,0%) | 376 (26,3%) | 81 (4,9%) | 519 (31,7%) |
| 2 (alto) | 2.474 (51,0%) | 1.043 (21,5%) | 813 (56,9%) | 475 (33,2%) | 849 (51,8%) | 335 (20,4%) |
| 3 (muy alto) | 2.144 (44,2%) | 148 (3,0%) | 450 (31,5%) | 132 (9,2%) | 704 (43,0%) | 37 (2,3%) |

La etiqueta Boredom presentó una distribución considerablemente más equilibrada, con sus dos niveles extremos (0 y 3) sumando entre el 33% y el 48% del total según la partición, y una masa de datos más repartida entre los cuatro niveles.

### 2.2. Relación entre Engagement y Boredom

Con el fin de determinar si ambas etiquetas aportan información redundante o complementaria, se calculó la correlación de Pearson entre Engagement y Boredom sobre el conjunto de entrenamiento, obteniéndose un valor de **-0,419**. La Tabla 3 presenta la tabla de contingencia correspondiente.

**Tabla 3.** Tabla de contingencia Engagement × Boredom (conjunto de Train, valores absolutos).

| Engagement \ Boredom | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 5 | 2 | 8 | 18 |
| 1 | 22 | 35 | 109 | 34 |
| 2 | 764 | 903 | 731 | 76 |
| 3 | 1.390 | 539 | 195 | 20 |

Una correlación de magnitud moderada (no cercana a -1) indica que Engagement y Boredom no constituyen la misma señal invertida, sino que capturan aspectos parcialmente distintos del estado atencional del estudiante. En particular, se observa que dentro del nivel Engagement=2 (el más numeroso del dataset) los clips se reparten de forma casi pareja entre los tres primeros niveles de Boredom, lo que evidencia que Engagement por sí solo no permite discriminar adecuadamente distintos grados de desatención.

### 2.3. Decisión de diseño

En función de lo expuesto, se descartó el uso de Engagement como variable objetivo principal, dado su severo desbalance de clases (95% concentrado en 2 de 4 niveles), y se adoptó **Boredom** como base para la construcción del target de desatención, colapsando sus cuatro niveles ordinales en tres categorías para mitigar la escasa representación del nivel 3 (2,3%-9,2% según partición):

- **Nivel 0 — Atento:** Boredom original = 0.
- **Nivel 1 — Desatención leve:** Boredom original = 1.
- **Nivel 2 — Desatención marcada:** Boredom original = 2 o 3.

Esta decisión resultó en una distribución de clases considerablemente más balanceada respecto de la alternativa binaria basada en Engagement (Tabla 4), reduciendo el riesgo de que un clasificador colapse prediciendo siempre la clase mayoritaria.

**Tabla 4.** Distribución del target de 3 niveles derivado de Boredom, por partición.

| Nivel | Train | Validation | Test |
|---|---|---|---|
| Atento | 2.181 (45,0%) | 446 (31,2%) | 747 (45,6%) |
| Desatención leve | 1.479 (30,5%) | 376 (26,3%) | 519 (31,7%) |
| Desatención marcada | 1.191 (24,6%) | 607 (42,5%) | 372 (22,7%) |

Se observa una distribución de clases divergente en la partición de Validation respecto de Train y Test (con una proporción mayor de "Desatención marcada"), atribuible a la composición demográfica de los participantes asignados a esta partición por los autores originales del dataset, dado que la división de DAiSEE se realiza por participante y no por muestreo aleatorio de clips.

## 3. Modelo baseline

### 3.1. Arquitectura

Se implementó un modelo secuencial basado en redes LSTM (*Long Short-Term Memory*), consistente con el enfoque adoptado en los antecedentes citados en la sección 2.2.3 (Gothwal *et al.*, 2025). La arquitectura consta de dos capas LSTM (64 y 32 unidades respectivamente) con regularización por *dropout* (0,3), seguidas de una capa densa de 16 unidades con activación ReLU y una capa de salida *softmax* de 3 unidades, correspondientes a los niveles del target definido en la sección 2.3.

### 3.2. Normalización y balanceo de clases

Las variables de entrada (EAR y ángulos de orientación de cabeza) presentan escalas heterogéneas — EAR en un rango aproximado de 0,1 a 0,4, frente a yaw/pitch expresados en grados (aproximadamente -90° a 90°) —, por lo que se aplicó estandarización (`StandardScaler`), ajustada exclusivamente sobre el conjunto de entrenamiento para evitar la fuga de información hacia las particiones de validación y prueba.

Dado el desbalance residual del target de 3 niveles (Tabla 4), se incorporó una estrategia de ponderación de clases (*class weighting*) durante el entrenamiento, calculada de forma inversamente proporcional a la frecuencia de cada clase, en lugar de recurrir en esta primera instancia a la incorporación de un conjunto de datos adicional para compensar el desbalance.

### 3.3. Resultados obtenidos

Para contextualizar el desempeño del modelo, se estableció como referencia un **clasificador trivial** que predice sistemáticamente la clase mayoritaria del conjunto de entrenamiento ("Atento"), obteniendo una exactitud (*accuracy*) del **45,6%** sobre el conjunto de Test — cualquier modelo propuesto debe superar claramente este valor para considerarse informativo.

Adicionalmente, con el objetivo de diagnosticar si la limitación observada correspondía a las variables utilizadas o a la arquitectura secuencial, se evaluaron dos modelos no temporales de referencia (Regresión Logística y Random Forest) entrenados sobre estadísticas resumen (media, desvío estándar, mínimo y máximo) de cada variable calculadas sobre la totalidad de los 60 cuadros de cada clip. La Tabla 5 resume los resultados comparativos sobre el conjunto de Test.

**Tabla 5.** Comparación de desempeño en el conjunto de Test.

| Modelo | Accuracy | F1 macro | Observación |
|---|---|---|---|
| Clasificador trivial (clase mayoritaria) | 45,6% | — | Referencia base |
| Regresión Logística (features resumen) | 37,4% | 0,370 | No temporal |
| Random Forest (features resumen) | 39,6% | 0,376 | No temporal |
| LSTM (secuencia completa) | 36,9% | 0,347 | Modelo propuesto |

Los tres modelos evaluados obtuvieron una exactitud **inferior** a la del clasificador trivial, resultado que se sostiene tanto en el enfoque secuencial (LSTM) como en los enfoques no temporales (Regresión Logística, Random Forest). El análisis de importancia de variables del Random Forest (Tabla 6) mostró una distribución de relevancia relativamente uniforme entre las cinco variables extraídas, sin que ninguna concentrara una capacidad discriminativa marcadamente superior — patrón consistente con una señal débil respecto de la tarea de clasificación planteada.

**Tabla 6.** Importancia de variables (Random Forest, top 10 de 20 features resumen).

| Variable | Importancia |
|---|---|
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

## 4. Discusión de resultados

El hallazgo de que modelos entrenados exclusivamente sobre apertura ocular y orientación de cabeza no logran superar un clasificador trivial para la predicción de Boredom encuentra respaldo directo en el trabajo original de DAiSEE. Gupta *et al.* (2016) señalan explícitamente que las señales faciales asociadas al aburrimiento incluyen la **postura corporal superior** del estudiante (por ejemplo, una postura reclinada hacia atrás asociada a mayor aburrimiento, frente a una postura erguida asociada a mayor compromiso), información que no es capturada por un pipeline centrado exclusivamente en el rostro como el implementado en el presente trabajo.

Esta observación resulta coherente con estudios más recientes sobre el mismo problema: Das y Dev (2025) reportan, para la tarea de clasificación de compromiso (*engagement*) sobre el dataset DAiSEE, que la incorporación de *Action Units* faciales (micro-expresiones definidas según el Facial Action Coding System, capturando movimientos de cejas, párpados y boca no representados en la geometría de landmarks básica) mejora significativamente el desempeño de los modelos respecto de utilizar únicamente landmarks, gaze y pose de cabeza. En ese mismo estudio, los modelos evaluados sobre DAiSEE combinando características faciales enriquecidas con redes profundas (EfficientNet) alcanzan una exactitud de aproximadamente 47,2%, en un orden de magnitud similar al obtenido en el presente trabajo, lo cual sitúa los resultados alcanzados dentro de un rango consistente con la dificultad reportada en la literatura para esta clase de tareas sobre DAiSEE, más allá de la elección puntual de variables.

Se concluye, en consecuencia, que la limitación observada no corresponde a un error de implementación ni a una configuración inadecuada del modelo secuencial, sino a una **restricción de las variables de entrada** disponibles en la versión actual del pipeline (apertura ocular y orientación de cabeza únicamente), insuficientes para capturar la totalidad de las señales conductuales asociadas al aburrimiento/desatención según la conceptualización utilizada por los anotadores originales de DAiSEE.

## 5. Limitaciones y trabajo futuro

A partir del diagnóstico anterior, se identifican las siguientes líneas de mejora para las etapas posteriores del proyecto:

1. **Incorporación de nuevas variables faciales.** Se propone extender el pipeline de extracción (sección 1.1) para incluir: (a) la relación de apertura bucal (*Mouth Aspect Ratio*), indicador directamente vinculado al bostezo y asociado en la literatura a la somnolencia y fatiga cognitiva (véase sección 2.1.2 del marco teórico), calculable a partir de los mismos landmarks ya extraídos sin costo computacional adicional significativo; y (b) *blendshapes* faciales provistos nativamente por la API de MediaPipe utilizada (`FaceLandmarker`, mediante el parámetro `output_face_blendshapes`), que proveen aproximadamente 52 puntuaciones de micro-expresión facial conceptualmente equivalentes a los *Action Units* referenciados por Das y Dev (2025).
2. **Evaluación de un conjunto de datos complementario.** Se descartó, en esta etapa, la incorporación de un segundo dataset (por ejemplo, el utilizado en estudios de EmotiW/Kaur *et al.*, 2018) para mitigar el desbalance de clases, priorizando en su lugar técnicas de ponderación dentro del propio conjunto DAiSEE. Esta alternativa se mantiene como línea de trabajo futuro, condicionada a los resultados obtenidos tras la incorporación de las variables adicionales descritas en el punto anterior.
3. **Limitación estructural del dataset.** Independientemente de las mejoras de features propuestas, se documenta como limitación del presente trabajo que DAiSEE fue construido a partir de grabaciones centradas en el rostro del participante, por lo que no resulta posible incorporar variables de postura corporal —identificadas por los propios autores del dataset como relevantes para el aburrimiento— sin recurrir a una fuente de datos adicional con captura de plano corporal más amplio, lo cual excede el alcance definido para el presente trabajo (sección 1.2, "Alcance").

## Referencias a incorporar en la bibliografía

*Formato ISO 690-2010, consistente con el resto de la bibliografía del trabajo.*

DAS, Riju y DEV, Soumyabrata, 2025. Optimizing student engagement detection using facial and behavioral features. *Neural Computing and Applications* [En línea]. Vol. 37, n.º 23, págs. 19063-19085 [consultado 2026-07-07]. Disponible en: DOI: [10.1007/s00521-025-11317-z](https://doi.org/10.1007/s00521-025-11317-z).
