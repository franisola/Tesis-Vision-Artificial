# Plan de implementación — Fase 3: hacia ≥70% de exactitud en detección de desatención (Sparkle V2)

**Restricción innegociable:** todo procesamiento visual ocurre en memoria local del dispositivo; la imagen se descarta inmediatamente tras la extracción de características. Nada en este plan la viola.

---

## 0. Diagnóstico ejecutivo

La evidencia acumulada en Fases 1–2 (convergencia de RegLog, RF y BiGRU en 40–45%, ablación con blendshapes como mejor aporte, variabilidad run-to-run) apunta a un **techo de señal de las features geométricas**, no a un déficit de modelado. La literatura lo confirma: ningún trabajo que use solo landmarks/pose supera ~50% en DAiSEE; los que superan 65–73% (ResNet+TCN 63,9%, TCN ordinal con features afectivas 67,4%, ViBED-Net 73,4%) usan **apariencia/textura vía CNN sobre el frame o el crop facial**.

Por lo tanto, la palanca principal es el **Eje 2 (apariencia)**; los Ejes 1, 3 y 4 son necesarios pero insuficientes por sí solos. Estimación de contribución a la brecha de ~25 puntos:

| Palanca | Ganancia esperada (acc.) | Costo edge | Riesgo |
| :--- | :--- | :--- | :--- |
| Embeddings CNN del crop facial (Eje 2) | +10 a +18 pts | Moderado (ver §2.4) | Medio |
| Target binario + protocolo correcto (Eje 1) | +8 a +10 pts *nominales* (desplazamiento de baseline) | Nulo | Bajo |
| Re-split por participante + multi-task + augmentación (Eje 4) | +2 a +5 pts | Nulo (solo entrenamiento) | Bajo |
| TCN / Transformer liviano (Eje 3) | +1 a +3 pts | Neutro o menor que BiGRU | Bajo |

Lectura honesta: **70% sobre Boredom en 3 niveles es improbable** (nadie lo reporta; el SOTA de 73,4% es Engagement, etiqueta con trivial >90%). **70% sobre una formulación binaria de atención/desatención con features de apariencia es alcanzable y defendible.** La recomendación es negociar esa definición con la cátedra (ya lo identifican en su §10) y ejecutar este plan para que el 70% se logre con brecha real sobre el trivial, no por construcción.

---

## 1. Eje 1 — Reformulación del target: análisis crítico

### 1.1. ¿La binarización acerca genuinamente al 70%?

Aritmética sobre sus propias tablas (Test): binarizando Boredom=0 → "Atento" vs Boredom≥1 → "Desatención", el trivial pasa de 45,6% a **54,4%** (891/1.638 clips en la clase Desatención). Efectos:

- **Desplazamiento superficial:** ~+9 pts del trivial son gratis. Si el modelo hoy pierde contra el trivial por −1 a −5 pts, la binarización sola lo dejaría en ~50–54%. **No alcanza el 70% por sí sola.** Su hipótesis de la sección 9.3 es correcta y debe validarse empíricamente (medio día de trabajo: re-mapear etiquetas y re-entrenar el BiGRU actual).
- **Ganancia genuina (no trivial):** la binarización elimina la frontera Boredom 1 vs 2/3, que es donde vive la mayor parte del ruido de anotación de DAiSEE (acuerdo inter-anotador bajo en niveles intermedios; las etiquetas son crowdsourced). Fusionar clases ruidosas mejora la relación señal/ruido del target, no solo la métrica. Esperable: +2 a +4 pts *reales* sobre la brecha contra el trivial.
- **Alineación con el producto:** Sparkle toma una decisión binaria (alertar / no alertar). El target de 3 niveles puede conservarse como salida ordinal secundaria para el mapa de calor, con la decisión binaria como métrica principal.

### 1.2. Recomendaciones concretas

1. **Target principal:** binario, corte Boredom {0} vs {1,2,3}. Evaluar también el corte {0,1} vs {2,3} (trivial Test ≈ 77% — descartarlo como métrica principal justamente por eso, pero documentar la comparación).
2. **Métricas de reporte:** *balanced accuracy* y F1 macro **junto con** accuracy, y siempre la brecha vs. trivial. Esto los protege metodológicamente de la acusación de "métrica desplazada" y es lo que un jurado técnico va a mirar.
3. **Cabeza ordinal auxiliar (CORAL/CORN):** en lugar de softmax de 3 clases, una salida ordinal de K−1 sigmoides explota que Boredom es ordinal. Costo nulo en inferencia; típicamente +1–2 pts en tareas ordinales con ruido. Permite derivar el binario y el de 3 niveles del mismo modelo.
4. **Suavizado de etiquetas** (label smoothing 0,1) como mitigación barata del ruido de anotación.

**Veredicto:** binarizar es correcto y necesario, pero es la palanca *de protocolo*; la palanca *de señal* es el Eje 2. Presentar ambas juntas: "binario + apariencia" es la combinación que la literatura respalda para cruzar 70%.

---

## 2. Eje 2 — Apariencia/textura con CNN ultra-liviana (palanca principal)

### 2.1. Principio de diseño compatible con Edge AI

El pipeline actual ya reduce cada frame a un vector (blendshapes) y descarta la imagen. La propuesta **no cambia ese contrato**: se agrega un segundo extractor que convierte el crop facial en un vector de embedding en memoria, e inmediatamente descarta el píxel. Un embedding CNN es, conceptualmente, un "blendshape aprendido" de mayor capacidad: no es imagen, no es reconstruible en la práctica si se aplican las mitigaciones de §2.5, y nunca se persiste.

Pipeline por frame muestreado:

```
frame (webcam, en RAM) → detección facial (ya la hace MediaPipe)
  → crop facial 112×112 ó 160×160 (alineado con los landmarks ya disponibles)
  → CNN liviana congelada → embedding 1280-d → proyección a 64–128-d
  → descarte inmediato del frame y del crop
  → buffer temporal de embeddings + 22 features geométricas
```

### 2.2. Elección del backbone

| Backbone | MFLOPs @112–160 | Tamaño int8 | Nota |
| :--- | :--- | :--- | :--- |
| **MobileNetV2 α=0,35 @112** | ~20 | ~1,7 MB | Recomendado para arrancar: soporte TFLite impecable, delegado XNNPACK |
| MobileNetV3-Small @160 | ~45 | ~2,5 MB | Mejor accuracy/FLOP; hard-swish ya soportado en TFLite |
| EfficientNet-Lite0 @160 | ~200 | ~4,5 MB | Solo si los dos anteriores se quedan cortos |

Punto crítico: **no usar pesos ImageNet a secas**. Un backbone preentrenado en rostros/afecto rinde mucho más con pocos datos:

- Opción A (recomendada): tomar MobileNetV2/V3 y hacer *fine-tuning* previo en un dataset de expresión facial abierto (AffectNet, FER+, RAF-DB) como tarea proxy de emociones. Eso produce un extractor de "textura afectiva" (ojeras, tensión, mirada apagada, sonrisa) que los blendshapes no capturan.
- Opción B: usar un modelo FER liviano ya publicado y quedarse con la penúltima capa.

### 2.3. Estrategia de muestreo temporal: la clave del costo

No hay que pasar 60 frames por la CNN. La textura facial cambia lento; la dinámica fina ya la capturan las 22 features geométricas a 6 fps. Esquema TSN (segmentos):

- **8–16 frames por clip de 10 s** (0,8–1,6 fps) para el flujo CNN.
- 60 frames para el flujo geométrico (sin cambios).
- Fusión: *upsampling* del embedding CNN por repetición/interpolación a 60 pasos y concatenación por frame, o arquitectura de dos ramas con fusión tardía (§3.3).

Costo resultante en producción: MobileNetV2 α0,35 @112 ≈ 20 MFLOPs × 1,6 fps ≈ **32 MFLOPs/s — despreciable** (un teléfono de gama media ejecuta >10 GFLOPs/s en CPU con XNNPACK; una notebook, mucho más). El costo real del sistema seguirá dominado por FaceLandmarker de MediaPipe, que ya corre hoy.

### 2.4. Estrategia de entrenamiento en dos etapas (crítica para no re-procesar video)

1. **Extracción única offline:** correr el pipeline (crop → CNN congelada → embedding) sobre DAiSEE una sola vez y persistir **solo los embeddings** (npz, igual que hoy con las features). Es exactamente el mismo contrato de privacidad que ya usan: se guarda el derivado, jamás el píxel. 8.416 clips × 16 frames ≈ 135k inferencias ≈ 1–2 h en una GPU modesta o una tarde en CPU.
2. **Entrenamiento del modelo temporal** sobre secuencias (embeddings + geométricas): iteración rápida, sin tocar video de nuevo.
3. (Opcional, si hay margen) *fine-tuning* del último bloque de la CNN con el modelo temporal, con lr bajo. Solo si la etapa 2 se estanca; requiere re-extraer.

### 2.5. Privacidad de los embeddings (para la defensa de tesis)

Anticipar la objeción "un embedding facial es dato biométrico": (a) el embedding se usa solo en memoria durante la sesión y se descarta tras la inferencia de la ventana; (b) proyección a baja dimensión (64–128-d) + cuantización int8 degradan fuertemente cualquier inversión; (c) el backbone se entrena para afecto, no para identidad (no es un embedding de reconocimiento facial tipo FaceNet); (d) nunca sale del dispositivo — al servidor solo viaja el evento de desatención con timestamp, igual que hoy.

---

## 3. Eje 3 — Evolución de la arquitectura temporal

### 3.1. TCN: el reemplazo recomendado

Razones, en orden de importancia para este proyecto:

1. **Compatibilidad TFLite:** ya sufrieron el error `TensorListReserve` de `Bidirectional(GRU)`. Una TCN es convolución 1D pura (Conv1D dilatada + residual): convierte a TFLite con `TFLITE_BUILTINS` sin trucos de batch fijo, cuantiza a int8 limpiamente y corre en XNNPACK. Elimina de raíz su cuello de botella de despliegue.
2. **Evidencia en DAiSEE:** ResNet+TCN (63,9%) y TCN ordinal con features afectivas (67,4%, Abedi & Khan) son precisamente los antecedentes más cercanos a su enfoque de "features por frame + modelo temporal".
3. Entrenamiento estable y paralelo (sin recurrencia), menos varianza run-to-run — problema que ya documentaron.

Configuración inicial sugerida: 3–4 bloques residuales, dilataciones {1,2,4,8} (campo receptivo ≈ 31–62 pasos ≈ la secuencia completa), 64 filtros, kernel 3, ~50–80k parámetros. Pooling final por **atención** (attention pooling) en lugar del último estado: pondera qué frames del clip importan, y los pesos de atención son interpretables para el mapa de calor de Sparkle.

### 3.2. Transformer liviano: segunda opción

2 bloques encoder, d_model 64, 4 cabezas, embedding posicional aprendido, secuencia 60: ~100–150k parámetros. TFLite ya soporta MHA vía ops estándar. Ventaja marginal esperada sobre TCN con solo ~4.850 clips de entrenamiento; los Transformers rinden con más datos. **Probarlo después de la TCN, no antes.** El BiGRU actual queda como baseline de comparación en la tesis.

### 3.3. Arquitectura de fusión propuesta (modelo objetivo de Fase 3)

```
Rama A: embeddings CNN (16×128)  → Conv1D ligera / upsample a 60
Rama B: geométricas (60×22)      → BatchNorm
Concat por frame (60×~150) → TCN (3 bloques, 64 filtros)
  → Attention pooling → Densa 32 ReLU + dropout
  → Cabeza 1: sigmoide binaria (atención/desatención)  ← métrica principal
  → Cabeza 2: ordinal CORAL 3 niveles                  ← mapa de calor
  → Cabezas auxiliares (solo entrenamiento): Engagement, Confusion, Frustration
```

Total estimado < 300k parámetros, < 1 MB int8 (+ backbone ~2 MB). Las cabezas auxiliares implementan *multi-task learning*: las 4 etiquetas de DAiSEE están correlacionadas y actúan como regularización (típicamente +1–3 pts en la tarea principal); se podan en la exportación TFLite.

---

## 4. Eje 4 — Estrategia de datos

### 4.1. Desalineamiento Train/Validation (el problema que ya midieron)

Su Tabla 4 muestra 42,5% de "Desatención marcada" en Validation vs 24,6% en Train. Con esa Validation, el early stopping y la selección de β optimizan contra una distribución que no es la de Test. Acciones, en orden:

1. **Re-split agrupado por participante:** fusionar Train+Validation oficiales (6.280 clips) y re-particionar con `StratifiedGroupKFold` (grupo = participante, estratificación = target). **Test oficial queda intacto** para comparabilidad con la literatura. Esto es estándar en la literatura de DAiSEE cuando se diagnostica el shift y es defendible en la tesis con la evidencia de su §2.3.
2. **Validación cruzada 3–5 folds por participante** para la selección de hiperparámetros, reportando media ± desvío. Ataca simultáneamente el shift y la variabilidad run-to-run de su §7.2.
3. Si se mantiene la partición oficial en algún experimento comparativo: selección de modelo por **balanced accuracy en Validation** (ya usan macro-F1, correcto) y *threshold tuning* de la sigmoide binaria sobre Validation re-balanceada por importancia (*importance weighting* por frecuencia de clase Train/Val).

### 4.2. Augmentación temporal (multiplica el dataset sin tocar imágenes)

Todas operan sobre las secuencias ya extraídas — costo cero en el pipeline edge:

- **Ventanas deslizantes:** de cada clip de 60 pasos, extraer sub-ventanas de 40–48 con solapamiento → 3–5× más muestras de entrenamiento. En inferencia, promediar predicciones de varias ventanas (TTA) — también aplicable en producción con ventanas de ~7 s.
- *Jitter* gaussiano leve sobre features (σ = 0,01–0,05 post-estandarización), *feature dropout* (enmascarar aleatoriamente un grupo de features por muestra), *time warping* suave, **mixup** sobre secuencias y etiquetas (α=0,2).
- **Sobremuestreo estratificado por clase** en el sampler (complementario al class weighting con β; con focal loss γ=1–2 como alternativa a comparar).

### 4.3. Datos externos (opcional, Fase 4+)

EngageNet (~11k clips, mayor escala y mejor balance) para pre-entrenar el modelo temporal, y/o test cruzado (entrenar DAiSEE → evaluar EngageNet) como evidencia de generalización para la tesis. No es prerequisito para el 70%.

---

## 5. Roadmap por fases

### Fase 3.0 — Protocolo y quick wins (1 semana)
- Binarizar target y re-entrenar el BiGRU actual tal cual → valida empíricamente la hipótesis de §9.3 de la tesis (resultado esperado: ~52–56%, brecha vs trivial sin cambios). Documentar como evidencia de "desplazamiento superficial".
- Implementar re-split agrupado por participante + CV por folds. Re-entrenar BiGRU binario sobre el nuevo split con ventanas deslizantes y label smoothing.
- **Criterio de salida:** brecha vs trivial > +3 pts de forma robusta (media de 5 semillas). Esperado: 57–61%.

### Fase 3.1 — TCN + multi-task + ordinal (1–2 semanas)
- Reemplazar BiGRU por TCN con attention pooling; cabezas binaria + CORAL + auxiliares; focal loss vs class weighting.
- Verificar conversión TFLite int8 desde el día 1 (debe ser trivial con Conv1D).
- **Criterio de salida:** ≥ 60–63% binario en Test oficial; TFLite int8 < 500 kB funcionando.

### Fase 3.2 — Rama de apariencia (3–4 semanas, la palanca grande)
- Semana 1: fine-tuning del backbone (MobileNetV2 α0,35) en AffectNet/FER+; congelar.
- Semana 2: extracción offline de embeddings sobre DAiSEE (16 frames/clip, crop desde landmarks ya disponibles).
- Semanas 3–4: entrenar el modelo de fusión (§3.3); ablación embeddings-solo vs geométricas-solo vs fusión.
- **Criterio de salida:** ≥ 68–72% binario en Test oficial con brecha > +12 pts vs trivial. Este es el experimento que decide el proyecto.

### Fase 3.3 — Optimización edge y validación final (1–2 semanas)
- Cuantización int8 post-entrenamiento (calibración con ~200 secuencias de Train); si cae > 1,5 pts, QAT.
- Benchmark de latencia/energía en hardware objetivo real (notebook gama media y/o Android): presupuesto ≤ 15% de un núcleo de CPU sostenido, memoria pico < 150 MB.
- Evaluación final: 5 semillas × bootstrap CI en Test oficial; matriz de confusión; análisis por participante (¿el error se concentra en pocos sujetos?).
- **Entregable:** modelo .tflite final + tabla comparativa Fase 1/2/3 para la tesis.

---

## 6. Cuellos de botella Edge AI previstos

1. **Dos grafos de MediaPipe + CNN + TCN concurrentes:** el detector facial ya consume el mayor presupuesto. Mitigación: la CNN corre a 1–2 fps (no 6), en el mismo hilo de post-proceso; TCN infiere 1 vez por ventana (cada 5–10 s), costo despreciable.
2. **Conversión TFLite:** resuelto por diseño al eliminar capas recurrentes bidireccionales (TCN = Conv1D). Regla de trabajo: **probar la conversión int8 en la semana 1 de cada arquitectura nueva**, nunca al final.
3. **Cuantización de la rama de fusión:** BatchNorm + concat de escalas heterogéneas puede degradar int8. Mitigación: estandarizar features geométricas antes de concatenar; calibración representativa; fallback a float16 (2× tamaño, aún < 3 MB total).
4. **Variabilidad de webcams reales** (15–30 fps, autoexposición, luz): el muestreo por índice uniforme sobre buffer temporal (no por timestamp fijo) y la augmentación con jitter mitigan el domain shift DAiSEE→producción. Prever una mini-validación con grabaciones propias de los autores (n=2) antes de la prueba con usuarios.
5. **Fragmentación térmica/CPU en notebooks de estudiantes:** medir con `tflite_benchmark_tool` en al menos 2 equipos reales; si hace falta, degradación elegante (bajar fps de la CNN antes que apagar el módulo).

---

## 7. Qué decir sobre el 70% (recomendación estratégica)

Presentar a la cátedra dos números, no uno: (a) exactitud binaria de atención/desatención sobre Boredom — objetivo ≥70%, alcanzable con este plan; (b) la brecha sobre el clasificador trivial y balanced accuracy — la evidencia de que el 70% es real y no un artefacto del desbalance, cosa que ningún benchmark de Engagement sobre DAiSEE puede decir (trivial > 90%). Eso convierte la restricción de Edge AI/privacidad de debilidad aparente en contribución diferencial de la tesis: *mismo orden de exactitud que enfoques full-frame, sin que un solo píxel salga del dispositivo*.

---

## Referencias del plan

- Gothwal, Banerjee & Biswas (2025). ViBED-Net. arXiv:2510.18016 — 73,43% (Engagement, 4 clases, EfficientNetV2 + LSTM/Transformer).
- Abedi & Khan (2021). ResNet + TCN híbrido para engagement en DAiSEE. arXiv:2104.10122 — 63,9%.
- Abedi & Khan (2023). Affect-driven ordinal engagement measurement (TCN ordinal). arXiv:2106.10882 — 67,4%.
- Gupta et al. (2016). DAiSEE. arXiv:1609.01885.
- Cao et al. CORAL/CORN: ordinal regression con redes neuronales.
- Das & Dev (2025). Neural Comput. & Applic. 37(23) — ya citado en la tesis.
