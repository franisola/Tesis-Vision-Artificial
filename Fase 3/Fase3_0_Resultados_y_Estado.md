# Fase 3.0 — Binarización + re-split por participante: estado y resultados

**Fecha:** 2026-07-23
**Alcance ejecutado en esta sesión:** primer escalón del roadmap de `Plan_Fase3_Deteccion_Desatencion_70pct.md` (Roadmap, Fase 3.0).

## Qué se hizo

1. **Verificación de datos existentes.** Se confirmó que `dataset_final/dataset_v2.npz` (Fase 2) contiene exactamente el target de 3 niveles usado en el diagnóstico del plan: el reparto Test (747 Atento / 519 leve / 372 marcada) reproduce el 891/1.638 = 54,4% citado en §1.1 al binarizar. También se confirmó que `id_train`/`id_val`/`id_test` codifican el participante en los primeros 6 dígitos del ClipID, sin overlap entre splits oficiales (63 participantes en Train, 19 en Validation, sin intersección).

2. **Paso A — binarización sobre el split oficial (smoke test CPU).** Se instaló TensorFlow-CPU en el entorno de trabajo (sin GPU disponible aquí) y se re-entrenó la arquitectura BiGRU actual con target binario (Atento vs Desatención), misma configuración de la Fase 2 (BiGRU 32 unidades, dropout 0,4, L2 1e-4, class weighting suavizado β=0,5). Por las limitaciones de cómputo del entorno (CPU, 2 núcleos, sin ejecución en segundo plano persistente entre llamadas), el entrenamiento se corrió en chunks manuales y se detuvo en la época 17/24 de una sola semilla (no convergido, no son las 5 semillas que pide el plan):

   | Métrica | Valor (parcial, 1 semilla, no convergido) |
   | :--- | :--- |
   | Accuracy Test | 0,583 |
   | F1 macro | 0,578 |
   | Balanced accuracy | 0,578 |
   | Brecha vs. trivial (0,544) | **+3,9 pts** |

   **Lectura:** direccionalmente consistente con la hipótesis del plan (~52–56% esperado, sin ganancia real grande) — incluso sin convergencia completa ya aparece una brecha positiva pequeña. **No es el resultado final**: falta correr las 5 semillas completas hasta early stopping real. Eso está listo para ejecutar en el notebook entregado (abajo), en GPU tarda minutos en vez de horas.

3. **Paso B — pipeline de re-split validado (sin entrenamiento completo).** Se implementó y probó el pipeline completo de re-split agrupado por participante: reconstrucción de secuencias crudas desde `features_v2`, tratamiento de NaN + deltas (32 features, igual que Fase 2), `StratifiedGroupKFold` (5 folds) agrupando por participante, y augmentación por ventanas deslizantes. Se verificó contra un subconjunto real de datos que:
   - No hay overlap de participantes entre train/val en ningún fold (los 5 folds dieron overlap = 0).
   - Las 32 columnas de features se reconstruyen correctamente y sin NaN residuales.
   - La augmentación por ventanas deslizantes multiplica el train por 8× (ventanas de 40/44/48 pasos con stride 8, zero-padding a 60 para ser compatibles con el `Masking` del modelo).

   El entrenamiento profundo de los 5 folds × 5 semillas no se corrió aquí por el mismo límite de cómputo (con datos completos, cada fold tarda del orden de minutos en CPU fragmentada en llamadas de ~40 s; en GPU es cuestión de minutos en total).

## Entregable principal: notebook listo para Colab/Kaggle GPU

**`Fase 3/Sparkle_V2_Fase3_0_Binarizacion_Resplit.ipynb`** — implementa el protocolo completo tal como lo pide el plan (Paso A: 5 semillas, 120 épocas, patience 15, split oficial; Paso B: re-split + CV 5 folds + ventanas deslizantes + label smoothing 0,1 + modelo final evaluado una única vez sobre Test oficial intacto). Solo requiere apuntar `PROJECT_DIR` a la carpeta de Drive/Kaggle donde ya están `features_v2` y `dataset_final` (mismo layout que los notebooks de Fase 2). Con GPU T4 el notebook completo (Paso A + Paso B) debería tardar en el orden de 15–30 minutos, no las horas que insumió el intento parcial en este entorno.

## Próximos pasos sugeridos

1. Correr `Sparkle_V2_Fase3_0_Binarizacion_Resplit.ipynb` en Colab/Kaggle (GPU) y volcar los CSV de resultados (`paso_a_resultados_5semillas.csv`, `paso_b_cv_5folds.csv`, `paso_b_final_5semillas.csv`) de vuelta a esta carpeta.
2. Verificar el criterio de salida de Fase 3.0: brecha vs. trivial > +3 pts robusta (media ± desvío de 5 semillas); esperado 57–61% de accuracy en Test oficial (Paso B).
3. Si se cumple, avanzar a **Fase 3.1** (reemplazo del BiGRU por TCN con attention pooling, cabezas binaria + CORAL ordinal + auxiliares, verificación de conversión TFLite desde el día 1) — puedo dejar preparado ese notebook a continuación.
4. Si no se cumple con claridad, barrer β de class weighting (0 / 0,25 / 0,5 / 0,75 / 1,0) y revisar la agresividad de la augmentación (ventanas más largas) antes de pasar a Fase 3.1.

## Archivos en esta carpeta

- `Sparkle_V2_Fase3_0_Binarizacion_Resplit.ipynb` — notebook completo, listo para GPU.
- `Sparkle_V2_Fase3_1_TCN_MultiTask.ipynb` — notebook de Fase 3.1 (ver sección siguiente).
- `bigru_binario_smoketest_seed0.keras` — checkpoint parcial del smoke test CPU (Paso A, 17 épocas, no convergido; solo como evidencia de que el pipeline corre de punta a punta).
- Este documento.

---

# Fase 3.1 — TCN + attention pooling + multi-task (binaria + CORAL-aprox + auxiliares)

**Alcance ejecutado en esta sesión:** construcción y validación completa del notebook `Sparkle_V2_Fase3_1_TCN_MultiTask.ipynb`. No se corrió el entrenamiento completo (5 semillas × 120 épocas) por el mismo límite de cómputo CPU del entorno; en cambio se validó exhaustivamente que el código es correcto antes de entregarlo, incluyendo un hallazgo de diseño importante.

## Hallazgo clave: CORAL "puro" rompe la conversión a TFLite

Se implementó primero la formulación CORAL estándar (Cao et al. 2020): un único logit con **peso compartido** entre los `K-1` umbrales + sesgos con orden garantizado (softplus + cumsum). Al intentar la conversión TFLite (con batch fijo, la misma mitigación que ya usan para evitar `TensorListReserve` en el BiGRU), la conversión **falla**: `'tfl.fully_connected' op num_input_elements / z_in != num_output_elements / z_out`. Se probaron tres variantes (broadcast implícito, `tf.tile` explícito, bias fijo vs. aprendido) y las tres fallan por la misma causa: la fusión de la capa `Dense(1)` compartida con el broadcast/tile posterior no es legalizable con batch fijo.

**Solución adoptada y verificada:** `Dense(K-1)` con pesos **independientes** por umbral (sin compartir peso) — convierte limpio a TFLite (float y con cuantización int8 sobre dataset representativo, ambas verificadas). Se pierde la garantía arquitectónica de monotonicidad estricta de CORAL puro; el loss ordinal (BCE sobre `y > k` para cada umbral) sigue empujando en la práctica hacia monotonicidad, pero para garantizarla en producción hay que aplicar `cummin` sobre las 2 probabilidades en el código de inferencia (fuera del grafo, trivial en costo). Esto se documentó en el propio notebook para que quede en la tesis como una decisión de diseño explícita, no un detalle perdido.

## Validación realizada (sin entrenamiento completo)

1. **Arquitectura (datos sintéticos):** TCN de 4 bloques residuales (dilataciones 1/2/4/8, 64 filtros, kernel 3) + attention pooling + 5 cabezas (binaria, ordinal, 3 auxiliares) — 99.472 parámetros totales (el tronco TCN solo, sin las cabezas, está en el rango ~85-90k que estima el plan). Un paso de entrenamiento con `GradientTape` no arrojó gradientes `None` en ninguna variable.
2. **Conversión TFLite "desde el día 1"** (tal como pide el plan): modelo de exportación (binaria + ordinal, sin cabezas auxiliares) convierte a **120 KB en float** y **129,5 KB cuantizado a int8** — ambos muy por debajo del umbral de 500 KB del criterio de salida. Nota: en este modelo tan chico, el int8 salió *más pesado* que el float (129,5 vs 120 KB) por el overhead de los parámetros de cuantización por tensor dominando sobre el ahorro; vale la pena revisar el tamaño real con el modelo entrenado sobre datos reales (los pesos sintéticos no son representativos de la distribución que vería la cuantización).
3. **Integración con datos reales (subset):** se corrió el pipeline completo (carga con labels auxiliares de Engagement/Confusion/Frustration, limpieza de NaN, deltas, `StandardScaler`, augmentación por ventanas con propagación de las 5 etiquetas, proyección del split `StratifiedGroupKFold` a los índices aumentados) contra un subconjunto real de `features_v2`. Se verificó explícitamente que **ningún participante del train aumentado aparece en validation** (0 overlap) — este es el punto donde más fácil se cuela un leakage al combinar re-split por grupo con augmentación por ventanas, y quedó comprobado que la implementación lo evita correctamente.

## Criterios de salida de Fase 3.1 (pendientes de correr en GPU)

| Métrica | Umbral del plan | Estado |
| :--- | :--- | :--- |
| Accuracy binaria Test oficial (media 5 semillas) | ≥ 60–63% | Pendiente — correr notebook en Colab/Kaggle |
| TFLite int8 | < 500 kB, funcionando | ✅ Verificado con arquitectura (120-130 KB); repetir con pesos entrenados reales |
| Delta de exactitud por cuantización | Idealmente < 1,5 pts | Pendiente — el notebook ya incluye la celda de verificación |

## Próximos pasos

1. Correr `Sparkle_V2_Fase3_1_TCN_MultiTask.ipynb` en Colab/Kaggle GPU (debería tardar bastante menos que el BiGRU porque la TCN paraleliza sobre toda la secuencia, a diferencia de la recurrencia).
2. Si se cumple el criterio de salida, avanzar a **Fase 3.2** (ver sección siguiente — ya está lista).
3. Si no se cumple con claridad, el plan sugiere probar el Transformer liviano (§3.2) antes de invertir en Fase 3.2, o revisar los pesos de las pérdidas auxiliares/ordinal.

---

# Fase 3.2 — Rama de apariencia (CNN sobre el crop facial): la palanca principal

**Alcance ejecutado en esta sesión:** construcción y validación arquitectónica de los dos notebooks de Fase 3.2. **A diferencia de 3.0 y 3.1, esta fase no se puede correr en el sandbox de trabajo en absoluto**: requiere el video crudo de DAiSEE (no solo las features ya extraídas), que nunca se persistió en este entorno por diseño de privacidad — ni siquiera para pruebas. Por eso el video tiene que descargarse en Colab/Kaggle (vía el mismo mirror de Kaggle que ya usan, `olgaparfenova/daisee`) y procesarse ahí.

## Qué se validó (arquitectura, con datos sintéticos — no con video real)

1. **Extractor CNN** (`MobileNetV2 α=0,35 @112`, sin top, pooling avg, proyección a 128-d): 574.176 parámetros, convierte a TFLite sin operadores Flex (672 KB en float sin cuantizar; con int8 y pesos reales entrenados debería bajar bastante más, en línea con el ~1,7 MB que estima el plan para este backbone). No se pudieron descargar los pesos preentrenados de ImageNet en este sandbox (el dominio `storage.googleapis.com` está bloqueado por el proxy de red), así que la validación fue solo estructural (pesos aleatorios); en Colab/Kaggle esto no es un problema.
2. **Modelo de fusión** (embeddings 16×128 + geométricas 60×32 → upsample/pad + concat + TCN + attention pooling + cabezas binaria/ordinal): 131.972 parámetros, un paso de entrenamiento real no arrojó gradientes `None`, y la conversión a TFLite con batch fijo funciona limpia (154,3 KB en float) — el `UpSampling1D`/`ZeroPadding1D`/`Concatenate` que fusiona las dos ramas no introdujo el mismo tipo de problema que el CORAL puro de Fase 3.1.
3. **Los 3 modos de la ablación** (`fusion`, `geom_only`, `emb_only`) se probaron con datos sintéticos empaquetados en `tf.data.Dataset` con número variable de inputs (1 o 2 tensores según el modo) — los tres entrenan sin errores de shape ni de grafo.

## Entregables

- **`Sparkle_V2_Fase3_2a_Extraccion_Embeddings_CNN.ipynb`** — extracción offline (corre una sola vez): descarga DAiSEE crudo, reutiliza el mismo `FaceLandmarker` de MediaPipe ya validado en Fase 2 para obtener el bounding box del rostro (sin agregar un segundo detector), recorta 112×112, opcionalmente hace fine-tuning de MobileNetV2 en FER2013 como tarea proxy de afecto (Opción A del plan, recomendada — con fallback a pesos ImageNet directos si se quiere iterar más rápido primero), y persiste solo los embeddings (16 frames/clip × 128-d), nunca el crop ni el frame.
- **`Sparkle_V2_Fase3_2b_Fusion_Apariencia_Geometria.ipynb`** — carga embeddings + geométricas alineadas por `ClipID`, re-split por participante (mismo criterio que 3.0/3.1), entrena y compara `geom_only` vs `emb_only` vs `fusion` (5 semillas cada uno), exporta el modelo de fusión a TFLite int8.

## Punto de atención para cuando se corra en Colab/Kaggle

- El fine-tuning en FER2013 requiere verificar que el slug de Kaggle (`msambare/fer2013` en el notebook) sigue vigente antes de lanzar la celda.
- El tiempo estimado de extracción offline es 1-2 h en GPU T4 para el dataset completo (bastante menos que la extracción de landmarks de Fase 2 porque son 16 frames/clip en vez de 60).
- Criterio de salida: ≥ 68-72% accuracy binaria en Test oficial con brecha > +12 pts vs. trivial, y que `fusion` supere claramente a `geom_only` — esa comparación es, según el plan, "el experimento que decide el proyecto".
