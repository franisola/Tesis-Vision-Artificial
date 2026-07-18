# Plan de Implementación — Fase 2: Modelado y Optimización

**Proyecto:** Sparkle V2 — Módulo de detección de desatención/fatiga en estudiantes
**Dataset:** DAiSEE (Gupta *et al.*, 2016) — mirror Kaggle `olgaparfenova/daisee`
**Entorno de ejecución:** Google Colab (GPU T4)
**Punto de partida:** Baseline LSTM Fase 1 → Test accuracy **36,9%** / F1 macro **0,347**, por debajo del clasificador trivial (**45,6%**).
**Objetivo de la fase:** Re-entrenar y optimizar el modelo con el set de features enriquecido (`features_v2`) hasta superar con claridad el baseline trivial y aproximarse al rango reportado en la literatura para DAiSEE (~47% accuracy; Das y Dev, 2025).

---

## 0. Estado real de los datos de entrada (leer antes de arrancar)

El pipeline `Sparkle_V2_Fase2_3_v2_MAR_Blendshapes.ipynb` ya generó, por clip, un archivo `.npz` en `features_v2/{Train,Validation,Test}/` con esta estructura:

```
sequence        → np.ndarray  shape (60, 22)   # 60 cuadros × 22 features
feature_names   → lista de 22 strings
boredom         → int 0..3    # etiqueta cruda
engagement, confusion, frustration → int 0..3
```

Las **22 features por cuadro** son:

| # | Grupo | Variables |
|---|---|---|
| 1–3 | Apertura ocular | `ear_avg`, `ear_left`, `ear_right` |
| 4–5 | Orientación de cabeza | `yaw`, `pitch` (grados, de la matriz de transformación facial) |
| 6 | Boca | `mar` (Mouth Aspect Ratio) |
| 7–8 | Parpadeo (blendshapes) | `eyeBlinkLeft`, `eyeBlinkRight` |
| 9–10 | Entrecerrar ojos | `eyeSquintLeft`, `eyeSquintRight` |
| 11–18 | Dirección de mirada (gaze) | `eyeLookInLeft/Right`, `eyeLookOutLeft/Right`, `eyeLookUpLeft/Right`, `eyeLookDownLeft/Right` |
| 19–21 | Cejas | `browDownLeft`, `browDownRight`, `browInnerUp` |
| 22 | Mandíbula | `jawOpen` |

> **Discrepancia importante a resolver en Fase 2.** El brief habla de "21 variables incluyendo deltas frame-a-frame", pero los `.npz` guardados contienen **22 features base y NO incluyen todavía los deltas** (velocidad de cambio). Los deltas se calculan en el **Sprint 1 (Consolidación)** a partir de estas 22 columnas. Es decir: la extracción entrega el nivel (posición) y la Fase 2 deriva la dinámica (velocidad). Esto es correcto metodológicamente —los deltas son una transformación determinística de las features base y no requieren re-procesar video— y conviene documentarlo así en la tesis.

**Target (definido en Fase 1, sección 2.3):** `Boredom` colapsado a 3 niveles.

| Nivel | Regla | Train | Validation | Test |
|---|---|---|---|---|
| 0 — Atento | Boredom = 0 | 45,0% | 31,2% | 45,6% |
| 1 — Desatención leve | Boredom = 1 | 30,5% | 26,3% | 31,7% |
| 2 — Desatención marcada | Boredom ∈ {2,3} | 24,6% | 42,5% | 22,7% |

> **Alerta de diseño (crítica para toda la fase):** la partición **Validation está desalineada** respecto de Train/Test (42,5% de "Desatención marcada" vs ~23%). Como DAiSEE se divide *por participante*, Validation no es una muestra representativa de Test. **Consecuencia práctica:** la selección de modelo (early stopping, elección de hiperparámetros) debe hacerse con una métrica robusta al desbalance (**macro-F1**), y el número reportado en la tesis es siempre el de **Test**. Se discute una mitigación opcional (re-split agrupado) en el Sprint 4.

---

## Estructura de la fase (sprints con compuertas de decisión)

| Sprint | Entregable | Compuerta para avanzar |
|---|---|---|
| S0 | Entorno + loader de `features_v2` a tensores `(N,60,F)` | Shapes y distribución de clases coinciden con Tabla 4 |
| S1 | Consolidación: NaN, deltas, normalización, `.npz` consolidado | 0% NaN residual; scaler ajustado solo en Train |
| S2 | Modelos (baseline no-temporal + LSTM/GRU ajustada) | Al menos un modelo supera 45,6% en Validation |
| S3 | Manejo de desbalance integrado al entrenamiento | Mejora de macro-F1 sin colapsar a clase mayoritaria |
| S4 | Evaluación final + criterios de éxito + ablation | Test accuracy > 45,6% con IC que no cruza el baseline |
| S5 | Export del modelo (opcional, Edge AI) | Modelo serializado y verificado |

---

## Sprint 0 — Setup del entorno y carga de datos

### 0.1 Runtime y dependencias

```python
# Colab: Runtime → Change runtime type → GPU (T4)
!pip install -q scikit-learn tensorflow imbalanced-learn

import os, glob, json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TF:", tf.__version__, "| GPU:", tf.config.list_physical_devices('GPU'))

from google.colab import drive
drive.mount('/content/drive')

PROJECT_DIR  = '/content/drive/MyDrive/SparkleV2'
FEATURES_DIR = f'{PROJECT_DIR}/features_v2'
OUT_DIR      = f'{PROJECT_DIR}/dataset_final'      # consolidado de la Fase 2
CKPT_DIR     = f'{PROJECT_DIR}/checkpoints'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
```

### 0.2 Loader: de `.npz` por clip a tensores por split

```python
N_FRAMES = 60

def collapse_boredom(b):
    # 0->Atento, 1->leve, {2,3}->marcada
    return 0 if b == 0 else (1 if b == 1 else 2)

def load_split(split):
    """Devuelve X (N,60,22), y (N,), ids (N,) y la lista de feature_names."""
    files = sorted(glob.glob(f'{FEATURES_DIR}/{split}/*.npz'))
    X, y, ids = [], [], []
    feat_names = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        seq = d['sequence'].astype(np.float32)           # (60, 22)
        if seq.shape != (N_FRAMES, 22):
            continue                                      # descarta clips mal formados
        if feat_names is None:
            feat_names = list(d['feature_names'])
        X.append(seq)
        y.append(collapse_boredom(int(d['boredom'])))
        ids.append(os.path.splitext(os.path.basename(f))[0])
    return np.stack(X), np.array(y, dtype=np.int64), np.array(ids), feat_names

X_train_raw, y_train, id_train, FEATS = load_split('Train')
X_val_raw,   y_val,   id_val,   _     = load_split('Validation')
X_test_raw,  y_test,  id_test,  _     = load_split('Test')

for name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    dist = np.bincount(y, minlength=3) / len(y)
    print(f'{name}: {len(y)} clips | dist {dist.round(3)}')
```

**Compuerta S0:** las proporciones impresas deben coincidir con la Tabla 4 (Train ≈ [0.45, 0.31, 0.25], Test ≈ [0.46, 0.32, 0.23], Val ≈ [0.31, 0.26, 0.43]). Si no coinciden, revisar el mapeo `collapse_boredom` o clips faltantes.

---

## Sprint 1 — Consolidación de datos

Tres sub-tareas en orden estricto: **(1)** tratar NaN → **(2)** derivar deltas → **(3)** normalizar. El orden importa: los deltas deben calcularse sobre secuencias ya interpoladas (sin NaN), y la normalización debe ajustarse *después* de tener el set final de columnas y *solo* con Train.

### 1.1 Tratamiento de valores faltantes (NaN)

**Origen de los NaN:** cuadros donde MediaPipe no detectó rostro (fila completa a NaN) o donde un denominador geométrico se anuló (EAR/MAR aislados). En Fase 1 el % de NaN fue marginal (~0,03% promedio por clip), pero blendshapes y gaze pueden introducir NaN adicionales cuando el rostro está muy oblicuo.

**Estrategia (coherente con la metodología de Fase 1, sección 1.2):**

1. Interpolación **lineal por columna, dentro de cada secuencia** (nunca entre clips): rellena gaps cortos preservando la dinámica temporal.
2. Relleno de bordes con `ffill`/`bfill` (la interpolación lineal no cubre NaN al inicio/fin de la secuencia).
3. Si tras esto queda alguna columna 100% NaN en un clip (feature nunca detectada), se rellena con la **mediana de esa feature en Train** (fallback robusto, calculado más abajo).
4. **Descarte de clips** con más de **30%** de cuadros con algún NaN (umbral de Fase 1). En v1 esto descartó solo 1 clip; se re-aplica por consistencia.

```python
def interpolate_sequence(seq):
    """seq: (60, F). Interpola por columna dentro de la secuencia."""
    df = pd.DataFrame(seq)
    df = df.interpolate(method='linear', axis=0, limit_direction='both')
    return df.values.astype(np.float32)

def clean_split(X, y, ids, nan_frame_thresh=0.30, train_medians=None):
    keep, X_clean = [], []
    for i in range(len(X)):
        frac_nan_frames = np.isnan(X[i]).any(axis=1).mean()
        if frac_nan_frames > nan_frame_thresh:
            continue                                  # descarta clip
        seq = interpolate_sequence(X[i])
        if np.isnan(seq).any() and train_medians is not None:
            idx = np.where(np.isnan(seq))
            seq[idx] = np.take(train_medians, idx[1]) # fallback: mediana de Train
        X_clean.append(seq); keep.append(i)
    keep = np.array(keep)
    return np.stack(X_clean), y[keep], ids[keep]

# Medianas por feature calculadas SOLO en Train (ignorando NaN)
train_medians = np.nanmedian(X_train_raw.reshape(-1, X_train_raw.shape[-1]), axis=0)

X_train, y_train, id_train = clean_split(X_train_raw, y_train, id_train, train_medians=train_medians)
X_val,   y_val,   id_val   = clean_split(X_val_raw,   y_val,   id_val,   train_medians=train_medians)
X_test,  y_test,  id_test  = clean_split(X_test_raw,  y_test,  id_test,  train_medians=train_medians)

assert not np.isnan(X_train).any(), "Quedan NaN en Train"
print("NaN residuales:", np.isnan(X_train).sum(), np.isnan(X_val).sum(), np.isnan(X_test).sum())
```

### 1.2 Derivación de deltas (velocidad de cambio → inquietud/fidgeting)

Los deltas son la primera derivada temporal `Δ_t = x_t − x_{t−1}`, con `Δ_0 = 0`. Capturan **movimiento** (parpadeos rápidos, sacádicos de mirada, cabeceo, apertura de boca) que el valor absoluto no expresa: un estudiante inquieto tiene EAR/gaze/pose con alta varianza de deltas aunque su promedio sea normal.

**Decisión de diseño — sobre qué columnas calcular deltas.** Calcular deltas sobre las 22 duplicaría la dimensionalidad a 44. Dos opciones:

- **Opción A (recomendada para arrancar): deltas sobre un subconjunto curado** de 10 señales dinámicamente informativas → 22 + 10 = **32 features**. Evita inflar la dimensión con derivadas ruidosas (p. ej., squint casi constante).
- **Opción B: deltas sobre las 22** → 44 features. Máxima información, mayor riesgo de sobreajuste; dejarla para el ablation del Sprint 4.

```python
# Subconjunto curado para deltas (Opción A): apertura ocular, pose, boca, gaze, parpadeo
DELTA_FEATURES = ['ear_avg', 'yaw', 'pitch', 'mar', 'jawOpen',
                  'eyeBlinkLeft', 'eyeBlinkRight',
                  'eyeLookInLeft', 'eyeLookOutLeft', 'browInnerUp']
delta_idx = [FEATS.index(f) for f in DELTA_FEATURES]

def add_deltas(X, idx):
    d = np.zeros_like(X[:, :, idx])
    d[:, 1:, :] = np.diff(X[:, :, idx], axis=1)     # Δ_t; Δ_0 = 0
    return np.concatenate([X, d], axis=-1)

X_train = add_deltas(X_train, delta_idx)
X_val   = add_deltas(X_val,   delta_idx)
X_test  = add_deltas(X_test,  delta_idx)

FEATS_FULL = FEATS + [f'd_{f}' for f in DELTA_FEATURES]
print("Dim final de features:", X_train.shape[-1], "->", FEATS_FULL)
```

### 1.3 Normalización

**Justificación:** las escalas son fuertemente heterogéneas — EAR/MAR en ~0,1–0,6, ángulos yaw/pitch en grados (~−90..90), blendshapes acotados en [0,1], y deltas centrados en 0 con colas. Una LSTM/GRU converge mal con features de rangos tan distintos. Se aplica **estandarización por feature** (`StandardScaler`), **ajustada exclusivamente sobre Train** para evitar fuga de información (idéntico criterio que Fase 1, sección 3.2).

Aunque los blendshapes ya están acotados en [0,1], se estandarizan igual para que todas las columnas entren a la red con media 0 y varianza 1 (uniformidad numérica). Se ajusta *después* de agregar deltas, para que el scaler cubra las 32 columnas.

```python
from sklearn.preprocessing import StandardScaler
import joblib

def fit_scaler(X_train):
    F = X_train.shape[-1]
    scaler = StandardScaler().fit(X_train.reshape(-1, F))   # (N*60, F)
    return scaler

def apply_scaler(X, scaler):
    N, T, F = X.shape
    return scaler.transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)

scaler = fit_scaler(X_train)
X_train = apply_scaler(X_train, scaler)
X_val   = apply_scaler(X_val,   scaler)
X_test  = apply_scaler(X_test,  scaler)

joblib.dump(scaler, f'{OUT_DIR}/scaler_v2.joblib')          # necesario para producción/Edge
```

### 1.4 Persistencia del dataset consolidado

```python
np.savez_compressed(
    f'{OUT_DIR}/dataset_v2.npz',
    X_train=X_train, y_train=y_train, id_train=id_train,
    X_val=X_val,     y_val=y_val,     id_val=id_val,
    X_test=X_test,   y_test=y_test,   id_test=id_test,
    feature_names=np.array(FEATS_FULL),
)
print("Consolidado guardado. Shapes:", X_train.shape, X_val.shape, X_test.shape)
```

**Compuerta S1:** 0 NaN residuales en los tres splits; scaler serializado; `dataset_v2.npz` re-cargable. A partir de aquí el modelado no vuelve a tocar los `.npz` por clip.

---

## Sprint 2 — Arquitectura del modelo

### 2.1 Diagnóstico y principio de diseño

El baseline de Fase 1 (2 capas LSTM de 64 y 32 unidades) falló por **falta de señal en las features**, no por la arquitectura — así lo confirmó el diagnóstico (LogReg y RF sobre features resumen también quedaron por debajo del trivial). Ahora que las features son más ricas (32 columnas, con gaze, blendshapes y dinámica), el principio rector cambia:

1. **Re-confirmar señal antes de modelar en profundidad.** Correr primero un baseline *no temporal* fuerte sobre las nuevas features. Si con las 22+deltas un modelo tabular ya supera 45,6%, se demuestra que el enriquecimiento resolvió el problema de raíz. Este paso es barato y da una cota inferior.
2. **Reducir capacidad, no aumentarla.** Con ~4.800 secuencias de entrenamiento, una LSTM de 64+32 (≈40k parámetros) sobreajusta fácil. Se propone **una sola capa recurrente pequeña** con regularización fuerte.
3. **Preferir GRU sobre LSTM en datos escasos:** menos parámetros (3 puertas vs 4), converge mejor y más rápido con datasets chicos, a igualdad de desempeño típico.

### 2.2 Paso previo obligatorio — baseline no temporal (re-diagnóstico)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

def summary_features(X):
    # media, std, min, max, y percentiles por feature -> vector por clip
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1),
                           np.percentile(X, 25, 1), np.percentile(X, 75, 1)], axis=1)

Xtr_s, Xva_s, Xte_s = map(summary_features, [X_train, X_val, X_test])

for name, clf in [('LogReg', LogisticRegression(max_iter=2000, class_weight='balanced')),
                  ('RandomForest', RandomForestClassifier(n_estimators=400, class_weight='balanced_subsample', random_state=SEED))]:
    clf.fit(Xtr_s, y_train)
    p = clf.predict(Xte_s)
    print(f'{name}: acc={accuracy_score(y_test, p):.3f} | f1_macro={f1_score(y_test, p, average="macro"):.3f}')
```

> Interpretación: en Fase 1 estos modelos daban acc ≈ 0,37–0,40 (bajo el trivial). Si ahora superan 0,456, el enriquecimiento de features funcionó y la parte temporal solo debe igualarlos o superarlos. Es el resultado más importante para la narrativa de la tesis.

### 2.3 Arquitectura recomendada — GRU compacta y regularizada

```python
def build_gru(input_shape, n_classes=3, units=32, dropout=0.4, l2=1e-4, lr=5e-4):
    reg = keras.regularizers.l2(l2)
    inp = keras.Input(shape=input_shape)                      # (60, F)
    # Masking es opcional: todos los clips tienen 60 cuadros (sin padding). Se puede
    # omitir; se incluye solo si más adelante se usa padding o time-masking (aug).
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Bidirectional(
            layers.GRU(units, dropout=dropout, recurrent_dropout=0.2,
                       kernel_regularizer=reg))(x)            # 1 capa BiGRU
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(lr, clipnorm=1.0),    # clip de gradiente
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

model = build_gru(X_train.shape[1:])
model.summary()
```

**Ajustes clave respecto del baseline de Fase 1:**

| Aspecto | Baseline v1 | Propuesta v2 | Motivo |
|---|---|---|---|
| Capas recurrentes | 2 (LSTM 64+32) | 1 (BiGRU 32) | Menos capacidad → menos sobreajuste |
| Celda | LSTM | GRU (bidireccional) | Mejor con pocos datos; ve la secuencia completa |
| Dropout | 0,3 | 0,4 + `recurrent_dropout` 0,2 | Regularización más fuerte |
| Regularización L2 | — | 1e-4 | Penaliza pesos grandes |
| Learning rate | (por defecto ~1e-3) | 5e-4 + `ReduceLROnPlateau` | Convergencia estable |
| Clip de gradiente | — | `clipnorm=1.0` | Evita explosión en recurrentes |
| Normalización interna | — | `BatchNormalization` | Estabiliza el entrenamiento |

### 2.4 Arquitecturas alternativas (para el ablation / si BiGRU no despega)

- **CNN1D + GRU (híbrido temporal).** Una o dos `Conv1D` extraen patrones locales de movimiento (especialmente de los canales delta) antes de la recurrente. Suele rendir bien cuando la señal está en micro-eventos (parpadeos, sacádicos):

```python
def build_cnn_gru(input_shape, n_classes=3, lr=5e-4):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, 5, padding='same', activation='relu')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.Bidirectional(layers.GRU(32, dropout=0.4, recurrent_dropout=0.2))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.0),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m
```

- **LSTM ajustada (1 capa, 32 unidades).** Igual que la BiGRU pero con celda LSTM; útil para aislar en el ablation si la mejora viene de la celda o de la reducción de capacidad, manteniendo comparabilidad con el baseline documentado.
- **Encoder Transformer ligero (stretch goal).** Un bloque `MultiHeadAttention` + pooling temporal. Solo si sobra tiempo y los anteriores plateauan; con ~4.800 muestras es propenso a sobreajustar y requiere más regularización/augmentation.

### 2.5 Data augmentation temporal (opcional, barato, alto impacto)

Con pocos datos, augmentar secuencias en Train mejora generalización sin re-extraer video: jitter gaussiano leve sobre features estandarizadas, *scaling* temporal (estirar/comprimir con re-muestreo), y *time masking* (poner a 0 una ventana corta, que la capa `Masking` ya tolera). Aplicar **solo a Train**, en un `tf.data.Dataset`.

---

## Sprint 3 — Manejo del desbalance de clases

El target de 3 niveles está moderadamente desbalanceado en Train (45 / 30,5 / 24,6). Fase 1 ya usó *class weighting* inverso a la frecuencia y aun así el modelo no aprendió; el ponderado no arregla features débiles, pero con las features nuevas sí es la palanca correcta para que la red no colapse en la clase mayoritaria. Se proponen tres estrategias, en orden de preferencia.

### 3.1 Estrategia recomendada — *class weighting* suavizado

El ponderado **inverso puro** (`w_c ∝ 1/f_c`) puede sobre-corregir: le da tanto peso a la clase minoritaria que el modelo sobre-predice desatención marcada y pierde precisión global. Un **exponente de suavizado β ∈ (0,1)** modera esa corrección:

```
w_c = (1 / f_c)^β ,  luego normalizado a media 1.
```

- `β = 1.0` → inverso puro (agresivo, el de Fase 1).
- `β = 0.5` → raíz cuadrada inversa (**recomendado**: buen equilibrio entre recall de minoritarias y accuracy global).
- `β = 0.0` → sin ponderación.

```python
from collections import Counter

def smoothed_class_weights(y, beta=0.5):
    counts = Counter(y)
    total = len(y)
    w = {c: (total / counts[c]) ** beta for c in counts}
    mean_w = np.mean(list(w.values()))
    return {c: w[c] / mean_w for c in w}          # normalizado a media 1

class_weight = smoothed_class_weights(y_train, beta=0.5)
print("Pesos (β=0.5):", {k: round(v, 3) for k, v in class_weight.items()})
# β es un hiperparámetro: barrer {0.0, 0.5, 0.75, 1.0} y elegir por macro-F1 en Validation.
```

Se pasa directamente a `model.fit(..., class_weight=class_weight)`.

### 3.2 Alternativa — *Focal Loss*

Cuando el ponderado por clase no basta, la **focal loss** re-pesa por *dificultad de ejemplo* (γ baja el peso de los ya bien clasificados), enfocando el aprendizaje en los casos difíciles. Útil si "Desatención leve" (la clase intermedia, la más confundible) sigue con recall bajo.

```python
# La focal loss "sparse" no está en todas las versiones de Keras. Dos rutas seguras:
# (a) paquete focal-loss:  !pip install focal-loss
from focal_loss import SparseCategoricalFocalLoss
loss = SparseCategoricalFocalLoss(gamma=2.0)
# (b) nativo Keras 3 (espera one-hot): keras.losses.CategoricalFocalCrossentropy(gamma=2.0)
#     -> requiere y en formato one-hot: keras.utils.to_categorical(y_train, 3)
# alternativa: combinar focal loss + alpha por clase = pesos suavizados de 3.1
```

### 3.3 Alternativa — *Resampling* (con cuidado de leakage)

- **Oversampling de secuencias minoritarias:** duplicar (con augmentation del Sprint 2.5, no copias idénticas) clips de "Desatención leve/marcada" **solo en Train**. Nunca tocar Validation ni Test.
- **SMOTE: desaconsejado aquí.** Interpola en espacio de features aplanado y rompe la coherencia temporal de la secuencia; si se usa, hacerlo sobre las features resumen del baseline tabular, no sobre los tensores `(60, F)`.

**Regla de oro:** cualquier resampling se ajusta y aplica **exclusivamente sobre Train**, después del split y del scaler, para no contaminar la evaluación.

### 3.4 El problema del Validation desalineado (mitigación)

Como Validation tiene 42,5% de "marcada" vs 22,7% en Test, un modelo seleccionado por *accuracy* en Validation optimiza una distribución equivocada. Mitigaciones, de menor a mayor intervención:

1. **(Por defecto)** Seleccionar por **macro-F1 en Validation** (robusto a la proporción de clases) y reportar en Test. Simple y defendible.
2. **Re-split agrupado por participante** uniendo Train+Validation y regenerando un split estratificado *por participante* (mismo criterio de DAiSEE, sin leakage) con `StratifiedGroupKFold`. Deja Test intacto. Documentar como decisión metodológica.
3. **Validación cruzada por participante (k-fold)** sobre Train+Val para estimaciones más estables de hiperparámetros; más costosa en Colab pero da intervalos de confianza sobre la selección de modelo.

> Recomendación: empezar con (1); si la brecha Validation↔Test genera selección inestable de hiperparámetros, pasar a (2). El `id_train/id_val/id_test` guardado en el consolidado permite recuperar el participante desde los primeros dígitos del ClipID para el re-split agrupado.

---

## Sprint 4 — Entrenamiento, métricas y criterios de éxito

### 4.1 Bucle de entrenamiento con selección por macro-F1

```python
class MacroF1(keras.callbacks.Callback):
    """Calcula macro-F1 en Validation al final de cada época (para early stopping)."""
    def __init__(self, val_data):
        super().__init__(); self.Xv, self.yv = val_data; self.best = -1
    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.Xv, verbose=0).argmax(1)
        logs['val_macro_f1'] = f1_score(self.yv, p, average='macro')

callbacks = [
    MacroF1((X_val, y_val)),
    keras.callbacks.EarlyStopping(monitor='val_macro_f1', mode='max',
                                  patience=15, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_macro_f1', mode='max',
                                      factor=0.5, patience=6, min_lr=1e-5),
    keras.callbacks.ModelCheckpoint(f'{CKPT_DIR}/bigru_v2_best.keras',
                                    monitor='val_macro_f1', mode='max', save_best_only=True),
]

model = build_gru(X_train.shape[1:])
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=120, batch_size=64,
    class_weight=class_weight,
    callbacks=callbacks, verbose=2)
```

### 4.2 Métricas reportadas (todas sobre Test)

No basta con *accuracy*: con clases desbalanceadas el trivial ya da 45,6%. Se reporta un panel completo:

- **Accuracy** — comparación directa con el trivial (45,6%) y el LSTM v1 (36,9%).
- **F1 macro** — métrica principal (peso igual a las 3 clases). Baseline a superar: **0,347**.
- **Balanced accuracy** y **Cohen's κ** — cuánto supera al azar considerando el desbalance.
- **Precision/Recall/F1 por clase** y **matriz de confusión** — para verificar que el modelo no colapsa en "Atento".

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, cohen_kappa_score)
import matplotlib.pyplot as plt

best = keras.models.load_model(f'{CKPT_DIR}/bigru_v2_best.keras')
p_test = best.predict(X_test, verbose=0).argmax(1)

TARGET_NAMES = ['Atento', 'Desatención leve', 'Desatención marcada']
print(classification_report(y_test, p_test, target_names=TARGET_NAMES, digits=3))
print("Balanced acc:", round(balanced_accuracy_score(y_test, p_test), 3))
print("Cohen kappa :", round(cohen_kappa_score(y_test, p_test), 3))

cm = confusion_matrix(y_test, p_test, normalize='true')
plt.imshow(cm); plt.colorbar()
plt.xticks(range(3), TARGET_NAMES, rotation=45); plt.yticks(range(3), TARGET_NAMES)
plt.title('Matriz de confusión (normalizada) — Test'); plt.tight_layout()
plt.savefig(f'{CKPT_DIR}/bigru_v2_confusion_matrix.png', dpi=120)
```

### 4.3 Robustez estadística — ¿supera *con claridad* el 45,6%?

Un modelo a 46% podría no ser distinguible del trivial por ruido de muestreo. Dos verificaciones:

**(a) Intervalo de confianza bootstrap sobre la accuracy de Test.** El criterio de éxito fuerte es que el **límite inferior del IC 95% quede por encima de 45,6%**.

```python
def bootstrap_ci(y_true, y_pred, metric=accuracy_score, n=2000, seed=SEED):
    rng = np.random.default_rng(seed); n_obs = len(y_true); scores = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        scores.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return np.mean(scores), lo, hi

m, lo, hi = bootstrap_ci(y_test, p_test)
print(f'Accuracy Test = {m:.3f}  IC95% [{lo:.3f}, {hi:.3f}]  (trivial = 0.456)')
print('¿Supera con claridad el trivial?', lo > 0.456)
```

**(b) Estabilidad entre semillas.** Re-entrenar la arquitectura con ≥5 semillas y reportar **media ± desvío** de accuracy y macro-F1. Evita conclusiones sobre un buen resultado por azar de inicialización.

```python
accs, f1s = [], []
for s in [0, 1, 2, 3, 4]:
    tf.random.set_seed(s); np.random.seed(s)
    m = build_gru(X_train.shape[1:])
    m.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=120, batch_size=64, class_weight=class_weight,
          callbacks=[MacroF1((X_val, y_val)),
                     keras.callbacks.EarlyStopping('val_macro_f1', mode='max',
                                                   patience=15, restore_best_weights=True)],
          verbose=0)
    pp = m.predict(X_test, verbose=0).argmax(1)
    accs.append(accuracy_score(y_test, pp)); f1s.append(f1_score(y_test, pp, average='macro'))
print(f'Accuracy Test: {np.mean(accs):.3f} ± {np.std(accs):.3f}')
print(f'F1 macro Test: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}')
```

### 4.4 Ablation — cuantificar la contribución de cada bloque de features

Directamente al servicio de la narrativa de la tesis (demostrar *por qué* v2 funciona donde v1 falló). Re-entrenar la misma arquitectura variando el set de entrada:

| Config | Features | Hipótesis |
|---|---|---|
| A — v1 | 5 (EAR, yaw, pitch) | Reproduce el fracaso (~37%) |
| B — +MAR | 6 | Aporte del bostezo |
| C — +blendshapes | 22 | Salto principal (gaze + AUs) |
| D — +deltas (curados) | 32 | Aporte de la dinámica/inquietud |
| E — +deltas (todas) | 44 | ¿Rinde o sobreajusta? |

Reportar accuracy y macro-F1 de cada uno en una tabla. Se espera que el salto grande ocurra en C (consistente con Das y Dev, 2025: los Action Units mejoran significativamente sobre landmarks+pose).

### 4.5 Criterios de éxito (definición de "hecho")

| Nivel | Criterio | Umbral |
|---|---|---|
| **Mínimo (informativo)** | Accuracy Test > trivial | > 45,6% |
| **Sólido** | Límite inferior IC95% de accuracy sobre el trivial **y** macro-F1 > baseline | LI > 0,456 y F1 > 0,347 |
| **Objetivo** | Alineado con literatura DAiSEE | Accuracy ≳ 50%, macro-F1 ≳ 0,45 |
| **Cualitativo** | Matriz de confusión sin colapso a "Atento"; recall > 0 en las 3 clases | — |

Referencias de contexto para la discusión: LSTM v1 = 36,9% / 0,347; trivial = 45,6%; Das y Dev (2025) con features enriquecidas + EfficientNet ≈ 47,2% en DAiSEE. Superar ~47% situaría el modelo en el estado del arte para esta tarea con este dataset.

---

## Sprint 5 — Export para Edge AI (opcional)

Coherente con el alcance de procesamiento local/cliente del proyecto:

```python
# TensorFlow Lite (móvil/edge)
conv = tf.lite.TFLiteConverter.from_keras_model(best)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
open(f'{CKPT_DIR}/bigru_v2.tflite', 'wb').write(conv.convert())
# Para navegador (extensión): tensorflowjs_converter --input_format=keras bigru_v2_best.keras web_model/
```

Empaquetar junto al modelo el `scaler_v2.joblib` (o sus `mean_`/`scale_` embebidos) y la lista `feature_names`: la inferencia en producción debe replicar exactamente el mismo pre-procesamiento (NaN → deltas → estandarización) que en entrenamiento.

---

## Resumen ejecutivo

La Fase 1 dejó demostrado que el cuello de botella eran las *features*, no el modelo. La Fase 2, por tanto, no reinventa la arquitectura: **(1)** consolida las 22 features nuevas tratando NaN por interpolación y derivando deltas de movimiento (→ 32 columnas), con estandarización ajustada solo en Train; **(2)** reduce la capacidad del modelo (BiGRU de 1 capa, regularizada) y valida primero con un baseline tabular que re-confirme la señal; **(3)** maneja el desbalance con *class weighting suavizado* (β=0,5) como palanca principal, con focal loss y oversampling como alternativas; **(4)** evalúa con macro-F1 como métrica primaria, selecciona sobre Validation por macro-F1 (dado su desbalance divergente) y exige que el IC95% bootstrap de la accuracy de Test supere el 45,6% del trivial, complementado con un ablation que aísla el aporte de cada bloque de features. El criterio de éxito objetivo es accuracy ≳ 50% y macro-F1 ≳ 0,45, en línea con el estado del arte reportado sobre DAiSEE.
