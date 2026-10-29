#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Reconocimiento de Comandos de Voz mediante K-NN y Análisis PCA
Trabajo Final - Inteligencia Artificial I (Ingeniería en Mecatrónica - UNCUYO)

Funciones principales:
1. Preprocesamiento acústico (eliminación DC, filtro pasa-banda Butterworth,
   recorte adaptativo de silencios VAD, normalización RMS y pre-énfasis).
2. Extracción de vectores de características temporales-espectrales (MFCC + CMN,
   Delta-MFCC, Espectrograma Log-Mel, Contraste Espectral, ZCR, Centroide y
   Envolvente de Energía dividida en bloques fonéticos).
3. Entrenamiento y evaluación estadística del clasificador K-NN (Validación
   Cruzada Estratificada 5-Fold, Leave-One-Person-Out, Matriz de Confusión,
   curva de exactitud según K) y proyección por Componentes Principales (PCA 2D y 3D).
4. Reconocimiento de comandos en vivo por micrófono («Reconocer», «Identificar»,
   «Salir») exportable en una sola función para el Agente final.
5. Generación de mosaicos comparativos de etapas de preprocesamiento y extracción
   acústica (análogo al mosaico de huellas dactilares para el informe final).
"""

import argparse
import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-ia1"

import joblib
import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler

import grabador_voz as gv

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# CONFIGURACIÓN PREDETERMINADA
# ==============================================================================

DIRECTORIO_DATASET_DEFAULT = gv.DIRECTORIO_DATASET_DEFAULT
DIRECTORIO_RESULTADOS_DEFAULT = "resultados_voz"
CARPETA_COMPARATIVAS_VOZ = os.path.join(
    DIRECTORIO_RESULTADOS_DEFAULT, "comparativas_etapas"
)
RUTA_MODELO_CACHE = os.path.join(
    DIRECTORIO_RESULTADOS_DEFAULT, "modelo_knn_voz.joblib"
)
RUTA_GRAFICO_ANALISIS = os.path.join(
    DIRECTORIO_RESULTADOS_DEFAULT, "analisis_knn_pca_voz.png"
)
RUTA_GRAFICO_TEST_VIVO = os.path.join(
    DIRECTORIO_RESULTADOS_DEFAULT, "ultima_prueba_microfono.png"
)

COMANDOS_ORDEN: Tuple[str, ...] = ("Reconocer", "Identificar", "Salir")
COLORES_COMANDOS: Dict[str, str] = {
    "Reconocer": "#1f77b4",    # Azul
    "Identificar": "#2ca02c",  # Verde
    "Salir": "#d62728",        # Rojo
}
MARCADORES_COMANDOS: Dict[str, str] = {
    "Reconocer": "o",
    "Identificar": "s",
    "Salir": "^",
}

# Hiperparámetros óptimos del clasificador y extractor acústico
K_VECINOS_DEFAULT = 5
N_SEGMENTOS_TEMPORALES = 6
ALPHA_PONDERACION_ANOVA = 0.20
UMBRAL_CONFIANZA_DEFAULT = 0.50
UMBRAL_PICO_SILENCIO = 0.015


# ==============================================================================
# 1. PREPROCESAMIENTO ACÚSTICO Y EXTRACCIÓN DE CARACTERÍSTICAS
# ==============================================================================

def _construir_filtro_pasabanda(
    sr: int = gv.SAMPLE_RATE_DEFAULT,
    low_hz: float = 120.0,
    high_hz: float = 7200.0,
    orden: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diseña un filtro Butterworth pasa-banda para eliminar zumbido de red y ruido ultrasónico."""
    nyq = 0.5 * sr
    low = max(20.0, low_hz) / nyq
    high = min(nyq - 100.0, high_hz) / nyq
    return butter(orden, [low, high], btype="band")


def preprocesar_senal_voz(
    senal: np.ndarray, sr: int = gv.SAMPLE_RATE_DEFAULT
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Acondiciona la señal cruda de voz antes de extraer características:
      1. Eliminación de componente continua (DC offset).
      2. Filtrado pasa-banda Butterworth (120 Hz - 7200 Hz).
      3. Detección de actividad de voz (VAD adaptativo por envolvente RMS) para
         recortar silencios al inicio y al final de la grabación.
      4. Normalización de energía RMS sobre el segmento activo de voz.

    Retorna:
        (senal_activa_norm, rms_envolvente_completa, umbral_vad, (indice_ini, indice_fin))
    """
    s = np.asarray(senal, dtype=np.float32).reshape(-1)
    if s.size == 0:
        return np.zeros(int(0.5 * sr), dtype=np.float32), np.zeros(10), 0.0, (0, 0)

    # 1. Quitar offset DC
    s = s - float(np.mean(s))

    # 2. Filtrado pasa-banda (120 Hz a 7200 Hz)
    b_bp, a_bp = _construir_filtro_pasabanda(sr=sr)
    if len(s) > 32:
        s_filtrada = filtfilt(b_bp, a_bp, s).astype(np.float32)
    else:
        s_filtrada = s

    # 3. Recorte adaptativo de silencio mediante envolvente RMS (ventanas de 25 ms, salto 10 ms)
    frame_len = int(0.025 * sr)  # 400 muestras @ 16 kHz
    hop_len = int(0.010 * sr)    # 160 muestras @ 16 kHz
    rms_env = librosa.feature.rms(
        y=s_filtrada, frame_length=frame_len, hop_length=hop_len
    )[0]

    piso_ruido = float(np.percentile(rms_env, 15))
    pico_max = float(np.max(rms_env))
    umbral_vad = piso_ruido + 0.18 * (pico_max - piso_ruido)

    frames_activos = np.where(rms_env >= umbral_vad)[0]
    if len(frames_activos) >= 4:
        ini_frame = max(0, int(frames_activos[0]) - 2)
        fin_frame = min(len(rms_env), int(frames_activos[-1]) + 3)
        muestra_ini = ini_frame * hop_len
        muestra_fin = min(len(s_filtrada), fin_frame * hop_len)
        s_voz = s_filtrada[muestra_ini:muestra_fin]
    else:
        muestra_ini, muestra_fin = 0, len(s_filtrada)
        s_voz = s_filtrada

    # Garantizar longitud mínima de 150 ms
    if len(s_voz) < int(0.15 * sr):
        muestra_ini, muestra_fin = 0, len(s_filtrada)
        s_voz = s_filtrada

    # 4. Normalización por valor eficaz (RMS = 1.0)
    rms_voz = float(np.sqrt(np.mean(np.square(s_voz)))) + 1e-8
    s_voz_norm = (s_voz / rms_voz).astype(np.float32)

    return s_voz_norm, rms_env, umbral_vad, (muestra_ini, muestra_fin)


def extraer_caracteristicas_voz(
    senal: np.ndarray,
    sr: int = gv.SAMPLE_RATE_DEFAULT,
    n_seg: int = N_SEGMENTOS_TEMPORALES,
) -> np.ndarray:
    """
    Extrae un vector discriminante de características acústicas y fonético-temporales:
      - Descriptores de duración activa, relación señal/silencio y perfil de envolvente
        silábica en 10 puntos equiespaciados.
      - Pre-énfasis de alta frecuencia (coef = 0.97).
      - División de la palabra recortada en `n_seg` bloques secuenciales (capturando la
        trayectoria fonética desde la sílaba inicial hasta la final).
      - En cada bloque temporal extrae:
          * 13 MFCC con Normalización de Media Cepstral (CMN, invariante al micrófono/locutor)
          * 13 MFCC absolutos
          * 13 derivadas de primer orden (Delta-MFCC)
          * 16 bandas de espectrograma Log-Mel normalizadas
          * 6 bandas de contraste espectral
          * Tasa de cruces por cero (ZCR, clave para fricativas como 'S' en Salir/Reconocer)
          * Centroide espectral normalizado
    """
    s_voz, rms_full, umbral_vad, _ = preprocesar_senal_voz(senal, sr=sr)

    duracion_activa = len(s_voz) / float(sr)
    fraccion_activa = float(np.mean(rms_full >= umbral_vad))

    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    n_fft = 512

    rms_voz_env = librosa.feature.rms(
        y=s_voz, frame_length=frame_len, hop_length=hop_len
    )[0]
    rms_norm = rms_voz_env / (float(np.max(rms_voz_env)) + 1e-8)

    # Pre-énfasis para realzar formantes superiores y consonantes
    s_pre = librosa.effects.preemphasis(s_voz, coef=0.97)

    # MFCC + CMN + Delta
    mfcc = librosa.feature.mfcc(
        y=s_pre,
        sr=sr,
        n_mfcc=13,
        n_mels=32,
        n_fft=n_fft,
        hop_length=hop_len,
        fmin=120,
        fmax=7200,
    )
    mfcc_cmn = mfcc - np.mean(mfcc, axis=1, keepdims=True)
    delta_mfcc = librosa.feature.delta(mfcc_cmn)

    # Espectrograma Log-Mel
    mel = librosa.feature.melspectrogram(
        y=s_pre,
        sr=sr,
        n_mels=16,
        n_fft=n_fft,
        hop_length=hop_len,
        fmin=120,
        fmax=7200,
    )
    log_mel = librosa.power_to_db(mel + 1e-9)
    log_mel_cmn = log_mel - np.mean(log_mel, axis=1, keepdims=True)

    # ZCR, Centroide Espectral y Contraste Espectral
    zcr = librosa.feature.zero_crossing_rate(
        y=s_voz, frame_length=frame_len, hop_length=hop_len
    )
    cent = librosa.feature.spectral_centroid(
        y=s_voz, sr=sr, n_fft=n_fft, hop_length=hop_len
    ) / (sr / 2.0)
    contrast = librosa.feature.spectral_contrast(
        y=s_voz, sr=sr, n_fft=n_fft, hop_length=hop_len, fmin=150, n_bands=5
    )

    # Ensamblar vector global + bloques temporales
    num_frames = mfcc.shape[1]
    bloques = np.array_split(np.arange(num_frames), n_seg)

    vector: List[float] = [
        duracion_activa,
        fraccion_activa,
        float(np.std(rms_norm)),
        float(np.mean(rms_norm)),
    ]

    # Perfil de energía silábica interpolado a 10 puntos
    env_10 = np.interp(
        np.linspace(0.0, 1.0, 10),
        np.linspace(0.0, 1.0, len(rms_norm)),
        rms_norm,
    )
    vector.extend(float(v) for v in env_10)

    for b in bloques:
        if len(b) == 0:
            b = np.arange(num_frames)
        vector.extend(float(v) for v in np.mean(mfcc_cmn[:, b], axis=1))
        vector.extend(float(v) for v in np.mean(mfcc[:, b], axis=1))
        vector.extend(float(v) for v in np.mean(delta_mfcc[:, b], axis=1))
        vector.extend(float(v) for v in np.mean(log_mel_cmn[:, b], axis=1))
        vector.extend(float(v) for v in np.mean(contrast[:, b], axis=1))
        vector.append(float(np.mean(zcr[:, b])))
        vector.append(float(np.mean(cent[:, b])))

    return np.asarray(vector, dtype=np.float32)


# ==============================================================================
# 2. TRANSFORMADOR SUPERVISADO Y PIPELINE K-NN + PCA
# ==============================================================================

class PonderadorANOVA(BaseEstimator, TransformerMixin):
    """
    Pondera cada característica estandarizada según su poder discriminante entre
    los comandos de voz usando el estadístico F de ANOVA (calculado exclusivamente
    sobre el conjunto de entrenamiento):
        w_j = (F_j / max(F))^alpha
    Esto atenúa las características que varían entre locutores y realza las que
    distinguen fonéticamente «Reconocer», «Identificar» y «Salir».
    """

    def __init__(self, alpha: float = ALPHA_PONDERACION_ANOVA):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PonderadorANOVA":
        f_vals, _ = f_classif(X, y)
        f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
        pesos = np.power(np.maximum(f_vals, 0.0), self.alpha)
        max_p = float(np.max(pesos))
        self.pesos_ = pesos / max_p if max_p > 0 else np.ones_like(pesos)
        self.f_vals_ = f_vals
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.pesos_


def construir_pipeline_knn(
    k_vecinos: int = K_VECINOS_DEFAULT,
    alpha: float = ALPHA_PONDERACION_ANOVA,
) -> Pipeline:
    """Construye el pipeline completo de escalado, ponderación ANOVA, normalización L2 y K-NN."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ponderador", PonderadorANOVA(alpha=alpha)),
            ("normalizer", Normalizer(norm="l2")),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k_vecinos,
                    weights="distance",
                    metric="euclidean",
                ),
            ),
        ]
    )


def extraer_matriz_dataset(
    directorio_base: str = DIRECTORIO_DATASET_DEFAULT,
    mostrar_progreso: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, str]]]:
    """
    Carga todos los audios registrados en `metadata.csv` y calcula su matriz `X`
    de características acústicas.
    Retorna `(X, y_comandos, grupos_personas, metadatos)`.
    """
    audios, etiquetas, personas, metas = gv.cargar_dataset_desde_csv(
        directorio_base=directorio_base, normalizar_float=True
    )
    if not audios:
        raise RuntimeError(
            f"No se encontraron grabaciones válidas en '{directorio_base}/metadata.csv'."
        )

    if mostrar_progreso:
        print(
            f"[*] Extrayendo características acústicas (MFCC + Delta + Log-Mel + ZCR) "
            f"de {len(audios)} grabaciones..."
        )

    vectores = []
    for idx, senal in enumerate(audios):
        sr_fila = int(metas[idx].get("frecuencia_muestreo", gv.SAMPLE_RATE_DEFAULT))
        vec = extraer_caracteristicas_voz(senal, sr=sr_fila)
        vectores.append(vec)

    X = np.vstack(vectores).astype(np.float32)
    y = np.asarray(etiquetas)
    grupos = np.asarray(personas)
    return X, y, grupos, metas


def entrenar_modelo_completo(
    directorio_base: str = DIRECTORIO_DATASET_DEFAULT,
    k_vecinos: int = K_VECINOS_DEFAULT,
    guardar_cache: bool = True,
) -> Dict[str, object]:
    """
    Entrena el clasificador K-NN sobre todo el dataset disponible y ajusta también
    la proyección PCA (2D y 3D) para visualización y análisis.
    Guarda el modelo en `resultados_voz/modelo_knn_voz.joblib`.
    """
    X, y, personas, metas = extraer_matriz_dataset(
        directorio_base=directorio_base, mostrar_progreso=True
    )

    pipeline_knn = construir_pipeline_knn(k_vecinos=k_vecinos)
    pipeline_knn.fit(X, y)

    # Ajustar transformación PCA sobre las características más discriminantes
    # para visualización gráfica clara en 2D y 3D
    scaler_pca = StandardScaler()
    X_std = scaler_pca.fit_transform(X)
    selector_pca = SelectKBest(f_classif, k=min(35, X.shape[1]))
    X_sel = selector_pca.fit_transform(X_std, y)
    ponderador_pca = PonderadorANOVA(alpha=0.60)
    X_pond = ponderador_pca.fit_transform(X_sel, y)
    norm_pca = Normalizer(norm="l2")
    X_pre_pca = norm_pca.fit_transform(X_pond)

    n_comp_max = min(15, X.shape[0], X_pre_pca.shape[1])
    pca_modelo = PCA(n_components=n_comp_max, random_state=42)
    X_pca = pca_modelo.fit_transform(X_pre_pca)

    # Clasificador K-NN auxiliar en el plano 2D (PC1, PC2) para dibujar regiones de decisión
    knn_2d = KNeighborsClassifier(
        n_neighbors=k_vecinos, weights="distance", metric="euclidean"
    )
    knn_2d.fit(X_pca[:, :2], y)

    paquete_modelo: Dict[str, object] = {
        "pipeline_knn": pipeline_knn,
        "X": X,
        "y": y,
        "personas": personas,
        "metas": metas,
        "k_vecinos": k_vecinos,
        "scaler_pca": scaler_pca,
        "selector_pca": selector_pca,
        "ponderador_pca": ponderador_pca,
        "norm_pca": norm_pca,
        "pca_modelo": pca_modelo,
        "X_pca": X_pca,
        "knn_2d": knn_2d,
        "num_audios_entrenamiento": len(y),
    }

    if guardar_cache:
        os.makedirs(DIRECTORIO_RESULTADOS_DEFAULT, exist_ok=True)
        joblib.dump(paquete_modelo, RUTA_MODELO_CACHE)

    return paquete_modelo


def proyectar_vector_en_pca(
    paquete_modelo: Dict[str, object], vector_x: np.ndarray
) -> np.ndarray:
    """Proyecta un nuevo vector de características en el espacio PCA del modelo."""
    x_2d = np.asarray(vector_x, dtype=np.float32).reshape(1, -1)
    x_std = paquete_modelo["scaler_pca"].transform(x_2d)
    x_sel = paquete_modelo["selector_pca"].transform(x_std)
    x_pond = paquete_modelo["ponderador_pca"].transform(x_sel)
    x_norm = paquete_modelo["norm_pca"].transform(x_pond)
    return paquete_modelo["pca_modelo"].transform(x_norm)[0]


def entrenar_o_cargar_modelo_voz(
    directorio_base: str = DIRECTORIO_DATASET_DEFAULT,
    k_vecinos: int = K_VECINOS_DEFAULT,
    forzar_reentrenamiento: bool = False,
) -> Dict[str, object]:
    """
    Carga el modelo K-NN desde caché si existe y coincide con la cantidad actual de
    audios en `metadata.csv`; de lo contrario lo entrena automáticamente.
    Ideal para llamar al inicio del Agente final.
    """
    registros_csv = gv.leer_registros_metadata(directorio_base)
    if not forzar_reentrenamiento and os.path.isfile(RUTA_MODELO_CACHE):
        try:
            # Garantizar compatibilidad de pickle/joblib tanto al ejecutar como __main__ como al importar
            main_mod = sys.modules.get("__main__")
            if main_mod is not None and not hasattr(main_mod, "PonderadorANOVA"):
                setattr(main_mod, "PonderadorANOVA", PonderadorANOVA)
            paquete = joblib.load(RUTA_MODELO_CACHE)
            if (
                paquete.get("num_audios_entrenamiento") == len(registros_csv)
                and paquete.get("k_vecinos") == k_vecinos
            ):
                return paquete
        except Exception:
            pass

    return entrenar_modelo_completo(
        directorio_base=directorio_base, k_vecinos=k_vecinos, guardar_cache=True
    )


# ==============================================================================
# 3. EVALUACIÓN ESTADÍSTICA Y GRÁFICOS (PCA 2D/3D + MATRIZ DE CONFUSIÓN)
# ==============================================================================

def _dibujar_elipse_confianza(
    ax: plt.Axes, points_xy: np.ndarray, color: str, n_std: float = 1.8
) -> None:
    """Dibuja una elipse de covarianza alrededor del clúster de una clase en 2D."""
    if points_xy.shape[0] < 3:
        return
    cov = np.cov(points_xy, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    orden = vals.argsort()[::-1]
    vals, vecs = vals[orden], vecs[:, orden]
    angulo = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ancho, alto = 2.0 * n_std * np.sqrt(np.maximum(vals, 1e-6))
    centro = np.mean(points_xy, axis=0)
    elipse = Ellipse(
        xy=centro,
        width=ancho,
        height=alto,
        angle=angulo,
        edgecolor=color,
        facecolor=color,
        alpha=0.14,
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(elipse)


def analizar_y_graficar_dataset(
    directorio_base: str = DIRECTORIO_DATASET_DEFAULT,
    k_vecinos: int = K_VECINOS_DEFAULT,
    mostrar_ventana: bool = True,
) -> Dict[str, object]:
    """
    Ejecuta la evaluación completa del clasificador K-NN sobre `dataset_voz/`,
    imprime las estadísticas requeridas por el informe de la cátedra y genera
    el gráfico de 4 paneles en `resultados_voz/analisis_knn_pca_voz.png`.
    """
    paquete = entrenar_modelo_completo(
        directorio_base=directorio_base, k_vecinos=k_vecinos, guardar_cache=True
    )
    X: np.ndarray = paquete["X"]
    y: np.ndarray = paquete["y"]
    personas: np.ndarray = paquete["personas"]
    metas: List[Dict[str, str]] = paquete["metas"]
    X_pca: np.ndarray = paquete["X_pca"]
    pca_modelo: PCA = paquete["pca_modelo"]
    knn_2d: KNeighborsClassifier = paquete["knn_2d"]

    # 1. Validación Cruzada Estratificada (5-Fold CV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe_eval = construir_pipeline_knn(k_vecinos=k_vecinos)
    y_pred_cv = cross_val_predict(pipe_eval, X, y, cv=skf)
    acc_cv = accuracy_score(y, y_pred_cv)
    matriz_conf = confusion_matrix(y, y_pred_cv, labels=list(COMANDOS_ORDEN))

    # 2. Validación Leave-One-Person-Out (Independencia del locutor)
    personas_unicas = sorted(set(personas))
    if len(personas_unicas) >= 2:
        logo = LeaveOneGroupOut()
        scores_lopo = cross_val_score(
            construir_pipeline_knn(k_vecinos=k_vecinos),
            X,
            y,
            groups=personas,
            cv=logo,
        )
        acc_lopo = float(np.mean(scores_lopo))
    else:
        acc_lopo = acc_cv

    # 3. Barrido de valores de K (K = 1..13)
    valores_k = [1, 2, 3, 4, 5, 7, 9, 11, 13]
    curva_acc_k = []
    for k_test in valores_k:
        p_k = construir_pipeline_knn(k_vecinos=k_test)
        sc = cross_val_score(p_k, X, y, cv=skf).mean()
        curva_acc_k.append(float(sc * 100.0))

    # 4. Imprimir reporte estadístico detallado en consola
    var_ratio = pca_modelo.explained_variance_ratio_ * 100.0
    var_2d = float(np.sum(var_ratio[:2]))
    var_3d = float(np.sum(var_ratio[:3]))

    print("\n" + "=" * 76)
    print("   RESULTADOS DEL CLASIFICADOR DE VOZ K-NN Y ANÁLISIS PCA (IA1 - UNCUYO)")
    print("=" * 76)
    print(f" - Total de audios analizados        : {len(y)} muestras ({len(personas_unicas)} personas)")
    print(f" - Dimensión del vector acústico     : {X.shape[1]} características por audio")
    print(f" - Clasificador configurado          : K-NN (K = {k_vecinos}, pesos por distancia, L2)")
    print(f" - Exactitud 5-Fold Cross-Validation : {acc_cv * 100:.2f}% ({int(round(acc_cv * len(y)))}/{len(y)} aciertos)")
    print(f" - Exactitud Leave-One-Person-Out    : {acc_lopo * 100:.2f}% (generalización a nuevos locutores)")
    print(f" - Varianza explicada PCA (Subespacio): 2D = {var_2d:.1f}% | 3D = {var_3d:.1f}%")
    print("-" * 76)
    print("\n MÉTRICAS POR CLASE (Validación Cruzada 5-Fold):")
    print(
        classification_report(
            y, y_pred_cv, labels=list(COMANDOS_ORDEN), digits=4
        )
    )

    print(" MATRIZ DE CONFUSIÓN (Filas = Real | Columnas = Predicho por K-NN):")
    encabezado_mc = f" {'Clase Real':<14} | " + " | ".join(
        f"{c:>12}" for c in COMANDOS_ORDEN
    )
    print(encabezado_mc)
    print(" " + "-" * (len(encabezado_mc) - 1))
    for idx_fila, cmd_real in enumerate(COMANDOS_ORDEN):
        celdas = " | ".join(
            f"{matriz_conf[idx_fila, j]:>12d}" for j in range(len(COMANDOS_ORDEN))
        )
        print(f" {cmd_real:<14} | {celdas}")

    # Mostrar qué audios específicos se confundieron en validación cruzada
    errores = [
        (metas[i]["ruta_audio"], y[i], y_pred_cv[i])
        for i in range(len(y))
        if y[i] != y_pred_cv[i]
    ]
    if errores:
        print(f"\n Audios confundidos en Validación Cruzada ({len(errores)} de {len(y)}):")
        for ruta_err, real_c, pred_c in errores:
            print(f"   * {ruta_err} -> Real: «{real_c}» | Predicho: «{pred_c}»")
    else:
        print("\n [!] 0 errores en Validación Cruzada (100% de separación).")

    # 5. Generar figura de 4 paneles
    os.makedirs(DIRECTORIO_RESULTADOS_DEFAULT, exist_ok=True)
    fig = plt.figure(figsize=(16, 12), dpi=150)
    fig.suptitle(
        f"Reconocimiento de Comandos de Voz con K-NN (K={k_vecinos}) y Separación de Clases PCA\n"
        f"Dataset: {len(y)} grabaciones ({len(personas_unicas)} personas) | "
        f"Exactitud 5-Fold CV: {acc_cv * 100:.2f}% | Leave-One-Person-Out: {acc_lopo * 100:.2f}%",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # --- PANEL 1: Separación de Clases en PCA 2D + Regiones de decisión K-NN ---
    ax1 = fig.add_subplot(2, 2, 1)
    x_min, x_max = X_pca[:, 0].min() - 0.20, X_pca[:, 0].max() + 0.20
    y_min, y_max = X_pca[:, 1].min() - 0.20, X_pca[:, 1].max() + 0.20
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180)
    )
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    le_grid = LabelEncoder().fit(list(COMANDOS_ORDEN))
    zz_labels = knn_2d.predict(grid_pts)
    zz_num = le_grid.transform(zz_labels).reshape(xx.shape)
    ax1.contourf(
        xx,
        yy,
        zz_num,
        alpha=0.10,
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=[
            COLORES_COMANDOS[c] for c in le_grid.classes_
        ],
    )

    for cmd in COMANDOS_ORDEN:
        mask = y == cmd
        pts = X_pca[mask, :2]
        _dibujar_elipse_confianza(ax1, pts, COLORES_COMANDOS[cmd], n_std=1.8)
        ax1.scatter(
            pts[:, 0],
            pts[:, 1],
            c=COLORES_COMANDOS[cmd],
            marker=MARCADORES_COMANDOS[cmd],
            s=65,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.7,
            label=f"«{cmd}» (n={np.sum(mask)})",
        )
        centro = np.mean(pts, axis=0)
        ax1.scatter(
            centro[0],
            centro[1],
            c=COLORES_COMANDOS[cmd],
            marker="X",
            s=210,
            edgecolors="black",
            linewidth=1.5,
            zorder=6,
        )

    ax1.set_title(
        f"1. Separación de Clases en PCA 2D (PC1 vs PC2: {var_2d:.1f}% var.)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel(f"Componente Principal 1 (PC1 - {var_ratio[0]:.1f}%)")
    ax1.set_ylabel(f"Componente Principal 2 (PC2 - {var_ratio[1]:.1f}%)")
    ax1.legend(loc="best", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # --- PANEL 2: Separación de Clases en PCA 3D (PC1, PC2, PC3) ---
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    for cmd in COMANDOS_ORDEN:
        mask = y == cmd
        pts3 = X_pca[mask, :3]
        ax2.scatter(
            pts3[:, 0],
            pts3[:, 1],
            pts3[:, 2],
            c=COLORES_COMANDOS[cmd],
            marker=MARCADORES_COMANDOS[cmd],
            s=55,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.5,
            label=f"«{cmd}»",
        )
        c3 = np.mean(pts3, axis=0)
        ax2.scatter(
            [c3[0]],
            [c3[1]],
            [c3[2]],
            c=COLORES_COMANDOS[cmd],
            marker="X",
            s=180,
            edgecolors="black",
            linewidth=1.5,
        )

    ax2.set_title(
        f"2. Proyección Tridimensional PCA 3D ({var_3d:.1f}% var. acumulada)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel(f"PC1 ({var_ratio[0]:.1f}%)")
    ax2.set_ylabel(f"PC2 ({var_ratio[1]:.1f}%)")
    ax2.set_zlabel(f"PC3 ({var_ratio[2]:.1f}%)")
    ax2.legend(loc="upper right")

    # --- PANEL 3: Matriz de Confusión (Validación Cruzada 5-Fold) ---
    ax3 = fig.add_subplot(2, 2, 3)
    im = ax3.imshow(matriz_conf, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title(
        f"3. Matriz de Confusión K-NN (5-Fold CV - Exactitud: {acc_cv * 100:.2f}%)",
        fontsize=11,
        fontweight="bold",
    )
    ticks = np.arange(len(COMANDOS_ORDEN))
    ax3.set_xticks(ticks)
    ax3.set_yticks(ticks)
    ax3.set_xticklabels([f"«{c}»" for c in COMANDOS_ORDEN])
    ax3.set_yticklabels([f"«{c}»" for c in COMANDOS_ORDEN])
    ax3.set_ylabel("Comando Real (Etiqueta)")
    ax3.set_xlabel("Comando Predicho por K-NN")

    umbral_texto = matriz_conf.max() / 2.0
    for i in range(matriz_conf.shape[0]):
        total_fila = max(1, np.sum(matriz_conf[i]))
        for j in range(matriz_conf.shape[1]):
            valor = matriz_conf[i, j]
            pct = (valor / total_fila) * 100.0
            color_txt = "white" if valor > umbral_texto else "black"
            ax3.text(
                j,
                i,
                f"{valor}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color_txt,
                fontweight="bold",
                fontsize=10,
            )

    # --- PANEL 4: Exactitud vs K y Varianza Acumulada PCA ---
    ax4 = fig.add_subplot(2, 2, 4)
    color_k = "#1f77b4"
    ax4.plot(
        valores_k,
        curva_acc_k,
        marker="o",
        linewidth=2.2,
        color=color_k,
        label="Exactitud 5-Fold CV (%) según K",
    )
    ax4.axvline(
        k_vecinos,
        color="#d62728",
        linestyle="--",
        alpha=0.8,
        label=f"K seleccionado = {k_vecinos} ({acc_cv * 100:.1f}%)",
    )
    for kv, av in zip(valores_k, curva_acc_k):
        ax4.annotate(
            f"{av:.1f}%",
            (kv, av),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8.5,
        )

    ax4.set_title(
        "4. Desempeño de K-NN según número de vecinos (K)",
        fontsize=11,
        fontweight="bold",
    )
    ax4.set_xlabel("Número de vecinos más cercanos (K)")
    ax4.set_ylabel("Exactitud en Validación Cruzada 5-Fold (%)")
    ax4.set_xticks(valores_k)
    ax4.set_ylim(max(65.0, min(curva_acc_k) - 8.0), 102.0)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="lower left")

    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(RUTA_GRAFICO_ANALISIS, dpi=200)
    print(f"\n[+] Gráfico de análisis guardado en: {os.path.abspath(RUTA_GRAFICO_ANALISIS)}")
    print("=" * 76)

    if mostrar_ventana:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)

    return {
        "acc_cv": acc_cv,
        "acc_lopo": acc_lopo,
        "matriz_confusion": matriz_conf,
        "ruta_grafico": RUTA_GRAFICO_ANALISIS,
        "paquete_modelo": paquete,
    }


# ==============================================================================
# 4. CLASIFICACIÓN DE VOZ EN VIVO (FUNCIÓN EXPORTABLE PARA EL AGENTE FINAL)
# ==============================================================================

def clasificar_senal_audio(
    senal: np.ndarray,
    sr: int = gv.SAMPLE_RATE_DEFAULT,
    paquete_modelo: Optional[Dict[str, object]] = None,
    directorio_base: str = DIRECTORIO_DATASET_DEFAULT,
    umbral_confianza: float = UMBRAL_CONFIANZA_DEFAULT,
) -> Dict[str, object]:
    """
    Recibe una señal de audio 1D y devuelve el diagnóstico completo del K-NN:
      - `comando`: etiqueta predicha ('Reconocer', 'Identificar', 'Salir' o 'SILENCIO' / 'DUDOSO')
      - `comando_mas_votado`: etiqueta con mayor probabilidad aún si no supera el umbral
      - `confianza`: probabilidad ponderada por distancia en [0.0, 1.0]
      - `probabilidades`: diccionario {comando: probabilidad}
      - `vecinos`: lista de los K audios del dataset más cercanos (ruta, persona, comando, distancia)
      - `es_valido`: True si hubo voz suficiente y superó `umbral_confianza`
    """
    if paquete_modelo is None:
        paquete_modelo = entrenar_o_cargar_modelo_voz(directorio_base=directorio_base)

    arr_i16 = (
        np.clip(np.asarray(senal) * 32768.0, -32768, 32767).astype(np.int16)
        if np.max(np.abs(senal)) <= 1.5
        else np.asarray(senal, dtype=np.int16)
    )
    rms_ac, pico_ac = gv.calcular_metricas_senal(arr_i16)

    if gv._es_captura_saturada_dc(arr_i16):
        return {
            "comando": "ERROR_MICROFONO",
            "comando_mas_votado": None,
            "confianza": 0.0,
            "probabilidades": {c: 0.0 for c in COMANDOS_ORDEN},
            "vecinos": [],
            "es_valido": False,
            "pico": pico_ac,
            "rms": rms_ac,
            "motivo": (
                "La entrada de audio devolvió una señal bloqueada en continua (posible "
                "jack analógico desconectado). Verificá el micrófono seleccionado."
            ),
        }

    if pico_ac < UMBRAL_PICO_SILENCIO or rms_ac < 0.004:
        return {
            "comando": "SILENCIO",
            "comando_mas_votado": None,
            "confianza": 0.0,
            "probabilidades": {c: 0.0 for c in COMANDOS_ORDEN},
            "vecinos": [],
            "es_valido": False,
            "pico": pico_ac,
            "rms": rms_ac,
            "motivo": (
                f"Nivel de voz demasiado bajo (pico AC={pico_ac * 100:.1f}%, "
                f"RMS={rms_ac * 100:.1f}%). No se detectó voz clara."
            ),
        }

    vector_x = extraer_caracteristicas_voz(senal, sr=sr)
    pipeline: Pipeline = paquete_modelo["pipeline_knn"]

    probs_arr = pipeline.predict_proba(vector_x.reshape(1, -1))[0]
    clases_modelo = list(pipeline.classes_)
    mapa_probs = {c: 0.0 for c in COMANDOS_ORDEN}
    for c_mod, p_val in zip(clases_modelo, probs_arr):
        mapa_probs[str(c_mod)] = float(p_val)

    comando_ganador = max(mapa_probs, key=mapa_probs.get)
    confianza = float(mapa_probs[comando_ganador])

    # Obtener los K vecinos más cercanos reales del dataset para explicabilidad
    x_trans = pipeline[:-1].transform(vector_x.reshape(1, -1))
    knn_est: KNeighborsClassifier = pipeline.named_steps["knn"]
    distancias, indices = knn_est.kneighbors(x_trans)

    metas_ds: List[Dict[str, str]] = paquete_modelo["metas"]
    vecinos_info = []
    for dist, idx_ds in zip(distancias[0], indices[0]):
        m_vec = metas_ds[int(idx_ds)]
        vecinos_info.append(
            {
                "indice": int(idx_ds),
                "distancia": float(dist),
                "comando": m_vec["comando"],
                "persona": m_vec["persona"],
                "ruta_audio": m_vec["ruta_audio"],
            }
        )

    coord_pca = proyectar_vector_en_pca(paquete_modelo, vector_x)
    es_valido = confianza >= umbral_confianza
    comando_final = comando_ganador if es_valido else "NO_RECONOCIDO"

    return {
        "comando": comando_final,
        "comando_mas_votado": comando_ganador,
        "confianza": confianza,
        "probabilidades": mapa_probs,
        "vecinos": vecinos_info,
        "es_valido": es_valido,
        "pico": pico_ac,
        "rms": rms_ac,
        "vector_pca": coord_pca,
        "motivo": "OK" if es_valido else f"Confianza ({confianza * 100:.1f}%) menor al umbral ({umbral_confianza * 100:.1f}%).",
    }


def reconocer_comando_microfono(
    paquete_modelo: Optional[Dict[str, object]] = None,
    directorio_base: str = DIRECTORIO_DATASET_DEFAULT,
    duracion_seg: float = gv.DURACION_DEFAULT,
    sample_rate: int = gv.SAMPLE_RATE_DEFAULT,
    cuenta_regresiva: int = 3,
    umbral_confianza: float = UMBRAL_CONFIANZA_DEFAULT,
    dispositivo: Optional[int] = None,
    mostrar_detalle: bool = True,
    graficar_resultado: bool = False,
) -> Dict[str, object]:
    """
    ==========================================================================
    FUNCIÓN PRINCIPAL PARA LLAMAR DESDE EL AGENTE FINAL
    ==========================================================================
    Escucha una orden por el micrófono, extrae sus características acústicas y
    la clasifica con K-NN entre «Reconocer», «Identificar» y «Salir».

    Ejemplo de uso en `agente.py`:
        from reconocimiento_voz_knn import entrenar_o_cargar_modelo_voz, reconocer_comando_microfono

        modelo_voz = entrenar_o_cargar_modelo_voz()
        resultado = reconocer_comando_microfono(modelo_voz, cuenta_regresiva=2)
        if resultado["es_valido"]:
            orden = resultado["comando"]  # "Reconocer", "Identificar" o "Salir"
    """
    if paquete_modelo is None:
        paquete_modelo = entrenar_o_cargar_modelo_voz(directorio_base=directorio_base)

    senal_capturada, sr = gv.escuchar_orden_en_vivo(
        duracion_seg=duracion_seg,
        sample_rate=sample_rate,
        dispositivo=dispositivo,
        cuenta_regresiva=cuenta_regresiva,
        normalizar_float=True,
        mostrar_mensajes=True,
    )

    resultado = clasificar_senal_audio(
        senal=senal_capturada,
        sr=sr,
        paquete_modelo=paquete_modelo,
        directorio_base=directorio_base,
        umbral_confianza=umbral_confianza,
    )
    resultado["senal_audio"] = senal_capturada
    resultado["sample_rate"] = sr

    if mostrar_detalle:
        imprimir_resultado_prediccion(resultado)

    if graficar_resultado and resultado.get("vector_pca") is not None:
        graficar_prediccion_en_pca(
            paquete_modelo=paquete_modelo,
            senal=senal_capturada,
            sr=sr,
            resultado=resultado,
            mostrar_ventana=True,
        )

    return resultado


def imprimir_resultado_prediccion(resultado: Dict[str, object]) -> None:
    """Muestra por consola el resultado del reconocimiento K-NN, probabilidades y vecinos."""
    print("\n" + "=" * 72)
    print(" RESULTADO DEL RECONOCIMIENTO DE VOZ (K-NN)")
    print("=" * 72)

    if not resultado["es_valido"]:
        print(f" [!] Estado           : {resultado['comando']}")
        print(f" [!] Detalle          : {resultado.get('motivo', '')}")
        if resultado.get("comando_mas_votado"):
            print(
                f" [i] Clase más cercana: «{resultado['comando_mas_votado']}» "
                f"({float(resultado['confianza']) * 100:.1f}%)"
            )
    else:
        cmd = resultado["comando"]
        conf = float(resultado["confianza"]) * 100.0
        pico_pct = float(resultado.get("pico", 0.0)) * 100.0
        rms_pct = float(resultado.get("rms", 0.0)) * 100.0
        print(f" >>> COMANDO RECONOCIDO: «{cmd.upper()}» (Confianza: {conf:.1f}%) <<<")
        print(f"     Nivel de voz captado: pico AC = {pico_pct:.1f}% | energía RMS = {rms_pct:.1f}%")

    probs: Dict[str, float] = resultado.get("probabilidades", {})
    if any(probs.values()):
        print("\n Distribución de votos ponderados por distancia (K-NN):")
        for c in COMANDOS_ORDEN:
            p_pct = probs.get(c, 0.0) * 100.0
            n_barras = int(round(p_pct / 4.0))
            barra = "█" * n_barras + "░" * (25 - n_barras)
            print(f"   - {c:<12}: [{barra}] {p_pct:5.1f}%")

    vecinos: List[Dict[str, object]] = resultado.get("vecinos", [])
    if vecinos:
        print(f"\n Los {len(vecinos)} vecinos más cercanos en el dataset:")
        for rank, v in enumerate(vecinos, 1):
            print(
                f"   #{rank} Distancia={v['distancia']:.4f} -> «{v['comando']}» "
                f"({v['persona']}: {os.path.basename(str(v['ruta_audio']))})"
            )
    print("=" * 72)


def _generar_ruta_unica_prueba_vivo(comando_mostrado: str) -> str:
    """
    Genera una ruta correlativa única dentro de `resultados_voz/` para guardar
    cada gráfico de prueba por micrófono sin sobrescribir los anteriores:
    Ejemplo: resultados_voz/prueba_microfono_001_Reconocer.png
    """
    os.makedirs(DIRECTORIO_RESULTADOS_DEFAULT, exist_ok=True)
    patron = re.compile(r"^prueba_microfono_(\d+)_", flags=re.IGNORECASE)
    max_idx = 0
    for nombre in os.listdir(DIRECTORIO_RESULTADOS_DEFAULT):
        m = patron.match(nombre)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    etiqueta_limpia = re.sub(r"[^A-Za-z0-9_\-]", "", str(comando_mostrado or "Audio"))
    siguiente = max_idx + 1
    while True:
        candidata = os.path.join(
            DIRECTORIO_RESULTADOS_DEFAULT,
            f"prueba_microfono_{siguiente:03d}_{etiqueta_limpia}.png",
        )
        if not os.path.exists(candidata):
            return candidata
        siguiente += 1


def graficar_prediccion_en_pca(
    paquete_modelo: Dict[str, object],
    senal: np.ndarray,
    sr: int,
    resultado: Dict[str, object],
    mostrar_ventana: bool = True,
    ruta_salida_personalizada: Optional[str] = None,
    silencioso: bool = False,
) -> str:
    """
    Genera una lámina visual mostrando:
      1. La onda de voz capturada y el recorte automático VAD de la palabra activa.
      2. La ubicación exacta del audio recién pronunciado (estrella dorada ★) en el
         mapa PCA 2D respecto de los clústeres de «Reconocer», «Identificar» y «Salir».
    """
    if ruta_salida_personalizada:
        dir_dest = os.path.dirname(os.path.abspath(ruta_salida_personalizada))
        if dir_dest:
            os.makedirs(dir_dest, exist_ok=True)
    else:
        os.makedirs(DIRECTORIO_RESULTADOS_DEFAULT, exist_ok=True)
    fig, (ax_wav, ax_pca) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140)

    cmd_mostrado = resultado.get("comando_mas_votado") or resultado.get("comando")
    conf_pct = float(resultado.get("confianza", 0.0)) * 100.0
    fig.suptitle(
        f"Prueba de Reconocimiento de Voz en Vivo -> Predicción K-NN: «{cmd_mostrado}» ({conf_pct:.1f}%)",
        fontsize=13,
        fontweight="bold",
    )

    # Panel izquierdo: Onda + Recorte VAD
    _, _, _, (idx_ini, idx_fin) = preprocesar_senal_voz(senal, sr=sr)
    t_eje = np.arange(len(senal)) / float(sr)
    ax_wav.plot(t_eje, senal, color="#555555", alpha=0.65, linewidth=0.8, label="Señal capturada")
    ax_wav.axvspan(
        idx_ini / float(sr),
        idx_fin / float(sr),
        color="#2ca02c",
        alpha=0.20,
        label="Palabra activa detectada (VAD)",
    )
    ax_wav.set_title("1. Señal de Micrófono y Recorte de Voz Activa (VAD)")
    ax_wav.set_xlabel("Tiempo (segundos)")
    ax_wav.set_ylabel("Amplitud normalizada")
    ax_wav.legend(loc="upper right")
    ax_wav.grid(True, linestyle=":", alpha=0.5)

    # Panel derecho: Mapa PCA 2D con la muestra proyectada
    X_pca: np.ndarray = paquete_modelo["X_pca"]
    y: np.ndarray = paquete_modelo["y"]
    for cmd in COMANDOS_ORDEN:
        mask = y == cmd
        pts = X_pca[mask, :2]
        _dibujar_elipse_confianza(ax_pca, pts, COLORES_COMANDOS[cmd], n_std=1.8)
        ax_pca.scatter(
            pts[:, 0],
            pts[:, 1],
            c=COLORES_COMANDOS[cmd],
            marker=MARCADORES_COMANDOS[cmd],
            s=50,
            alpha=0.65,
            label=f"Dataset «{cmd}»",
        )

    pt_nuevo = resultado["vector_pca"][:2]
    # Dibujar líneas hacia los K vecinos más cercanos en el gráfico
    for v in resultado.get("vecinos", []):
        pt_vecino = X_pca[int(v["indice"]), :2]
        ax_pca.plot(
            [pt_nuevo[0], pt_vecino[0]],
            [pt_nuevo[1], pt_vecino[1]],
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

    ax_pca.scatter(
        [pt_nuevo[0]],
        [pt_nuevo[1]],
        c="#ffcc00",
        marker="*",
        s=380,
        edgecolors="black",
        linewidth=1.5,
        zorder=10,
        label=f"Tu voz («{cmd_mostrado}»)",
    )
    ax_pca.set_title("2. Proyección de la Orden en el Espacio PCA 2D y K Vecinos")
    ax_pca.set_xlabel("Componente Principal 1 (PC1)")
    ax_pca.set_ylabel("Componente Principal 2 (PC2)")
    ax_pca.legend(loc="best", fontsize=8.5)
    ax_pca.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    ruta_salida = (
        ruta_salida_personalizada
        if ruta_salida_personalizada
        else _generar_ruta_unica_prueba_vivo(str(cmd_mostrado))
    )
    fig.savefig(ruta_salida, dpi=160)
    if not silencioso:
        print(f"[+] Gráfico de la prueba guardado en: {os.path.abspath(ruta_salida)}")

    if mostrar_ventana:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)
    return ruta_salida


def _generar_ruta_unica_mosaico_vivo(comando_mostrado: str) -> str:
    """
    Genera una ruta correlativa única dentro de `resultados_voz/comparativas_etapas/`
    para guardar cada mosaico de etapas capturado en vivo sin sobrescribir los anteriores:
    Ejemplo: resultados_voz/comparativas_etapas/mosaico_microfono_001_Reconocer.png
    """
    os.makedirs(CARPETA_COMPARATIVAS_VOZ, exist_ok=True)
    patron = re.compile(r"^mosaico_microfono_(\d+)_", flags=re.IGNORECASE)
    max_idx = 0
    for nombre in os.listdir(CARPETA_COMPARATIVAS_VOZ):
        m = patron.match(nombre)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    etiqueta_limpia = re.sub(r"[^A-Za-z0-9_\-]", "", str(comando_mostrado or "Audio"))
    siguiente = max_idx + 1
    while True:
        candidata = os.path.join(
            CARPETA_COMPARATIVAS_VOZ,
            f"mosaico_microfono_{siguiente:03d}_{etiqueta_limpia}.png",
        )
        if not os.path.exists(candidata):
            return candidata
        siguiente += 1


def crear_mosaico_etapas_audio(
    senal: np.ndarray,
    sr: int = gv.SAMPLE_RATE_DEFAULT,
    ruta_guardado: Optional[str] = None,
    nombre_archivo: str = "",
    etiqueta: str = "",
    mostrar_ventana: bool = False,
    silencioso: bool = False,
) -> str:
    """
    Construye y guarda una lámina comparativa en mosaico (3 filas x 2 columnas)
    que ilustra paso a paso cada una de las 6 etapas del procesamiento acústico
    y extracción espectro-temporal (análogo a `crear_mosaico_comparativo` en huellas):

      Panel 1: Señal Original Cruda (onda en el dominio del tiempo con offset DC y ruido ambiental).
      Panel 2: Filtrado Pasa-Banda Butterworth (120 Hz a 7200 Hz).
      Panel 3: Detección de Actividad de Voz (VAD adaptativo por envolvente RMS, umbral y segmento activo).
      Panel 4: Palabra Recortada y Normalizada (segmento activo s_voz con energía eficaz RMS = 1.0).
      Panel 5: Pre-énfasis de Alta Frecuencia (coef = 0.97) y Partición Fonética en 6 Bloques Temporales (B1..B6).
      Panel 6: Espectrograma MFCC con Normalización de Media Cepstral (CMN) y delimitación de bloques.

    Retorna:
        Ruta absoluta del archivo PNG guardado.
    """
    s_cruda = np.asarray(senal, dtype=np.float32).reshape(-1)
    if s_cruda.size == 0:
        s_cruda = np.zeros(int(0.5 * sr), dtype=np.float32)

    # 1. Quitar offset DC
    dc_offset = float(np.mean(s_cruda))
    s_centrada = s_cruda - dc_offset

    # 2. Filtrado pasa-banda Butterworth
    b_bp, a_bp = _construir_filtro_pasabanda(sr=sr)
    if len(s_centrada) > 32:
        s_filtrada = filtfilt(b_bp, a_bp, s_centrada).astype(np.float32)
    else:
        s_filtrada = s_centrada

    # 3. VAD por envolvente RMS
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    rms_env = librosa.feature.rms(
        y=s_filtrada, frame_length=frame_len, hop_length=hop_len
    )[0]
    t_env = np.arange(len(rms_env)) * (hop_len / float(sr))

    piso_ruido = float(np.percentile(rms_env, 15))
    pico_max = float(np.max(rms_env))
    umbral_vad = piso_ruido + 0.18 * (pico_max - piso_ruido)

    frames_activos = np.where(rms_env >= umbral_vad)[0]
    if len(frames_activos) >= 4:
        ini_frame = max(0, int(frames_activos[0]) - 2)
        fin_frame = min(len(rms_env), int(frames_activos[-1]) + 3)
        muestra_ini = ini_frame * hop_len
        muestra_fin = min(len(s_filtrada), fin_frame * hop_len)
        s_voz = s_filtrada[muestra_ini:muestra_fin]
    else:
        muestra_ini, muestra_fin = 0, len(s_filtrada)
        s_voz = s_filtrada

    if len(s_voz) < int(0.15 * sr):
        muestra_ini, muestra_fin = 0, len(s_filtrada)
        s_voz = s_filtrada

    # 4. Normalización RMS
    rms_voz = float(np.sqrt(np.mean(np.square(s_voz)))) + 1e-8
    s_voz_norm = (s_voz / rms_voz).astype(np.float32)

    # 5. Pre-énfasis
    s_pre = librosa.effects.preemphasis(s_voz_norm, coef=0.97)

    # 6. MFCC con CMN
    mfcc = librosa.feature.mfcc(
        y=s_pre,
        sr=sr,
        n_mfcc=13,
        n_mels=32,
        n_fft=512,
        hop_length=hop_len,
        fmin=120,
        fmax=7200,
    )
    mfcc_cmn = mfcc - np.mean(mfcc, axis=1, keepdims=True)

    # Partición en 6 bloques temporales
    n_frames = mfcc.shape[1]
    bloques = np.array_split(np.arange(n_frames), N_SEGMENTOS_TEMPORALES)

    # Construcción de la lámina
    fig, axs = plt.subplots(3, 2, figsize=(15.5, 10.5), dpi=140)

    sub_titulo = "Mosaico Comparativo de Etapas de Procesamiento de Audio"
    detalles = []
    if nombre_archivo:
        detalles.append(f"Archivo: {nombre_archivo}")
    if etiqueta:
        detalles.append(f"Comando: «{etiqueta}»")
    detalles.append(f"Fs: {sr} Hz ({sr / 1000.0:.1f} kHz)")
    detalles.append(f"Pre-énfasis: 0.97")
    detalles.append(f"Bloques temporales: {N_SEGMENTOS_TEMPORALES}")

    fig.suptitle(
        f"{sub_titulo}\n{' | '.join(detalles)}",
        fontsize=12,
        fontweight="bold",
    )

    t_cruda = np.arange(len(s_cruda)) / float(sr)

    # Panel 1: Onda cruda
    ax1 = axs[0, 0]
    ax1.plot(t_cruda, s_cruda, color="#455a64", linewidth=0.85, label="Señal capturada")
    ax1.axhline(
        dc_offset,
        color="#c62828",
        linestyle=":",
        linewidth=1.1,
        label=f"Offset DC ({dc_offset:+.4f})",
    )
    ax1.set_title("1. Señal Original Cruda (Dominio del Tiempo)", fontweight="bold", fontsize=10.5)
    ax1.set_ylabel("Amplitud cruda", fontsize=9)
    ax1.set_xlabel("Tiempo (segundos)", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: Señal filtrada
    ax2 = axs[0, 1]
    ax2.plot(t_cruda, s_filtrada, color="#0288d1", linewidth=0.85, label="Filtro pasa-banda")
    ax2.set_title(
        "2. Filtrado Pasa-Banda Butterworth (120 Hz – 7200 Hz)",
        fontweight="bold",
        fontsize=10.5,
    )
    ax2.set_ylabel("Amplitud filtrada", fontsize=9)
    ax2.set_xlabel("Tiempo (segundos)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3: VAD
    ax3 = axs[1, 0]
    ax3.plot(t_env, rms_env, color="#e65100", linewidth=1.5, label="Envolvente RMS (25 ms)")
    ax3.axhline(
        umbral_vad,
        color="#d32f2f",
        linestyle="--",
        linewidth=1.2,
        label=f"Umbral VAD adaptativo ({umbral_vad:.3f})",
    )
    t_ini_s = muestra_ini / float(sr)
    t_fin_s = muestra_fin / float(sr)
    ax3.axvspan(
        t_ini_s,
        t_fin_s,
        color="#2ca02c",
        alpha=0.25,
        label=f"Voz activa detectada ({t_ini_s:.2f}s – {t_fin_s:.2f}s)",
    )
    ax3.set_title(
        "3. Detección de Actividad de Voz (VAD por Envolvente RMS)",
        fontweight="bold",
        fontsize=10.5,
    )
    ax3.set_ylabel("Energía RMS", fontsize=9)
    ax3.set_xlabel("Tiempo (segundos)", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8)

    # Panel 4: Palabra recortada y normalizada
    ax4 = axs[1, 1]
    t_voz = np.arange(len(s_voz_norm)) / float(sr)
    dur_util_s = len(s_voz_norm) / float(sr)
    ax4.plot(t_voz, s_voz_norm, color="#2e7d32", linewidth=0.85, label="Palabra activa")
    ax4.set_title(
        f"4. Palabra Recortada y Normalizada (RMS = 1.0, Duración = {dur_util_s:.3f} s)",
        fontweight="bold",
        fontsize=10.5,
    )
    ax4.set_ylabel("Amplitud normalizada", fontsize=9)
    ax4.set_xlabel("Tiempo relativo de la palabra (segundos)", fontsize=9)
    ax4.grid(True, linestyle=":", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=8)

    # Panel 5: Pre-énfasis y Bloques temporales
    ax5 = axs[2, 0]
    ax5.plot(t_voz, s_pre, color="#7b1fa2", linewidth=0.85, label="Pre-énfasis (coef=0.97)")
    colores_bloques = ["#e3f2fd", "#f3e5f5", "#e8f5e9", "#fff3e0", "#fce4ec", "#e0f2f1"]
    max_pre = float(np.max(np.abs(s_pre))) if len(s_pre) > 0 else 1.0
    for idx_b, b in enumerate(bloques):
        t_b_ini = (b[0] * hop_len) / float(sr)
        t_b_fin = ((b[-1] + 1) * hop_len) / float(sr)
        ax5.axvspan(
            t_b_ini,
            t_b_fin,
            color=colores_bloques[idx_b % len(colores_bloques)],
            alpha=0.40,
        )
        if idx_b > 0:
            ax5.axvline(t_b_ini, color="#8e24aa", linestyle="--", linewidth=0.9, alpha=0.7)
        ax5.text(
            (t_b_ini + t_b_fin) / 2.0,
            max_pre * 0.80,
            f"B{idx_b + 1}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#4a148c",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="#8e24aa",
                alpha=0.85,
                linewidth=0.7,
            ),
        )
    ax5.set_title(
        f"5. Filtro Pre-énfasis y Partición Fonética en {N_SEGMENTOS_TEMPORALES} Bloques Temporales",
        fontweight="bold",
        fontsize=10.5,
    )
    ax5.set_ylabel("Amplitud pre-énfasis", fontsize=9)
    ax5.set_xlabel("Tiempo relativo de la palabra (segundos)", fontsize=9)
    ax5.grid(True, linestyle=":", alpha=0.5)
    ax5.legend(loc="lower right", fontsize=8)

    # Panel 6: MFCC con CMN
    ax6 = axs[2, 1]
    im = ax6.imshow(
        mfcc_cmn,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, dur_util_s, 0.5, 13.5],
    )
    for idx_b, b in enumerate(bloques[:-1]):
        t_b_div = ((b[-1] + 1) * hop_len) / float(sr)
        ax6.axvline(t_b_div, color="white", linestyle="--", linewidth=1.0, alpha=0.85)
    ax6.set_title(
        "6. Espectrograma MFCC con Normalización CMN (13 Coeficientes)",
        fontweight="bold",
        fontsize=10.5,
    )
    ax6.set_ylabel("Coeficiente Cepstral (1 a 13)", fontsize=9)
    ax6.set_xlabel("Tiempo relativo de la palabra (segundos)", fontsize=9)
    ax6.set_yticks(np.arange(1, 14, 2))
    cbar = fig.colorbar(im, ax=ax6, pad=0.02)
    cbar.set_label("Energía cepstral (CMN)", fontsize=8)

    plt.tight_layout()

    if not ruta_guardado:
        os.makedirs(CARPETA_COMPARATIVAS_VOZ, exist_ok=True)
        nombre_limpio = os.path.splitext(os.path.basename(nombre_archivo or "audio"))[0]
        nombre_limpio = re.sub(r"[^A-Za-z0-9_\-]", "_", nombre_limpio)
        ruta_guardado = os.path.join(
            CARPETA_COMPARATIVAS_VOZ, f"{nombre_limpio}_mosaico_etapas.png"
        )
    else:
        dir_dest = os.path.dirname(os.path.abspath(ruta_guardado))
        if dir_dest:
            os.makedirs(dir_dest, exist_ok=True)

    fig.savefig(ruta_guardado, dpi=150)
    if not silencioso:
        print(f"[+] Mosaico comparativo de etapas guardado en: {os.path.abspath(ruta_guardado)}")

    if mostrar_ventana:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)
    return os.path.abspath(ruta_guardado)


def generar_mosaicos_dataset(
    directorio_base: str = DIRECTORIO_DATASET_DEFAULT,
    carpeta_salida: str = CARPETA_COMPARATIVAS_VOZ,
    max_por_clase: Optional[int] = 1,
    mostrar_ventana: bool = False,
) -> List[str]:
    """
    Genera mosaicos comparativos de las etapas de procesamiento para los audios
    del dataset. Si `max_por_clase` es un entero (ej: 1 o 2), genera esa cantidad
    de ejemplos representativos por cada comando («Reconocer», «Identificar», «Salir»).
    Si es None, procesa el dataset completo.
    """
    registros = gv.leer_registros_metadata(directorio_base)
    if not registros:
        print(f"[!] No hay grabaciones registradas en '{directorio_base}/metadata.csv'.")
        return []

    os.makedirs(carpeta_salida, exist_ok=True)
    seleccionados: List[Dict[str, str]] = []

    if max_por_clase is not None and max_por_clase > 0:
        conteo_clase: Dict[str, int] = {cmd: 0 for cmd in COMANDOS_ORDEN}
        for reg in registros:
            cmd = reg.get("comando", "")
            if cmd in conteo_clase and conteo_clase[cmd] < max_por_clase:
                seleccionados.append(reg)
                conteo_clase[cmd] += 1
    else:
        seleccionados = list(registros)

    print(
        f"\n[*] Generando {len(seleccionados)} mosaicos comparativos de etapas de audio "
        f"en '{carpeta_salida}/'..."
    )
    rutas_generadas: List[str] = []
    for idx, reg in enumerate(seleccionados, start=1):
        rel_path = reg.get("ruta_audio", "")
        ruta_wav = gv.resolver_ruta_en_disco(directorio_base, rel_path)
        if not os.path.isfile(ruta_wav):
            continue
        senal, sr = gv.leer_wav_mono(ruta_wav, normalizar_float=True)
        nombre_base = os.path.basename(ruta_wav)
        nombre_sin_ext = os.path.splitext(nombre_base)[0]
        ruta_salida = os.path.join(carpeta_salida, f"{nombre_sin_ext}_mosaico_etapas.png")

        cmd = reg.get("comando", "")
        persona = reg.get("persona", "")
        etiqueta_desc = f"{cmd} ({persona})" if persona else cmd

        print(f" [{idx}/{len(seleccionados)}] Procesando '{nombre_base}' -> «{cmd}»...")
        crear_mosaico_etapas_audio(
            senal=senal,
            sr=sr,
            ruta_guardado=ruta_salida,
            nombre_archivo=nombre_base,
            etiqueta=etiqueta_desc,
            mostrar_ventana=mostrar_ventana,
            silencioso=True,
        )
        rutas_generadas.append(ruta_salida)

    print(f"\n[+] ¡Listo! Se generaron {len(rutas_generadas)} mosaicos comparativos en:")
    print(f"    {os.path.abspath(carpeta_salida)}/")
    return rutas_generadas


# ==============================================================================
# 5. MENÚ INTERACTIVO DE CONSOLA
# ==============================================================================

def _probar_archivo_existente_menu(
    paquete_modelo: Dict[str, object], directorio_base: str
) -> None:
    """Permite elegir un archivo .wav del dataset o una ruta externa para probar el K-NN."""
    metas: List[Dict[str, str]] = paquete_modelo["metas"]
    print("\n" + "-" * 72)
    print(" PRUEBA DEL CLASIFICADOR SOBRE UN ARCHIVO WAV")
    print("-" * 72)
    print(f" Hay {len(metas)} archivos registrados en '{directorio_base}/metadata.csv'.")
    entrada = input(
        f" Ingresá un número de fila (1 a {len(metas)}) o la ruta a un archivo .wav [Default: 1]: "
    ).strip()

    if not entrada:
        entrada = "1"

    if entrada.isdigit() and 1 <= int(entrada) <= len(metas):
        elegido = metas[int(entrada) - 1]
        ruta_wav = gv.resolver_ruta_en_disco(directorio_base, elegido["ruta_audio"])
        print(
            f"\n [*] Evaluando archivo #{entrada}: {elegido['ruta_audio']} "
            f"(Etiqueta real: «{elegido['comando']}», {elegido['persona']})"
        )
    else:
        ruta_wav = entrada
        if not os.path.isfile(ruta_wav):
            print(f" [!] No se encontró el archivo: {ruta_wav}")
            return

    senal, sr = gv.leer_wav_mono(ruta_wav, normalizar_float=True)
    res = clasificar_senal_audio(
        senal=senal,
        sr=sr,
        paquete_modelo=paquete_modelo,
        directorio_base=directorio_base,
    )
    imprimir_resultado_prediccion(res)

    ver_graf = input(
        "\n Opciones de visualización: 'g' Gráfico PCA | 'm' Mosaico comparativo de etapas | [Enter] Continuar: "
    ).strip().lower()
    if ver_graf in ("g", "s", "si", "sí", "y"):
        graficar_prediccion_en_pca(
            paquete_modelo=paquete_modelo,
            senal=senal,
            sr=sr,
            resultado=res,
            mostrar_ventana=True,
        )
    elif ver_graf == "m":
        cmd_det = str(res.get("comando_mas_votado") or res.get("comando") or "")
        crear_mosaico_etapas_audio(
            senal=senal,
            sr=sr,
            nombre_archivo=os.path.basename(ruta_wav),
            etiqueta=cmd_det,
            mostrar_ventana=True,
        )


def menu_interactivo(
    directorio_base: str = DIRECTORIO_DATASET_DEFAULT,
    k_inicial: int = K_VECINOS_DEFAULT,
) -> None:
    """Menú principal para analizar el dataset con K-NN + PCA y probar voz en vivo."""
    k_vecinos = k_inicial
    paquete_modelo: Optional[Dict[str, object]] = None

    while True:
        registros = gv.leer_registros_metadata(directorio_base)
        ok_mic, _, desc_mic = gv.verificar_disponibilidad_microfono()
        estado_mic = f"OK ({desc_mic})" if ok_mic else f"NO DISPONIBLE ({desc_mic})"

        print("\n" + "=" * 76)
        print("   MÓDULO DE RECONOCIMIENTO DE VOZ (K-NN + PCA) - IA1 UNCUYO")
        print("=" * 76)
        print(f" Dataset cargado       : {directorio_base}/ ({len(registros)} grabaciones registradas)")
        print(f" Vecinos K-NN (K)      : K = {k_vecinos} (pesos por distancia en espacio L2-ANOVA)")
        print(f" Estado del micrófono  : {estado_mic}")
        print("-" * 76)
        print(" MENÚ DE OPCIONES:")
        print("   [1] Analizar dataset completo y mostrar gráficos (PCA 2D/3D + Matriz de Confusión)")
        print("   [2] Testear reconocimiento de voz en vivo con el MICRÓFONO")
        print("   [3] Probar el clasificador sobre un archivo .wav existente")
        print("   [4] Cambiar el valor de K (número de vecinos) y re-entrenar")
        print("   [5] Ver / cambiar micrófono activo del sistema (Ubuntu PipeWire)")
        print("   [6] Generar mosaico comparativo de etapas de audio (como en huellas)")
        print("   [0] Salir")
        print("=" * 76)

        try:
            op = input(" Seleccioná una opción [Default: 1]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[+] Saliendo del módulo de reconocimiento de voz.")
            break

        if not op:
            op = "1"

        if op == "0" or op.lower() in ("salir", "q", "exit"):
            print("\n[+] Módulo finalizado correctamente.")
            break

        elif op == "1":
            try:
                info = analizar_y_graficar_dataset(
                    directorio_base=directorio_base,
                    k_vecinos=k_vecinos,
                    mostrar_ventana=True,
                )
                paquete_modelo = info["paquete_modelo"]
            except Exception as exc:
                print(f"\n[ERROR] No se pudo completar el análisis: {exc}")

        elif op == "2":
            if paquete_modelo is None:
                paquete_modelo = entrenar_o_cargar_modelo_voz(
                    directorio_base=directorio_base, k_vecinos=k_vecinos
                )

            while True:
                try:
                    res = reconocer_comando_microfono(
                        paquete_modelo=paquete_modelo,
                        directorio_base=directorio_base,
                        duracion_seg=gv.DURACION_DEFAULT,
                        cuenta_regresiva=3,
                        mostrar_detalle=True,
                        graficar_resultado=False,
                    )
                except KeyboardInterrupt:
                    print("\n[i] Prueba de micrófono cancelada.")
                    break
                except Exception as exc:
                    print(f"\n[ERROR] Error al capturar o clasificar: {exc}")
                    break

                try:
                    post = input(
                        "\n Opciones: [Enter] Probar de nuevo | 'g' Gráfico PCA | 'm' Mosaico de etapas | '0' Volver al menú: "
                    ).strip().lower()
                except (KeyboardInterrupt, EOFError):
                    break

                if post == "g" and res.get("vector_pca") is not None:
                    graficar_prediccion_en_pca(
                        paquete_modelo=paquete_modelo,
                        senal=res["senal_audio"],
                        sr=int(res["sample_rate"]),
                        resultado=res,
                        mostrar_ventana=True,
                    )
                    cont = input(
                        "\n Presioná [Enter] para hacer otra prueba de micrófono o '0' para volver al menú: "
                    ).strip()
                    if cont == "0":
                        break
                elif post == "m" and res.get("senal_audio") is not None:
                    cmd_det = res.get("comando_mas_votado") or res.get("comando") or "Voz"
                    ruta_mos = _generar_ruta_unica_mosaico_vivo(str(cmd_det))
                    crear_mosaico_etapas_audio(
                        senal=res["senal_audio"],
                        sr=int(res["sample_rate"]),
                        ruta_guardado=ruta_mos,
                        nombre_archivo="Prueba de micrófono en vivo",
                        etiqueta=str(cmd_det),
                        mostrar_ventana=True,
                    )
                    cont = input(
                        "\n Presioná [Enter] para hacer otra prueba de micrófono o '0' para volver al menú: "
                    ).strip()
                    if cont == "0":
                        break
                elif post in ("0", "q", "salir"):
                    break

        elif op == "3":
            if paquete_modelo is None:
                paquete_modelo = entrenar_o_cargar_modelo_voz(
                    directorio_base=directorio_base, k_vecinos=k_vecinos
                )
            _probar_archivo_existente_menu(paquete_modelo, directorio_base)

        elif op == "4":
            nuevo_k = input(
                f" Ingresá el nuevo número de vecinos K (impar recomendado, ej. 3, 5, 7) [Actual: {k_vecinos}]: "
            ).strip()
            if nuevo_k.isdigit() and 1 <= int(nuevo_k) <= 25:
                k_vecinos = int(nuevo_k)
                paquete_modelo = entrenar_modelo_completo(
                    directorio_base=directorio_base,
                    k_vecinos=k_vecinos,
                    guardar_cache=True,
                )
                print(f" [+] Modelo re-entrenado con K = {k_vecinos}.")
            else:
                print(" [!] Valor de K no válido.")

        elif op == "5":
            gv.menu_cambiar_microfono_sistema()

        elif op == "6":
            print("\n" + "-" * 72)
            print(" GENERAR MOSAICO COMPARATIVO DE ETAPAS DE AUDIO")
            print("-" * 72)
            print("   [1] Generar 1 mosaico representativo por comando (Reconocer, Identificar, Salir)")
            print("   [2] Generar mosaico de un archivo .wav específico (o número del metadata)")
            print("   [3] Generar mosaicos para TODO el dataset")
            print("   [0] Cancelar / Volver")
            sub_op = input("\n Seleccioná una opción [Default: 1]: ").strip()
            if not sub_op or sub_op == "1":
                generar_mosaicos_dataset(
                    directorio_base=directorio_base,
                    max_por_clase=1,
                    mostrar_ventana=True,
                )
            elif sub_op == "2":
                metas = gv.leer_registros_metadata(directorio_base)
                entrada_m = input(
                    f" Ingresá número de fila (1 a {len(metas)}) o ruta a archivo .wav [Default: 1]: "
                ).strip() or "1"
                if entrada_m.isdigit() and 1 <= int(entrada_m) <= len(metas):
                    elegido = metas[int(entrada_m) - 1]
                    ruta_m = gv.resolver_ruta_en_disco(directorio_base, elegido["ruta_audio"])
                    cmd_m = elegido.get("comando", "")
                    pers_m = elegido.get("persona", "")
                    lbl_m = f"{cmd_m} ({pers_m})" if pers_m else cmd_m
                else:
                    ruta_m = entrada_m
                    lbl_m = ""
                if os.path.isfile(ruta_m):
                    s_m, sr_m = gv.leer_wav_mono(ruta_m, normalizar_float=True)
                    crear_mosaico_etapas_audio(
                        senal=s_m,
                        sr=sr_m,
                        nombre_archivo=os.path.basename(ruta_m),
                        etiqueta=lbl_m,
                        mostrar_ventana=True,
                    )
                else:
                    print(f" [!] No se encontró el archivo: {ruta_m}")
            elif sub_op == "3":
                conf = input(
                    " ¿Seguro que deseás generar mosaicos para todo el dataset? [s/N]: "
                ).strip().lower()
                if conf in ("s", "si", "sí", "y"):
                    generar_mosaicos_dataset(
                        directorio_base=directorio_base,
                        max_por_clase=None,
                        mostrar_ventana=False,
                    )

        else:
            print(" [!] Opción no reconocida. Elegí entre 0 y 6.")


# ==============================================================================
# PUNTO DE ENTRADA CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clasificador de comandos de voz K-NN con análisis PCA y prueba en vivo."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DIRECTORIO_DATASET_DEFAULT,
        help=f"Carpeta del dataset de voz (default: {DIRECTORIO_DATASET_DEFAULT})",
    )
    parser.add_argument(
        "-k",
        "--vecinos",
        type=int,
        default=K_VECINOS_DEFAULT,
        help=f"Número de vecinos para K-NN (default: {K_VECINOS_DEFAULT})",
    )
    parser.add_argument(
        "--analizar",
        action="store_true",
        help="Ejecutar directamente el análisis estadístico y generar el gráfico PNG sin abrir menú",
    )
    parser.add_argument(
        "--mosaico",
        type=str,
        default=None,
        help="Ruta a un archivo .wav para generar directamente su mosaico comparativo de etapas",
    )
    parser.add_argument(
        "--mosaicos-dataset",
        type=int,
        nargs="?",
        const=1,
        default=None,
        help="Generar mosaicos comparativos para el dataset (por default 1 por clase, ej: --mosaicos-dataset 2)",
    )
    parser.add_argument(
        "--sin-ventana",
        action="store_true",
        help="Guardar los gráficos en disco sin abrir ventana interactiva de Matplotlib",
    )

    args = parser.parse_args()

    if args.mosaico:
        if not os.path.isfile(args.mosaico):
            print(f"[!] Archivo no encontrado: {args.mosaico}")
            sys.exit(1)
        s, sr = gv.leer_wav_mono(args.mosaico, normalizar_float=True)
        crear_mosaico_etapas_audio(
            senal=s,
            sr=sr,
            nombre_archivo=os.path.basename(args.mosaico),
            mostrar_ventana=not args.sin_ventana,
        )
        return

    if args.mosaicos_dataset is not None:
        generar_mosaicos_dataset(
            directorio_base=args.dataset_dir,
            max_por_clase=args.mosaicos_dataset if args.mosaicos_dataset > 0 else None,
            mostrar_ventana=not args.sin_ventana,
        )
        return

    if args.analizar:
        analizar_y_graficar_dataset(
            directorio_base=args.dataset_dir,
            k_vecinos=args.vecinos,
            mostrar_ventana=not args.sin_ventana,
        )
        return

    menu_interactivo(directorio_base=args.dataset_dir, k_inicial=args.vecinos)


if __name__ == "__main__":
    main()
