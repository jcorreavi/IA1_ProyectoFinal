#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Reconocimiento de Huellas Dactilares Sintéticas mediante K-Means y PCA
Trabajo Final - Inteligencia Artificial I (2026) - Ingeniería en Mecatrónica (UNCUYO)

Funciones principales:
1. Reutilización íntegra de `preprocesamiento.py` (sin recorte de ROI, conservando 275x400 px).
2. Extracción de vectores de características discriminantes invariantes a traslación y rotación:
   - Singularidades topológicas mediante Índice de Poincaré (Núcleo/Core y Delta) y conteo de
     crestas de Galton-Henry sobre la imagen realzada con Gabor.
   - Campo direccional intrínseco de orientación de crestas (cos 2θ, sin 2θ) alineado al eje
     principal del óvalo dactilar.
   - Descriptores de textura direccional de Gabor por franjas anatómicas.
   - Distribución espacial de minucias (Crossing Number sobre esqueleto de Zhang-Suen) y
     Momentos Invariantes de Hu.
3. Agrupamiento no supervisado K-Means (k = 3 para las identidades autorizadas) entrenado
   exclusivamente con el conjunto de entrenamiento autorizado (`dataset_huellas/Personas/`).
4. Métrica de distancia euclídea estandarizada y calibración experimental del umbral de
   aceptación/rechazo para discriminar huellas autorizadas frente a intrusas (`huellas_prueba/`).
5. Proyección por Componentes Principales (PCA 2D/3D), estadísticas de clasificación y funciones
   exportables listas para el Agente gráfico (`agente.py`).
"""

import argparse
import os
import re
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union

import cv2
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

import preprocesamiento as prep

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# CONFIGURACIÓN PREDETERMINADA
# ==============================================================================

DIRECTORIO_ENTRENAMIENTO_DEFAULT = os.path.join("dataset_huellas", "Personas")
DIRECTORIO_PRUEBA_DEFAULT = "huellas_prueba"
DIRECTORIO_RESULTADOS_DEFAULT = "resultados_huellas"

RUTA_MODELO_CACHE = os.path.join(
    DIRECTORIO_RESULTADOS_DEFAULT, "modelo_kmeans_huellas.joblib"
)
RUTA_GRAFICO_ANALISIS = os.path.join(
    DIRECTORIO_RESULTADOS_DEFAULT, "analisis_kmeans_pca_huellas.png"
)
RUTA_GRAFICO_EXPERIMENTO = os.path.join(
    DIRECTORIO_RESULTADOS_DEFAULT, "experimento_umbral_kmeans.png"
)

# Nota conceptual importante:
# `K_CLUSTERS_DEFAULT = 3` corresponde al número de clústeres (centroides) de K-Means
# para las 3 identidades sintéticas autorizadas. NO confundir con el parámetro K de
# vecinos más cercanos del clasificador supervisado K-NN usado en el módulo de voz.
K_CLUSTERS_DEFAULT = 3
FACTOR_SIGMA_UMBRAL_DEFAULT = 4.35
FACTOR_PISO_ESCALA = 0.075
VERSION_EXTRACTOR_HUELLAS = 2

MAPA_CARPETA_A_IDENTIDAD: Dict[str, str] = {
    "Persona_1": "Identidad Sintética #1",
    "Persona_2": "Identidad Sintética #2",
    "Persona_3": "Identidad Sintética #3",
}
IDENTIDADES_ORDEN: Tuple[str, ...] = (
    "Identidad Sintética #1",
    "Identidad Sintética #2",
    "Identidad Sintética #3",
)
ETIQUETA_NO_AUTORIZADO = "NO AUTORIZADO"
ETIQUETA_INTRUSO_REPORTE = "Intruso (No Aut.)"

COLORES_IDENTIDADES: Dict[str, str] = {
    "Identidad Sintética #1": "#1f77b4",  # Azul
    "Identidad Sintética #2": "#2ca02c",  # Verde
    "Identidad Sintética #3": "#ff7f0e",  # Naranja
    ETIQUETA_INTRUSO_REPORTE: "#d62728",  # Rojo
}
MARCADORES_IDENTIDADES: Dict[str, str] = {
    "Identidad Sintética #1": "o",
    "Identidad Sintética #2": "s",
    "Identidad Sintética #3": "^",
    ETIQUETA_INTRUSO_REPORTE: "X",
}


# ==============================================================================
# 1. ESCALADOR REGULARIZADO Y EXTRACCIÓN DE CARACTERÍSTICAS DACTILARES
# ==============================================================================

class EscaladorEstandarRegularizado(BaseEstimator, TransformerMixin):
    """
    Estandariza centrando en la media de entrenamiento y escalando cada característica
    según su dispersión local no supervisada entre los `m_vecinos` más cercanos dentro
    de `X` (sin utilizar etiquetas `y`), con cota inferior física (`factor_piso`).

    Esto preserva la resolución discriminante fina (ej. distinguir dos presillas del
    mismo tipo) incluso cuando el conjunto autorizado combina patrones heterogéneos de
    Galton-Henry / Vucetich (Arcos, Presillas Internas/Externas y Verticilos).
    """

    def __init__(self, factor_piso: float = FACTOR_PISO_ESCALA, m_vecinos: int = 8):
        self.factor_piso = factor_piso
        self.m_vecinos = m_vecinos

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "EscaladorEstandarRegularizado":
        X_arr = np.asarray(X, dtype=np.float32)
        n = X_arr.shape[0]
        self.mean_ = np.mean(X_arr, axis=0)

        sc0 = StandardScaler().fit(X_arr)
        piso0 = max(1e-4, 0.20 * float(np.median(sc0.scale_)))
        X0 = (X_arr - sc0.mean_) / np.maximum(sc0.scale_, piso0)

        d2 = np.sum((X0[:, None, :] - X0[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        m = min(self.m_vecinos, max(1, n - 1))
        nn_idx = np.argsort(d2, axis=1)[:, :m]

        diffs = X_arr[:, None, :] - X_arr[nn_idx]
        local_rms = np.sqrt(np.mean(diffs**2, axis=(0, 1)) / 2.0)

        pisos = np.full_like(local_rms, float(self.factor_piso))
        # Piso suave mayor en los 3 indicadores enteros de presencia (carga_core, pres_L, pres_R)
        for idx_disc in (0, 3, 12):
            if idx_disc < len(pisos):
                pisos[idx_disc] = max(pisos[idx_disc], 0.20)
        self.scale_ = np.maximum(local_rms, pisos)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        return (X_arr - self.mean_) / self.scale_


def estimar_marco_intrinseco_huella(
    peso_textura: np.ndarray, imagen_binaria: np.ndarray
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """
    Estima el centro de masa `(cx, cy)`, la inclinación del eje mayor `phi_rot` y
    las dispersiones principales `(std_min, std_maj)` de la huella activa a partir
    de su envolvente convexa depurada morfológicamente, SIN recortar la imagen
    (preserva W x H íntegros).

    Esto proporciona un sistema de referencia intrínseco al dedo que compensa las
    traslaciones y rotaciones operativas e ignora puntos aislados de ruido externo.
    """
    cresta_limpia = cv2.morphologyEx(
        (imagen_binaria < 128).astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    tex_limpia = cv2.morphologyEx(
        (peso_textura > 0.28).astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    mask = ((cresta_limpia > 0) | (tex_limpia > 0)).astype(np.uint8) * 255
    mask_open = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull_mask = np.zeros_like(mask_closed)
    if contours:
        contorno_mayor = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contorno_mayor)
        cv2.drawContours(hull_mask, [hull], -1, 255, -1)

    ys, xs = np.where(hull_mask > 0)
    if len(xs) < 100:
        h, w = imagen_binaria.shape[:2]
        return w / 2.0, h / 2.0, 0.0, 65.0, 95.0, hull_mask

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    dx = xs - cx
    dy = ys - cy
    cov = np.cov(np.vstack([dx, dy]))
    vals, vecs = np.linalg.eigh(cov)

    idx_maj = int(np.argmax(vals))
    idx_min = 1 - idx_maj
    v_maj = vecs[:, idx_maj]
    # Convención consistente: el eje longitudinal apunta hacia la base de la huella (+y)
    if v_maj[1] < 0:
        v_maj = -v_maj

    phi_rot = float(np.arctan2(-v_maj[0], v_maj[1]))
    std_maj = float(np.sqrt(max(vals[idx_maj], 1.0)))
    std_min = float(np.sqrt(max(vals[idx_min], 1.0)))
    return cx, cy, phi_rot, std_min, std_maj, hull_mask


def transformar_a_coordenadas_intrinsecas(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    cx: float,
    cy: float,
    phi_rot: float,
    std_min: float,
    std_maj: float,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Convierte coordenadas de píxel `(x, y)` al marco normalizado `(u, v)` del óvalo dactilar."""
    dx = x - cx
    dy = y - cy
    c, s = np.cos(phi_rot), np.sin(phi_rot)
    u = (dx * c + dy * s) / std_min
    v = (-dx * s + dy * c) / std_maj
    return u, v


def transformar_intrinseco_a_pixel(
    u: float,
    v: float,
    cx: float,
    cy: float,
    phi_rot: float,
    std_min: float,
    std_maj: float,
) -> Tuple[float, float]:
    """Transformación inversa: proyecta coordenadas canónicas `(u, v)` a píxeles `(x, y)`."""
    c, s = np.cos(phi_rot), np.sin(phi_rot)
    dx_can = u * std_min
    dy_can = v * std_maj
    x = cx + dx_can * c - dy_can * s
    y = cy + dx_can * s + dy_can * c
    return float(x), float(y)


def contar_crestas_galton_henry(
    imagen_gabor: np.ndarray,
    imagen_binaria: np.ndarray,
    p1_xy: Tuple[float, float],
    p2_xy: Tuple[float, float],
) -> float:
    """
    Cuenta el número de crestas que atraviesan el segmento recto que une dos puntos
    singulares (ej. Núcleo y Delta), evaluado sobre la señal realzada con filtros de
    Gabor para mantener inmunidad frente a micro-poros o cicatrices sintéticas.
    """
    dist = float(np.hypot(p2_xy[0] - p1_xy[0], p2_xy[1] - p1_xy[1]))
    if dist < 6.0:
        return 0.0

    n_pts = max(16, int(round(dist * 2.0)))
    xs = np.clip(
        np.linspace(p1_xy[0], p2_xy[0], n_pts), 0, imagen_binaria.shape[1] - 1
    )
    ys = np.clip(
        np.linspace(p1_xy[1], p2_xy[1], n_pts), 0, imagen_binaria.shape[0] - 1
    )

    perfil = cv2.remap(
        imagen_gabor.astype(np.float32),
        xs.reshape(1, -1).astype(np.float32),
        ys.reshape(1, -1).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
    ).reshape(-1)

    media_p = float(np.mean(perfil))
    es_cresta = (perfil < media_p).astype(np.int32)
    cruces = int(np.sum((es_cresta[1:] == 1) & (es_cresta[:-1] == 0)))
    return float(cruces)


def detectar_minucias_crossing_number(
    imagen_esqueleto: np.ndarray, hull_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detecta terminaciones (CN = 1) y bifurcaciones (CN = 3) mediante el algoritmo
    clásico de Crossing Number (Rutovitz) sobre el esqueleto topológico de 1 píxel
    generado por `preprocesamiento.py`, excluyendo el perímetro exterior para evitar
    falsas terminaciones de borde.
    """
    s = (imagen_esqueleto < 128).astype(np.int16)
    interior = (
        cv2.erode(
            hull_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)),
            iterations=1,
        )
        > 0
    )

    P2 = s[:-2, 1:-1]
    P3 = s[:-2, 2:]
    P4 = s[1:-1, 2:]
    P5 = s[2:, 2:]
    P6 = s[2:, 1:-1]
    P7 = s[2:, :-2]
    P8 = s[1:-1, :-2]
    P9 = s[:-2, :-2]
    P = s[1:-1, 1:-1]

    cn = (
        np.abs(P2 - P3)
        + np.abs(P3 - P4)
        + np.abs(P4 - P5)
        + np.abs(P5 - P6)
        + np.abs(P6 - P7)
        + np.abs(P7 - P8)
        + np.abs(P8 - P9)
        + np.abs(P9 - P2)
    ) // 2

    mask_in = interior[1:-1, 1:-1] & (P == 1)
    term_map = np.zeros_like(s, dtype=np.uint8)
    bif_map = np.zeros_like(s, dtype=np.uint8)
    term_map[1:-1, 1:-1] = (mask_in & (cn == 1)).astype(np.uint8)
    bif_map[1:-1, 1:-1] = (mask_in & (cn == 3)).astype(np.uint8)
    return term_map, bif_map


def detectar_singularidades_poincare(
    imagen_referencia: np.ndarray,
    hull_mask: np.ndarray,
    cx: float,
    cy: float,
    phi_rot: float,
    std_min: float,
    std_maj: float,
) -> Dict[str, object]:
    """
    Localiza de manera general los puntos singulares de cualquier clase dactilar de
    Galton-Henry / Vucetich mediante el Índice de Poincaré sobre el campo realzado:
      - Arcos (Arch)            : 0 núcleos (`carga_core = 0`), 0 deltas (sin respaldos ruidosos).
      - Presillas (Left / Right): 1 núcleo (`+1/2`) y 1 delta (`-1/2`, izquierdo o derecho).
      - Verticilos (Whorl)      : 1 espiral `+1` o 2 núcleos (`carga_core = 2`) y 2 deltas
                                  (`-1/2` izquierdo y `-1/2` derecho).
    Incluye enmascarado previo del escalón de borde exterior y aniquilación topológica
    de dipolos espurios de corta distancia (< 24 px) causados por ruido o cicatrices.
    """
    mask_hull_in = cv2.erode(
        hull_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    vals_in = imagen_referencia[mask_hull_in > 0]
    nivel_medio = float(np.mean(vals_in)) if len(vals_in) > 0 else 128.0
    lienzo = np.where(
        mask_hull_in > 0, imagen_referencia.astype(np.float32), nivel_medio
    )

    base = cv2.GaussianBlur(lienzo, (5, 5), 1.0)
    gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)

    mask_grad = (
        cv2.erode(
            hull_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    ).astype(np.float32)
    gx = gx * mask_grad
    gy = gy * mask_grad

    gxx = cv2.GaussianBlur(gx * gx, (31, 31), 6.5)
    gyy = cv2.GaussianBlur(gy * gy, (31, 31), 6.5)
    gxy = cv2.GaussianBlur(gx * gy, (31, 31), 6.5)

    step = 4
    vx = cv2.GaussianBlur((gxx - gyy)[::step, ::step], (7, 7), 1.5)
    vy = cv2.GaussianBlur((2.0 * gxy)[::step, ::step], (7, 7), 1.5)
    theta_s = 0.5 * np.arctan2(vy, vx) + (np.pi / 2.0)

    interior = cv2.erode(
        hull_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)),
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )[::step, ::step]

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    gh, gw = theta_s.shape
    poincare = np.zeros((gh, gw), dtype=np.float32)
    pts = [theta_s[1 + dy : gh - 1 + dy, 1 + dx : gw - 1 + dx] for dy, dx in offsets]
    suma = np.zeros((gh - 2, gw - 2), dtype=np.float32)
    for k in range(8):
        d = pts[(k + 1) % 8] - pts[k]
        d = (d + np.pi / 2.0) % np.pi - np.pi / 2.0
        suma += d
    poincare[1:-1, 1:-1] = suma / np.pi

    def _extraer_blobs(mask_bin: np.ndarray, signo: float) -> List[Dict[str, object]]:
        num_labels, labels = cv2.connectedComponents(
            mask_bin.astype(np.uint8), connectivity=8
        )
        blobs: List[Dict[str, object]] = []
        for lbl in range(1, num_labels):
            ys, xs = np.where(labels == lbl)
            vals = signo * poincare[ys, xs]
            fuerza = float(np.sum(vals))
            pico = float(np.max(vals))
            bx = float(np.average(xs, weights=vals) * step)
            by = float(np.average(ys, weights=vals) * step)
            blobs.append({"xy": (bx, by), "fuerza": fuerza, "pico": pico})
        blobs.sort(key=lambda b: float(b["fuerza"]), reverse=True)
        fusionados: List[Dict[str, object]] = []
        for b in blobs:
            agregado = False
            b_xy = b["xy"]
            for f in fusionados:
                f_xy = f["xy"]
                if np.hypot(b_xy[0] - f_xy[0], b_xy[1] - f_xy[1]) < 20.0:
                    wt = float(f["fuerza"]) + float(b["fuerza"])
                    f["xy"] = (
                        (f_xy[0] * float(f["fuerza"]) + b_xy[0] * float(b["fuerza"])) / wt,
                        (f_xy[1] * float(f["fuerza"]) + b_xy[1] * float(b["fuerza"])) / wt,
                    )
                    f["fuerza"] = wt
                    f["pico"] = max(float(f["pico"]), float(b["pico"]))
                    agregado = True
                    break
            if not agregado:
                fusionados.append(dict(b))
        return fusionados[:2]

    cores_blobs = _extraer_blobs((poincare > 0.70) & (interior > 0), signo=+1.0)
    deltas_blobs = _extraer_blobs((poincare < -0.70) & (interior > 0), signo=-1.0)

    # Aniquilación topológica de dipolos espurios (< 24 px entre +1/2 y -1/2)
    cores_filtrados = []
    deltas_usados = set()
    for cb in cores_blobs:
        es_dipolo = False
        cb_xy = cb["xy"]
        for j, db in enumerate(deltas_blobs):
            if j in deltas_usados:
                continue
            db_xy = db["xy"]
            if np.hypot(cb_xy[0] - db_xy[0], cb_xy[1] - db_xy[1]) < 24.0:
                deltas_usados.add(j)
                es_dipolo = True
                break
        if not es_dipolo:
            cores_filtrados.append(cb)
    deltas_blobs = [db for j, db in enumerate(deltas_blobs) if j not in deltas_usados]
    cores_blobs = cores_filtrados

    carga_core = 0
    for cb in cores_blobs:
        carga_core += 2 if float(cb["pico"]) > 1.50 else 1
    carga_core = min(2, carga_core)

    cores_xy: List[Tuple[float, float]] = [cb["xy"] for cb in cores_blobs]
    if cores_xy:
        tiene_core = True
        pesos_c = [float(cb["fuerza"]) for cb in cores_blobs]
        core_x = float(np.average([p[0] for p in cores_xy], weights=pesos_c))
        core_y = float(np.average([p[1] for p in cores_xy], weights=pesos_c))
    else:
        tiene_core = False
        core_x, core_y = float(cx), float(cy)

    deltas_info: List[Dict[str, object]] = []
    for db in deltas_blobs:
        dx, dy = db["xy"]
        du, dv = transformar_a_coordenadas_intrinsecas(
            dx, dy, cx, cy, phi_rot, std_min, std_maj
        )
        deltas_info.append(
            {
                "xy": (float(dx), float(dy)),
                "uv": (float(du), float(dv)),
                "fuerza": float(db["fuerza"]),
            }
        )
    deltas_info.sort(key=lambda d: d["uv"][0])

    n_deltas = len(deltas_info)
    if not tiene_core and n_deltas == 0:
        tipo_topo = "Arco"
    elif n_deltas >= 2 or carga_core >= 2:
        tipo_topo = "Verticilo"
    elif n_deltas == 1:
        du0 = float(deltas_info[0]["uv"][0])
        if du0 < -0.18:
            tipo_topo = "Presilla Derecha"
        elif du0 > 0.18:
            tipo_topo = "Presilla Izquierda"
        else:
            tipo_topo = "Arco en Tienda"
    else:
        tipo_topo = "Presilla"

    return {
        "tiene_core": tiene_core,
        "carga_core": float(carga_core),
        "core_xy": (core_x, core_y),
        "cores_xy": cores_xy,
        "deltas": deltas_info,
        "tipo_topologico": tipo_topo,
    }


def extraer_caracteristicas_huella(
    etapas: Dict[str, np.ndarray],
    retornar_info_geometrica: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, object]]]:
    """
    Representa una huella preprocesada mediante un vector discriminante de 55
    características biométricas invariantes a traslación y rotación, compatible con
    cualquier familia dactilar (Arcos, Presillas Internas/Externas y Verticilos):
      - [0..2]   (3) : Carga topológica de núcleo (0/1/2) y coordenadas `(core_u, core_v)`.
      - [3..20]  (18): Canales laterales suaves Izquierdo (`u < 0`) y Derecho (`u > 0`)
                       para Deltas (presencia, posición intrínseca, vector Core->Delta,
                       distancia escalada, cos/sin angular y conteo de crestas Galton-Henry).
      - [21..23] (3) : Perfiles longitudinales de densidad de crestas (`u = -0.45, 0, +0.45`).
      - [24..47] (24): Campo de orientación intrínseco de crestas (`cos 2(θ-φ)`,
                       `sin 2(θ-φ)`) ponderado por coherencia en grilla elíptica 4x3.
      - [48..50] (3) : Descriptores de textura direccional de Gabor por franjas.
      - [51..54] (4) : Centroide intrínseco de bifurcaciones (`CN=3` en esqueleto) y
                       primeros 2 Momentos Invariantes de Hu del mapa de singularidad.
    """
    img_suav = etapas["sin_ruido"]
    img_gabor = etapas["gabor"]
    img_bin = etapas["binarizada"]
    img_skel = etapas["esqueleto"]

    theta_denso, peso_tex = prep.estimar_orientacion_y_coherencia(img_suav)
    cx, cy, phi_rot, std_min, std_maj, hull = estimar_marco_intrinseco_huella(
        peso_tex, img_bin
    )
    escala_ref = float(np.sqrt(std_min * std_maj))

    # 1. Singularidades de Poincaré multi-clase y conteo de crestas Galton-Henry
    sing = detectar_singularidades_poincare(
        img_gabor, hull, cx, cy, phi_rot, std_min, std_maj
    )
    core_x, core_y = sing["core_xy"]
    tiene_core = bool(sing["tiene_core"])
    if tiene_core:
        cu_raw, cv_raw = transformar_a_coordenadas_intrinsecas(
            core_x, core_y, cx, cy, phi_rot, std_min, std_maj
        )
        core_u, core_v = float(cu_raw), float(cv_raw)
    else:
        core_u, core_v = 0.0, 0.0

    pres_L = dLu = dLv = duL = dvL = distL = cosL = sinL = crL = 0.0
    pres_R = dRu = dRv = duR = dvR = distR = cosR = sinR = crR = 0.0
    crestas_por_delta: List[int] = []
    deltas_xy_lista: List[Tuple[float, float]] = []

    for d_item in sing["deltas"]:
        dx, dy = d_item["xy"]
        du, dv = d_item["uv"]
        deltas_xy_lista.append((dx, dy))

        # Partición suave de la unidad centrada en u = 0 (evita saltos discretos en Arcos en Tienda)
        z = float(np.clip(du / 0.08, -30.0, 30.0))
        w_R = 1.0 / (1.0 + np.exp(-z))
        w_L = 1.0 - w_R

        rel_u = du - core_u
        rel_v = dv - core_v
        ang = float(np.arctan2(rel_v, rel_u))
        dist_n = float(np.hypot(dx - core_x, dy - core_y)) / (escala_ref + 1e-6)
        cr_raw = (
            contar_crestas_galton_henry(img_gabor, img_bin, (core_x, core_y), (dx, dy))
            if tiene_core
            else 0.0
        )
        crestas_por_delta.append(int(round(cr_raw)))
        cr_n = cr_raw / 15.0

        pres_L += w_L
        dLu += w_L * du
        dLv += w_L * dv
        duL += w_L * rel_u
        dvL += w_L * rel_v
        distL += w_L * dist_n
        cosL += w_L * float(np.cos(ang))
        sinL += w_L * float(np.sin(ang))
        crL += w_L * cr_n

        pres_R += w_R
        dRu += w_R * du
        dRv += w_R * dv
        duR += w_R * rel_u
        dvR += w_R * rel_v
        distR += w_R * dist_n
        cosR += w_R * float(np.cos(ang))
        sinR += w_R * float(np.sin(ang))
        crR += w_R * cr_n

    caracteristicas: List[float] = [
        float(sing["carga_core"]),
        core_u,
        core_v,
        pres_L,
        dLu,
        dLv,
        duL,
        dvL,
        distL,
        cosL,
        sinL,
        crL,
        pres_R,
        dRu,
        dRv,
        duR,
        dvR,
        distR,
        cosR,
        sinR,
        crR,
    ]

    # Perfiles longitudinales de densidad de crestas (discriminan incluso Arcos sin singularidades)
    for u_col in (-0.45, 0.0, 0.45):
        p_ini = transformar_intrinseco_a_pixel(
            u_col, -0.85, cx, cy, phi_rot, std_min, std_maj
        )
        p_fin = transformar_intrinseco_a_pixel(
            u_col, +0.85, cx, cy, phi_rot, std_min, std_maj
        )
        dens = contar_crestas_galton_henry(img_gabor, img_bin, p_ini, p_fin) / 22.0
        caracteristicas.append(float(dens))

    # 2. Orientación local de crestas en grilla canónica 4x3 alineada al dedo
    h, w = img_suav.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    uu, vv = transformar_a_coordenadas_intrinsecas(
        xx, yy, cx, cy, phi_rot, std_min, std_maj
    )

    theta_rel = theta_denso - phi_rot
    cos2t = np.cos(2.0 * theta_rel)
    sin2t = np.sin(2.0 * theta_rel)

    u_edges = np.linspace(-1.15, 1.15, 4)
    v_edges = np.linspace(-1.25, 1.25, 5)
    for i in range(4):
        for j in range(3):
            celda = (
                (vv >= v_edges[i])
                & (vv < v_edges[i + 1])
                & (uu >= u_edges[j])
                & (uu < u_edges[j + 1])
            )
            w_c = peso_tex[celda]
            sw = float(np.sum(w_c)) + 1e-6
            caracteristicas.append(float(np.sum(cos2t[celda] * w_c) / sw))
            caracteristicas.append(float(np.sum(sin2t[celda] * w_c) / sw))

    # 3. Descriptores de textura direccional sobre la respuesta de Gabor
    grad_gabor = (
        np.hypot(
            cv2.Sobel(img_gabor.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(img_gabor.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3),
        )
        / 255.0
    )
    for i in range(3):
        franja = (
            (vv >= -1.2 + i * 0.8)
            & (vv < -1.2 + (i + 1) * 0.8)
            & (np.abs(uu) < 1.1)
        )
        w_f = peso_tex[franja]
        sw = float(np.sum(w_f)) + 1e-6
        caracteristicas.append(float(np.sum(grad_gabor[franja] * w_f) / sw))

    # 4. Distribución espacial de minucias en el esqueleto + Momentos de Hu
    term_map, bif_map = detectar_minucias_crossing_number(img_skel, hull)
    by, bx = np.where(bif_map > 0)
    if len(bx) > 3:
        bu, bv = transformar_a_coordenadas_intrinsecas(
            bx.astype(np.float32),
            by.astype(np.float32),
            cx,
            cy,
            phi_rot,
            std_min,
            std_maj,
        )
        caracteristicas.extend([float(np.mean(bu)), float(np.mean(bv))])
    else:
        caracteristicas.extend([0.0, 0.0])

    int_soft = cv2.GaussianBlur(
        (
            cv2.erode(
                hull,
                np.ones((25, 25), np.uint8),
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        ).astype(np.float32),
        (15, 15),
        4.0,
    )
    mapa_sing = np.clip((1.0 - peso_tex) * int_soft, 0.0, 1.0)
    hu = cv2.HuMoments(cv2.moments(mapa_sing)).flatten()
    caracteristicas.append(float(-np.sign(hu[0]) * np.log10(np.abs(hu[0]) + 1e-12)))
    caracteristicas.append(float(-np.sign(hu[1]) * np.log10(np.abs(hu[1]) + 1e-12)))

    vector = np.asarray(caracteristicas, dtype=np.float32)

    if not retornar_info_geometrica:
        return vector

    delta_principal_xy = deltas_xy_lista[0] if deltas_xy_lista else (cx, cy)
    conteo_total = int(sum(crestas_por_delta))
    info_geo: Dict[str, object] = {
        "cx": cx,
        "cy": cy,
        "phi_grados": float(np.degrees(phi_rot)),
        "std_min": std_min,
        "std_maj": std_maj,
        "tiene_core": tiene_core,
        "carga_core": int(round(float(sing["carga_core"]))),
        "core_xy": (core_x, core_y),
        "cores_xy": sing["cores_xy"],
        "delta_xy": delta_principal_xy,
        "deltas_xy": deltas_xy_lista,
        "crestas_por_delta": crestas_por_delta,
        "conteo_crestas": conteo_total,
        "tipo_topologico": sing["tipo_topologico"],
        "num_terminaciones": int(np.sum(term_map > 0)),
        "num_bifurcaciones": int(np.sum(bif_map > 0)),
    }
    return vector, info_geo


# ==============================================================================
# 2. CARGA DE DATASETS, ENTRENAMIENTO K-MEANS (k=3) Y CALIBRACIÓN DE UMBRAL
# ==============================================================================

def _normalizar_etiqueta_carpeta(nombre_subcarpeta: str) -> str:
    """Traduce el nombre de carpeta (`Persona_1`, `intrusos`, etc.) a la etiqueta oficial."""
    if nombre_subcarpeta in MAPA_CARPETA_A_IDENTIDAD:
        return MAPA_CARPETA_A_IDENTIDAD[nombre_subcarpeta]
    if "intrus" in nombre_subcarpeta.lower() or "no_aut" in nombre_subcarpeta.lower():
        return ETIQUETA_INTRUSO_REPORTE
    return nombre_subcarpeta


def listar_muestras_por_subcarpeta(
    directorio_base: str,
) -> List[Tuple[str, str, str]]:
    """
    Recorre las subcarpetas directas de `directorio_base` y devuelve una lista de
    tuplas `(ruta_absoluta, nombre_subcarpeta, etiqueta_legible)`.
    """
    muestras: List[Tuple[str, str, str]] = []
    if not os.path.isdir(directorio_base):
        return muestras

    # 1. Archivos directos en la raíz si la carpeta es plana (ej. huellas scrambled)
    for archivo in sorted(os.listdir(directorio_base)):
        ruta_arch = os.path.join(directorio_base, archivo)
        if os.path.isfile(ruta_arch) and archivo.lower().endswith(prep.EXTENSIONES_VALIDAS):
            ruta_abs = os.path.abspath(ruta_arch)
            muestras.append((ruta_abs, ".", "Desconocido"))

    # 2. Archivos en subcarpetas (Persona_1, Persona_2, intrusos, etc.)
    for sub in sorted(os.listdir(directorio_base)):
        ruta_sub = os.path.join(directorio_base, sub)
        if not os.path.isdir(ruta_sub) or sub.startswith("."):
            continue
        etiqueta = _normalizar_etiqueta_carpeta(sub)
        for archivo in sorted(os.listdir(ruta_sub)):
            if archivo.lower().endswith(prep.EXTENSIONES_VALIDAS):
                ruta_abs = os.path.abspath(os.path.join(ruta_sub, archivo))
                muestras.append((ruta_abs, sub, etiqueta))
    return muestras


def extraer_matriz_huellas(
    lista_muestras: List[Tuple[str, str, str]],
    mostrar_progreso: bool = True,
    titulo_lote: str = "huellas",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Ejecuta `preprocesar_imagen_completa` y `extraer_caracteristicas_huella` sobre
    cada imagen de `lista_muestras`, garantizando un pipeline idéntico entre el
    entrenamiento y la inferencia en tiempo real del Agente.
    """
    if not lista_muestras:
        raise RuntimeError(f"No se encontraron imágenes para procesar en {titulo_lote}.")

    if mostrar_progreso:
        print(
            f"[*] Preprocesando y extrayendo características de {len(lista_muestras)} "
            f"imágenes ({titulo_lote})..."
        )

    vectores: List[np.ndarray] = []
    etiquetas: List[str] = []
    carpetas: List[str] = []
    rutas: List[str] = []

    for ruta_abs, sub, etiq in lista_muestras:
        img = cv2.imread(ruta_abs, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[!] Advertencia: no se pudo leer '{ruta_abs}', omitiendo.")
            continue
        etapas = prep.preprocesar_imagen_completa(img)
        vec = extraer_caracteristicas_huella(etapas, retornar_info_geometrica=False)
        vectores.append(vec)
        etiquetas.append(etiq)
        carpetas.append(sub)
        rutas.append(ruta_abs)

    X = np.vstack(vectores).astype(np.float32)
    return X, np.asarray(etiquetas), np.asarray(carpetas), rutas


def entrenar_modelo_huellas(
    directorio_entrenamiento: str = DIRECTORIO_ENTRENAMIENTO_DEFAULT,
    k_clusters: int = K_CLUSTERS_DEFAULT,
    factor_sigma_umbral: float = FACTOR_SIGMA_UMBRAL_DEFAULT,
    guardar_cache: bool = True,
    mostrar_progreso: bool = True,
) -> Dict[str, object]:
    """
    Entrena el agrupamiento no supervisado K-Means (`k = 3`) exclusivamente con las
    muestras de las 3 identidades autorizadas en `directorio_entrenamiento`, asigna
    cada centroide a su identidad sintética correspondiente, calibra el umbral
    estadístico de distancia y ajusta el subespacio PCA para visualización.
    """
    muestras_train = [
        m
        for m in listar_muestras_por_subcarpeta(directorio_entrenamiento)
        if m[1] in MAPA_CARPETA_A_IDENTIDAD
    ]
    if not muestras_train:
        raise RuntimeError(
            f"No se hallaron muestras autorizadas en '{directorio_entrenamiento}'."
        )

    X_train, y_train, carpetas_train, rutas_train = extraer_matriz_huellas(
        muestras_train,
        mostrar_progreso=mostrar_progreso,
        titulo_lote="entrenamiento autorizado",
    )

    escalador = EscaladorEstandarRegularizado(factor_piso=FACTOR_PISO_ESCALA)
    X_train_std = escalador.fit_transform(X_train)

    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=30)
    clusters_train = kmeans.fit_predict(X_train_std)

    # Asociar cada centroide descubierto por K-Means a la identidad autorizada mayoritaria
    # (Asignación óptima de Kuhn-Munkres cuando k_clusters == 3; mayoría simple para otros k)
    identidades_unicas = list(IDENTIDADES_ORDEN)
    mapa_cluster_a_identidad: Dict[int, str] = {}
    mapa_cluster_a_carpeta: Dict[int, str] = {}
    inv_identidad_carpeta = {v: k for k, v in MAPA_CARPETA_A_IDENTIDAD.items()}

    if k_clusters == len(identidades_unicas):
        matriz_costo = np.zeros((k_clusters, len(identidades_unicas)), dtype=int)
        for ci in range(k_clusters):
            for cj, id_str in enumerate(identidades_unicas):
                matriz_costo[ci, cj] = -int(
                    np.sum((clusters_train == ci) & (y_train == id_str))
                )
        filas_ind, cols_ind = linear_sum_assignment(matriz_costo)
        for ci, cj in zip(filas_ind, cols_ind):
            id_asignada = identidades_unicas[int(cj)]
            mapa_cluster_a_identidad[int(ci)] = id_asignada
            mapa_cluster_a_carpeta[int(ci)] = inv_identidad_carpeta[id_asignada]
    else:
        for ci in range(k_clusters):
            mask_ci = clusters_train == ci
            if np.any(mask_ci):
                valores, conteos = np.unique(y_train[mask_ci], return_counts=True)
                id_ganadora = str(valores[int(np.argmax(conteos))])
            else:
                id_ganadora = identidades_unicas[ci % len(identidades_unicas)]
            mapa_cluster_a_identidad[int(ci)] = id_ganadora
            mapa_cluster_a_carpeta[int(ci)] = inv_identidad_carpeta.get(
                id_ganadora, "Persona_1"
            )

    # Distancias euclídeas equiparadas por compacidad radial no supervisada de cada clúster
    dist_matriz_raw = kmeans.transform(X_train_std)
    radios_c = np.ones(k_clusters, dtype=np.float32)
    dist_propia_raw = dist_matriz_raw[np.arange(len(clusters_train)), clusters_train]
    for ci in range(k_clusters):
        vals_ci = dist_propia_raw[clusters_train == ci]
        if len(vals_ci) >= 3:
            radios_c[ci] = float(np.percentile(vals_ci, 75))
    radio_medio = float(np.mean(radios_c))
    escala_cluster = np.maximum(radios_c / max(radio_medio, 1e-4), 0.65).astype(np.float32)

    dist_matriz_train = dist_matriz_raw / escala_cluster[None, :]
    dist_min_train = np.min(dist_matriz_train, axis=1)

    media_dist_train = float(np.mean(dist_min_train))
    std_dist_train = float(np.std(dist_min_train))
    mediana_dist_train = float(np.median(dist_min_train))
    sigma_mad_train = float(
        1.4826 * np.median(np.abs(dist_min_train - mediana_dist_train))
    )
    max_dist_train = float(np.max(dist_min_train))
    p95_dist_train = float(np.percentile(dist_min_train, 95))

    # Calibración estadística robusta (MAD + P95) sobre el conjunto autorizado:
    # evita que una sola impresión distorsionada infle artificialmente el umbral τ
    umbral_distancia = float(
        max(
            mediana_dist_train + factor_sigma_umbral * sigma_mad_train,
            p95_dist_train * 1.15,
        )
    )

    # Ajuste de Análisis por Componentes Principales (PCA) sobre X_train_std
    n_comp_max = min(15, X_train_std.shape[0], X_train_std.shape[1])
    pca_modelo = PCA(n_components=n_comp_max, random_state=42)
    X_train_pca = pca_modelo.fit_transform(X_train_std)
    centroides_pca = pca_modelo.transform(kmeans.cluster_centers_)

    firmas_archivos = [
        (
            str(carpetas_train[idx_r]),
            os.path.basename(r),
            os.path.getsize(r),
            int(os.path.getmtime(r)),
        )
        for idx_r, r in enumerate(rutas_train)
        if os.path.isfile(r)
    ]

    paquete_modelo: Dict[str, object] = {
        "version_extractor": VERSION_EXTRACTOR_HUELLAS,
        "escalador": escalador,
        "kmeans": kmeans,
        "escala_cluster": escala_cluster,
        "k_clusters": k_clusters,
        "mapa_cluster_a_identidad": mapa_cluster_a_identidad,
        "mapa_cluster_a_carpeta": mapa_cluster_a_carpeta,
        "umbral_distancia": umbral_distancia,
        "factor_sigma_umbral": factor_sigma_umbral,
        "estadisticas_dist_train": {
            "media": media_dist_train,
            "mediana": mediana_dist_train,
            "std": std_dist_train,
            "sigma_mad": sigma_mad_train,
            "p95": p95_dist_train,
            "max": max_dist_train,
            "min": float(np.min(dist_min_train)),
        },
        "X_train": X_train,
        "X_train_std": X_train_std,
        "y_train": y_train,
        "carpetas_train": carpetas_train,
        "rutas_train": rutas_train,
        "clusters_train": clusters_train,
        "dist_min_train": dist_min_train,
        "pca_modelo": pca_modelo,
        "X_train_pca": X_train_pca,
        "centroides_pca": centroides_pca,
        "num_imagenes_entrenamiento": len(rutas_train),
        "firmas_entrenamiento": firmas_archivos,
    }

    if guardar_cache and k_clusters == K_CLUSTERS_DEFAULT:
        os.makedirs(DIRECTORIO_RESULTADOS_DEFAULT, exist_ok=True)
        joblib.dump(paquete_modelo, RUTA_MODELO_CACHE)

    return paquete_modelo


def entrenar_o_cargar_modelo_huellas(
    directorio_entrenamiento: str = DIRECTORIO_ENTRENAMIENTO_DEFAULT,
    k_clusters: int = K_CLUSTERS_DEFAULT,
    forzar_reentrenamiento: bool = False,
) -> Dict[str, object]:
    """
    Carga el modelo K-Means de huellas desde `resultados_huellas/modelo_kmeans_huellas.joblib`
    en milisegundos si está actualizado respecto de las imágenes en `dataset_huellas/Personas/`;
    en caso contrario lo entrena y guarda automáticamente.
    Ideal para invocar al arranque de `agente.py`.
    """
    muestras_actuales = [
        m
        for m in listar_muestras_por_subcarpeta(directorio_entrenamiento)
        if m[1] in MAPA_CARPETA_A_IDENTIDAD
    ]
    firmas_actuales = [
        (
            str(m[1]),
            os.path.basename(m[0]),
            os.path.getsize(m[0]),
            int(os.path.getmtime(m[0])),
        )
        for m in muestras_actuales
        if os.path.isfile(m[0])
    ]

    if not forzar_reentrenamiento and os.path.isfile(RUTA_MODELO_CACHE):
        try:
            main_mod = sys.modules.get("__main__")
            if main_mod is not None and not hasattr(
                main_mod, "EscaladorEstandarRegularizado"
            ):
                setattr(
                    main_mod,
                    "EscaladorEstandarRegularizado",
                    EscaladorEstandarRegularizado,
                )
            paquete = joblib.load(RUTA_MODELO_CACHE)
            if (
                paquete.get("version_extractor") == VERSION_EXTRACTOR_HUELLAS
                and paquete.get("k_clusters") == k_clusters
                and paquete.get("firmas_entrenamiento") == firmas_actuales
            ):
                return paquete
        except Exception:
            pass

    return entrenar_modelo_huellas(
        directorio_entrenamiento=directorio_entrenamiento,
        k_clusters=k_clusters,
        guardar_cache=(k_clusters == K_CLUSTERS_DEFAULT),
    )


# ==============================================================================
# 3. EVALUACIÓN DE UNA NUEVA HUELLA (FUNCIÓN PRINCIPAL PARA EL AGENTE)
# ==============================================================================

def clasificar_vector_huella(
    vector_x: np.ndarray,
    paquete_modelo: Dict[str, object],
    umbral_personalizado: Optional[float] = None,
) -> Dict[str, object]:
    """
    Evalúa un vector de 55 características frente a los centroides K-Means y aplica
    el criterio de aceptación/rechazo por umbral de distancia euclídea equiparada.
    """
    escalador: EscaladorEstandarRegularizado = paquete_modelo["escalador"]
    kmeans: KMeans = paquete_modelo["kmeans"]
    pca_modelo: PCA = paquete_modelo["pca_modelo"]
    mapa_id: Dict[int, str] = paquete_modelo["mapa_cluster_a_identidad"]
    mapa_carp: Dict[int, str] = paquete_modelo["mapa_cluster_a_carpeta"]
    umbral = (
        float(umbral_personalizado)
        if umbral_personalizado is not None
        else float(paquete_modelo["umbral_distancia"])
    )

    x_2d = np.asarray(vector_x, dtype=np.float32).reshape(1, -1)
    x_std = escalador.transform(x_2d)
    escala_cluster = np.asarray(
        paquete_modelo.get("escala_cluster", np.ones(kmeans.n_clusters, dtype=np.float32)),
        dtype=np.float32,
    )
    distancias = (kmeans.transform(x_std)[0]) / escala_cluster

    cluster_ganador = int(np.argmin(distancias))
    distancia_min = float(distancias[cluster_ganador])
    identidad_cercana = mapa_id[cluster_ganador]
    carpeta_cercana = mapa_carp[cluster_ganador]

    autorizado = bool(distancia_min <= umbral)
    estado_acceso = "AUTORIZADO" if autorizado else "NO AUTORIZADO"
    identidad_final = identidad_cercana if autorizado else ETIQUETA_NO_AUTORIZADO

    distancias_por_identidad_raw: Dict[str, float] = {}
    for c_idx, d_val in enumerate(distancias):
        etiqueta_c = mapa_id[int(c_idx)]
        if (
            etiqueta_c not in distancias_por_identidad_raw
            or float(d_val) < distancias_por_identidad_raw[etiqueta_c]
        ):
            distancias_por_identidad_raw[etiqueta_c] = float(d_val)

    distancias_por_identidad: Dict[str, float] = {
        et: distancias_por_identidad_raw[et]
        for et in IDENTIDADES_ORDEN
        if et in distancias_por_identidad_raw
    }
    for et, dv in distancias_por_identidad_raw.items():
        if et not in distancias_por_identidad:
            distancias_por_identidad[et] = dv

    coord_pca = pca_modelo.transform(x_std)[0]

    return {
        "autorizado": autorizado,
        "estado_acceso": estado_acceso,
        "identidad": identidad_final,
        "identidad_cercana": identidad_cercana,
        "persona_carpeta": carpeta_cercana if autorizado else "Intruso",
        "cluster_idx": cluster_ganador,
        "distancia_min": distancia_min,
        "umbral": umbral,
        "margen_umbral": float(umbral - distancia_min),
        "distancias_clusters": [float(d) for d in distancias],
        "distancias_por_identidad": distancias_por_identidad,
        "vector_pca": coord_pca,
    }


def evaluar_huella(
    imagen_o_ruta: Union[str, np.ndarray],
    paquete_modelo: Optional[Dict[str, object]] = None,
    umbral_personalizado: Optional[float] = None,
    incluir_mosaico: bool = True,
) -> Dict[str, object]:
    """
    ==========================================================================
    FUNCIÓN PRINCIPAL EXPORTABLE PARA EL AGENTE FINAL (`agente.py`)
    ==========================================================================
    Recibe la ruta de una imagen de huella (o una matriz NumPy), ejecuta el
    preprocesamiento completo sin recorte de ROI (`preprocesamiento.py`), extrae
    el vector de características y determina mediante K-Means (k=3) + umbral:
      - `estado_acceso`: "AUTORIZADO" o "NO AUTORIZADO" (para el comando «Reconocer»).
      - `identidad`    : "Identidad Sintética #1/#2/#3" si fue autorizada (para «Identificar»).
      - `distancia_min`, `umbral`, `etapas`, `mosaico_etapas`, `vector_pca`, `info_geometrica`.
    """
    if paquete_modelo is None:
        paquete_modelo = entrenar_o_cargar_modelo_huellas()

    nombre_archivo = ""
    ruta_origen = None
    if isinstance(imagen_o_ruta, str):
        ruta_origen = os.path.abspath(imagen_o_ruta)
        nombre_archivo = os.path.basename(ruta_origen)
        img = cv2.imread(ruta_origen, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"No se pudo abrir la imagen de huella: '{imagen_o_ruta}'"
            )
    else:
        img = imagen_o_ruta

    etapas = prep.preprocesar_imagen_completa(img)
    vector_x, info_geo = extraer_caracteristicas_huella(
        etapas, retornar_info_geometrica=True
    )
    resultado = clasificar_vector_huella(
        vector_x=vector_x,
        paquete_modelo=paquete_modelo,
        umbral_personalizado=umbral_personalizado,
    )

    resultado["ruta_imagen"] = ruta_origen
    resultado["nombre_archivo"] = nombre_archivo
    resultado["etapas"] = etapas
    resultado["vector_caracteristicas"] = vector_x
    resultado["info_geometrica"] = info_geo
    if incluir_mosaico:
        resultado["mosaico_etapas"] = prep.crear_mosaico_comparativo(
            etapas, nombre_archivo=nombre_archivo or "Muestra evaluada"
        )
    return resultado


# ==============================================================================
# 4. GRÁFICOS INDIVIDUALES Y ESTADÍSTICOS (INFORME + CARPETAS DEL AGENTE)
# ==============================================================================

def _dibujar_elipse_confianza(
    ax: plt.Axes, points_xy: np.ndarray, color: str, n_std: float = 2.0
) -> None:
    """Dibuja una elipse de covarianza 2D alrededor de las muestras de una identidad."""
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


def crear_imagen_huella_anotada(resultado_huella: Dict[str, object]) -> np.ndarray:
    """
    Genera una imagen BGR (mismas dimensiones 275x400 px) superponiendo sobre la
    huella realzada con Gabor las singularidades realmente presentes según la clase
    dactilar:
      - Arco     : centroide intrínseco (+) e indicación "Arco (sin Delta)".
      - Presilla : Core, Delta y segmento de conteo de crestas Galton-Henry.
      - Verticilo: Core(s), ambos Deltas (Izq./Der.) y ambos conteos de crestas.
    """
    gabor = resultado_huella["etapas"]["gabor"]
    vis = cv2.cvtColor(gabor, cv2.COLOR_GRAY2BGR)
    info = resultado_huella["info_geometrica"]

    cx, cy = int(round(info["cx"])), int(round(info["cy"]))
    tiene_core = bool(info.get("tiene_core", True))
    core_x, core_y = (
        int(round(info["core_xy"][0])),
        int(round(info["core_xy"][1])),
    )
    deltas_xy: List[Tuple[float, float]] = list(
        info.get("deltas_xy") or ([info["delta_xy"]] if tiene_core else [])
    )

    # Centroide del óvalo (+)
    cv2.drawMarker(
        vis, (cx, cy), (180, 105, 30), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA
    )

    # Líneas de conteo Galton-Henry hacia cada Delta detectado
    if tiene_core and deltas_xy:
        for dx_f, dy_f in deltas_xy:
            dx_i, dy_i = int(round(dx_f)), int(round(dy_f))
            cv2.line(vis, (core_x, core_y), (dx_i, dy_i), (0, 165, 255), 2, cv2.LINE_AA)

    # Dibujar Core(s) solo si la topología presenta núcleo real
    if tiene_core:
        cores_lista = list(info.get("cores_xy") or [(core_x, core_y)])
        for cxf, cyf in cores_lista:
            cv2.circle(
                vis,
                (int(round(cxf)), int(round(cyf))),
                6,
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.circle(vis, (core_x, core_y), 7, (0, 180, 0), 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            "Core",
            (min(vis.shape[1] - 46, max(4, core_x + 9)), max(16, core_y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 140, 0),
            1,
            cv2.LINE_AA,
        )

    # Dibujar cada Delta real detectado (0 en Arco, 1 en Presilla, 2 en Verticilo)
    for idx_d, (dx_f, dy_f) in enumerate(deltas_xy):
        delta_x, delta_y = int(round(dx_f)), int(round(dy_f))
        cv2.drawMarker(
            vis,
            (delta_x, delta_y),
            (20, 30, 220),
            cv2.MARKER_TRIANGLE_UP,
            13,
            2,
            cv2.LINE_AA,
        )
        etiq_d = (
            "Delta"
            if len(deltas_xy) == 1
            else ("Delta Izq." if idx_d == 0 else "Delta Der.")
        )
        cv2.putText(
            vis,
            etiq_d,
            (
                min(vis.shape[1] - 72, max(4, delta_x - 16 if idx_d > 0 else delta_x + 6)),
                min(vis.shape[0] - 8, delta_y + 15),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 30, 220),
            1,
            cv2.LINE_AA,
        )

    if not tiene_core and not deltas_xy:
        cv2.putText(
            vis,
            "Tipo: Arco (sin Core/Delta)",
            (16, vis.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (150, 60, 10),
            1,
            cv2.LINE_AA,
        )
    return vis


def graficar_evaluacion_huella(
    paquete_modelo: Dict[str, object],
    resultado_huella: Dict[str, object],
    mostrar_identidad_en_titulo: bool = True,
    mostrar_ventana: bool = False,
    ruta_salida_personalizada: Optional[str] = None,
) -> str:
    """
    Genera una lámina diagnóstica de 3 paneles para una huella evaluada:
      1. Huella realzada con singularidades detectadas (Core, Delta y crestas).
      2. Proyección de la huella (estrella dorada ★) en el mapa PCA 2D respecto de
         los 3 clústeres autorizados y los centroides K-Means.
      3. Comparación de distancias euclídeas a cada centroide K-Means frente al
         umbral experimental de aceptación/rechazo.
    """
    if ruta_salida_personalizada:
        dir_dest = os.path.dirname(os.path.abspath(ruta_salida_personalizada))
        if dir_dest:
            os.makedirs(dir_dest, exist_ok=True)
        ruta_salida = ruta_salida_personalizada
    else:
        os.makedirs(DIRECTORIO_RESULTADOS_DEFAULT, exist_ok=True)
        base_nom = os.path.splitext(
            resultado_huella.get("nombre_archivo") or "muestra"
        )[0]
        base_limpio = re.sub(r"[^A-Za-z0-9_\-]", "_", base_nom)
        ruta_salida = os.path.join(
            DIRECTORIO_RESULTADOS_DEFAULT,
            f"evaluacion_huella_{base_limpio}.png",
        )

    fig, (ax_img, ax_pca, ax_bar) = plt.subplots(
        1, 3, figsize=(16.5, 5.6), dpi=150, gridspec_kw={"width_ratios": [1.0, 1.45, 1.15]}
    )

    estado = resultado_huella["estado_acceso"]
    autorizado = bool(resultado_huella["autorizado"])
    dist_min = float(resultado_huella["distancia_min"])
    umbral = float(resultado_huella["umbral"])
    nombre_arch = resultado_huella.get("nombre_archivo") or "Muestra"

    if mostrar_identidad_en_titulo and autorizado:
        sub_id = f" -> {estado} ({resultado_huella['identidad']})"
    else:
        sub_id = f" -> {estado}"

    color_titulo = "#1b7837" if autorizado else "#b2182b"
    fig.suptitle(
        f"Evaluación Biométrica K-Means (k=3): {nombre_arch}{sub_id} | "
        f"Distancia mín.: {dist_min:.2f} (Umbral τ = {umbral:.2f})",
        fontsize=12.5,
        fontweight="bold",
        color=color_titulo,
    )

    # Panel 1: Imagen anotada según clase topológica (Arco, Presilla o Verticilo)
    vis_bgr = crear_imagen_huella_anotada(resultado_huella)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    info = resultado_huella["info_geometrica"]
    tipo_topo = str(info.get("tipo_topologico", "Huella"))
    cr_lista = list(info.get("crestas_por_delta") or [])
    if len(cr_lista) >= 2:
        sub_topo = f"{tipo_topo} | Crestas C-D: {cr_lista[0]}/{cr_lista[1]}"
    elif len(cr_lista) == 1:
        sub_topo = f"{tipo_topo} | Crestas C-D: {cr_lista[0]}"
    else:
        sub_topo = f"{tipo_topo} | 0 Deltas"
    ax_img.imshow(vis_rgb)
    ax_img.set_title(
        f"1. Topología Detectada\n({sub_topo} | Eje: {info['phi_grados']:+.1f}°)",
        fontsize=10.2,
        fontweight="bold",
    )
    ax_img.axis("off")

    # Panel 2: Espacio PCA 2D + Centroides K-Means + Muestra evaluada
    X_pca: np.ndarray = paquete_modelo["X_train_pca"]
    y_train: np.ndarray = paquete_modelo["y_train"]
    centroides_pca: np.ndarray = paquete_modelo["centroides_pca"]
    mapa_id: Dict[int, str] = paquete_modelo["mapa_cluster_a_identidad"]
    var_ratio = paquete_modelo["pca_modelo"].explained_variance_ratio_ * 100.0

    for id_label in IDENTIDADES_ORDEN:
        mask = y_train == id_label
        if not np.any(mask):
            continue
        pts = X_pca[mask, :2]
        col = COLORES_IDENTIDADES[id_label]
        _dibujar_elipse_confianza(ax_pca, pts, col, n_std=2.0)
        ax_pca.scatter(
            pts[:, 0],
            pts[:, 1],
            c=col,
            marker=MARCADORES_IDENTIDADES[id_label],
            s=52,
            alpha=0.70,
            edgecolors="white",
            linewidth=0.6,
            label=id_label,
        )

    pt_muestra = resultado_huella["vector_pca"][:2]
    for c_idx in range( centroides_pca.shape[0] ):
        c_xy = centroides_pca[c_idx, :2]
        id_c = mapa_id[int(c_idx)]
        col_c = COLORES_IDENTIDADES.get(id_c, "#333333")
        ax_pca.scatter(
            [c_xy[0]],
            [c_xy[1]],
            c=col_c,
            marker="P",
            s=210,
            edgecolors="black",
            linewidth=1.5,
            zorder=7,
        )
        estilo_lin = "-" if c_idx == resultado_huella["cluster_idx"] else ":"
        ancho_lin = 1.6 if c_idx == resultado_huella["cluster_idx"] else 0.9
        ax_pca.plot(
            [pt_muestra[0], c_xy[0]],
            [pt_muestra[1], c_xy[1]],
            color=col_c,
            linestyle=estilo_lin,
            linewidth=ancho_lin,
            alpha=0.75,
        )

    ax_pca.scatter(
        [pt_muestra[0]],
        [pt_muestra[1]],
        c="#ffcc00" if autorizado else "#ff3333",
        marker="*",
        s=380,
        edgecolors="black",
        linewidth=1.5,
        zorder=10,
        label=f"Muestra ({estado})",
    )
    ax_pca.set_title(
        "2. Proyección en Espacio PCA 2D y Centroides K-Means",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_pca.set_xlabel(f"PC1 ({var_ratio[0]:.1f}% var.)")
    ax_pca.set_ylabel(f"PC2 ({var_ratio[1]:.1f}% var.)")
    ax_pca.legend(loc="best", fontsize=8.3)
    ax_pca.grid(True, linestyle=":", alpha=0.5)

    # Panel 3: Distancias euclídeas a los 3 centroides vs Umbral
    dists_id = resultado_huella["distancias_por_identidad"]
    etiquetas_bar = [
        et for et in IDENTIDADES_ORDEN if et in dists_id
    ]
    valores_bar = [dists_id[et] for et in etiquetas_bar]
    colores_bar = [
        "#2ca02c" if (d <= umbral and d == min(valores_bar)) else "#7f8c8d"
        for d in valores_bar
    ]
    if not autorizado:
        colores_bar = ["#d62728" if d == min(valores_bar) else "#95a5a6" for d in valores_bar]

    y_pos = np.arange(len(etiquetas_bar))
    barras = ax_bar.barh(
        y_pos, valores_bar, color=colores_bar, edgecolor="black", height=0.52, alpha=0.88
    )
    ax_bar.axvline(
        umbral,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"Umbral τ = {umbral:.2f}",
    )
    for rect, val in zip(barras, valores_bar):
        ax_bar.text(
            val + max(valores_bar + [umbral]) * 0.02,
            rect.get_y() + rect.get_height() / 2.0,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([et.replace("Identidad Sintética ", "Id. ") for et in etiquetas_bar])
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, max(max(valores_bar, default=1.0), umbral) * 1.25)
    ax_bar.set_xlabel("Distancia Euclídea al Centroide")
    ax_bar.set_title(
        "3. Distancia a Centroides vs. Umbral",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_bar.legend(loc="lower right", fontsize=9)
    ax_bar.grid(True, axis="x", linestyle=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0.01, 1, 0.92])
    fig.savefig(ruta_salida, dpi=160)

    if mostrar_ventana:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)
    return ruta_salida


def analizar_y_graficar_dataset_huellas(
    directorio_entrenamiento: str = DIRECTORIO_ENTRENAMIENTO_DEFAULT,
    directorio_prueba: str = DIRECTORIO_PRUEBA_DEFAULT,
    k_clusters: int = K_CLUSTERS_DEFAULT,
    umbral_override: Optional[float] = None,
    mostrar_ventana: bool = True,
) -> Dict[str, object]:
    """
    Ejecuta la evaluación integral del sistema de huellas (entrenamiento K-Means k=3,
    validación de umbral, prueba sobre autorizados e intrusos, métricas FAR/FRR,
    pureza de clústeres, coeficiente de silueta y PCA 2D/3D) y guarda el panel de
    4 gráficos en `resultados_huellas/analisis_kmeans_pca_huellas.png`.
    """
    paquete = entrenar_modelo_huellas(
        directorio_entrenamiento=directorio_entrenamiento,
        k_clusters=k_clusters,
        guardar_cache=(k_clusters == K_CLUSTERS_DEFAULT and umbral_override is None),
    )

    escalador: EscaladorEstandarRegularizado = paquete["escalador"]
    kmeans: KMeans = paquete["kmeans"]
    pca_modelo: PCA = paquete["pca_modelo"]
    mapa_id: Dict[int, str] = paquete["mapa_cluster_a_identidad"]
    umbral = (
        float(umbral_override)
        if umbral_override is not None
        else float(paquete["umbral_distancia"])
    )

    X_train_std: np.ndarray = paquete["X_train_std"]
    y_train: np.ndarray = paquete["y_train"]
    clusters_train: np.ndarray = paquete["clusters_train"]
    dist_min_train: np.ndarray = paquete["dist_min_train"]
    X_train_pca: np.ndarray = paquete["X_train_pca"]
    centroides_pca: np.ndarray = paquete["centroides_pca"]

    pred_train_id = np.asarray([mapa_id[int(c)] for c in clusters_train])
    pureza_train = float(accuracy_score(y_train, pred_train_id))
    silueta_train = (
        float(silhouette_score(X_train_std, clusters_train))
        if k_clusters >= 2
        else 0.0
    )

    # Evaluar conjunto de prueba completo (autorizados + intrusos)
    muestras_test = listar_muestras_por_subcarpeta(directorio_prueba)
    X_test, y_test_real, carpetas_test, rutas_test = extraer_matriz_huellas(
        muestras_test, mostrar_progreso=True, titulo_lote="evaluación y prueba (con intrusos)"
    )
    X_test_std = escalador.transform(X_test)
    X_test_pca = pca_modelo.transform(X_test_std)

    escala_cluster = np.asarray(
        paquete.get("escala_cluster", np.ones(kmeans.n_clusters, dtype=np.float32)),
        dtype=np.float32,
    )
    dists_test_mat = kmeans.transform(X_test_std) / escala_cluster[None, :]
    dist_min_test = np.min(dists_test_mat, axis=1)
    clusters_test = np.argmin(dists_test_mat, axis=1)

    y_test_pred = np.asarray(
        [
            mapa_id[int(c)] if d <= umbral else ETIQUETA_INTRUSO_REPORTE
            for c, d in zip(clusters_test, dist_min_test)
        ]
    )

    mask_aut_test = y_test_real != ETIQUETA_INTRUSO_REPORTE
    mask_int_test = y_test_real == ETIQUETA_INTRUSO_REPORTE

    exactitud_global_test = float(accuracy_score(y_test_real, y_test_pred))
    frr = (
        float(np.mean(dist_min_test[mask_aut_test] > umbral))
        if np.any(mask_aut_test)
        else 0.0
    )
    far = (
        float(np.mean(dist_min_test[mask_int_test] <= umbral))
        if np.any(mask_int_test)
        else 0.0
    )

    etiquetas_4 = list(IDENTIDADES_ORDEN) + [ETIQUETA_INTRUSO_REPORTE]
    matriz_conf = confusion_matrix(y_test_real, y_test_pred, labels=etiquetas_4)

    var_ratio = pca_modelo.explained_variance_ratio_ * 100.0
    var_2d = float(np.sum(var_ratio[:2]))
    var_3d = float(np.sum(var_ratio[:3]))

    # Imprimir reporte detallado en consola
    print("\n" + "=" * 78)
    print("   RESULTADOS DE RECONOCIMIENTO DE HUELLAS K-MEANS + PCA (IA1 - UNCUYO)")
    print("=" * 78)
    print(f" - Muestras entrenamiento autorizado : {len(y_train)} imágenes (3 identidades sintéticas)")
    print(
        f" - Muestras conjunto de prueba       : {len(y_test_real)} imágenes "
        f"({int(np.sum(mask_aut_test))} autorizadas + {int(np.sum(mask_int_test))} intrusas)"
    )
    print(f" - Dimensión del vector dactilar     : {X_train_std.shape[1]} características intrínsecas")
    print(f" - Agrupamiento no supervisado       : K-Means (k = {k_clusters} centroides)")
    print(f" - Pureza de clústeres entrenamiento : {pureza_train * 100:.2f}% | Silueta: {silueta_train:.4f}")
    print(
        f" - Umbral de distancia calibrado (τ) : {umbral:.3f} "
        f"(mediana + {paquete['factor_sigma_umbral']}·σ_MAD / P95)"
    )
    print(f" - Exactitud global en Prueba        : {exactitud_global_test * 100:.2f}% ({int(round(exactitud_global_test * len(y_test_real)))}/{len(y_test_real)})")
    print(f" - Falso Rechazo (FRR autorizados)   : {frr * 100:.2f}% | Falsa Aceptación (FAR intrusos): {far * 100:.2f}%")
    print(f" - Varianza explicada PCA            : 2D = {var_2d:.1f}% | 3D = {var_3d:.1f}%")
    print("-" * 78)
    print("\n MÉTRICAS POR CLASE EN PRUEBA (Incluyendo detección de Intrusos):")
    print(
        classification_report(
            y_test_real, y_test_pred, labels=etiquetas_4, digits=4, zero_division=0
        )
    )

    print(" MATRIZ DE CONFUSIÓN EN PRUEBA (Filas = Real | Columnas = Asignado por K-Means + Umbral):")
    nombres_cortos = ["Id. #1", "Id. #2", "Id. #3", "Intruso"]
    enc = f" {'Clase Real':<22} | " + " | ".join(f"{n:>9}" for n in nombres_cortos)
    print(enc)
    print(" " + "-" * (len(enc) - 1))
    for i, et_real in enumerate(etiquetas_4):
        fila_str = " | ".join(f"{matriz_conf[i, j]:>9d}" for j in range(len(etiquetas_4)))
        print(f" {et_real:<22} | {fila_str}")

    # Generar gráfico de 4 paneles para el informe
    os.makedirs(DIRECTORIO_RESULTADOS_DEFAULT, exist_ok=True)
    fig = plt.figure(figsize=(16.5, 12.2), dpi=150)
    fig.suptitle(
        f"Reconocimiento de Huellas Sintéticas con K-Means (k={k_clusters}), Umbral Experimental y PCA\n"
        f"Entrenamiento: {len(y_train)} muestras (Pureza: {pureza_train * 100:.1f}%) | "
        f"Prueba: {len(y_test_real)} muestras (Exactitud: {exactitud_global_test * 100:.1f}%, FRR={frr*100:.1f}%, FAR={far*100:.1f}%)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    # --- PANEL 1: Separación de Clases en PCA 2D (Entrenamiento + Prueba + Intrusos) ---
    ax1 = fig.add_subplot(2, 2, 1)
    for id_label in IDENTIDADES_ORDEN:
        m_tr = y_train == id_label
        col = COLORES_IDENTIDADES[id_label]
        if np.any(m_tr):
            pts_tr = X_train_pca[m_tr, :2]
            _dibujar_elipse_confianza(ax1, pts_tr, col, n_std=2.0)
            ax1.scatter(
                pts_tr[:, 0],
                pts_tr[:, 1],
                c=col,
                marker=MARCADORES_IDENTIDADES[id_label],
                s=58,
                alpha=0.75,
                edgecolors="white",
                linewidth=0.6,
                label=f"{id_label} (Train n={np.sum(m_tr)})",
            )
        m_te = y_test_real == id_label
        if np.any(m_te):
            pts_te = X_test_pca[m_te, :2]
            ax1.scatter(
                pts_te[:, 0],
                pts_te[:, 1],
                facecolors="none",
                edgecolors=col,
                marker=MARCADORES_IDENTIDADES[id_label],
                s=85,
                linewidth=1.8,
                label=f"{id_label.replace('Identidad Sintética ', 'Prueba Id. ')}",
            )

    if np.any(mask_int_test):
        pts_int = X_test_pca[mask_int_test, :2]
        ax1.scatter(
            pts_int[:, 0],
            pts_int[:, 1],
            c=COLORES_IDENTIDADES[ETIQUETA_INTRUSO_REPORTE],
            marker="X",
            s=115,
            edgecolors="black",
            linewidth=0.9,
            zorder=8,
            label=f"Intrusos Prueba (n={np.sum(mask_int_test)})",
        )

    for c_idx in range(centroides_pca.shape[0]):
        c_xy = centroides_pca[c_idx, :2]
        col_c = COLORES_IDENTIDADES.get(mapa_id[int(c_idx)], "black")
        ax1.scatter(
            [c_xy[0]],
            [c_xy[1]],
            c=col_c,
            marker="*",
            s=320,
            edgecolors="black",
            linewidth=1.5,
            zorder=9,
            label="Centroides K-Means" if c_idx == 0 else None,
        )

    ax1.set_title(
        f"1. Separación de Clases e Intrusos en PCA 2D ({var_2d:.1f}% var.)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel(f"Componente Principal 1 (PC1 - {var_ratio[0]:.1f}%)")
    ax1.set_ylabel(f"Componente Principal 2 (PC2 - {var_ratio[1]:.1f}%)")
    ymin_1, ymax_1 = ax1.get_ylim()
    ax1.set_ylim(ymin_1, ymax_1 + 0.30 * (ymax_1 - ymin_1))
    ax1.legend(loc="upper right", fontsize=7.8, ncol=2, framealpha=0.92)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # --- PANEL 2: Proyección Tridimensional PCA 3D (Autorizados + Centroides) ---
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    for id_label in IDENTIDADES_ORDEN:
        m_tr = y_train == id_label
        col = COLORES_IDENTIDADES[id_label]
        if np.any(m_tr):
            p3 = X_train_pca[m_tr, :3]
            ax2.scatter(
                p3[:, 0],
                p3[:, 1],
                p3[:, 2],
                c=col,
                marker=MARCADORES_IDENTIDADES[id_label],
                s=52,
                alpha=0.82,
                edgecolors="white",
                linewidth=0.5,
                label=id_label,
            )
    for c_idx in range(centroides_pca.shape[0]):
        c3 = centroides_pca[c_idx, :3]
        col_c = COLORES_IDENTIDADES.get(mapa_id[int(c_idx)], "black")
        ax2.scatter(
            [c3[0]],
            [c3[1]],
            [c3[2]],
            c=col_c,
            marker="*",
            s=250,
            edgecolors="black",
            linewidth=1.4,
        )
    ax2.set_title(
        f"2. Clústeres Autorizados en PCA 3D ({var_3d:.1f}% var. acumulada)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel(f"PC1 ({var_ratio[0]:.1f}%)")
    ax2.set_ylabel(f"PC2 ({var_ratio[1]:.1f}%)")
    ax2.set_zlabel(f"PC3 ({var_ratio[2]:.1f}%)")
    ax2.legend(loc="upper right", fontsize=8.2)

    # --- PANEL 3: Matriz de Confusión en Prueba (4x4 con Intrusos) ---
    ax3 = fig.add_subplot(2, 2, 3)
    # Normalización por fila (Recall / Sensibilidad) para que el color refleje fielmente la
    # tasa de acierto (0% a 100%) y no se distorsione por diferencias en el conteo de muestras.
    totales_por_fila = np.maximum(1, np.sum(matriz_conf, axis=1, keepdims=True))
    matriz_conf_pct = (matriz_conf.astype(np.float32) / totales_por_fila) * 100.0

    im_mc = ax3.imshow(
        matriz_conf_pct,
        interpolation="nearest",
        cmap=plt.cm.Blues,
        vmin=0.0,
        vmax=100.0,
    )
    cbar_mc = fig.colorbar(im_mc, ax=ax3, fraction=0.046, pad=0.04)
    cbar_mc.set_label("Tasa de Acierto por Clase (%)", fontsize=8.5)

    ax3.set_title(
        f"3. Matriz de Confusión en Prueba (Exactitud: {exactitud_global_test * 100:.1f}%)",
        fontsize=11,
        fontweight="bold",
    )
    ticks = np.arange(len(nombres_cortos))
    ax3.set_xticks(ticks)
    ax3.set_yticks(ticks)
    ax3.set_xticklabels(nombres_cortos)
    ax3.set_yticklabels(nombres_cortos)
    ax3.set_ylabel("Identidad Real")
    ax3.set_xlabel("Decisión del Sistema (K-Means + Umbral)")

    for i in range(matriz_conf.shape[0]):
        tot_f = max(1, int(np.sum(matriz_conf[i])))
        for j in range(matriz_conf.shape[1]):
            val = int(matriz_conf[i, j])
            pct = (val / tot_f) * 100.0
            # Contraste de texto: blanco para celdas de fondo oscuro (> 50% de intensidad), negro para el resto
            c_txt = "white" if pct > 50.0 else "black"
            ax3.text(
                j,
                i,
                f"{val}\n({pct:.0f}%)",
                ha="center",
                va="center",
                color=c_txt,
                fontweight="bold",
                fontsize=9.5,
            )

    # --- PANEL 4: Justificación Experimental del Umbral de Distancia ---
    ax4 = fig.add_subplot(2, 2, 4)
    dists_aut_test = dist_min_test[mask_aut_test]
    dists_int_test = dist_min_test[mask_int_test]
    max_x = max(
        float(np.max(dist_min_test)) if len(dist_min_test) else 15.0,
        umbral * 1.4,
    )
    bins = np.linspace(0.0, max_x * 1.05, 28)

    ax4.hist(
        dist_min_train,
        bins=bins,
        alpha=0.55,
        color="#1f77b4",
        edgecolor="black",
        label=f"Entrenamiento Autorizado (n={len(dist_min_train)}, máx={np.max(dist_min_train):.2f})",
    )
    if len(dists_aut_test) > 0:
        ax4.hist(
            dists_aut_test,
            bins=bins,
            alpha=0.68,
            color="#2ca02c",
            edgecolor="black",
            label=f"Prueba Autorizada (n={len(dists_aut_test)}, máx={np.max(dists_aut_test):.2f})",
        )
    if len(dists_int_test) > 0:
        ax4.hist(
            dists_int_test,
            bins=bins,
            alpha=0.78,
            color="#d62728",
            edgecolor="black",
            label=f"Intrusos No Autorizados (n={len(dists_int_test)}, mín={np.min(dists_int_test):.2f})",
        )

    ax4.axvline(
        umbral,
        color="#b2182b",
        linestyle="--",
        linewidth=2.3,
        label=f"Umbral experimental τ = {umbral:.2f}",
    )
    ax4.axvspan(0, umbral, color="#2ca02c", alpha=0.06)
    ax4.axvspan(umbral, max_x * 1.08, color="#d62728", alpha=0.05)

    ax4.set_title(
        "4. Distribución de Distancias al Centroide y Umbral de Rechazo",
        fontsize=11,
        fontweight="bold",
    )
    ax4.set_xlabel("Distancia Euclídea al Centroide K-Means más cercano")
    ax4.set_ylabel("Cantidad de huellas")
    ax4.set_xlim(0, max_x * 1.06)
    ax4.legend(loc="upper right", fontsize=8.3)
    ax4.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(RUTA_GRAFICO_ANALISIS, dpi=200)
    print(f"\n[+] Gráfico integral guardado en: {os.path.abspath(RUTA_GRAFICO_ANALISIS)}")
    print("=" * 78)

    if mostrar_ventana:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)

    return {
        "pureza_train": pureza_train,
        "silueta_train": silueta_train,
        "exactitud_test": exactitud_global_test,
        "frr": frr,
        "far": far,
        "matriz_confusion": matriz_conf,
        "ruta_grafico": RUTA_GRAFICO_ANALISIS,
        "paquete_modelo": paquete,
    }


# ==============================================================================
# 5. MENÚ INTERACTIVO DE CONSOLA Y CLI
# ==============================================================================

def _mostrar_resumen_dataset(
    dir_train: str = DIRECTORIO_ENTRENAMIENTO_DEFAULT,
    dir_test: str = DIRECTORIO_PRUEBA_DEFAULT,
) -> None:
    """Inspecciona y cuenta las imágenes disponibles por identidad e intrusos."""
    m_tr = listar_muestras_por_subcarpeta(dir_train)
    m_te = listar_muestras_por_subcarpeta(dir_test)

    print("\n" + "=" * 74)
    print(" RESUMEN DEL DATASET DE HUELLAS SINTÉTICAS")
    print("=" * 74)
    print(f" 1. Conjunto de Entrenamiento Autorizado ('{dir_train}'): {len(m_tr)} imágenes")
    for sub in sorted({m[1] for m in m_tr}):
        n_s = sum(1 for m in m_tr if m[1] == sub)
        print(f"    - {sub:<15} -> {_normalizar_etiqueta_carpeta(sub)} ({n_s} muestras)")

    print(f"\n 2. Conjunto de Prueba y Validación ('{dir_test}'): {len(m_te)} imágenes")
    for sub in sorted({m[1] for m in m_te}):
        n_s = sum(1 for m in m_te if m[1] == sub)
        print(f"    - {sub:<15} -> {_normalizar_etiqueta_carpeta(sub)} ({n_s} muestras)")
    print("=" * 74)


def _experimentar_k_y_umbral_menu(
    dir_train: str = DIRECTORIO_ENTRENAMIENTO_DEFAULT,
    dir_test: str = DIRECTORIO_PRUEBA_DEFAULT,
) -> None:
    """
    Permite experimentar temporalmente cambiando el número de clústeres `k` de K-Means
    y el umbral `τ` de aceptación/rechazo sin alterar el modelo principal en caché.
    """
    print("\n" + "-" * 74)
    print(" EXPERIMENTACIÓN DE HIPERPARÁMETROS (K-MEANS Y UMBRAL DE DISTANCIA)")
    print(" Nota: 'k' en K-Means es la cantidad de centroides no supervisados")
    print("       (distinto del número de vecinos 'K' del clasificador K-NN de voz).")
    print("-" * 74)

    k_in = input(" Ingresá k de K-Means a probar [Default: 3]: ").strip()
    k_exp = int(k_in) if k_in.isdigit() and 2 <= int(k_in) <= 8 else 3

    paq = entrenar_modelo_huellas(
        directorio_entrenamiento=dir_train,
        k_clusters=k_exp,
        guardar_cache=False,
        mostrar_progreso=False,
    )
    tau_sug = float(paq["umbral_distancia"])
    t_in = input(
        f" Ingresá umbral de distancia τ a evaluar [Default calibrado: {tau_sug:.2f}]: "
    ).strip()
    try:
        tau_exp = float(t_in) if t_in else tau_sug
    except ValueError:
        tau_exp = tau_sug

    analizar_y_graficar_dataset_huellas(
        directorio_entrenamiento=dir_train,
        directorio_prueba=dir_test,
        k_clusters=k_exp,
        umbral_override=tau_exp,
        mostrar_ventana=True,
    )


def _probar_huella_menu(paquete_modelo: Dict[str, object], dir_test: str) -> None:
    """Solicita una huella y muestra su diagnóstico completo por consola y gráfico."""
    muestras = listar_muestras_por_subcarpeta(dir_test)
    print("\n" + "-" * 74)
    print(" PRUEBA INDIVIDUAL DE UNA HUELLA DACTILAR")
    print("-" * 74)
    for idx, (ruta, sub, etiq) in enumerate(muestras, 1):
        print(f"  [{idx:02d}] {os.path.basename(ruta):<16} (Carpeta origen: {sub} -> {etiq})")

    ent = input(
        f"\n Elegí un número (1 a {len(muestras)}) o escribí la ruta de una imagen [Default: 1]: "
    ).strip()
    if not ent:
        ent = "1"

    if ent.isdigit() and 1 <= int(ent) <= len(muestras):
        ruta_elegida = muestras[int(ent) - 1][0]
    else:
        ruta_elegida = ent
        if not os.path.isfile(ruta_elegida):
            print(f" [!] No existe el archivo: {ruta_elegida}")
            return

    res = evaluar_huella(ruta_elegida, paquete_modelo=paquete_modelo)
    print("\n" + "=" * 72)
    print(f" RESULTADO SOBRE '{res['nombre_archivo']}'")
    print("=" * 72)
    print(f" - Estado («Reconocer»)   : {res['estado_acceso']}")
    print(f" - Identidad («Identif.») : {res['identidad']}")
    print(f" - Distancia mínima       : {res['distancia_min']:.3f} (Umbral τ = {res['umbral']:.3f})")
    print(" - Distancias por centroide:")
    for id_k, d_v in res["distancias_por_identidad"].items():
        print(f"     * {id_k:<26}: {d_v:.3f}")
    print("=" * 72)

    ver = input(" ¿Deseás generar y abrir el gráfico diagnóstico de esta huella? [s/N]: ").strip().lower()
    if ver in ("s", "si", "sí", "y"):
        ruta_g = graficar_evaluacion_huella(
            paquete_modelo=paquete_modelo,
            resultado_huella=res,
            mostrar_ventana=True,
        )
        print(f" [+] Gráfico guardado en: {os.path.abspath(ruta_g)}")


def menu_interactivo() -> None:
    """Menú interactivo del módulo de reconocimiento de huellas con K-Means."""
    paquete_modelo: Optional[Dict[str, object]] = None

    while True:
        print("\n" + "=" * 76)
        print("   MÓDULO DE RECONOCIMIENTO DE HUELLAS (K-MEANS k=3 + PCA) - IA1 UNCUYO")
        print("=" * 76)
        print("   [1] Analizar cantidades del dataset (autorizadas e intrusas)")
        print("   [2] Entrenar K-Means (k=3), evaluar prueba + intrusos y ver gráficos PCA")
        print("   [3] Probar una nueva huella individual")
        print("   [4] Experimentar variando k de K-Means y umbral de rechazo τ")
        print("   [5] Ejecutar preprocesamiento por lotes (`preprocesamiento.py`)")
        print("   [0] Salir")
        print("=" * 76)

        try:
            op = input(" Seleccioná una opción [Default: 2]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[+] Saliendo del módulo de huellas.")
            break

        if not op:
            op = "2"

        if op in ("0", "salir", "q", "exit"):
            break
        elif op == "1":
            _mostrar_resumen_dataset()
        elif op == "2":
            info = analizar_y_graficar_dataset_huellas(mostrar_ventana=True)
            paquete_modelo = info["paquete_modelo"]
        elif op == "3":
            if paquete_modelo is None:
                paquete_modelo = entrenar_o_cargar_modelo_huellas()
            _probar_huella_menu(paquete_modelo, DIRECTORIO_PRUEBA_DEFAULT)
        elif op == "4":
            _experimentar_k_y_umbral_menu()
        elif op == "5":
            prep.menu_interactivo()
        else:
            print(" [!] Opción no válida.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconocimiento de huellas dactilares sintéticas con K-Means (k=3), umbral experimental y PCA."
    )
    parser.add_argument(
        "--analizar",
        action="store_true",
        help="Entrenar el modelo K-Means, evaluar prueba+intrusos y generar gráficos PNG sin menú",
    )
    parser.add_argument(
        "--sin-ventana",
        action="store_true",
        help="Guardar gráficos sin abrir ventana interactiva de Matplotlib",
    )
    args = parser.parse_args()

    if args.analizar:
        analizar_y_graficar_dataset_huellas(mostrar_ventana=not args.sin_ventana)
        return

    menu_interactivo()


if __name__ == "__main__":
    main()
