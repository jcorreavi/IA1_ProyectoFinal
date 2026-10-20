#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Preprocesamiento de Huellas Dactilares Sintéticas
Trabajo Final - Inteligencia Artificial I (2026) - Ingeniería Mecatrónica (UNCUYO)

Secuencia de acondicionamiento (conservando dimensiones exactas sin recorte de ROI):
  1. Normalización de contraste (estiramiento robusto por percentiles + CLAHE ponderado por textura).
  2. Filtrado espacial de reducción de ruido (filtro de mediana 3x3 + filtro Gaussiano suave 3x3).
  3. Realce direccional de crestas mediante banco de filtros de Gabor (16 orientaciones guiadas por
     el tensor de estructura local + mapa continuo de coherencia direccional).
  4. Binarización adaptativa local y depuración morfológica (cierre en cruz 3x3, eliminación de
     componentes conexas espurias y relleno de micro-poros internos).
  5. Esqueletizado topológico (algoritmo de Zhang-Suen / scikit-image si está disponible) con
     poda conservadora de espolones parásitos.
"""

import os
import sys
import argparse
import cv2
import numpy as np

# Intentar importar skeletonize de scikit-image si está instalado en el entorno;
# en caso contrario, utilizar la implementación vectorizada exacta de Zhang-Suen en NumPy.
try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
    TIENE_SKIMAGE = True
except ImportError:
    TIENE_SKIMAGE = False


# ==============================================================================
# PARÁMETROS POR DEFECTO (Calibrados para huellas sintéticas Anguli 275x400 px)
# ==============================================================================
PARAMETROS_DEFAULT = {
    # 1. Normalización de contraste
    "percentil_bajo": 2.0,
    "percentil_alto": 95.0,
    "clahe_clip": 1.8,
    "clahe_grid": (8, 8),

    # 2. Filtrado espacial (reducción de ruido)
    "mediana_ksize": 3,
    "gauss_ksize": 3,
    "gauss_sigma": 0.65,

    # 3. Estimación de orientación y banco de filtros de Gabor
    "sigma_gradiente": 1.0,
    "sigma_tensor": 5.5,
    "num_orientaciones": 16,
    "gabor_ksize": 17,
    "gabor_sigma_x": 2.6,   # Selectividad perpendicular a la cresta (evita fusionar crestas vecinas)
    "gabor_sigma_y": 4.2,   # Continuidad longitudinal (reconecta micro-cortes y rasguños)
    "gabor_lambda": 8.5,    # Periodo espacial medio cresta-valle en píxeles (~8.5 px)

    # 4. Binarización adaptativa y limpieza morfológica
    "umbral_ventana": 17,
    "umbral_constante_c": 4.0,
    "min_area_cresta": 28,  # Píxeles mínimos de una cresta válida (elimina ruido exterior)
    "max_area_poro": 10,    # Área máxima de micro-poros internos a rellenar

    # 5. Esqueletizado y poda de espolones
    "max_long_espolon": 3   # Longitud máxima (px) de espolones parásitos adyacentes a bifurcaciones
}

EXTENSIONES_VALIDAS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


# ==============================================================================
# ETAPA 1: NORMALIZACIÓN DEL CONTRASTE
# ==============================================================================

def normalizar_contraste(imagen_gris, params=None):
    """
    Estandariza la distribución de intensidades de la huella dactilar conservando sus
    dimensiones originales (H x W) y manteniendo limpio el fondo blanco (255):
      1. Estiramiento lineal robusto entre los percentiles [2%, 95%] para compensar
         variaciones globales de iluminación o correcciones gamma.
      2. Ecualización adaptativa de histograma limitada en contraste (CLAHE) para
         resaltar crestas con baja presión local.
      3. Ponderación suave por desviación estándar local para evitar que CLAHE
         amplifique el ruido de fondo en las regiones blancas sin huella.
    """
    p = params or PARAMETROS_DEFAULT
    h_orig, w_orig = imagen_gris.shape[:2]

    # 1. Estiramiento por percentiles
    p_low, p_high = np.percentile(imagen_gris, [p["percentil_bajo"], p["percentil_alto"]])
    if (p_high - p_low) > 10.0:
        estirada = np.clip(
            (imagen_gris.astype(np.float32) - p_low) * 255.0 / (p_high - p_low),
            0.0, 255.0
        ).astype(np.uint8)
    else:
        estirada = cv2.normalize(imagen_gris, None, 0, 255, cv2.NORM_MINMAX)

    # 2. CLAHE controlado
    clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=p["clahe_grid"])
    ecualizada = clahe.apply(estirada)

    # 3. Ponderación por actividad local para preservar el fondo blanco puro (255)
    img_f = estirada.astype(np.float32)
    media_loc = cv2.GaussianBlur(img_f, (25, 25), 5.0)
    var_loc = cv2.GaussianBlur((img_f - media_loc) ** 2, (25, 25), 5.0)
    std_loc = np.sqrt(np.maximum(0.0, var_loc))

    peso_actividad = np.clip((std_loc - 8.0) / 20.0, 0.0, 1.0)
    resultado = peso_actividad * ecualizada.astype(np.float32) + (1.0 - peso_actividad) * 255.0
    resultado = np.clip(resultado, 0.0, 255.0).astype(np.uint8)

    assert resultado.shape == (h_orig, w_orig), "Dimensiones alteradas en normalización"
    return resultado


# ==============================================================================
# ETAPA 2: REDUCCIÓN DE RUIDO (FILTRADO ESPACIAL PRE-REALCE)
# ==============================================================================

def reducir_ruido(imagen_norm, params=None):
    """
    Aplica filtrado espacial no lineal y lineal de soporte compacto (3x3):
      1. Filtro de mediana (3x3): suprime ruido impulsivo (sal y pimienta) y sella los
         micro-poros blancos de 1 píxel característicos del generador Anguli dentro de
         las crestas, sin difuminar los bordes.
      2. Filtro Gaussiano suave (3x3, sigma=0.65): atenúa el ruido electrónico gaussiano
         de alta frecuencia y regulariza los contornos antes de calcular gradientes.
    Justificación de parámetros: un kernel mayor (5x5 o 7x7) invadiría los valles de
    3 a 4 píxeles de ancho y fusionaría crestas paralelas adyacentes.
    """
    p = params or PARAMETROS_DEFAULT
    h_orig, w_orig = imagen_norm.shape[:2]

    k_med = int(p["mediana_ksize"]) | 1
    k_gau = int(p["gauss_ksize"]) | 1

    sin_impulsos = cv2.medianBlur(imagen_norm, k_med)
    suavizada = cv2.GaussianBlur(sin_impulsos, (k_gau, k_gau), p["gauss_sigma"])

    assert suavizada.shape == (h_orig, w_orig), "Dimensiones alteradas en reducción de ruido"
    return suavizada


# ==============================================================================
# ETAPA 3: ESTIMACIÓN DE ORIENTACIÓN LOCAL Y REALCE CON FILTROS DE GABOR
# ==============================================================================

def estimar_orientacion_y_coherencia(imagen_suavizada, params=None):
    """
    Calcula el campo denso de orientación de crestas theta(x, y) en [0, pi) y un mapa
    continuo de presencia de textura estructurada mediante el tensor de estructura
    (Hong, Wan & Jain, 1998).
    No recorta la imagen (conserva W x H exactos), sino que pondera suavemente las regiones
    con estructura periódica de crestas frente al fondo uniforme o ruidoso.
    """
    p = params or PARAMETROS_DEFAULT

    base = cv2.GaussianBlur(imagen_suavizada.astype(np.float32), (5, 5), p["sigma_gradiente"])
    gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)

    # Tensor de estructura suavizado en vecindario local (31x31)
    sigma_t = p["sigma_tensor"]
    gxx = cv2.GaussianBlur(gx * gx, (31, 31), sigma_t)
    gyy = cv2.GaussianBlur(gy * gy, (31, 31), sigma_t)
    gxy = cv2.GaussianBlur(gx * gy, (31, 31), sigma_t)

    # Vector doble de ángulo del gradiente: (gxx - gyy, 2*gxy)
    # La orientación de la cresta es ortogonal al gradiente dominante (+ pi/2)
    theta_cresta = 0.5 * np.arctan2(2.0 * gxy, gxx - gyy) + (np.pi / 2.0)
    theta_cresta = np.mod(theta_cresta, np.pi)

    # Coherencia direccional local: mide cuán alineados están los gradientes en el bloque
    # (alta en crestas paralelas ~0.7-1.0; baja en ruido aleatorio de fondo <0.35)
    energia_total = gxx + gyy
    anisotropia = np.sqrt((gxx - gyy) ** 2 + 4.0 * (gxy ** 2))
    coherencia = anisotropia / (energia_total + 1e-4)

    magnitud_grad = np.sqrt(energia_total)
    peso_energia = np.clip((magnitud_grad - 14.0) / 35.0, 0.0, 1.0)
    peso_coherencia = np.clip((coherencia - 0.25) / 0.35, 0.0, 1.0)

    peso_textura = cv2.GaussianBlur(peso_energia * peso_coherencia, (17, 17), 3.5)
    return theta_cresta, peso_textura


def realzar_crestas_gabor(imagen_suavizada, params=None):
    """
    Realza selectivamente las crestas dactilares mediante un banco de N=16 filtros de Gabor
    simétricos (fase psi=0) sintonizados al periodo espacial medio de las huellas (lambda=8.5 px).
      - Elimina previamente la componente continua (DC) local para evitar anillos de borde.
      - Interpola linealmente entre los dos filtros cuya orientación encuadra el ángulo local
        de la cresta en cada píxel.
      - Reconecta interrupciones por cicatrices/rasguños gracias a la envolvente alargada
        a lo largo de la cresta (sigma_y = 4.2 > sigma_x = 2.6).
    Retorna la imagen realzada en escala de grises (8 bits, mismas dimensiones) y el mapa
    de peso de textura para las etapas siguientes.
    """
    p = params or PARAMETROS_DEFAULT
    h_orig, w_orig = imagen_suavizada.shape[:2]

    theta_cresta, peso_textura = estimar_orientacion_y_coherencia(imagen_suavizada, p)

    # Invertir intensidades para que las crestas sean picos positivos y quitar media local (DC)
    inv = 255.0 - imagen_suavizada.astype(np.float32)
    media_local = cv2.GaussianBlur(inv, (15, 15), 3.5)
    senal_ac = inv - media_local

    num_ori = int(p["num_orientaciones"])
    ksize = int(p["gabor_ksize"]) | 1
    sigma_x = float(p["gabor_sigma_x"])
    sigma_y = float(p["gabor_sigma_y"])
    lambd = float(p["gabor_lambda"])
    gamma = sigma_x / sigma_y  # En OpenCV, gamma = sigma_x / sigma_y

    angulos = np.linspace(0.0, np.pi, num_ori, endpoint=False)
    respuestas = np.zeros((num_ori, h_orig, w_orig), dtype=np.float32)

    for idx, ang_cresta in enumerate(angulos):
        # En cv2.getGaborKernel, la dirección paralela a las franjas del coseno es theta_cv + pi/2
        theta_cv = ang_cresta - (np.pi / 2.0)
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma=sigma_x, theta=theta_cv,
            lambd=lambd, gamma=gamma, psi=0.0, ktype=cv2.CV_32F
        )
        # Garantizar media cero exacta y ganancia L1 unitaria
        kernel -= np.mean(kernel)
        kernel /= (np.sum(np.abs(kernel)) + 1e-6)
        respuestas[idx] = cv2.filter2D(senal_ac, cv2.CV_32F, kernel)

    # Interpolación angular continua entre los dos filtros adyacentes por píxel
    paso = np.pi / num_ori
    idx_float = theta_cresta / paso
    idx0 = np.floor(idx_float).astype(np.int32) % num_ori
    idx1 = (idx0 + 1) % num_ori
    w1 = idx_float - np.floor(idx_float)
    w0 = 1.0 - w1

    filas, cols = np.indices((h_orig, w_orig))
    resp_orientada = w0 * respuestas[idx0, filas, cols] + w1 * respuestas[idx1, filas, cols]
    resp_modulada = resp_orientada * peso_textura

    # Normalización robusta de la respuesta de Gabor a [0, 255] (crestas negras = 0, valles = 255)
    pixeles_activos = np.abs(resp_modulada[peso_textura > 0.25])
    escala = float(np.percentile(pixeles_activos, 90)) + 1e-5 if pixeles_activos.size > 0 else 25.0

    gabor_gris = np.clip(
        255.0 - (np.maximum(0.0, resp_modulada + 0.35 * escala) / (1.35 * escala)) * 255.0,
        0.0, 255.0
    )

    # Fusión ponderada (75% Gabor direccional + 25% imagen suavizada) para conservar nitidez en bifurcaciones
    combinada = (
        peso_textura * (0.75 * gabor_gris + 0.25 * imagen_suavizada.astype(np.float32))
        + (1.0 - peso_textura) * 255.0
    )
    resultado = np.clip(combinada, 0.0, 255.0).astype(np.uint8)

    assert resultado.shape == (h_orig, w_orig), "Dimensiones alteradas en realce de Gabor"
    return resultado, peso_textura


# ==============================================================================
# ETAPA 4: BINARIZACIÓN ADAPTATIVA Y LIMPIEZA MORFOLÓGICA
# ==============================================================================

def binarizar_huella(imagen_gabor, peso_textura=None, params=None):
    """
    Convierte la imagen realzada en una representación binaria pura {0, 255}
    (0 = cresta negra, 255 = valle/fondo blanco) conservando el tamaño original:
      1. Umbralización adaptativa local comparando cada píxel con la media Gaussiana
         de su entorno (17x17 px ~ 2 periodos de cresta).
      2. Cierre morfológico suave con elemento estructurante en cruz (3x3) para sellar
         poros residuales de 1 px sin engrosar ni unir crestas vecinas.
      3. Filtrado de componentes conexas para eliminar manchas aisladas (< 28 px) y
         rellenar micro-huecos internos (< 10 px).
    """
    p = params or PARAMETROS_DEFAULT
    h_orig, w_orig = imagen_gabor.shape[:2]

    if peso_textura is None:
        _, peso_textura = estimar_orientacion_y_coherencia(imagen_gabor, p)

    k_win = int(p["umbral_ventana"]) | 1
    c_val = float(p["umbral_constante_c"])

    media_local = cv2.GaussianBlur(imagen_gabor.astype(np.float32), (k_win, k_win), 3.5)
    es_cresta = (imagen_gabor.astype(np.float32) < (media_local - c_val)) & (peso_textura > 0.25)
    crestas_blancas = (es_cresta.astype(np.uint8)) * 255

    # Cierre morfológico con cruz 3x3 (preserva separación entre crestas paralelas)
    kernel_cruz = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    crestas_blancas = cv2.morphologyEx(crestas_blancas, cv2.MORPH_CLOSE, kernel_cruz, iterations=1)

    # Eliminar partículas o ruido aislado fuera o dentro de la huella
    min_area = int(p["min_area_cresta"])
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crestas_blancas, connectivity=8)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < min_area:
            crestas_blancas[labels == lbl] = 0

    # Rellenar micro-poros blancos residuales encerrados dentro de las crestas
    max_poro = int(p["max_area_poro"])
    fondo_blanco = 255 - crestas_blancas
    num_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(fondo_blanco, connectivity=4)
    for lbl in range(1, num_h):
        if stats_h[lbl, cv2.CC_STAT_AREA] < max_poro:
            crestas_blancas[labels_h == lbl] = 255

    # Convención visual de huella: crestas negras (0), fondo/valles blancos (255)
    binaria = (255 - crestas_blancas).astype(np.uint8)
    assert binaria.shape == (h_orig, w_orig), "Dimensiones alteradas en binarización"
    return binaria


# ==============================================================================
# ETAPA 5: ESQUELETIZADO TOPOLÓGICO Y PODA DE ESPOLONES
# ==============================================================================

def _zhang_suen_vectorizado(bin_crestas_blancas):
    """
    Implementación vectorizada en NumPy del algoritmo de adelgazamiento topológico
    de Zhang-Suen (1984). Reduce las crestas binarias a un grosor de exactamente
    1 píxel preservando la conectividad en 8-vecindad y la ubicación de terminaciones
    y bifurcaciones.
    """
    img = (bin_crestas_blancas > 0).astype(np.uint8)
    skel = np.pad(img, 1, mode='constant', constant_values=0)

    while True:
        # --- Sub-iteración 1 ---
        P2 = skel[:-2, 1:-1]
        P3 = skel[:-2, 2:]
        P4 = skel[1:-1, 2:]
        P5 = skel[2:, 2:]
        P6 = skel[2:, 1:-1]
        P7 = skel[2:, :-2]
        P8 = skel[1:-1, :-2]
        P9 = skel[:-2, :-2]
        P1 = skel[1:-1, 1:-1]

        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        A = (
            ((P2 == 0) & (P3 == 1)).astype(np.uint8) +
            ((P3 == 0) & (P4 == 1)).astype(np.uint8) +
            ((P4 == 0) & (P5 == 1)).astype(np.uint8) +
            ((P5 == 0) & (P6 == 1)).astype(np.uint8) +
            ((P6 == 0) & (P7 == 1)).astype(np.uint8) +
            ((P7 == 0) & (P8 == 1)).astype(np.uint8) +
            ((P8 == 0) & (P9 == 1)).astype(np.uint8) +
            ((P9 == 0) & (P2 == 1)).astype(np.uint8)
        )
        m1 = P2 * P4 * P6
        m2 = P4 * P6 * P8
        eliminar_1 = (P1 == 1) & (B >= 2) & (B <= 6) & (A == 1) & (m1 == 0) & (m2 == 0)
        skel[1:-1, 1:-1][eliminar_1] = 0

        # --- Sub-iteración 2 ---
        P2 = skel[:-2, 1:-1]
        P3 = skel[:-2, 2:]
        P4 = skel[1:-1, 2:]
        P5 = skel[2:, 2:]
        P6 = skel[2:, 1:-1]
        P7 = skel[2:, :-2]
        P8 = skel[1:-1, :-2]
        P9 = skel[:-2, :-2]
        P1 = skel[1:-1, 1:-1]

        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        A = (
            ((P2 == 0) & (P3 == 1)).astype(np.uint8) +
            ((P3 == 0) & (P4 == 1)).astype(np.uint8) +
            ((P4 == 0) & (P5 == 1)).astype(np.uint8) +
            ((P5 == 0) & (P6 == 1)).astype(np.uint8) +
            ((P6 == 0) & (P7 == 1)).astype(np.uint8) +
            ((P7 == 0) & (P8 == 1)).astype(np.uint8) +
            ((P8 == 0) & (P9 == 1)).astype(np.uint8) +
            ((P9 == 0) & (P2 == 1)).astype(np.uint8)
        )
        m1 = P2 * P4 * P8
        m2 = P2 * P6 * P8
        eliminar_2 = (P1 == 1) & (B >= 2) & (B <= 6) & (A == 1) & (m1 == 0) & (m2 == 0)
        skel[1:-1, 1:-1][eliminar_2] = 0

        if not (np.any(eliminar_1) or np.any(eliminar_2)):
            break

    return (skel[1:-1, 1:-1] * 255).astype(np.uint8)


def _podar_espolones(skel_crestas_blancas, max_long=3):
    """
    Elimina únicamente micro-ramas parásitas (espolones <= max_long píxeles) que nacen
    directamente de una bifurcación por rugosidades del borde, sin acortar las crestas
    principales ni desplazar terminaciones legítimas.
    """
    skel = (skel_crestas_blancas > 0).astype(np.uint8)
    kernel_vec = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)

    for _ in range(max_long):
        vecinos = cv2.filter2D(skel, cv2.CV_8U, kernel_vec)
        extremos = (skel == 1) & (vecinos == 1)
        bifurcaciones = (skel == 1) & (vecinos >= 3)
        if not np.any(extremos) or not np.any(bifurcaciones):
            break

        bif_vecina = cv2.dilate(bifurcaciones.astype(np.uint8), np.ones((3, 3), dtype=np.uint8))
        espolones = extremos & (bif_vecina == 1)
        if not np.any(espolones):
            break
        skel[espolones] = 0

    # Eliminar píxeles sueltos remanentes (< 5 px)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel * 255, connectivity=8)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < 5:
            skel[labels == lbl] = 0

    return (skel * 255).astype(np.uint8)


def esqueletizar_huella(imagen_binaria, params=None, usar_skimage=False):
    """
    Obtiene el esqueleto topológico de 1 píxel de grosor (0 = cresta negra de 1 px,
    255 = fondo blanco) a partir de la huella binarizada, aplicando adelgazamiento de
    Zhang-Suen y poda conservadora de espolones.
    """
    p = params or PARAMETROS_DEFAULT
    h_orig, w_orig = imagen_binaria.shape[:2]

    # Invertir para que las crestas sean el objeto activo (1 / 255)
    crestas_activas = (imagen_binaria < 128).astype(np.uint8) * 255

    if usar_skimage and TIENE_SKIMAGE:
        skel_bool = _skimage_skeletonize(crestas_activas > 0, method='zhang')
        skel_blancas = (skel_bool.astype(np.uint8)) * 255
    else:
        skel_blancas = _zhang_suen_vectorizado(crestas_activas)

    skel_limpio = _podar_espolones(skel_blancas, max_long=int(p["max_long_espolon"]))
    esqueleto = (255 - skel_limpio).astype(np.uint8)

    assert esqueleto.shape == (h_orig, w_orig), "Dimensiones alteradas en esqueletizado"
    return esqueleto


# ==============================================================================
# PIPELINE COMPLETO Y VISUALIZACIÓN COMPARATIVA
# ==============================================================================

def preprocesar_imagen_completa(imagen, params=None):
    """
    Ejecuta la secuencia completa de preprocesamiento sobre una imagen de huella dactilar

    Retorna un diccionario con todas las etapas en escala de grises uint8 (H x W):
      - 'original'    : Imagen original en escala de grises.
      - 'normalizada' : Etapa 1 - Normalización de contraste (Percentiles + CLAHE ponderado).
      - 'sin_ruido'   : Etapa 2 - Filtrado espacial (Mediana 3x3 + Gaussiano 3x3).
      - 'gabor'       : Etapa 3 - Realce direccional de crestas con banco de filtros de Gabor.
      - 'binarizada'  : Etapa 4 - Binarización adaptativa y limpieza morfológica.
      - 'esqueleto'   : Etapa 5 - Esqueletizado topológico (1 píxel de grosor).
    """
    if imagen is None:
        raise ValueError("La imagen de entrada es None.")

    if len(imagen.shape) == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen.copy()

    h_orig, w_orig = gris.shape[:2]
    p = params or PARAMETROS_DEFAULT

    etapa1_norm = normalizar_contraste(gris, p)
    etapa2_ruido = reducir_ruido(etapa1_norm, p)
    etapa3_gabor, peso_textura = realzar_crestas_gabor(etapa2_ruido, p)
    etapa4_bin = binarizar_huella(etapa3_gabor, peso_textura, p)
    etapa5_skel = esqueletizar_huella(etapa4_bin, p)

    etapas = {
        "original": gris,
        "normalizada": etapa1_norm,
        "sin_ruido": etapa2_ruido,
        "gabor": etapa3_gabor,
        "binarizada": etapa4_bin,
        "esqueleto": etapa5_skel,
    }

    # Verificación estricta de dimensiones en todas las etapas (prohibido recortar ROI)
    for nombre_etapa, img_etapa in etapas.items():
        if img_etapa.shape[:2] != (h_orig, w_orig):
            raise RuntimeError(
                f"Discrepancia de dimensiones en etapa '{nombre_etapa}': "
                f"{img_etapa.shape[:2]} != {(h_orig, w_orig)}"
            )

    return etapas


def crear_mosaico_comparativo(etapas, nombre_archivo=""):
    """
    Construye una lámina horizontal que permite comparar visualmente la imagen original
    con cada una de las 5 etapas del preprocesamiento.
    """
    orden = [
        ("original", "0. Original"),
        ("normalizada", "1. Normalizada"),
        ("sin_ruido", "2. Mediana+Gauss"),
        ("gabor", "3. Realce Gabor"),
        ("binarizada", "4. Binarizada"),
        ("esqueleto", "5. Esqueleto"),
    ]

    h, w = etapas["original"].shape[:2]
    alto_cabecera = 34
    margen = 6
    paneles = []

    for clave, titulo in orden:
        img = etapas[clave]
        lienzo = np.full((h + alto_cabecera + margen, w + 2 * margen, 3), 245, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        lienzo[alto_cabecera:alto_cabecera + h, margen:margen + w] = img_bgr
        cv2.rectangle(
            lienzo,
            (margen, alto_cabecera),
            (margen + w, alto_cabecera + h),
            (185, 185, 185),
            1
        )
        cv2.putText(
            lienzo, titulo, (margen + 4, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (25, 25, 25), 1, cv2.LINE_AA
        )
        paneles.append(lienzo)

    mosaico = np.hstack(paneles)

    if nombre_archivo:
        barra_sup = np.full((28, mosaico.shape[1], 3), 230, dtype=np.uint8)
        texto = f"Archivo: {nombre_archivo} | Dimensiones preservadas: {w}x{h} px (Sin recorte de ROI)"
        cv2.putText(
            barra_sup, texto, (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 40, 40), 1, cv2.LINE_AA
        )
        mosaico = np.vstack([barra_sup, mosaico])

    return mosaico


# ==============================================================================
# PROCESAMIENTO POR LOTES RESPETANDO ESTRUCTURA DE CARPETAS E IDENTIDADES
# ==============================================================================

def recolectar_imagenes(carpeta_entrada):
    """
    Recorre recursivamente 'carpeta_entrada' y devuelve una lista ordenada de tuplas
    (ruta_absoluta, ruta_relativa_desde_entrada) para todas las imágenes válidas.
    """
    carpeta_entrada = os.path.abspath(carpeta_entrada)
    encontradas = []
    if not os.path.isdir(carpeta_entrada):
        return encontradas

    for dirpath, dirnames, filenames in os.walk(carpeta_entrada):
        dirnames.sort()
        for f in sorted(filenames):
            if f.lower().endswith(EXTENSIONES_VALIDAS):
                ruta_abs = os.path.join(dirpath, f)
                ruta_rel = os.path.relpath(ruta_abs, carpeta_entrada)
                encontradas.append((ruta_abs, ruta_rel))
    return encontradas


def procesar_conjunto(carpeta_entrada, carpeta_salida, etapa_principal="esqueleto",
                      guardar_etapas=True, guardar_comparativas=True, params=None):
    """
    Procesa todas las imágenes de 'carpeta_entrada' y guarda los resultados en
    'carpeta_salida' sin modificar ni sobrescribir jamás los archivos originales:
      - En '<carpeta_salida>/<identidad>/<nombre_original>' guarda la imagen procesada
        final (por defecto 'esqueleto', o la etapa elegida) respetando exactamente la
        estructura de carpetas, identidades y nombres de archivo.
      - Si 'guardar_etapas=True', guarda además cada etapa intermedia en
        '<carpeta_salida>_etapas/<etapa>/<identidad>/<nombre_original>'.
      - Si 'guardar_comparativas=True', guarda el mosaico visual comparativo (0 a 5) en
        '<carpeta_salida>_comparativas/<identidad>/<nombre_sin_ext>_comparativa.png'.
    """
    abs_in = os.path.abspath(carpeta_entrada)
    abs_out = os.path.abspath(carpeta_salida)

    # Protección estricta contra sobrescritura de originales
    if abs_in == abs_out or abs_out.startswith(abs_in + os.sep):
        raise ValueError(
            f"La carpeta de salida ({abs_out}) no puede coincidir ni estar dentro "
            f"de la carpeta de entrada original ({abs_in})."
        )

    lista_imagenes = recolectar_imagenes(abs_in)
    total = len(lista_imagenes)
    if total == 0:
        print(f"[!] No se encontraron imágenes en: {carpeta_entrada}")
        return 0

    carpeta_etapas = f"{abs_out}_etapas"
    carpeta_comp = f"{abs_out}_comparativas"

    print("\n" + "=" * 72)
    print(" PREPROCESAMIENTO DE HUELLAS DACTILARES SINTÉTICAS")
    print("=" * 72)
    print(f" - Carpeta de entrada (intacta) : {carpeta_entrada}")
    print(f" - Carpeta de salida principal  : {carpeta_salida} (etapa: {etapa_principal})")
    if guardar_etapas:
        print(f" - Carpeta de etapas intermedias: {carpeta_etapas}")
    if guardar_comparativas:
        print(f" - Carpeta de comparativas (0-5): {carpeta_comp}")
    print(f" - Total de imágenes detectadas : {total}")
    print(" - Restricción de tamaño        : Imagen completa conservada (SIN recorte ROI)")
    print("=" * 72)

    procesadas = 0
    dimensiones_vistas = set()

    for idx, (ruta_abs, ruta_rel) in enumerate(lista_imagenes, 1):
        img = cv2.imread(ruta_abs, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[!] Error al leer la imagen: {ruta_abs}")
            continue

        h_orig, w_orig = img.shape[:2]
        dimensiones_vistas.add(f"{w_orig}x{h_orig} px")

        etapas = preprocesar_imagen_completa(img, params=params)

        # 1. Guardar resultado principal respetando identidad y nombre de archivo exactos
        ruta_dest_principal = os.path.join(abs_out, ruta_rel)
        os.makedirs(os.path.dirname(ruta_dest_principal), exist_ok=True)
        cv2.imwrite(ruta_dest_principal, etapas[etapa_principal])

        # 2. Guardar cada etapa individual para inspección o extracción de características futura
        if guardar_etapas:
            for nombre_etapa in ("normalizada", "sin_ruido", "gabor", "binarizada", "esqueleto"):
                ruta_etapa = os.path.join(carpeta_etapas, nombre_etapa, ruta_rel)
                os.makedirs(os.path.dirname(ruta_etapa), exist_ok=True)
                cv2.imwrite(ruta_etapa, etapas[nombre_etapa])

        # 3. Guardar lámina comparativa visual (Original + 5 etapas)
        if guardar_comparativas:
            sub_dir = os.path.dirname(ruta_rel)
            nombre_base, _ = os.path.splitext(os.path.basename(ruta_rel))
            ruta_comp = os.path.join(carpeta_comp, sub_dir, f"{nombre_base}_comparativa.png")
            os.makedirs(os.path.dirname(ruta_comp), exist_ok=True)
            mosaico = crear_mosaico_comparativo(etapas, nombre_archivo=ruta_rel)
            cv2.imwrite(ruta_comp, mosaico)

        procesadas += 1
        print(f"[{procesadas:02d}/{total:02d}] {ruta_rel} ({w_orig}x{h_orig} px) -> OK")

    dims_str = ", ".join(sorted(dimensiones_vistas))
    print("\n[+] Preprocesamiento finalizado exitosamente.")
    print(f"    Imágenes procesadas : {procesadas}/{total}")
    print(f"    Dimensiones intactas: {dims_str}")
    return procesadas


# ==============================================================================
# MENÚ INTERACTIVO Y LÍNEA DE COMANDOS (CLI)
# ==============================================================================

def menu_interactivo():
    print("\n" + "=" * 72)
    print("   MÓDULO DE PREPROCESAMIENTO DE HUELLAS SINTÉTICAS (IA1 - UNCUYO)")
    print("=" * 72)

    candidatos = [
        ("dataset_huellas/Personas", "preprocesadas/dataset_huellas/Personas"),
        ("huellas_prueba", "preprocesadas/huellas_prueba"),
        ("dataset_huellas/Fingerprints", "preprocesadas/dataset_huellas/Fingerprints"),
    ]
    disponibles = [(e, s) for e, s in candidatos if os.path.isdir(e)]

    print("\nConjuntos de imágenes detectados en el proyecto:")
    for i, (ent, sal) in enumerate(disponibles, 1):
        n_imgs = len(recolectar_imagenes(ent))
        print(f"  [{i}] {ent} ({n_imgs} imágenes) -> salida en '{sal}'")
    print(f"  [{len(disponibles) + 1}] Procesar TODOS los conjuntos anteriores (Personas + Prueba)")
    print("  [0] Especificar otra carpeta manualmente")

    op = input(f"\nSeleccioná una opción [Default: {len(disponibles) + 1}]: ").strip()

    lotes = []
    if not op or (op.isdigit() and int(op) == len(disponibles) + 1):
        # Por defecto procesa Personas y huellas_prueba
        for ent, sal in disponibles:
            if "Personas" in ent or "huellas_prueba" in ent:
                lotes.append((ent, sal))
    elif op.isdigit() and 1 <= int(op) <= len(disponibles):
        lotes.append(disponibles[int(op) - 1])
    elif op == "0":
        ent = input("Ruta de la carpeta de entrada: ").strip()
        sal = input("Ruta de la carpeta de salida separada [Default: preprocesadas/custom]: ").strip()
        if not sal:
            sal = "preprocesadas/custom"
        lotes.append((ent, sal))
    else:
        print("[!] Opción no reconocida. Abortando.")
        return

    print("\n¿Qué representación deseás guardar en la carpeta de salida principal?")
    print("  [1] Esqueleto topológico de 1 px (Recomendado para análisis de minucias)")
    print("  [2] Binarizada limpia (Crestas rellenas y depuradas)")
    print("  [3] Realce direccional de Gabor en escala de grises")
    op_et = input("Seleccioná etapa principal [Default: 1]: ").strip()
    mapa_et = {"1": "esqueleto", "2": "binarizada", "3": "gabor"}
    etapa_principal = mapa_et.get(op_et, "esqueleto")

    for ent, sal in lotes:
        procesar_conjunto(
            carpeta_entrada=ent,
            carpeta_salida=sal,
            etapa_principal=etapa_principal,
            guardar_etapas=True,
            guardar_comparativas=True
        )


def main():
    parser = argparse.ArgumentParser(
        description="Módulo de preprocesamiento de huellas dactilares sintéticas (sin recorte de ROI)."
    )
    parser.add_argument(
        "--entrada", type=str, default=None,
        help="Carpeta de entrada con las imágenes originales (ej: dataset_huellas/Personas)"
    )
    parser.add_argument(
        "--salida", type=str, default=None,
        help="Carpeta de salida separada donde se guardarán los resultados"
    )
    parser.add_argument(
        "--etapa-principal",
        choices=["normalizada", "sin_ruido", "gabor", "binarizada", "esqueleto"],
        default="esqueleto",
        help="Etapa que se guardará en la estructura principal de salida (default: esqueleto)"
    )
    parser.add_argument(
        "--sin-etapas", action="store_true",
        help="No guardar subcarpetas adicionales con cada etapa intermedia individual"
    )
    parser.add_argument(
        "--sin-comparativas", action="store_true",
        help="No generar láminas PNG comparativas de las 6 etapas"
    )
    parser.add_argument(
        "--procesar-todo", action="store_true",
        help="Procesar automáticamente 'dataset_huellas/Personas' y 'huellas_prueba' en 'preprocesadas/'"
    )

    args = parser.parse_args()

    if args.procesar_todo:
        conjuntos = [
            ("dataset_huellas/Personas", "preprocesadas/dataset_huellas/Personas"),
            ("huellas_prueba", "preprocesadas/huellas_prueba"),
        ]
        for ent, sal in conjuntos:
            if os.path.isdir(ent):
                procesar_conjunto(
                    carpeta_entrada=ent,
                    carpeta_salida=sal,
                    etapa_principal=args.etapa_principal,
                    guardar_etapas=not args.sin_etapas,
                    guardar_comparativas=not args.sin_comparativas,
                )
    elif args.entrada:
        salida = args.salida
        if not salida:
            norm_in = os.path.normpath(args.entrada)
            salida = os.path.join("preprocesadas", norm_in)
        procesar_conjunto(
            carpeta_entrada=args.entrada,
            carpeta_salida=salida,
            etapa_principal=args.etapa_principal,
            guardar_etapas=not args.sin_etapas,
            guardar_comparativas=not args.sin_comparativas,
        )
    else:
        menu_interactivo()


if __name__ == "__main__":
    main()

