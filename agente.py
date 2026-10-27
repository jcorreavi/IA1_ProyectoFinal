#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agente Inteligente de Control de Acceso Biométrico Multimodal
(Reconocimiento de Huellas Dactilares con K-Means Controlado por Voz con K-NN)
Trabajo Final - Inteligencia Artificial I (2026) - Ingeniería en Mecatrónica (UNCUYO)

Integra los módulos desarrollados en el proyecto:
  - `grabador_voz.py`                  : Captura acústica y gestión del micrófono.
  - `reconocimiento_voz_knn.py`        : Extracción MFCC/espectral, clasificador K-NN (K=5) y PCA de voz.
  - `preprocesamiento.py`              : Normalización, filtrado Mediana+Gauss, Gabor, binarización y esqueleto.
  - `reconocimiento_huellas_kmeans.py` : Extracción topológica/direccional, K-Means (k=3), umbral τ y PCA.

Comportamiento estricto según consignas del Trabajo Final:
  1. Selección y previsualización de huellas desde una carpeta unificada (`bandeja_agente/`),
     generada copiando las muestras de `huellas_prueba/` (autorizadas e intrusas) sin alterar
     los directorios originales.
  2. Comando de voz «Reconocer»: evalúa la huella seleccionada y muestra únicamente
     "AUTORIZADO" o "NO AUTORIZADO" según el umbral de distancia al centroide K-Means.
  3. Comando de voz «Identificar»: exige secuencia estricta (haber ejecutado antes «Reconocer»
     sobre la huella actual); si fue autorizada, revela "Identidad Sintética #1/#2/#3".
  4. Comando de voz «Salir»: concluye de forma limpia la ejecución del agente.
  5. Opción activable en la interfaz para guardar o no todas las fotos, etapas y gráficos PCA
     en una carpeta de salida (`salidas_agente/`) creando una subcarpeta única por cada análisis.
"""

import argparse
import datetime
import io
import os
import re
import shutil
import sys
import tempfile
import threading
from typing import Dict, List, Optional, Tuple

# Garantizar directorio de caché escribible para Matplotlib en cualquier entorno Linux
_mpl_default = os.path.expanduser("~/.config/matplotlib")
if not os.access(_mpl_default, os.W_OK):
    os.environ.setdefault(
        "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_cache_ia1")
    )

import matplotlib
matplotlib.use("Agg")  # Renderizado offscreen seguro para convivir con Tkinter sin conflictos

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import grabador_voz as gv
import preprocesamiento as prep
import reconocimiento_huellas_kmeans as rhk
import reconocimiento_voz_knn as rv


# ==============================================================================
# CONFIGURACIÓN DE CARPETAS DEL AGENTE
# ==============================================================================

DIRECTORIO_PRUEBA_ORIGEN = rhk.DIRECTORIO_PRUEBA_DEFAULT
DIRECTORIO_BANDEJA_DEFAULT = "bandeja_agente"
DIRECTORIO_SALIDAS_AGENTE_DEFAULT = "salidas_agente"

CUENTA_REGRESIVA_SEG = 3
DURACION_COMANDO_SEG = gv.DURACION_DEFAULT


# ==============================================================================
# PREPARACIÓN DE LA BANDEJA UNIFICADA DE HUELLAS (SIN TOCAR ORIGINALES)
# ==============================================================================

def sincronizar_bandeja_agente(
    carpeta_origen: str = DIRECTORIO_PRUEBA_ORIGEN,
    carpeta_bandeja: str = DIRECTORIO_BANDEJA_DEFAULT,
) -> int:
    """
    Copia las imágenes de `huellas_prueba/` (tanto de `Persona_1..3` como de `intrusos`)
    hacia una única carpeta plana `bandeja_agente/` para elegirlas cómodamente desde la
    interfaz gráfica, preservando intacta la estructura original de `huellas_prueba/`
    para los análisis estadísticos del informe.

    Nota: el clasificador K-Means evalúa exclusivamente la matriz de píxeles de la imagen
    y jamás inspecciona el nombre del archivo.
    """
    os.makedirs(carpeta_bandeja, exist_ok=True)
    if not os.path.isdir(carpeta_origen):
        return len(listar_imagenes_en_carpeta(carpeta_bandeja))

    copiadas = 0
    # 1. Copiar imágenes directas si la carpeta origen es plana (ej. carpeta scrambled o anonimizada)
    for elemento in sorted(os.listdir(carpeta_origen)):
        ruta_elem = os.path.join(carpeta_origen, elemento)
        if os.path.isfile(ruta_elem) and elemento.lower().endswith(prep.EXTENSIONES_VALIDAS):
            destino_abs = os.path.join(carpeta_bandeja, elemento)
            if not os.path.exists(destino_abs):
                shutil.copy2(ruta_elem, destino_abs)
                copiadas += 1

    # 2. Copiar imágenes desde subcarpetas (Persona_1, Persona_2, intrusos, etc.)
    for sub in sorted(os.listdir(carpeta_origen)):
        ruta_sub = os.path.join(carpeta_origen, sub)
        if not os.path.isdir(ruta_sub) or sub.startswith("."):
            continue

        es_intruso = "intrus" in sub.lower() or "no_aut" in sub.lower()
        for nombre in sorted(os.listdir(ruta_sub)):
            if not nombre.lower().endswith(prep.EXTENSIONES_VALIDAS):
                continue
            origen_abs = os.path.join(ruta_sub, nombre)
            # Evitar colisiones si en `intrusos/` hay nombres genéricos como `1.jpg`
            if es_intruso and not nombre.lower().startswith("intrus"):
                nombre_limpio = nombre.replace(" ", "_")
                nombre_dest = f"intruso_{nombre_limpio}"
            else:
                nombre_dest = nombre

            destino_abs = os.path.join(carpeta_bandeja, nombre_dest)
            if not os.path.exists(destino_abs):
                shutil.copy2(origen_abs, destino_abs)
                copiadas += 1

    return copiadas


def listar_imagenes_en_carpeta(carpeta: str) -> List[str]:
    """Devuelve las rutas absolutas ordenadas de todas las imágenes directas en `carpeta`."""
    if not os.path.isdir(carpeta):
        return []
    archivos = []
    for f in sorted(os.listdir(carpeta)):
        ruta = os.path.join(carpeta, f)
        if os.path.isfile(ruta) and f.lower().endswith(prep.EXTENSIONES_VALIDAS):
            archivos.append(os.path.abspath(ruta))
    return archivos


# ==============================================================================
# UTILIDADES DE CONVERSIÓN DE IMÁGENES PARA TKINTER
# ==============================================================================

def _cv2_a_pil(img_cv: np.ndarray) -> Image.Image:
    """Convierte una matriz NumPy en escala de grises o BGR a objeto PIL RGB."""
    if img_cv.ndim == 2:
        return Image.fromarray(img_cv).convert("RGB")
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _ajustar_pil_a_caja(
    img_pil: Image.Image, ancho_max: int, alto_max: int, color_fondo=(248, 249, 250)
) -> Image.Image:
    """Escala proporcionalmente `img_pil` para encajar con nitidez dentro de `(ancho_max, alto_max)`."""
    ancho_max = max(120, int(ancho_max))
    alto_max = max(120, int(alto_max))
    copia = img_pil.copy()
    copia.thumbnail((ancho_max, alto_max), Image.Resampling.LANCZOS)
    lienzo = Image.new("RGB", (ancho_max, alto_max), color_fondo)
    ox = (ancho_max - copia.width) // 2
    oy = (alto_max - copia.height) // 2
    lienzo.paste(copia, (ox, oy))
    return lienzo


def _componer_vista_triple_huella(
    img_orig: np.ndarray, img_topo_bgr: np.ndarray, img_skel: np.ndarray
) -> Image.Image:
    """Crea una lámina clara con: Original | Topología (Core/Delta) | Esqueleto 1 px."""
    paneles_datos = [
        (_cv2_a_pil(img_orig), "Huella Original"),
        (_cv2_a_pil(img_topo_bgr), "Topología (Core y Delta)"),
        (_cv2_a_pil(img_skel), "Esqueleto Topológico (1 px)"),
    ]
    h, w = img_orig.shape[:2]
    cab = 32
    pad = 10
    tarjetas = []
    for p_img, titulo in paneles_datos:
        arr = np.full((h + cab + pad, w + 2 * pad, 3), 246, dtype=np.uint8)
        bgr = cv2.cvtColor(np.asarray(p_img), cv2.COLOR_RGB2BGR)
        arr[cab : cab + h, pad : pad + w] = bgr
        cv2.rectangle(arr, (pad, cab), (pad + w, cab + h), (185, 190, 198), 1)
        cv2.putText(
            arr,
            titulo,
            (pad + 4, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (30, 41, 59),
            1,
            cv2.LINE_AA,
        )
        tarjetas.append(arr)
    unida_bgr = np.hstack(tarjetas)
    return _cv2_a_pil(unida_bgr)


# ==============================================================================
# INTERFAZ GRÁFICA PRINCIPAL DEL AGENTE BIOMÉTRICO
# ==============================================================================

class AgenteControlAccesoGUI:
    """
    Agente interactivo con interfaz gráfica Tkinter que coordina:
      - Sensores simulados/reales: lector de huellas (bandeja de imágenes) + micrófono en vivo.
      - Perceptos y deliberación: clasificador acústico K-NN y agrupamiento dactilar K-Means + umbral.
      - Actuadores de interfaz: estado de acceso (AUTORIZADO / NO AUTORIZADO), identificación
        de identidad sintética con secuencia estricta, cierre por voz y exportación organizada.
    """

    # Estados posibles para la huella seleccionada actualmente
    ESTADO_SIN_SELECCION = "SIN_SELECCION"
    ESTADO_PENDIENTE = "PENDIENTE"
    ESTADO_RECONOCIDO_AUTORIZADO = "RECONOCIDO_AUTORIZADO"
    ESTADO_RECONOCIDO_RECHAZADO = "RECONOCIDO_RECHAZADO"
    ESTADO_IDENTIFICADO = "IDENTIFICADO"

    def __init__(
        self,
        root: tk.Tk,
        carpeta_bandeja: str = DIRECTORIO_BANDEJA_DEFAULT,
        carpeta_salidas: str = DIRECTORIO_SALIDAS_AGENTE_DEFAULT,
    ) -> None:
        self.root = root
        self.root.title(
            "Agente de Control de Acceso Biométrico — Reconocimiento de Huellas (K-Means) por Voz (K-NN) | IA1 UNCUYO"
        )
        self.root.geometry("1380x840")
        self.root.minsize(1120, 720)
        self.root.configure(bg="#f1f5f9")

        # Variables de configuración de carpetas y exportación
        self.var_carpeta_bandeja = tk.StringVar(value=os.path.abspath(carpeta_bandeja))
        self.var_carpeta_salidas = tk.StringVar(value=os.path.abspath(carpeta_salidas))
        self.var_guardar_graficos = tk.BooleanVar(value=True)
        self.var_ultimo_directorio_analisis = tk.StringVar(value="(Ningún análisis guardado aún)")

        # Estado interno del agente sobre la huella activa
        self.archivos_bandeja: List[str] = []
        self.ruta_huella_actual: Optional[str] = None
        self.imagen_original_actual: Optional[np.ndarray] = None
        self.resultado_huella_actual: Optional[Dict[str, object]] = None
        self.estado_huella: str = self.ESTADO_SIN_SELECCION

        # Subcarpeta activa del análisis en curso (para agrupar Reconocer + Identificar de la misma huella)
        self.carpeta_analisis_actual: Optional[str] = None
        self.contador_ordenes_en_analisis: int = 0

        # Imágenes PIL listas para re-escalar limpiamente al redimensionar las pestañas
        self.pil_vista_principal: Optional[Image.Image] = None
        self.pil_mosaico_etapas: Optional[Image.Image] = None
        self.pil_grafico_huella: Optional[Image.Image] = None
        self.pil_grafico_voz: Optional[Image.Image] = None
        self._photo_refs: Dict[str, ImageTk.PhotoImage] = {}

        # Control de captura de micrófono no bloqueante
        self.escuchando_microfono: bool = False
        self._after_countdown_id: Optional[str] = None

        # Cargar modelos entrenados (desde caché .joblib en milisegundos o entrenando si faltan)
        self._configurar_estilos()
        self._construir_interfaz()

        self._registrar_evento("Inicializando modelos K-NN (voz) y K-Means (huellas)...")
        self.root.update_idletasks()
        self.modelo_voz = rv.entrenar_o_cargar_modelo_voz()
        self.modelo_huellas = rhk.entrenar_o_cargar_modelo_huellas()
        self._actualizar_estado_microfono()

        # Sincronizar bandeja inicial y poblar lista
        sincronizar_bandeja_agente(DIRECTORIO_PRUEBA_ORIGEN, self.var_carpeta_bandeja.get())
        self._recargar_lista_huellas(seleccionar_primera=True)

        tau = float(self.modelo_huellas["umbral_distancia"])
        k_voz = int(self.modelo_voz["k_vecinos"])
        self._registrar_evento(
            f"Modelos listos — Voz K-NN (K={k_voz}, {self.modelo_voz['num_audios_entrenamiento']} audios) | "
            f"Huellas K-Means (k=3, umbral τ={tau:.2f})."
        )

        # Atajo de teclado: Barra espaciadora para hablar comando por micrófono
        self.root.bind("<space>", self._atajo_espacio_hablar)
        self.root.protocol("WM_DELETE_WINDOW", self._cerrar_agente)

    # --------------------------------------------------------------------------
    # CONSTRUCCIÓN DE LA INTERFAZ GRÁFICA
    # --------------------------------------------------------------------------

    def _configurar_estilos(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background="#f1f5f9")
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure(
            "Title.TLabel",
            background="#0f172a",
            foreground="#ffffff",
            font=("DejaVu Sans", 13, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#0f172a",
            foreground="#cbd5e1",
            font=("DejaVu Sans", 9),
        )
        style.configure(
            "Section.TLabel",
            background="#ffffff",
            foreground="#0f172a",
            font=("DejaVu Sans", 10, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background="#ffffff",
            foreground="#334155",
            font=("DejaVu Sans", 9),
        )
        style.configure("TNotebook", background="#f1f5f9")
        style.configure("TNotebook.Tab", font=("DejaVu Sans", 9, "bold"), padding=(12, 6))

    def _construir_interfaz(self) -> None:
        # 1. Encabezado superior institucional
        header = tk.Frame(self.root, bg="#0f172a", height=62, padx=18, pady=10)
        header.pack(side=tk.TOP, fill=tk.X)

        tit_frame = tk.Frame(header, bg="#0f172a")
        tit_frame.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(
            tit_frame,
            text="SISTEMA DE CONTROL DE ACCESO POR HUELLAS DACTILARES Y COMANDOS DE VOZ",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            tit_frame,
            text="Agente Inteligente | K-Means No Supervisado (k=3) + Umbral Experimental y Control Acústico K-NN (K=5) — IA1 UNCUYO",
            style="Subtitle.TLabel",
        ).pack(anchor="w")

        self.lbl_estado_mic_top = tk.Label(
            header,
            text="Micrófono: verificando...",
            bg="#1e293b",
            fg="#e2e8f0",
            font=("DejaVu Sans", 9, "bold"),
            padx=12,
            pady=6,
        )
        self.lbl_estado_mic_top.pack(side=tk.RIGHT, pady=2)

        # 2. Contenedor principal de 3 columnas
        main_pane = tk.Frame(self.root, bg="#f1f5f9", padx=12, pady=10)
        main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        main_pane.columnconfigure(0, weight=0, minsize=280)
        main_pane.columnconfigure(1, weight=1, minsize=560)
        main_pane.columnconfigure(2, weight=0, minsize=380)
        main_pane.rowconfigure(0, weight=1)

        # --- COLUMNA IZQUIERDA: Selección de Huella en Carpeta Unificada ---
        col_izq = ttk.Frame(main_pane, style="Card.TFrame", padding=12)
        col_izq.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        ttk.Label(
            col_izq,
            text="1. Bandeja de Huellas a Evaluar",
            style="Section.TLabel",
        ).pack(anchor="w")

        lbl_ruta_bandeja = tk.Label(
            col_izq,
            textvariable=self.var_carpeta_bandeja,
            bg="#f8fafc",
            fg="#475569",
            font=("DejaVu Sans", 8),
            anchor="w",
            padx=6,
            pady=4,
            relief="groove",
        )
        lbl_ruta_bandeja.pack(fill=tk.X, pady=(6, 6))

        btn_bar_izq = tk.Frame(col_izq, bg="#ffffff")
        btn_bar_izq.pack(fill=tk.X, pady=(0, 8))
        tk.Button(
            btn_bar_izq,
            text="Elegir carpeta...",
            command=self._elegir_carpeta_bandeja,
            bg="#e2e8f0",
            fg="#0f172a",
            activebackground="#cbd5e1",
            relief="flat",
            font=("DejaVu Sans", 8, "bold"),
            padx=8,
            pady=4,
            cursor="hand2",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(
            btn_bar_izq,
            text="Sincronizar",
            command=self._accion_sincronizar_bandeja,
            bg="#e2e8f0",
            fg="#0f172a",
            activebackground="#cbd5e1",
            relief="flat",
            font=("DejaVu Sans", 8),
            padx=8,
            pady=4,
            cursor="hand2",
        ).pack(side=tk.LEFT)

        self.lbl_conteo_huellas = ttk.Label(
            col_izq, text="0 muestras disponibles", style="Body.TLabel"
        )
        self.lbl_conteo_huellas.pack(anchor="w", pady=(0, 4))

        list_frame = tk.Frame(col_izq, bg="#ffffff")
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.listbox_huellas = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("DejaVu Sans Mono", 9),
            bg="#f8fafc",
            fg="#0f172a",
            selectbackground="#2563eb",
            selectforeground="#ffffff",
            activestyle="none",
            exportselection=False,
            relief="solid",
            borderwidth=1,
        )
        scrollbar.config(command=self.listbox_huellas.yview)
        self.listbox_huellas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox_huellas.bind("<<ListboxSelect>>", self._al_seleccionar_huella_lista)

        nota_secuencia = tk.Label(
            col_izq,
            text=(
                "Flujo del Agente:\n"
                " • Elegí una huella para previsualizarla.\n"
                " • Decí «Reconocer» → AUTORIZADO / NO AUT.\n"
                " • Decí «Identificar» → Identidad Sintética #X.\n"
                " • Decí «Salir» → Finaliza el agente."
            ),
            bg="#eff6ff",
            fg="#1e3a8a",
            justify=tk.LEFT,
            anchor="w",
            font=("DejaVu Sans", 8),
            padx=8,
            pady=8,
            relief="solid",
            borderwidth=1,
        )
        nota_secuencia.pack(fill=tk.X, pady=(10, 0))

        # --- COLUMNA CENTRAL: Previsualización y Gráficos en Pestañas ---
        col_cen = ttk.Frame(main_pane, style="Card.TFrame", padding=10)
        col_cen.grid(row=0, column=1, sticky="nsew", padx=4)

        top_cen = tk.Frame(col_cen, bg="#ffffff")
        top_cen.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(
            top_cen,
            text="2. Previsualización de Muestra y Diagnóstico Visual",
            style="Section.TLabel",
        ).pack(side=tk.LEFT)
        self.lbl_archivo_actual = tk.Label(
            top_cen,
            text="Muestra: —",
            bg="#ffffff",
            fg="#2563eb",
            font=("DejaVu Sans", 9, "bold"),
        )
        self.lbl_archivo_actual.pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(col_cen)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_principal = tk.Frame(self.notebook, bg="#f8fafc")
        self.tab_etapas = tk.Frame(self.notebook, bg="#f8fafc")
        self.tab_pca_huella = tk.Frame(self.notebook, bg="#f8fafc")
        self.tab_pca_voz = tk.Frame(self.notebook, bg="#f8fafc")

        self.notebook.add(self.tab_principal, text=" Huella / Previsualización ")
        self.notebook.add(self.tab_etapas, text=" Etapas Preprocesamiento (0-5) ")
        self.notebook.add(self.tab_pca_huella, text=" K-Means + PCA (Huella) ")
        self.notebook.add(self.tab_pca_voz, text=" K-NN + PCA (Última Voz) ")

        self.lbl_canvas_principal = tk.Label(
            self.tab_principal,
            bg="#f8fafc",
            fg="#64748b",
            text="Seleccioná una huella de la lista izquierda para previsualizarla.",
            font=("DejaVu Sans", 10),
        )
        self.lbl_canvas_principal.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.lbl_canvas_etapas = tk.Label(
            self.tab_etapas,
            bg="#f8fafc",
            fg="#64748b",
            text="Ejecutá el comando «Reconocer» sobre la huella para visualizar las 5 etapas de preprocesamiento.",
            font=("DejaVu Sans", 10),
        )
        self.lbl_canvas_etapas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.lbl_canvas_pca_huella = tk.Label(
            self.tab_pca_huella,
            bg="#f8fafc",
            fg="#64748b",
            text="Ejecutá el comando «Reconocer» para ver la ubicación en PCA 2D y la distancia a los centroides K-Means.",
            font=("DejaVu Sans", 10),
        )
        self.lbl_canvas_pca_huella.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.lbl_canvas_pca_voz = tk.Label(
            self.tab_pca_voz,
            bg="#f8fafc",
            fg="#64748b",
            text="Pronunciá un comando por micrófono para visualizar la onda recortada (VAD) y su proyección PCA K-NN.",
            font=("DejaVu Sans", 10),
        )
        self.lbl_canvas_pca_voz.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tab_principal.bind("<Configure>", lambda _e: self._redibujar_pestanas())
        self.notebook.bind("<<NotebookTabChanged>>", lambda _e: self._redibujar_pestanas())

        # Bitácora inferior en la columna central
        log_box = tk.Frame(col_cen, bg="#ffffff")
        log_box.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(log_box, text="Bitácora de eventos del Agente:", style="Body.TLabel").pack(
            anchor="w"
        )
        self.txt_bitacora = tk.Text(
            log_box,
            height=5,
            bg="#0f172a",
            fg="#e2e8f0",
            font=("DejaVu Sans Mono", 8),
            state=tk.DISABLED,
            relief="flat",
            padx=8,
            pady=6,
        )
        self.txt_bitacora.pack(fill=tk.X, pady=(3, 0))

        # --- COLUMNA DERECHA: Control por Voz (K-NN), Estado de Acceso y Guardado ---
        col_der = ttk.Frame(main_pane, style="Card.TFrame", padding=12)
        col_der.grid(row=0, column=2, sticky="nsew", padx=(8, 0))

        ttk.Label(
            col_der,
            text="3. Módulo de Control por Voz (K-NN)",
            style="Section.TLabel",
        ).pack(anchor="w")

        self.btn_hablar = tk.Button(
            col_der,
            text="🎤  DAR COMANDO POR VOZ  (Espacio)",
            command=self._iniciar_escucha_microfono,
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            font=("DejaVu Sans", 10, "bold"),
            relief="flat",
            pady=12,
            cursor="hand2",
        )
        self.btn_hablar.pack(fill=tk.X, pady=(8, 6))

        # Botón auxiliar para evaluar un archivo .wav con el mismo K-NN (útil si no hay mic físico)
        aux_voz_frame = tk.Frame(col_der, bg="#ffffff")
        aux_voz_frame.pack(fill=tk.X, pady=(0, 6))
        tk.Button(
            aux_voz_frame,
            text="Cargar orden desde archivo .wav...",
            command=self._probar_comando_desde_wav,
            bg="#f1f5f9",
            fg="#334155",
            activebackground="#e2e8f0",
            relief="solid",
            borderwidth=1,
            font=("DejaVu Sans", 8),
            padx=8,
            pady=3,
            cursor="hand2",
        ).pack(side=tk.LEFT)
        tk.Button(
            aux_voz_frame,
            text="Verificar micrófono",
            command=self._actualizar_estado_microfono,
            bg="#f1f5f9",
            fg="#334155",
            activebackground="#e2e8f0",
            relief="solid",
            borderwidth=1,
            font=("DejaVu Sans", 8),
            padx=8,
            pady=3,
            cursor="hand2",
        ).pack(side=tk.RIGHT)

        self.lbl_estado_escucha = tk.Label(
            col_der,
            text="Listo para escuchar («Reconocer», «Identificar» o «Salir»)",
            bg="#f8fafc",
            fg="#334155",
            font=("DejaVu Sans", 9, "bold"),
            pady=8,
            padx=8,
            wraplength=340,
            relief="groove",
        )
        self.lbl_estado_escucha.pack(fill=tk.X, pady=(2, 8))

        # Barras de probabilidad K-NN del último comando
        probs_card = tk.Frame(col_der, bg="#f8fafc", padx=10, pady=8, relief="solid", borderwidth=1)
        probs_card.pack(fill=tk.X, pady=(0, 12))
        self.lbl_ultimo_cmd_voz = tk.Label(
            probs_card,
            text="Último comando K-NN: —",
            bg="#f8fafc",
            fg="#0f172a",
            font=("DejaVu Sans", 9, "bold"),
            anchor="w",
        )
        self.lbl_ultimo_cmd_voz.pack(fill=tk.X, pady=(0, 4))

        self.barras_voz: Dict[str, Tuple[ttk.Progressbar, tk.Label]] = {}
        for cmd in rv.COMANDOS_ORDEN:
            fila_p = tk.Frame(probs_card, bg="#f8fafc")
            fila_p.pack(fill=tk.X, pady=2)
            tk.Label(
                fila_p,
                text=f"«{cmd}»",
                width=13,
                anchor="w",
                bg="#f8fafc",
                fg="#334155",
                font=("DejaVu Sans", 8, "bold"),
            ).pack(side=tk.LEFT)
            pb = ttk.Progressbar(fila_p, orient=tk.HORIZONTAL, length=150, mode="determinate", maximum=100.0)
            pb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
            lbl_pct = tk.Label(
                fila_p,
                text="0.0%",
                width=6,
                anchor="e",
                bg="#f8fafc",
                fg="#475569",
                font=("DejaVu Sans Mono", 8),
            )
            lbl_pct.pack(side=tk.RIGHT)
            self.barras_voz[cmd] = (pb, lbl_pct)

        # Panel de Decisión Biométrica (K-Means + Umbral)
        ttk.Label(
            col_der,
            text="4. Decisión del Sistema de Control de Acceso",
            style="Section.TLabel",
        ).pack(anchor="w", pady=(2, 6))

        self.frame_cartel_acceso = tk.Frame(
            col_der, bg="#e2e8f0", padx=12, pady=14, relief="solid", borderwidth=1
        )
        self.frame_cartel_acceso.pack(fill=tk.X, pady=(0, 8))

        self.lbl_cartel_estado = tk.Label(
            self.frame_cartel_acceso,
            text="SELECCIONE UNA HUELLA",
            bg="#e2e8f0",
            fg="#1e293b",
            font=("DejaVu Sans", 15, "bold"),
            wraplength=330,
        )
        self.lbl_cartel_estado.pack(fill=tk.X)

        self.lbl_cartel_sub = tk.Label(
            self.frame_cartel_acceso,
            text="Elija una muestra de la lista y pronuncie «Reconocer».",
            bg="#e2e8f0",
            fg="#475569",
            font=("DejaVu Sans", 8),
            wraplength=330,
            pady=4,
        )
        self.lbl_cartel_sub.pack(fill=tk.X)

        # Tarjeta de Identidad Sintética («Identificar») y métricas de distancia
        id_card = tk.Frame(col_der, bg="#f8fafc", padx=10, pady=10, relief="solid", borderwidth=1)
        id_card.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            id_card,
            text="Identidad («Identificar»):",
            bg="#f8fafc",
            fg="#475569",
            font=("DejaVu Sans", 8, "bold"),
            anchor="w",
        ).pack(fill=tk.X)

        self.lbl_identidad_resultado = tk.Label(
            id_card,
            text="— (Sin identificar)",
            bg="#f8fafc",
            fg="#0f172a",
            font=("DejaVu Sans", 12, "bold"),
            anchor="w",
            pady=3,
        )
        self.lbl_identidad_resultado.pack(fill=tk.X)

        self.lbl_metricas_distancia = tk.Label(
            id_card,
            text="Distancia mín. a centroide: —  |  Umbral τ: —",
            bg="#f8fafc",
            fg="#334155",
            font=("DejaVu Sans Mono", 8),
            anchor="w",
            justify=tk.LEFT,
        )
        self.lbl_metricas_distancia.pack(fill=tk.X, pady=(4, 0))

        # Sección 5: Guardado opcional en carpeta de salida con subcarpeta por análisis
        ttk.Label(
            col_der,
            text="5. Exportación de Gráficos y Fotos por Análisis",
            style="Section.TLabel",
        ).pack(anchor="w", pady=(2, 4))

        save_card = tk.Frame(col_der, bg="#f8fafc", padx=10, pady=8, relief="solid", borderwidth=1)
        save_card.pack(fill=tk.X)

        chk = tk.Checkbutton(
            save_card,
            text="Guardar gráficos y fotos en carpeta de salida\n(crea 1 subcarpeta por cada análisis)",
            variable=self.var_guardar_graficos,
            command=self._al_cambiar_opcion_guardado,
            bg="#f8fafc",
            fg="#0f172a",
            activebackground="#f8fafc",
            selectcolor="#ffffff",
            justify=tk.LEFT,
            anchor="w",
            font=("DejaVu Sans", 8, "bold"),
        )
        chk.pack(fill=tk.X)

        dir_out_row = tk.Frame(save_card, bg="#f8fafc")
        dir_out_row.pack(fill=tk.X, pady=(6, 4))
        tk.Label(
            dir_out_row,
            textvariable=self.var_carpeta_salidas,
            bg="#ffffff",
            fg="#475569",
            font=("DejaVu Sans", 8),
            anchor="w",
            padx=5,
            pady=3,
            relief="solid",
            borderwidth=1,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(
            dir_out_row,
            text="Carpeta...",
            command=self._elegir_carpeta_salidas,
            bg="#e2e8f0",
            fg="#0f172a",
            relief="flat",
            font=("DejaVu Sans", 8),
            padx=6,
            pady=2,
            cursor="hand2",
        ).pack(side=tk.RIGHT)

        tk.Label(
            save_card,
            text="Última subcarpeta generada:",
            bg="#f8fafc",
            fg="#64748b",
            font=("DejaVu Sans", 7, "bold"),
            anchor="w",
        ).pack(fill=tk.X)
        tk.Label(
            save_card,
            textvariable=self.var_ultimo_directorio_analisis,
            bg="#f8fafc",
            fg="#15803d",
            font=("DejaVu Sans Mono", 7),
            anchor="w",
            wraplength=330,
            justify=tk.LEFT,
        ).pack(fill=tk.X)

    # --------------------------------------------------------------------------
    # GESTIÓN DE LISTA DE HUELLAS Y PREVISUALIZACIÓN
    # --------------------------------------------------------------------------

    def _actualizar_estado_microfono(self) -> None:
        ok, _, desc = gv.verificar_disponibilidad_microfono()
        if ok:
            self.lbl_estado_mic_top.config(
                text=f"🎙 Micrófono OK: {desc[:48]}", bg="#14532d", fg="#dcfce7"
            )
        else:
            self.lbl_estado_mic_top.config(
                text="⚠ Micrófono no detectado (podés usar WAV)", bg="#7f1d1d", fg="#fee2e2"
            )

    def _registrar_evento(self, mensaje: str) -> None:
        hora = datetime.datetime.now().strftime("%H:%M:%S")
        linea = f"[{hora}] {mensaje}\n"
        self.txt_bitacora.config(state=tk.NORMAL)
        self.txt_bitacora.insert(tk.END, linea)
        self.txt_bitacora.see(tk.END)
        self.txt_bitacora.config(state=tk.DISABLED)

    def _recargar_lista_huellas(self, seleccionar_primera: bool = False) -> None:
        carpeta = self.var_carpeta_bandeja.get()
        self.archivos_bandeja = listar_imagenes_en_carpeta(carpeta)
        self.listbox_huellas.delete(0, tk.END)
        for idx, ruta in enumerate(self.archivos_bandeja, 1):
            self.listbox_huellas.insert(tk.END, f" {idx:02d}. {os.path.basename(ruta)}")

        n = len(self.archivos_bandeja)
        self.lbl_conteo_huellas.config(text=f"{n} muestras en la carpeta seleccionada")

        if seleccionar_primera and n > 0:
            self.listbox_huellas.selection_set(0)
            self._cargar_previsualizacion_huella(self.archivos_bandeja[0])

    def _elegir_carpeta_bandeja(self) -> None:
        elegida = filedialog.askdirectory(
            parent=self.root,
            title="Seleccionar carpeta con huellas dactilares",
            initialdir=self.var_carpeta_bandeja.get(),
        )
        if elegida:
            self.var_carpeta_bandeja.set(os.path.abspath(elegida))
            self._recargar_lista_huellas(seleccionar_primera=True)
            self._registrar_evento(f"Carpeta de huellas cambiada a: {elegida}")

    def _accion_sincronizar_bandeja(self) -> None:
        nuevas = sincronizar_bandeja_agente(
            DIRECTORIO_PRUEBA_ORIGEN, self.var_carpeta_bandeja.get()
        )
        self._recargar_lista_huellas(seleccionar_primera=(self.ruta_huella_actual is None))
        self._registrar_evento(
            f"Bandeja sincronizada desde '{DIRECTORIO_PRUEBA_ORIGEN}' ({nuevas} archivos nuevos copiados)."
        )

    def _elegir_carpeta_salidas(self) -> None:
        elegida = filedialog.askdirectory(
            parent=self.root,
            title="Seleccionar carpeta raíz para guardar análisis",
            initialdir=self.var_carpeta_salidas.get(),
        )
        if elegida:
            self.var_carpeta_salidas.set(os.path.abspath(elegida))
            self._registrar_evento(f"Carpeta de salidas configurada en: {elegida}")

    def _al_seleccionar_huella_lista(self, _event=None) -> None:
        sel = self.listbox_huellas.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.archivos_bandeja):
            ruta = self.archivos_bandeja[idx]
            if ruta != self.ruta_huella_actual:
                self._cargar_previsualizacion_huella(ruta)

    def _cargar_previsualizacion_huella(self, ruta_imagen: str) -> None:
        """
        Previsualiza la huella elegida y reinicia el estado de deliberación a PENDIENTE,
        exigiendo ejecutar primero «Reconocer» antes de habilitar «Identificar».
        """
        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror(
                "Error de lectura", f"No se pudo cargar la imagen:\n{ruta_imagen}"
            )
            return

        self.ruta_huella_actual = ruta_imagen
        self.imagen_original_actual = img
        self.resultado_huella_actual = None
        self.estado_huella = self.ESTADO_PENDIENTE

        # Cerrar vínculo con la subcarpeta de la huella anterior para que el próximo análisis cree la suya
        self.carpeta_analisis_actual = None
        self.contador_ordenes_en_analisis = 0

        nombre = os.path.basename(ruta_imagen)
        h, w = img.shape[:2]
        self.lbl_archivo_actual.config(text=f"Muestra: {nombre} ({w}x{h} px)")

        # Mostrar únicamente la previsualización limpia de la huella original
        self.pil_vista_principal = _cv2_a_pil(img)
        self.pil_mosaico_etapas = None
        self.pil_grafico_huella = None
        self.lbl_canvas_etapas.config(
            image="",
            text="Huella aún sin procesar.\nPronunciá el comando «Reconocer» para ejecutar el preprocesamiento.",
        )
        self.lbl_canvas_pca_huella.config(
            image="",
            text="Huella aún sin evaluar.\nPronunciá «Reconocer» para calcular distancias a los centroides K-Means.",
        )
        self.notebook.select(self.tab_principal)
        self._redibujar_pestanas()

        # Actualizar cartel de estado a PENDIENTE
        self._actualizar_cartel_acceso(
            titulo="PENDIENTE DE RECONOCIMIENTO",
            subtitulo=f"Muestra '{nombre}' previsualizada. Diga «RECONOCER» para iniciar la evaluación.",
            bg="#e0f2fe",
            fg="#0369a1",
        )
        self.lbl_identidad_resultado.config(text="— (Requiere «Reconocer» previo)", fg="#475569")
        tau = float(self.modelo_huellas["umbral_distancia"])
        self.lbl_metricas_distancia.config(
            text=f"Distancia mín. a centroide: —  |  Umbral τ: {tau:.2f}"
        )
        self._registrar_evento(f"Huella seleccionada para previsualización: '{nombre}'.")

    def _actualizar_cartel_acceso(
        self, titulo: str, subtitulo: str, bg: str, fg: str
    ) -> None:
        self.frame_cartel_acceso.config(bg=bg)
        self.lbl_cartel_estado.config(text=titulo, bg=bg, fg=fg)
        self.lbl_cartel_sub.config(text=subtitulo, bg=bg, fg=fg)

    def _redibujar_pestanas(self) -> None:
        """Ajusta dinámicamente la imagen de cada pestaña al tamaño disponible del panel."""
        pares = [
            ("principal", self.pil_vista_principal, self.lbl_canvas_principal),
            ("etapas", self.pil_mosaico_etapas, self.lbl_canvas_etapas),
            ("pca_h", self.pil_grafico_huella, self.lbl_canvas_pca_huella),
            ("pca_v", self.pil_grafico_voz, self.lbl_canvas_pca_voz),
        ]
        for clave, img_pil, widget in pares:
            if img_pil is None:
                continue
            w = widget.winfo_width()
            h = widget.winfo_height()
            if w < 80 or h < 80:
                w = max(560, self.notebook.winfo_width() - 24)
                h = max(420, self.notebook.winfo_height() - 48)
            ajustada = _ajustar_pil_a_caja(img_pil, w - 12, h - 12)
            photo = ImageTk.PhotoImage(ajustada)
            self._photo_refs[clave] = photo
            widget.config(image=photo, text="")

    # --------------------------------------------------------------------------
    # CAPTURA DE VOZ NO BLOQUEANTE Y CLASIFICACIÓN K-NN
    # --------------------------------------------------------------------------

    def _atajo_espacio_hablar(self, event) -> None:
        # Evitar disparar si el foco está en un cuadro de texto editable
        if isinstance(event.widget, tk.Entry):
            return
        self._iniciar_escucha_microfono()

    def _iniciar_escucha_microfono(self) -> None:
        if self.escuchando_microfono:
            return
        ok_mic, _, desc_mic = gv.verificar_disponibilidad_microfono()
        if not ok_mic:
            messagebox.showwarning(
                "Micrófono no disponible",
                f"No se detectó un micrófono activo:\n{desc_mic}\n\n"
                "Verificá la configuración del sistema o probá con un archivo .wav.",
            )
            return

        self.escuchando_microfono = True
        self.btn_hablar.config(state=tk.DISABLED, bg="#94a3b8")
        self._ejecutar_paso_cuenta_regresiva(CUENTA_REGRESIVA_SEG)

    def _ejecutar_paso_cuenta_regresiva(self, segundos_restantes: int) -> None:
        if segundos_restantes > 0:
            self.lbl_estado_escucha.config(
                text=f"⏳ Preparate para hablar en {segundos_restantes}...",
                bg="#fef3c7",
                fg="#92400e",
            )
            self._after_countdown_id = self.root.after(
                850, self._ejecutar_paso_cuenta_regresiva, segundos_restantes - 1
            )
        else:
            self.lbl_estado_escucha.config(
                text="🔴 ¡HABLÁ AHORA! Pronunciá: «Reconocer», «Identificar» o «Salir»",
                bg="#fee2e2",
                fg="#b91c1c",
            )
            self.root.update_idletasks()
            hilo = threading.Thread(target=self._hilo_captura_audio, daemon=True)
            hilo.start()

    def _hilo_captura_audio(self) -> None:
        try:
            audio_i16 = gv.capturar_audio_microfono(
                duracion_seg=DURACION_COMANDO_SEG,
                sample_rate=gv.SAMPLE_RATE_DEFAULT,
            )
            self.root.after(
                0, self._procesar_audio_capturado, audio_i16, gv.SAMPLE_RATE_DEFAULT, None
            )
        except Exception as exc:
            self.root.after(0, self._procesar_audio_capturado, None, gv.SAMPLE_RATE_DEFAULT, str(exc))

    def _probar_comando_desde_wav(self) -> None:
        if self.escuchando_microfono:
            return
        ruta_wav = filedialog.askopenfilename(
            parent=self.root,
            title="Seleccionar grabación WAV de comando («Reconocer», «Identificar» o «Salir»)",
            initialdir=gv.DIRECTORIO_DATASET_DEFAULT
            if os.path.isdir(gv.DIRECTORIO_DATASET_DEFAULT)
            else ".",
            filetypes=[("Archivos de audio WAV", "*.wav"), ("Todos los archivos", "*.*")],
        )
        if not ruta_wav:
            return
        try:
            audio_i16, sr = gv.leer_wav_mono(ruta_wav, normalizar_float=False)
            self._registrar_evento(f"Evaluando comando desde archivo WAV: {os.path.basename(ruta_wav)}")
            self._procesar_audio_capturado(audio_i16, sr, None)
        except Exception as exc:
            messagebox.showerror("Error al leer WAV", str(exc))

    def _procesar_audio_capturado(
        self,
        audio_i16: Optional[np.ndarray],
        sr: int,
        error_msg: Optional[str],
    ) -> None:
        self.escuchando_microfono = False
        self.btn_hablar.config(state=tk.NORMAL, bg="#2563eb")

        if error_msg or audio_i16 is None:
            self.lbl_estado_escucha.config(
                text=f"⚠ Error de captura: {error_msg}", bg="#fee2e2", fg="#991b1b"
            )
            self._registrar_evento(f"Error de micrófono: {error_msg}")
            return

        senal_float = audio_i16.astype(np.float32) / 32768.0
        res_voz = rv.clasificar_senal_audio(
            senal=senal_float,
            sr=sr,
            paquete_modelo=self.modelo_voz,
        )
        res_voz["senal_audio"] = senal_float
        res_voz["audio_int16"] = audio_i16
        res_voz["sample_rate"] = sr

        # Actualizar barras de probabilidad K-NN en la GUI
        probs: Dict[str, float] = res_voz.get("probabilidades", {})
        for cmd in rv.COMANDOS_ORDEN:
            pct = float(probs.get(cmd, 0.0)) * 100.0
            pb, lbl_p = self.barras_voz[cmd]
            pb["value"] = pct
            lbl_p.config(text=f"{pct:5.1f}%")

        # Renderizar en memoria (o en carpeta de análisis) el gráfico PCA de esta voz si hubo señal
        if res_voz.get("vector_pca") is not None:
            fd_tmp, ruta_tmp_voz = tempfile.mkstemp(suffix="_voz_pca.png")
            os.close(fd_tmp)
            try:
                rv.graficar_prediccion_en_pca(
                    paquete_modelo=self.modelo_voz,
                    senal=senal_float,
                    sr=sr,
                    resultado=res_voz,
                    mostrar_ventana=False,
                    ruta_salida_personalizada=ruta_tmp_voz,
                    silencioso=True,
                )
                with Image.open(ruta_tmp_voz) as im_v:
                    self.pil_grafico_voz = im_v.convert("RGB")
                self._redibujar_pestanas()
            finally:
                if os.path.exists(ruta_tmp_voz):
                    os.remove(ruta_tmp_voz)

        if not res_voz["es_valido"]:
            estado_v = str(res_voz["comando"])
            motivo = str(res_voz.get("motivo", ""))
            self.lbl_ultimo_cmd_voz.config(
                text=f"Último audio: {estado_v}", fg="#b91c1c"
            )
            self.lbl_estado_escucha.config(
                text=f"⚠ Comando no válido ({estado_v}). {motivo}",
                bg="#fef2f2",
                fg="#991b1b",
            )
            self._registrar_evento(f"Voz descartada ({estado_v}): {motivo}")
            return

        comando = str(res_voz["comando"])
        conf_pct = float(res_voz["confianza"]) * 100.0
        self.lbl_ultimo_cmd_voz.config(
            text=f"Comando reconocido: «{comando.upper()}» ({conf_pct:.1f}%)",
            fg="#15803d",
        )
        self.lbl_estado_escucha.config(
            text=f"✔ Orden ejecutada: «{comando}» (Confianza K-NN: {conf_pct:.1f}%)",
            bg="#dcfce7",
            fg="#166534",
        )
        self._registrar_evento(
            f"K-NN Voz reconoció comando «{comando}» (confianza: {conf_pct:.1f}%)."
        )

        # Ejecutar deliberación del Agente según el comando reconocido
        self._ejecutar_comando_agente(comando, res_voz)

    # --------------------------------------------------------------------------
    # LÓGICA DEL AGENTE SEGÚN COMANDOS («Reconocer», «Identificar», «Salir»)
    # --------------------------------------------------------------------------

    def _ejecutar_comando_agente(
        self, comando: str, res_voz: Dict[str, object]
    ) -> None:
        if comando == "Salir":
            self._ejecutar_salir(res_voz)
            return

        if self.ruta_huella_actual is None or self.imagen_original_actual is None:
            self._actualizar_cartel_acceso(
                titulo="SIN HUELLA SELECCIONADA",
                subtitulo="Seleccioná primero una huella de la bandeja izquierda.",
                bg="#fef3c7",
                fg="#92400e",
            )
            return

        if comando == "Reconocer":
            self._ejecutar_reconocer(res_voz)
        elif comando == "Identificar":
            self._ejecutar_identificar(res_voz)

    def _ejecutar_reconocer(self, res_voz: Dict[str, object]) -> None:
        """
        Comando obligatorio «Reconocer» (PDF pág. 4):
        Inicia la evaluación de la huella sintética seleccionada. Si pertenece a una
        clase autorizada muestra "AUTORIZADO"; de lo contrario, "NO AUTORIZADO".
        (Oculta la etiqueta de identidad hasta que el usuario solicite «Identificar»).
        """
        assert self.ruta_huella_actual is not None
        res_h = rhk.evaluar_huella(
            self.ruta_huella_actual,
            paquete_modelo=self.modelo_huellas,
            incluir_mosaico=True,
        )
        self.resultado_huella_actual = res_h

        # Construir visualizaciones para las pestañas (sin revelar identidad en el título aún)
        vis_topo = rhk.crear_imagen_huella_anotada(res_h)
        self.pil_vista_principal = _componer_vista_triple_huella(
            res_h["etapas"]["original"], vis_topo, res_h["etapas"]["esqueleto"]
        )
        self.pil_mosaico_etapas = _cv2_a_pil(res_h["mosaico_etapas"])
        self._actualizar_grafico_pca_huella_en_memoria(mostrar_identidad=False)

        dist_min = float(res_h["distancia_min"])
        umbral = float(res_h["umbral"])
        autorizado = bool(res_h["autorizado"])

        self.lbl_metricas_distancia.config(
            text=f"Distancia mín. a centroide: {dist_min:.2f}  |  Umbral τ: {umbral:.2f}"
        )

        if autorizado:
            self.estado_huella = self.ESTADO_RECONOCIDO_AUTORIZADO
            self._actualizar_cartel_acceso(
                titulo="AUTORIZADO",
                subtitulo=(
                    f"Distancia euclídea ({dist_min:.2f}) ≤ Umbral ({umbral:.2f}). "
                    "Pronuncie «Identificar» para consultar la identidad."
                ),
                bg="#dcfce7",
                fg="#14532d",
            )
            self.lbl_identidad_resultado.config(
                text="[Autorizado — diga «Identificar»]", fg="#15803d"
            )
            self._registrar_evento(
                f"Evaluación «Reconocer» en '{res_h['nombre_archivo']}': AUTORIZADO "
                f"(distancia={dist_min:.2f} ≤ τ={umbral:.2f})."
            )
        else:
            self.estado_huella = self.ESTADO_RECONOCIDO_RECHAZADO
            self._actualizar_cartel_acceso(
                titulo="NO AUTORIZADO",
                subtitulo=(
                    f"Acceso denegado: Distancia al centroide más cercano ({dist_min:.2f}) "
                    f"supera el umbral τ = {umbral:.2f}."
                ),
                bg="#fee2e2",
                fg="#991b1b",
            )
            self.lbl_identidad_resultado.config(
                text="NO AUTORIZADO (Muestra rechazada)", fg="#b91c1c"
            )
            self._registrar_evento(
                f"Evaluación «Reconocer» en '{res_h['nombre_archivo']}': NO AUTORIZADO "
                f"(distancia={dist_min:.2f} > τ={umbral:.2f})."
            )

        self._redibujar_pestanas()

        # Si está habilitado guardar en carpeta de salida, crear la subcarpeta de este análisis
        if self.var_guardar_graficos.get():
            self._asegurar_y_guardar_artefactos_huella(mostrar_identidad=False)
            self._guardar_artefactos_orden_voz("Reconocer", res_voz)

    def _ejecutar_identificar(self, res_voz: Dict[str, object]) -> None:
        """
        Comando obligatorio «Identificar» con SECUENCIA ESTRICTA:
          - Si la huella aún no fue evaluada con «Reconocer», exige ejecutar «Reconocer» primero.
          - Si ya fue evaluada y resultó AUTORIZADO, muestra "Identidad Sintética #1/#2/#3".
          - Si ya fue evaluada y resultó NO AUTORIZADO, informa que el acceso fue denegado.
        """
        if self.estado_huella == self.ESTADO_PENDIENTE or self.resultado_huella_actual is None:
            self._actualizar_cartel_acceso(
                titulo="EJECUTE «RECONOCER» PRIMERO",
                subtitulo=(
                    "Secuencia obligatoria: antes de «Identificar» debe evaluar "
                    "si la huella está autorizada diciendo «Reconocer»."
                ),
                bg="#fef3c7",
                fg="#92400e",
            )
            self._registrar_evento(
                "«Identificar» bloqueado por secuencia estricta: la muestra aún no fue evaluada con «Reconocer»."
            )
            return

        res_h = self.resultado_huella_actual
        dist_min = float(res_h["distancia_min"])
        umbral = float(res_h["umbral"])

        if self.estado_huella in (
            self.ESTADO_RECONOCIDO_AUTORIZADO,
            self.ESTADO_IDENTIFICADO,
        ):
            self.estado_huella = self.ESTADO_IDENTIFICADO
            identidad_str = str(res_h["identidad"])
            self._actualizar_cartel_acceso(
                titulo=f"AUTORIZADO — {identidad_str}",
                subtitulo=(
                    f"Identificación completada | Distancia al centroide: {dist_min:.2f} "
                    f"(Umbral τ = {umbral:.2f})"
                ),
                bg="#dcfce7",
                fg="#14532d",
            )
            self.lbl_identidad_resultado.config(text=identidad_str, fg="#1d4ed8")
            self._actualizar_grafico_pca_huella_en_memoria(mostrar_identidad=True)
            self._redibujar_pestanas()
            self._registrar_evento(
                f"Resultado «Identificar» en '{res_h['nombre_archivo']}': {identidad_str}."
            )

            if self.var_guardar_graficos.get():
                self._asegurar_y_guardar_artefactos_huella(mostrar_identidad=True)
                self._guardar_artefactos_orden_voz("Identificar", res_voz)

        elif self.estado_huella == self.ESTADO_RECONOCIDO_RECHAZADO:
            self._actualizar_cartel_acceso(
                titulo="NO AUTORIZADO",
                subtitulo=(
                    "No se puede identificar: la huella fue rechazada en «Reconocer» "
                    f"(distancia {dist_min:.2f} > τ = {umbral:.2f}, identidad desconocida)."
                ),
                bg="#fee2e2",
                fg="#991b1b",
            )
            self.lbl_identidad_resultado.config(
                text="ACCESO DENEGADO (Intruso / No registrado)", fg="#b91c1c"
            )
            self._registrar_evento(
                f"«Identificar» sobre muestra rechazada '{res_h['nombre_archivo']}': permanece NO AUTORIZADO."
            )
            if self.var_guardar_graficos.get():
                self._asegurar_y_guardar_artefactos_huella(mostrar_identidad=False)
                self._guardar_artefactos_orden_voz("Identificar", res_voz)

    def _ejecutar_salir(self, res_voz: Dict[str, object]) -> None:
        """Comando obligatorio «Salir»: concluye limpiamente la ejecución del agente."""
        if self.var_guardar_graficos.get() and self.carpeta_analisis_actual is not None:
            self._guardar_artefactos_orden_voz("Salir", res_voz)

        self._actualizar_cartel_acceso(
            titulo="CERRANDO AGENTE («SALIR»)",
            subtitulo="Ejecución concluida por comando de voz.",
            bg="#cbd5e1",
            fg="#0f172a",
        )
        self._registrar_evento("Comando «Salir» recibido. Finalizando ejecución del agente...")
        self.root.after(850, self._cerrar_agente)

    def _actualizar_grafico_pca_huella_en_memoria(self, mostrar_identidad: bool) -> None:
        if self.resultado_huella_actual is None:
            return
        fd_tmp, ruta_tmp = tempfile.mkstemp(suffix="_huella_pca.png")
        os.close(fd_tmp)
        try:
            rhk.graficar_evaluacion_huella(
                paquete_modelo=self.modelo_huellas,
                resultado_huella=self.resultado_huella_actual,
                mostrar_identidad_en_titulo=mostrar_identidad,
                mostrar_ventana=False,
                ruta_salida_personalizada=ruta_tmp,
            )
            with Image.open(ruta_tmp) as im_h:
                self.pil_grafico_huella = im_h.convert("RGB")
        finally:
            if os.path.exists(ruta_tmp):
                os.remove(ruta_tmp)

    # --------------------------------------------------------------------------
    # EXPORTACIÓN ORGANIZADA: 1 SUBCARPETA POR ANÁLISIS CON TODAS LAS FOTOS/GRÁFICAS
    # --------------------------------------------------------------------------

    def _al_cambiar_opcion_guardado(self) -> None:
        activo = self.var_guardar_graficos.get()
        estado_txt = "ACTIVADO" if activo else "DESACTIVADO"
        self._registrar_evento(f"Guardado automático de análisis en disco: {estado_txt}.")
        # Si el usuario activa el guardado teniendo ya una huella evaluada en pantalla, volcarla ahora
        if activo and self.resultado_huella_actual is not None and self.carpeta_analisis_actual is None:
            mostrar_id = self.estado_huella == self.ESTADO_IDENTIFICADO
            self._asegurar_y_guardar_artefactos_huella(mostrar_identidad=mostrar_id)

    def _crear_nueva_subcarpeta_analisis(self, nombre_archivo_huella: str) -> str:
        raiz_salidas = self.var_carpeta_salidas.get()
        os.makedirs(raiz_salidas, exist_ok=True)

        patron = re.compile(r"^analisis_(\d+)_", flags=re.IGNORECASE)
        max_idx = 0
        for entrada in os.listdir(raiz_salidas):
            m = patron.match(entrada)
            if m:
                max_idx = max(max_idx, int(m.group(1)))

        base_sin_ext = os.path.splitext(nombre_archivo_huella)[0]
        base_limpia = re.sub(r"[^A-Za-z0-9_\-]", "_", base_sin_ext)
        marca_tiempo = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_sub = f"analisis_{max_idx + 1:03d}_{marca_tiempo}_{base_limpia}"
        ruta_sub = os.path.join(raiz_salidas, nombre_sub)
        os.makedirs(ruta_sub, exist_ok=True)
        return ruta_sub

    def _asegurar_y_guardar_artefactos_huella(self, mostrar_identidad: bool) -> None:
        if self.resultado_huella_actual is None:
            return
        res_h = self.resultado_huella_actual
        es_nueva = False

        if self.carpeta_analisis_actual is None:
            self.carpeta_analisis_actual = self._crear_nueva_subcarpeta_analisis(
                str(res_h["nombre_archivo"])
            )
            self.contador_ordenes_en_analisis = 0
            es_nueva = True
            rel_mostrar = os.path.relpath(self.carpeta_analisis_actual)
            self.var_ultimo_directorio_analisis.set(rel_mostrar)

        carp = self.carpeta_analisis_actual

        if es_nueva:
            # 1. Mosaico comparativo con las etapas de la huella juntas (`preprocesamiento.py`)
            cv2.imwrite(
                os.path.join(carp, "01_mosaico_etapas_huella.png"),
                res_h["mosaico_etapas"],
            )

        # 2. Gráfico diagnóstico K-Means + PCA 2D + distancias a centroides
        rhk.graficar_evaluacion_huella(
            paquete_modelo=self.modelo_huellas,
            resultado_huella=res_h,
            mostrar_identidad_en_titulo=mostrar_identidad,
            mostrar_ventana=False,
            ruta_salida_personalizada=os.path.join(
                carp, "02_grafico_kmeans_pca_huella.png"
            ),
        )

        # 3. Actualizar archivo de resumen textual del análisis
        self._escribir_resumen_analisis(mostrar_identidad=mostrar_identidad)
        if es_nueva:
            self._registrar_evento(
                f"Guardado en subcarpeta: '{os.path.relpath(carp)}'."
            )

    def _guardar_artefactos_orden_voz(
        self, etiqueta_orden: str, res_voz: Dict[str, object]
    ) -> None:
        if self.carpeta_analisis_actual is None:
            return
        self.contador_ordenes_en_analisis += 1
        idx_v = self.contador_ordenes_en_analisis
        carp = self.carpeta_analisis_actual

        # Guardar WAV de la orden capturada y su gráfico Onda + VAD + PCA 2D de K-NN
        audio_i16 = res_voz.get("audio_int16")
        sr = int(res_voz.get("sample_rate", gv.SAMPLE_RATE_DEFAULT))
        if audio_i16 is not None:
            ruta_wav = os.path.join(
                carp, f"03_voz_{idx_v:02d}_{etiqueta_orden}.wav"
            )
            gv.guardar_wav_seguro(ruta_wav, audio_i16, sr)

        senal = res_voz.get("senal_audio")
        if self.pil_grafico_voz is not None:
            ruta_png_voz = os.path.join(
                carp, f"03_voz_{idx_v:02d}_{etiqueta_orden}_knn_pca.png"
            )
            self.pil_grafico_voz.save(ruta_png_voz, format="PNG")
        elif senal is not None and res_voz.get("vector_pca") is not None:
            ruta_png_voz = os.path.join(
                carp, f"03_voz_{idx_v:02d}_{etiqueta_orden}_knn_pca.png"
            )
            rv.graficar_prediccion_en_pca(
                paquete_modelo=self.modelo_voz,
                senal=senal,
                sr=sr,
                resultado=res_voz,
                mostrar_ventana=False,
                ruta_salida_personalizada=ruta_png_voz,
                silencioso=True,
            )

        # Registrar orden de voz en el resumen del análisis
        ruta_txt = os.path.join(carp, "resumen_analisis.txt")
        conf = float(res_voz.get("confianza", 0.0)) * 100.0
        hora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ruta_txt, "a", encoding="utf-8") as f:
            f.write(
                f"[{hora}] Comando de voz #{idx_v:02d}: «{etiqueta_orden}» "
                f"(Confianza K-NN: {conf:.1f}%)\n"
            )

    def _escribir_resumen_analisis(self, mostrar_identidad: bool) -> None:
        if self.carpeta_analisis_actual is None or self.resultado_huella_actual is None:
            return
        res_h = self.resultado_huella_actual
        ruta_txt = os.path.join(self.carpeta_analisis_actual, "resumen_analisis.txt")

        lineas_previas_voz: List[str] = []
        if os.path.isfile(ruta_txt):
            with open(ruta_txt, "r", encoding="utf-8") as f:
                for lin in f:
                    if lin.startswith("[") and "Comando de voz" in lin:
                        lineas_previas_voz.append(lin)

        id_texto = (
            str(res_h["identidad"])
            if mostrar_identidad and res_h["autorizado"]
            else (
                "NO AUTORIZADO"
                if not res_h["autorizado"]
                else "(Pendiente de comando «Identificar»)"
            )
        )

        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write("====================================================================\n")
            f.write(" RESUMEN DE ANÁLISIS BIOMÉTRICO — AGENTE IA1 (UNCUYO 2026)\n")
            f.write("====================================================================\n")
            f.write(f" Archivo evaluado         : {res_h['nombre_archivo']}\n")
            f.write(f" Resultado («Reconocer»)  : {res_h['estado_acceso']}\n")
            f.write(f" Identidad («Identificar»): {id_texto}\n")
            f.write(f" Distancia mín. centroide : {res_h['distancia_min']:.4f}\n")
            f.write(f" Umbral de aceptación (τ) : {res_h['umbral']:.4f}\n")
            f.write("\n Distancias euclídeas a los 3 centroides K-Means:\n")
            for id_k, d_v in res_h["distancias_por_identidad"].items():
                f.write(f"   - {id_k:<26}: {d_v:.4f}\n")
            f.write("\n Historial de comandos de voz en este análisis:\n")
            for lv in lineas_previas_voz:
                f.write(lv)

    # --------------------------------------------------------------------------
    # CIERRE SEGURO
    # --------------------------------------------------------------------------

    def _cerrar_agente(self) -> None:
        if self._after_countdown_id is not None:
            try:
                self.root.after_cancel(self._after_countdown_id)
            except Exception:
                pass
        self.root.destroy()


# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agente gráfico de reconocimiento de huellas dactilares controlado por voz."
    )
    parser.add_argument(
        "--bandeja",
        type=str,
        default=DIRECTORIO_BANDEJA_DEFAULT,
        help=f"Carpeta unificada de huellas para elegir en la interfaz (default: {DIRECTORIO_BANDEJA_DEFAULT})",
    )
    parser.add_argument(
        "--salidas",
        type=str,
        default=DIRECTORIO_SALIDAS_AGENTE_DEFAULT,
        help=f"Carpeta raíz donde crear subcarpetas por análisis (default: {DIRECTORIO_SALIDAS_AGENTE_DEFAULT})",
    )
    args = parser.parse_args()

    root = tk.Tk()
    _app = AgenteControlAccesoGUI(
        root=root,
        carpeta_bandeja=args.bandeja,
        carpeta_salidas=args.salidas,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
