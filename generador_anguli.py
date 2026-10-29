#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de Huellas Dactilares Sintéticas utilizando Anguli (MinGW/Wine).
Proyecto Final - Inteligencia Artificial I (2026) - UNCUYO

Permite ejecutar Anguli.exe en Linux a través de Wine, generando huellas
dactilares sintéticas listas para ser procesadas por el agente de control de acceso.
"""

import argparse
import datetime
import os
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, List, Tuple


RUTA_ANGULI_DEFAULT = os.path.expanduser("~/Downloads/Anguli-MinGw")
RUTA_BANDEJA_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bandeja_agente"
)

CLASES_PERMITIDAS = ["natural", "arch", "tarch", "right", "left", "dloop", "whirl"]


def verificar_entorno(directorio_anguli: str = RUTA_ANGULI_DEFAULT) -> Tuple[bool, str]:
    """Verifica la presencia de Wine y de Anguli.exe."""
    wine_path = shutil.which("wine")
    if not wine_path:
        return False, "Wine no está instalado en el sistema. Ejecute: sudo apt install wine"

    exe_path = os.path.join(directorio_anguli, "Anguli.exe")
    if not os.path.isfile(exe_path):
        return False, f"No se encontró Anguli.exe en '{directorio_anguli}'."

    return True, "Entorno verificado correctamente."


def generar_huella_sintetica(
    directorio_destino: str = RUTA_BANDEJA_DEFAULT,
    nombre_archivo: Optional[str] = None,
    clase: str = "natural",
    con_impresion: bool = False,
    ruido: Optional[int] = None,
    rayones: Optional[int] = None,
    rotacion: Optional[int] = None,
    traslacion: Optional[int] = None,
    seed: Optional[int] = None,
    directorio_anguli: str = RUTA_ANGULI_DEFAULT,
    verbose: bool = True,
) -> str:
    """
    Ejecuta Anguli mediante Wine para generar una nueva huella dactilar sintética.

    Parámetros:
      - directorio_destino: Carpeta donde se guardará la imagen generada.
      - nombre_archivo: Nombre del archivo de salida (si es None, se genera uno con timestamp).
      - clase: 'natural', 'arch', 'tarch', 'right', 'left', 'dloop', 'whirl'.
      - con_impresion: Si True, genera una impresión simulada con distorsiones físicas.
                       Si False, genera la huella sintética maestra (ideal).
      - ruido: Nivel de ruido (0 a 8).
      - rayones: Nivel/cantidad de rayones (0 a 14).
      - rotacion: Grados de rotación máxima.
      - traslacion: Píxeles de traslación máxima.
      - seed: Semilla aleatoria (entero). Si es None, se genera una aleatoria.
      - directorio_anguli: Directorio donde reside Anguli.exe.
      - verbose: Imprimir mensajes de progreso en consola.

    Retorna:
      Ruta absoluta de la imagen de huella generada.
    """
    if clase not in CLASES_PERMITIDAS:
        raise ValueError(f"Clase inválida '{clase}'. Opciones: {CLASES_PERMITIDAS}")

    ok, msg = verificar_entorno(directorio_anguli)
    if not ok:
        raise RuntimeError(msg)

    os.makedirs(directorio_destino, exist_ok=True)

    if seed is None:
        seed = random.randint(1, 999999)

    # Creamos un directorio temporal para la salida intermedia de Anguli
    with tempfile.TemporaryDirectory(prefix="anguli_run_") as tmp_dir:
        cmd: List[str] = [
            "wine",
            "Anguli.exe",
            "-num", "1",
            "-cdist", clase,
            "-seed", str(seed),
            "-outdir", tmp_dir,
        ]

        if con_impresion:
            cmd.extend(["-ni", "1"])
            if ruido is not None:
                cmd.extend(["-noise", f"0 {ruido}"])
            if rayones is not None:
                cmd.extend(["-scratch", f"0 {rayones}"])
            if rotacion is not None:
                cmd.extend(["-rot", str(rotacion)])
            if traslacion is not None:
                cmd.extend(["-trans", str(traslacion)])

        if verbose:
            print(f"[Anguli] Generando huella sintética (clase='{clase}', seed={seed})...")

        # Anguli requiere ejecutarse con cwd en su carpeta para cargar sus DLLs y recursos
        env = os.environ.copy()
        env["WINEDEBUG"] = "-all"

        resultado = subprocess.run(
            cmd,
            cwd=directorio_anguli,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if resultado.returncode != 0:
            raise RuntimeError(
                f"Anguli falló con código {resultado.returncode}:\n{resultado.stderr}"
            )

        # Ubicar la imagen resultante
        if con_impresion:
            ruta_origen = os.path.join(tmp_dir, "Impression_1", "fp_1", "1.jpg")
        else:
            ruta_origen = os.path.join(tmp_dir, "Fingerprints", "fp_1", "1.jpg")

        if not os.path.isfile(ruta_origen):
            # Fallback de búsqueda si la estructura difiere
            encontrados = []
            for root, _, files in os.walk(tmp_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".png", ".bmp", ".tif")):
                        encontrados.append(os.path.join(root, f))
            if not encontrados:
                raise FileNotFoundError(
                    f"No se encontró ninguna imagen generada por Anguli en {tmp_dir}"
                )
            ruta_origen = encontrados[0]

        # Definir nombre de destino
        if not nombre_archivo:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            tipo = "impr" if con_impresion else "sintetica"
            nombre_archivo = f"huella_{tipo}_{clase}_{timestamp}.jpg"

        ruta_final = os.path.join(directorio_destino, nombre_archivo)
        shutil.copy2(ruta_origen, ruta_final)

        if verbose:
            print(f"[Anguli] ✓ Huella generada exitosamente en:\n  -> {ruta_final}")

        return os.path.abspath(ruta_final)


def main():
    parser = argparse.ArgumentParser(
        description="Generador de huellas sintéticas con Anguli vía Wine (Linux)."
    )
    parser.add_argument(
        "-d", "--destino",
        default=RUTA_BANDEJA_DEFAULT,
        help=f"Directorio de destino (por defecto: {RUTA_BANDEJA_DEFAULT})",
    )
    parser.add_argument(
        "-o", "--nombre",
        default=None,
        help="Nombre del archivo de salida (por defecto: huella_sintetica_<clase>_<timestamp>.jpg)",
    )
    parser.add_argument(
        "-c", "--clase",
        choices=CLASES_PERMITIDAS,
        default="natural",
        help="Distribución/tipo de huella (natural, arch, tarch, right, left, dloop, whirl)",
    )
    parser.add_argument(
        "--impresion",
        action="store_true",
        help="Generar impresión simulada (con variaciones físicas) en vez de la huella pura.",
    )
    parser.add_argument(
        "--ruido",
        type=int,
        default=None,
        help="Nivel de ruido de 0 a 8 (aplica con --impresion).",
    )
    parser.add_argument(
        "--rayones",
        type=int,
        default=None,
        help="Nivel de rayones de 0 a 14 (aplica con --impresion).",
    )
    parser.add_argument(
        "--rotacion",
        type=int,
        default=None,
        help="Rotación máxima en grados (aplica con --impresion).",
    )
    parser.add_argument(
        "--traslacion",
        type=int,
        default=None,
        help="Traslación máxima en píxeles (aplica con --impresion).",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Semilla aleatoria (entero).",
    )
    parser.add_argument(
        "--anguli-dir",
        default=RUTA_ANGULI_DEFAULT,
        help=f"Ruta a la carpeta de Anguli-MinGw (por defecto: {RUTA_ANGULI_DEFAULT})",
    )

    args = parser.parse_args()

    try:
        ruta = generar_huella_sintetica(
            directorio_destino=args.destino,
            nombre_archivo=args.nombre,
            clase=args.clase,
            con_impresion=args.impresion,
            ruido=args.ruido,
            rayones=args.rayones,
            rotacion=args.rotacion,
            traslacion=args.traslacion,
            seed=args.seed,
            directorio_anguli=args.anguli_dir,
            verbose=True,
        )
        print(f"\nListo: {ruta}")
    except Exception as e:
        print(f"Error al generar huella: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
