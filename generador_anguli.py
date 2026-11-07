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
SEMILLA_DATASET_DEFAULT = 1604


def detectar_wine() -> Optional[str]:
    """Detecta el ejecutable de Wine en el sistema (soporta Wine 32/64 y rutas estándar de Linux)."""
    candidatos = [
        "wine",
        "wine64",
        "/usr/bin/wine",
        "/usr/bin/wine64",
        "/opt/wine-stable/bin/wine",
        "/usr/lib/wine/wine",
        "/usr/lib/wine/wine64",
    ]
    for c in candidatos:
        p = shutil.which(c) or (c if os.path.isfile(c) and os.access(c, os.X_OK) else None)
        if p and os.path.exists(p):
            return p
    return None


def detectar_directorio_anguli(sugerido: Optional[str] = None) -> str:
    """Busca y retorna el directorio donde reside Anguli.exe."""
    candidatos = []
    if sugerido:
        candidatos.append(sugerido)
    candidatos.extend([
        RUTA_ANGULI_DEFAULT,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Anguli-MinGw"),
        os.path.expanduser("~/Anguli-MinGw"),
        os.path.expanduser("~/Descargas/Anguli-MinGw"),
        os.path.expanduser("~/Escritorio/Anguli-MinGw"),
        os.path.expanduser("~/Desktop/Anguli-MinGw"),
    ])
    for c in candidatos:
        if c and os.path.isfile(os.path.join(c, "Anguli.exe")):
            return os.path.abspath(c)
    return sugerido or RUTA_ANGULI_DEFAULT


def verificar_entorno(directorio_anguli: Optional[str] = None) -> Tuple[bool, str]:
    """Verifica la presencia de Wine y de Anguli.exe."""
    wine_path = detectar_wine()
    if not wine_path:
        return False, "Wine no está instalado en el sistema. Ejecute: sudo apt install wine"

    dir_anguli = detectar_directorio_anguli(directorio_anguli)
    exe_path = os.path.join(dir_anguli, "Anguli.exe")
    if not os.path.isfile(exe_path):
        return False, f"No se encontró Anguli.exe en '{dir_anguli}'."

    return True, f"Entorno verificado correctamente (Wine: {os.path.basename(wine_path)}, Anguli en '{dir_anguli}')."


def generar_huella_sintetica(
    directorio_destino: str = RUTA_BANDEJA_DEFAULT,
    nombre_archivo: Optional[str] = None,
    clase: str = "natural",
    dedo: int = 1,
    con_impresion: bool = False,
    ruido: Optional[int] = None,
    rayones: Optional[int] = None,
    rotacion: Optional[int] = None,
    traslacion: Optional[int] = None,
    seed: Optional[int] = None,
    directorio_anguli: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Ejecuta Anguli mediante Wine para generar una nueva huella dactilar sintética.

    Parámetros:
      - directorio_destino: Carpeta donde se guardará la imagen generada.
      - nombre_archivo: Nombre del archivo de salida (si es None, se genera uno con timestamp).
      - clase: 'natural', 'arch', 'tarch', 'right', 'left', 'dloop', 'whirl'.
      - dedo: Índice de dedo a extraer (1, 2, 3...) cuando se generan múltiples huellas.
      - con_impresion: Si True, genera una impresión simulada con distorsiones físicas.
                       Si False, genera la huella sintética maestra (ideal).
      - ruido: Nivel de ruido (0 a 10).
      - rayones: Nivel/cantidad de rayones (0 a 5 máximo).
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

    wine_bin = detectar_wine()
    if not wine_bin:
        raise RuntimeError("Wine no está instalado en el sistema. Ejecute: sudo apt install wine")

    dir_anguli = detectar_directorio_anguli(directorio_anguli)
    exe_path = os.path.join(dir_anguli, "Anguli.exe")
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(f"No se encontró Anguli.exe en '{dir_anguli}'.")

    os.makedirs(directorio_destino, exist_ok=True)

    if seed is None:
        seed = random.randint(1, 999999)

    dedo = max(1, int(dedo))
    num_a_generar = max(1, dedo)

    # Restricciones solicitadas: rayones máximo 5, ruido hasta 10
    if rayones is not None:
        rayones = max(0, min(5, int(rayones)))
    if ruido is not None:
        ruido = max(0, min(10, int(ruido)))

    # Creamos un directorio temporal para la salida intermedia de Anguli
    with tempfile.TemporaryDirectory(prefix="anguli_run_") as tmp_dir:
        cmd: List[str] = [
            wine_bin,
            "Anguli.exe",
            "-num", str(num_a_generar),
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
            print(f"[Anguli] Generando huella sintética (clase='{clase}', dedo={dedo}, seed={seed})...")

        # Anguli requiere ejecutarse con cwd en su carpeta para cargar sus DLLs y recursos
        env = os.environ.copy()
        env["WINEDEBUG"] = "-all"

        resultado = subprocess.run(
            cmd,
            cwd=dir_anguli,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if resultado.returncode != 0:
            raise RuntimeError(
                f"Anguli falló con código {resultado.returncode}:\n{resultado.stderr}"
            )

        # Ubicar la imagen resultante para el dedo solicitado
        if con_impresion:
            ruta_origen = os.path.join(tmp_dir, "Impression_1", "fp_1", f"{dedo}.jpg")
        else:
            ruta_origen = os.path.join(tmp_dir, "Fingerprints", "fp_1", f"{dedo}.jpg")

        if not os.path.isfile(ruta_origen):
            # Fallback de búsqueda si la estructura difiere
            encontrados = []
            for root, _, files in os.walk(tmp_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".png", ".bmp", ".tif")):
                        encontrados.append(os.path.join(root, f))
            # Filtrar si hay una con el número del dedo
            coincidentes = [f for f in encontrados if os.path.basename(f) == f"{dedo}.jpg"]
            if coincidentes:
                ruta_origen = coincidentes[0]
            elif encontrados:
                ruta_origen = encontrados[0]
            else:
                raise FileNotFoundError(
                    f"No se encontró ninguna imagen generada por Anguli para dedo {dedo} en {tmp_dir}"
                )

        # Definir nombre de destino
        if not nombre_archivo:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            tipo = "impr" if con_impresion else "sintetica"
            nombre_archivo = f"huella_{tipo}_{clase}_d{dedo}_{timestamp}.jpg"

        ruta_final = os.path.join(directorio_destino, nombre_archivo)
        shutil.copy2(ruta_origen, ruta_final)

        if verbose:
            print(f"[Anguli] ✓ Huella generada exitosamente en:\n  -> {ruta_final}")

        return os.path.abspath(ruta_final)


def generar_huella_autorizada_dataset(
    directorio_destino: str = RUTA_BANDEJA_DEFAULT,
    nombre_archivo: Optional[str] = None,
    persona_id: int = 1,
    clase: Optional[str] = None,
    con_impresion: bool = True,
    ruido: int = 2,
    rayones: int = 1,
    rotacion: int = 5,
    traslacion: int = 5,
    directorio_anguli: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Genera una huella sintética autorizada basada en la semilla oficial del dataset (1604):
      - Persona 1: clase 'arch', dedo 1
      - Persona 2: clase 'natural', dedo 2
      - Persona 3: clase 'natural', dedo 3
    Permite opcionalmente sobreescribir la clase.
    Límites: rayones máximo 5, ruido hasta 10.
    """
    persona_id = max(1, min(3, int(persona_id)))

    # Persona 1 es arch; Personas 2 y 3 son natural
    if clase is None:
        clase = "arch" if persona_id == 1 else "natural"

    dedo = persona_id

    # Restricciones solicitadas
    ruido = max(0, min(10, int(ruido)))
    rayones = max(0, min(5, int(rayones)))

    if not nombre_archivo:
        ts = datetime.datetime.now().strftime("%H%M%S")
        tipo = "impr" if con_impresion else "master"
        nombre_archivo = f"huella_anguli_persona{persona_id}_seed1604_{clase}_{tipo}_{ts}.jpg"

    return generar_huella_sintetica(
        directorio_destino=directorio_destino,
        nombre_archivo=nombre_archivo,
        clase=clase,
        dedo=dedo,
        con_impresion=con_impresion,
        ruido=ruido,
        rayones=rayones,
        rotacion=rotacion,
        traslacion=traslacion,
        seed=SEMILLA_DATASET_DEFAULT,
        directorio_anguli=directorio_anguli,
        verbose=verbose,
    )


def generar_huella_intrusa(
    directorio_destino: str = RUTA_BANDEJA_DEFAULT,
    nombre_archivo: Optional[str] = None,
    clase: str = "whirl",
    dedo: int = 1,
    con_impresion: bool = True,
    ruido: int = 4,
    rayones: int = 2,
    seed: Optional[int] = None,
    directorio_anguli: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Genera una huella sintética intrusa (no autorizada) con semilla aleatoria
    o clase distinta para testear el rechazo del agente de control de acceso.
    Límites: rayones máximo 5, ruido hasta 10.
    """
    if seed is None:
        seed = random.randint(100000, 999999)

    ruido = max(0, min(10, int(ruido)))
    rayones = max(0, min(5, int(rayones)))

    if not nombre_archivo:
        ts = datetime.datetime.now().strftime("%H%M%S")
        tipo = "impr" if con_impresion else "master"
        nombre_archivo = f"huella_anguli_intrusa_seed{seed}_{clase}_{tipo}_{ts}.jpg"

    return generar_huella_sintetica(
        directorio_destino=directorio_destino,
        nombre_archivo=nombre_archivo,
        clase=clase,
        dedo=dedo,
        con_impresion=con_impresion,
        ruido=ruido,
        rayones=rayones,
        rotacion=10,
        traslacion=10,
        seed=seed,
        directorio_anguli=directorio_anguli,
        verbose=verbose,
    )


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
        help="Nivel de ruido de 0 a 10 (aplica con --impresion).",
    )
    parser.add_argument(
        "--rayones",
        type=int,
        default=None,
        help="Nivel de rayones de 0 a 5 máximo (aplica con --impresion).",
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
        "-p", "--persona",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="ID de Persona para --autorizada (1: arch, 2: natural, 3: natural).",
    )
    parser.add_argument(
        "--dedo",
        type=int,
        default=1,
        help="Número de dedo/huella a extraer de Anguli (1, 2 o 3). Por defecto: 1.",
    )
    parser.add_argument(
        "--autorizada",
        action="store_true",
        help=f"Atajo: generar huella autorizada con semilla del dataset ({SEMILLA_DATASET_DEFAULT}).",
    )
    parser.add_argument(
        "--intrusa",
        action="store_true",
        help="Atajo: generar huella intrusa (no autorizada) con semilla aleatoria.",
    )
    parser.add_argument(
        "--anguli-dir",
        default=None,
        help=f"Ruta a la carpeta de Anguli-MinGw (por defecto detecta automáticamente)",
    )

    args = parser.parse_args()

    try:
        if args.autorizada:
            ruta = generar_huella_autorizada_dataset(
                directorio_destino=args.destino,
                nombre_archivo=args.nombre,
                persona_id=args.persona,
                clase=args.clase if args.clase != "natural" or args.persona != 1 else None,
                directorio_anguli=args.anguli_dir,
                verbose=True,
            )
        elif args.intrusa:
            ruta = generar_huella_intrusa(
                directorio_destino=args.destino,
                nombre_archivo=args.nombre,
                clase=args.clase if args.clase != "natural" else "whirl",
                dedo=args.dedo,
                seed=args.seed,
                directorio_anguli=args.anguli_dir,
                verbose=True,
            )
        else:
            ruta = generar_huella_sintetica(
                directorio_destino=args.destino,
                nombre_archivo=args.nombre,
                clase=args.clase,
                dedo=args.dedo,
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
