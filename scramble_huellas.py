#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script utilitario para anonimizar y mezclar (scramble) las huellas dactilares de prueba.

Toma las muestras contenidas en las distintas subcarpetas (Persona_1, Persona_2,
Persona_3, intrusos) de `huellas_prueba/`, las mezcla aleatoriamente y las copia
con nombres neutrales (ej: `huella_01.jpg`, `huella_02.jpg`, ...) a una nueva carpeta.

Genera además un archivo `mapeo_scramble.csv` y `mapeo_scramble.json` como clave de
respuestas/ground-truth para verificar los diagnósticos del modelo a posteriori.
"""

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def recolectar_muestras(carpeta_origen: str) -> List[Dict[str, str]]:
    """
    Busca todas las imágenes válidas en `carpeta_origen` y sus subcarpetas.
    """
    muestras: List[Dict[str, str]] = []
    if not os.path.isdir(carpeta_origen):
        raise FileNotFoundError(f"No existe el directorio de origen: {carpeta_origen}")

    for root, dirs, files in sorted(os.walk(carpeta_origen)):
        dirs.sort()
        for f in sorted(files):
            if f.lower().endswith(EXTENSIONES_VALIDAS):
                ruta_abs = os.path.abspath(os.path.join(root, f))
                subcarpeta = os.path.relpath(root, carpeta_origen)
                if subcarpeta == ".":
                    subcarpeta = "raiz"

                es_intruso = "intrus" in subcarpeta.lower() or "no_aut" in subcarpeta.lower()
                identidad = "Intruso" if es_intruso else subcarpeta.replace("_", " ")

                muestras.append({
                    "ruta_origen": ruta_abs,
                    "subcarpeta_origen": subcarpeta,
                    "archivo_original": f,
                    "identidad_esperada": identidad,
                    "es_autorizado": not es_intruso,
                })
    return muestras


def scramble_carpeta_huellas(
    carpeta_origen: str = "huellas_prueba",
    carpeta_destino: str = "huellas_prueba_scramble",
    prefijo: str = "huella_",
    semilla: Optional[int] = None,
    generar_clave: bool = True,
    normalizar_timestamps: bool = True,
) -> List[Dict[str, str]]:
    """
    Crea una copia anonimizada y aleatorizada de las fotos de `carpeta_origen`
    en `carpeta_destino`.
    """
    muestras = recolectar_muestras(carpeta_origen)
    if not muestras:
        print(f"[!] No se encontraron imágenes en {carpeta_origen}")
        return []

    # Barajar muestras
    rng = random.Random(semilla)
    muestras_shuffled = list(muestras)
    rng.shuffle(muestras_shuffled)

    # Preparar directorio de destino limpio
    os.makedirs(carpeta_destino, exist_ok=True)

    total = len(muestras_shuffled)
    pad = max(2, len(str(total)))
    tiempo_uniforme = time.time()

    mapeo_resultado: List[Dict[str, str]] = []

    for idx, item in enumerate(muestras_shuffled, start=1):
        _, ext = os.path.splitext(item["archivo_original"])
        nombre_nuevo = f"{prefijo}{idx:0{pad}d}{ext.lower()}"
        ruta_destino = os.path.join(carpeta_destino, nombre_nuevo)

        # Copiar contenido sin preservar metadatos de fechas originales
        shutil.copyfile(item["ruta_origen"], ruta_destino)

        if normalizar_timestamps:
            # Igualar fecha de acceso y modificación para que no delate orden de copia
            os.utime(ruta_destino, (tiempo_uniforme, tiempo_uniforme))

        registro = {
            "archivo_scramble": nombre_nuevo,
            "subcarpeta_origen": item["subcarpeta_origen"],
            "archivo_original": item["archivo_original"],
            "identidad_esperada": item["identidad_esperada"],
            "es_autorizado": str(item["es_autorizado"]),
            "ruta_destino": os.path.abspath(ruta_destino),
        }
        mapeo_resultado.append(registro)

    if generar_clave:
        ruta_csv = os.path.join(carpeta_destino, "mapeo_scramble.csv")
        with open(ruta_csv, mode="w", newline="", encoding="utf-8") as f_csv:
            campos = [
                "archivo_scramble",
                "identidad_esperada",
                "es_autorizado",
                "subcarpeta_origen",
                "archivo_original",
            ]
            writer = csv.DictWriter(f_csv, fieldnames=campos, extrasaction="ignore")
            writer.writeheader()
            for r in mapeo_resultado:
                writer.writerow(r)

        ruta_json = os.path.join(carpeta_destino, "mapeo_scramble.json")
        with open(ruta_json, mode="w", encoding="utf-8") as f_json:
            json.dump(mapeo_resultado, f_json, indent=2, ensure_ascii=False)

    return mapeo_resultado


def main():
    parser = argparse.ArgumentParser(
        description="Genera una copia scrambled/anonimizada de huellas de prueba."
    )
    parser.add_argument(
        "--origen",
        default="huellas_prueba",
        help="Carpeta origen a copiar (default: huellas_prueba)",
    )
    parser.add_argument(
        "--destino",
        default="huellas_prueba_scramble",
        help="Carpeta destino donde guardar las imágenes mezcladas (default: huellas_prueba_scramble)",
    )
    parser.add_argument(
        "--prefijo",
        default="huella_",
        help="Prefijo para los nombres de archivo (default: huella_)",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=None,
        help="Semilla aleatoria opcional para reproducibilidad.",
    )
    parser.add_argument(
        "--copiar-a-bandeja",
        action="store_true",
        help="Si se especifica, actualiza también 'bandeja_agente/' con estas huellas scrambled.",
    )

    args = parser.parse_args()

    print(f"[*] Leyendo imágenes desde '{args.origen}'...")
    mapeo = scramble_carpeta_huellas(
        carpeta_origen=args.origen,
        carpeta_destino=args.destino,
        prefijo=args.prefijo,
        semilla=args.semilla,
    )

    print(f"[✓] Se mezclaron y anonimizaron {len(mapeo)} huellas en '{args.destino}/'.")
    print(f"[✓] Clave de respuestas guardada en:")
    print(f"    - {os.path.join(args.destino, 'mapeo_scramble.csv')}")
    print(f"    - {os.path.join(args.destino, 'mapeo_scramble.json')}")

    if getattr(args, "copiar_a_bandeja", False):
        bandeja = "bandeja_agente"
        os.makedirs(bandeja, exist_ok=True)
        # Limpiar bandeja actual y copiar scrambled
        for f in os.listdir(bandeja):
            p = os.path.join(bandeja, f)
            if os.path.isfile(p):
                os.remove(p)
        for r in mapeo:
            shutil.copyfile(r["ruta_destino"], os.path.join(bandeja, r["archivo_scramble"]))
        print(f"[✓] 'bandeja_agente/' actualizada con las {len(mapeo)} huellas scrambled.")


if __name__ == "__main__":
    main()
