#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de Huellas Dactilares Sintéticas utilizando Anguli (MinGW/Wine).
Proyecto Final - Inteligencia Artificial I (2026) - UNCUYO

Permite ejecutar Anguli.exe en Linux a través de Wine utilizando archivos de
configuración XML (-cfg) compatibles con los parámetros originales del dataset
(NumThreads=5, semilla 1604), generando tanto identidades autorizadas como
huellas intrusas listas para evaluar en vivo en el agente de control de acceso.
"""

import argparse
import datetime
import os
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple


RUTA_ANGULI_DEFAULT = os.path.expanduser("~/Downloads/Anguli-MinGw")
RUTA_BANDEJA_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bandeja_agente"
)

CLASES_PERMITIDAS = ["natural", "arch", "tarch", "right", "left", "dloop", "whirl"]
MAPA_CLASE_A_ID_CFG: Dict[str, int] = {
    "natural": 0,
    "arch": 1,
    "tarch": 2,
    "right": 3,
    "left": 4,
    "dloop": 5,
    "whirl": 6,
}

SEMILLA_DATASET_DEFAULT = 1604
NUM_HILOS_ANGULI_DEFAULT = 5
MAX_RAYONES = 5
MAX_RUIDO = 10


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


def _ruta_linux_a_wine_z(ruta_linux: str) -> str:
    """Convierte una ruta absoluta de Linux al formato de unidad Z: esperado por Qt/Wine en Anguli."""
    ruta_abs = os.path.abspath(ruta_linux).replace("\\", "/")
    return f"Z:{ruta_abs}"


def _buscar_imagen_dedo_anguli(tmp_dir: str, subcarpeta: str, dedo: int) -> str:
    """
    Localiza la imagen generada por Anguli para `dedo` dentro de `subcarpeta`
    (contemplando nombres `N..jpg` emitidos por `<ImageType value="0"/>` en modo `-cfg`
    y `N.jpg` estándar).
    """
    carpeta_fp = os.path.join(tmp_dir, subcarpeta, "fp_1")
    for candidato in (f"{dedo}.jpg", f"{dedo}..jpg", f"{dedo}.png", f"{dedo}..png"):
        ruta = os.path.join(carpeta_fp, candidato)
        if os.path.isfile(ruta):
            return ruta

    # Búsqueda de respaldo recursiva
    encontrados: List[str] = []
    prefijos = (f"{dedo}.", f"{dedo}..")
    base_busqueda = os.path.join(tmp_dir, subcarpeta)
    if not os.path.isdir(base_busqueda):
        base_busqueda = tmp_dir

    for root, _, files in os.walk(base_busqueda):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif")):
                encontrados.append(os.path.join(root, f))

    coincidentes = [
        f for f in encontrados if os.path.basename(f).startswith(prefijos)
    ]
    if coincidentes:
        return coincidentes[0]
    if encontrados:
        return encontrados[0]

    raise FileNotFoundError(
        f"No se encontró ninguna imagen generada por Anguli para el dedo {dedo} en '{tmp_dir}'."
    )


def _ejecutar_anguli_en_directorio(
    tmp_dir: str,
    wine_bin: str,
    dir_anguli: str,
    clase: str,
    num_huellas: int,
    seed: int,
    impr_por_dedo: int,
    ruido_max: int,
    rayones_max: int,
    rotacion: int,
    traslacion: int,
    num_threads: int = NUM_HILOS_ANGULI_DEFAULT,
) -> None:
    """
    Escribe un archivo de configuración XML (`Anguli.conf`) y ejecuta `Anguli.exe -cfg`
    en Wine. El uso de `-cfg` con `NumThreads=5` reproduce exactamente las identidades
    maestras del dataset oficial (semilla 1604) y soporta niveles de ruido hasta 10.
    """
    class_id = MAPA_CLASE_A_ID_CFG[clase]
    z_out = _ruta_linux_a_wine_z(tmp_dir)
    cfg_path = os.path.join(tmp_dir, "anguli_live.conf")
    z_cfg = _ruta_linux_a_wine_z(cfg_path)

    xml_conf = (
        "<Anguli>\n"
        " <FingerprintGen>\n"
        f'  <Translation value="{max(0, int(traslacion))}"/>\n'
        f'  <Rotation value="{max(0, int(rotacion))}"/>\n'
        f'  <NumScratches high="{max(0, min(MAX_RAYONES, int(rayones_max)))}" low="0"/>\n'
        f'  <NoiseLevel high="{max(0, min(MAX_RUIDO, int(ruido_max)))}" low="0"/>\n'
        f'  <customSeed value="1" seedValue="{int(seed)}"/>\n'
        '  <SaveMetaInfo value="0"/>\n'
        f'  <NumThreads value="{max(1, int(num_threads))}"/>\n'
        '  <ImageType value="0"/>\n'
        '  <FingerPerDir value="1000"/>\n'
        '  <StartFinger value="1"/>\n'
        f'  <OutDir value="{z_out}"/>\n'
        f'  <ClassDistr value="{class_id}"/>\n'
        f'  <ImprPerFinger value="{max(0, int(impr_por_dedo))}"/>\n'
        f'  <NumFingerprints value="{max(1, int(num_huellas))}"/>\n'
        " </FingerprintGen>\n"
        "</Anguli>\n"
    )

    with open(cfg_path, "w", encoding="utf-8") as fh:
        fh.write(xml_conf)

    env = os.environ.copy()
    env["WINEDEBUG"] = "-all"

    resultado = subprocess.run(
        [wine_bin, "Anguli.exe", "-cfg", z_cfg],
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


def generar_huella_sintetica(
    directorio_destino: str = RUTA_BANDEJA_DEFAULT,
    nombre_archivo: Optional[str] = None,
    clase: str = "natural",
    dedo: int = 1,
    num_huellas: Optional[int] = None,
    con_impresion: bool = False,
    ruido: Optional[int] = None,
    rayones: Optional[int] = None,
    rotacion: Optional[int] = None,
    traslacion: Optional[int] = None,
    seed: Optional[int] = None,
    indice_impresion: Optional[int] = None,
    directorio_anguli: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Ejecuta Anguli mediante Wine para generar una nueva huella dactilar sintética.

    Parámetros:
      - directorio_destino: Carpeta donde se guardará la imagen generada.
      - nombre_archivo: Nombre del archivo de salida (si es None, se genera uno con timestamp).
      - clase: 'natural', 'arch', 'tarch', 'right', 'left', 'dloop', 'whirl'.
      - dedo: Índice de identidad/dedo a extraer (1, 2, 3...) cuando se generan múltiples huellas.
      - num_huellas: Cantidad total de identidades a generar en el lote de Anguli.
                     Para semilla 1604 + clase 'natural' con dedo 2 o 3 se fuerza al menos 3.
      - con_impresion: Si True, genera una impresión simulada con distorsiones físicas.
                       Si False, genera la huella sintética maestra (ideal).
      - ruido: Nivel máximo de ruido (0 a 10).
      - rayones: Nivel/cantidad máxima de rayones (0 a 5 máximo).
      - rotacion: Grados de rotación máxima.
      - traslacion: Píxeles de traslación máxima.
      - seed: Semilla aleatoria (entero). Si es None, se genera una aleatoria.
      - indice_impresion: Qué número de impresión elegir (1..N). Si es None y con_impresion=True,
                          se elige aleatoriamente entre 1 y 5 para obtener muestras variadas en
                          cada ejecución aún usando una semilla maestra fija como 1604.
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
        seed = random.randint(100000, 999999)

    dedo = max(1, int(dedo))
    if num_huellas is None:
        if int(seed) == SEMILLA_DATASET_DEFAULT and clase == "natural" and dedo in (2, 3):
            num_a_generar = 3
        else:
            num_a_generar = max(1, dedo)
    else:
        num_a_generar = max(dedo, int(num_huellas))

    # Restricciones solicitadas: rayones máximo 5, ruido hasta 10
    ruido_eff = max(0, min(MAX_RUIDO, int(ruido))) if ruido is not None else (4 if con_impresion else 0)
    rayones_eff = max(0, min(MAX_RAYONES, int(rayones))) if rayones is not None else (2 if con_impresion else 0)
    rot_eff = max(0, int(rotacion)) if rotacion is not None else (5 if con_impresion else 0)
    trans_eff = max(0, int(traslacion)) if traslacion is not None else (5 if con_impresion else 0)

    if con_impresion:
        if indice_impresion is not None:
            impr_idx = max(1, int(indice_impresion))
        else:
            max_impr_azar = 4 if num_a_generar >= 3 else 5
            impr_idx = random.randint(1, max_impr_azar)
        impr_por_dedo = impr_idx
    else:
        impr_idx = 0
        impr_por_dedo = 0

    if verbose:
        print(
            f"[Anguli] Generando huella sintética "
            f"(clase='{clase}', dedo={dedo}/{num_a_generar}, seed={seed}, impr={impr_idx})..."
        )

    with tempfile.TemporaryDirectory(prefix="anguli_run_") as tmp_dir:
        _ejecutar_anguli_en_directorio(
            tmp_dir=tmp_dir,
            wine_bin=wine_bin,
            dir_anguli=dir_anguli,
            clase=clase,
            num_huellas=num_a_generar,
            seed=int(seed),
            impr_por_dedo=impr_por_dedo,
            ruido_max=ruido_eff,
            rayones_max=rayones_eff,
            rotacion=rot_eff,
            traslacion=trans_eff,
            num_threads=NUM_HILOS_ANGULI_DEFAULT,
        )

        subcarpeta = f"Impression_{impr_idx}" if con_impresion else "Fingerprints"
        ruta_origen = _buscar_imagen_dedo_anguli(tmp_dir, subcarpeta, dedo)

        if not nombre_archivo:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            tipo = f"impr{impr_idx}" if con_impresion else "master"
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
    ruido: int = 4,
    rayones: int = 2,
    rotacion: int = 5,
    traslacion: int = 5,
    directorio_anguli: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Genera una muestra sintética autorizada basada en la semilla oficial del dataset (1604):
      - Persona 1: clase 'arch', 1 identidad (reemplazó a la 1ra identidad natural original).
      - Persona 2: clase 'natural', lote de 3 identidades simultáneas (descarta #1 y #3, extrae #2).
      - Persona 3: clase 'natural', lote de 3 identidades simultáneas (descarta #1 y #2, extrae #3).
    Límites estrictos: rayones máximo 5, ruido hasta 10.
    """
    persona_id = max(1, min(3, int(persona_id)))

    if clase is None:
        clase = "arch" if persona_id == 1 else "natural"

    dedo = 1 if (persona_id == 1 and clase == "arch") else persona_id
    num_huellas = 1 if (persona_id == 1 and clase == "arch") else max(3, dedo)

    ruido = max(0, min(MAX_RUIDO, int(ruido)))
    rayones = max(0, min(MAX_RAYONES, int(rayones)))

    if not nombre_archivo:
        ts = datetime.datetime.now().strftime("%H%M%S")
        suf_azar = random.randint(10, 99)
        tipo = "impr" if con_impresion else "master"
        nombre_archivo = (
            f"huella_anguli_persona{persona_id}_seed1604_{clase}_{tipo}_{ts}_{suf_azar}.jpg"
        )

    return generar_huella_sintetica(
        directorio_destino=directorio_destino,
        nombre_archivo=nombre_archivo,
        clase=clase,
        dedo=dedo,
        num_huellas=num_huellas,
        con_impresion=con_impresion,
        ruido=ruido,
        rayones=rayones,
        rotacion=rotacion,
        traslacion=traslacion,
        seed=SEMILLA_DATASET_DEFAULT,
        directorio_anguli=directorio_anguli,
        verbose=verbose,
    )


def generar_lote_personas_2_y_3_dataset(
    directorio_destino: str = RUTA_BANDEJA_DEFAULT,
    con_impresion: bool = True,
    ruido: int = 4,
    rayones: int = 2,
    rotacion: int = 5,
    traslacion: int = 5,
    directorio_anguli: Optional[str] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Ejecuta Anguli una sola vez pidiendo 1 muestra de las 3 identidades 'natural' con la
    semilla oficial 1604 (NumThreads=5), descarta la 1ra identidad (ya que Persona 1 fue
    reemplazada por una huella tipo 'arch') y guarda las muestras de Persona 2 y Persona 3
    directamente en `directorio_destino` (bandeja de entrada del agente).

    Límites estrictos: rayones máximo 5, ruido hasta 10.
    Retorna la lista `[ruta_persona_2, ruta_persona_3]`.
    """
    wine_bin = detectar_wine()
    if not wine_bin:
        raise RuntimeError("Wine no está instalado en el sistema. Ejecute: sudo apt install wine")

    dir_anguli = detectar_directorio_anguli(directorio_anguli)
    exe_path = os.path.join(dir_anguli, "Anguli.exe")
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(f"No se encontró Anguli.exe en '{dir_anguli}'.")

    os.makedirs(directorio_destino, exist_ok=True)

    ruido_eff = max(0, min(MAX_RUIDO, int(ruido)))
    rayones_eff = max(0, min(MAX_RAYONES, int(rayones)))
    rot_eff = max(0, int(rotacion))
    trans_eff = max(0, int(traslacion))

    impr_idx = random.randint(1, 4) if con_impresion else 0

    if verbose:
        print(
            f"[Anguli] Generando lote de 3 identidades 'natural' (seed={SEMILLA_DATASET_DEFAULT}, "
            f"impr={impr_idx}), descartando la 1ra y conservando Personas 2 y 3..."
        )

    rutas_guardadas: List[str] = []
    with tempfile.TemporaryDirectory(prefix="anguli_lote1604_") as tmp_dir:
        _ejecutar_anguli_en_directorio(
            tmp_dir=tmp_dir,
            wine_bin=wine_bin,
            dir_anguli=dir_anguli,
            clase="natural",
            num_huellas=3,
            seed=SEMILLA_DATASET_DEFAULT,
            impr_por_dedo=impr_idx,
            ruido_max=ruido_eff,
            rayones_max=rayones_eff,
            rotacion=rot_eff,
            traslacion=trans_eff,
            num_threads=NUM_HILOS_ANGULI_DEFAULT,
        )

        subcarpeta = f"Impression_{impr_idx}" if con_impresion else "Fingerprints"
        ts = datetime.datetime.now().strftime("%H%M%S")
        suf_azar = random.randint(10, 99)
        tipo = "impr" if con_impresion else "master"

        # La identidad #1 se descarta deliberadamente (reemplazada en el dataset por Arch).
        for persona_id in (2, 3):
            ruta_origen = _buscar_imagen_dedo_anguli(tmp_dir, subcarpeta, persona_id)
            nombre_dest = (
                f"huella_anguli_persona{persona_id}_seed1604_natural_{tipo}_{ts}_{suf_azar}.jpg"
            )
            ruta_final = os.path.abspath(os.path.join(directorio_destino, nombre_dest))
            shutil.copy2(ruta_origen, ruta_final)
            rutas_guardadas.append(ruta_final)
            if verbose:
                print(f"[Anguli] ✓ Persona {persona_id} guardada en: {ruta_final}")

    return rutas_guardadas


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

    ruido = max(0, min(MAX_RUIDO, int(ruido)))
    rayones = max(0, min(MAX_RAYONES, int(rayones)))

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
        indice_impresion=1 if con_impresion else None,
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
        "--personas-2-3",
        action="store_true",
        help=(
            f"Genera 1 muestra de las 3 identidades 'natural' con semilla {SEMILLA_DATASET_DEFAULT}, "
            "descarta la 1ra (reemplazada por Arch) y guarda Persona 2 y Persona 3 en la bandeja."
        ),
    )
    parser.add_argument(
        "--intrusa",
        action="store_true",
        help="Atajo: generar huella intrusa (no autorizada) con semilla aleatoria.",
    )
    parser.add_argument(
        "--anguli-dir",
        default=None,
        help="Ruta a la carpeta de Anguli-MinGw (por defecto detecta automáticamente)",
    )

    args = parser.parse_args()

    try:
        if args.personas_2_3:
            rutas = generar_lote_personas_2_y_3_dataset(
                directorio_destino=args.destino,
                ruido=args.ruido if args.ruido is not None else 4,
                rayones=args.rayones if args.rayones is not None else 2,
                rotacion=args.rotacion if args.rotacion is not None else 5,
                traslacion=args.traslacion if args.traslacion is not None else 5,
                directorio_anguli=args.anguli_dir,
                verbose=True,
            )
            print("\nListo (Personas 2 y 3 guardadas, 1ra descartada):")
            for r in rutas:
                print(f"  - {r}")
        elif args.autorizada:
            ruta = generar_huella_autorizada_dataset(
                directorio_destino=args.destino,
                nombre_archivo=args.nombre,
                persona_id=args.persona,
                clase=args.clase if args.clase != "natural" or args.persona != 1 else None,
                ruido=args.ruido if args.ruido is not None else 4,
                rayones=args.rayones if args.rayones is not None else 2,
                rotacion=args.rotacion if args.rotacion is not None else 5,
                traslacion=args.traslacion if args.traslacion is not None else 5,
                directorio_anguli=args.anguli_dir,
                verbose=True,
            )
            print(f"\nListo: {ruta}")
        elif args.intrusa:
            ruta = generar_huella_intrusa(
                directorio_destino=args.destino,
                nombre_archivo=args.nombre,
                clase=args.clase if args.clase != "natural" else "whirl",
                dedo=args.dedo,
                ruido=args.ruido if args.ruido is not None else 4,
                rayones=args.rayones if args.rayones is not None else 2,
                seed=args.seed,
                directorio_anguli=args.anguli_dir,
                verbose=True,
            )
            print(f"\nListo: {ruta}")
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
