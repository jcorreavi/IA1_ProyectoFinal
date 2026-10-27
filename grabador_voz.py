#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Captura y Organización de Dataset de Comandos de Voz
Trabajo Final - Inteligencia Artificial I (Ingeniería en Mecatrónica - UNCUYO)

Este programa permite grabar, organizar y catalogar muestras de voz para los
tres comandos obligatorios del sistema («Reconocer», «Identificar» y «Salir»),
almacenando archivos WAV mono con formato uniforme y actualizando de manera
segura el registro en `dataset_voz/metadata.csv`.
"""

import argparse
import csv
import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np

# Intentar importar bibliotecas principales de audio
try:
    import sounddevice as sd
    SOUNDDEVICE_DISPONIBLE = True
except ImportError:
    sd = None
    SOUNDDEVICE_DISPONIBLE = False

try:
    import soundfile as sf
    SOUNDFILE_DISPONIBLE = True
except ImportError:
    sf = None
    SOUNDFILE_DISPONIBLE = False


# ==============================================================================
# CONFIGURACIÓN PREDETERMINADA (Fácil de modificar)
# ==============================================================================

DIRECTORIO_DATASET_DEFAULT = "dataset_voz"
ARCHIVO_METADATA_NOMBRE = "metadata.csv"

# Comandos exigidos por la consigna del proyecto final
COMANDOS_PROYECTO: Tuple[str, ...] = ("Reconocer", "Identificar", "Salir")

# Parámetros acústicos uniformes predeterminados
SAMPLE_RATE_DEFAULT = 16000       # 16 kHz (estándar para reconocimiento de voz / MFCC)
CANALES_DEFAULT = 1               # Mono (1 canal)
DURACION_DEFAULT = 2.0            # Duración por toma en segundos
TOMAS_DEFAULT = 5                 # Cantidad de tomas sugeridas por sesión
CUENTA_REGRESIVA_DEFAULT = 3      # Segundos de cuenta regresiva antes de grabar
PAUSA_ENTRE_TOMAS_DEFAULT = 1.0   # Pausa breve antes de pasar a la siguiente toma

# Límites de validación para entradas del usuario
DURACION_MIN_SEG = 0.5
DURACION_MAX_SEG = 15.0
TOMAS_MAX_SESION = 200
SAMPLE_RATES_VALIDOS = (8000, 16000, 22050, 32000, 44100, 48000)

# Columnas del archivo metadata.csv
COLUMNAS_METADATA = [
    "ruta_audio",
    "persona",
    "comando",
    "numero_toma",
    "fecha_hora",
    "duracion_seg",
    "frecuencia_muestreo",
]


# ==============================================================================
# GESTIÓN DE DIRECTORIOS, NOMENCLATURA Y METADATOS
# ==============================================================================

def sanitizar_nombre_persona(nombre_crudo: str) -> str:
    """
    Normaliza el identificador de la persona para que sea un nombre de carpeta
    y de archivo seguro y consistente.
    - Si el usuario ingresa solo un número (ej. '1'), devuelve 'Persona_01'.
    - Si ingresa 'persona 1' o 'Persona_1', lo normaliza a 'Persona_01'.
    - Si ingresa un nombre propio (ej. 'Julian Correa'), reemplaza espacios por '_'
      y elimina caracteres especiales no válidos en rutas.
    """
    limpio = nombre_crudo.strip()
    if not limpio:
        return ""

    # Caso 1: solo dígitos -> Persona_XX
    if limpio.isdigit():
        return f"Persona_{int(limpio):02d}"

    # Caso 2: patrón "persona X" o "persona_X"
    match_persona = re.match(r"^persona[\s_\-]*(\d+)$", limpio, flags=re.IGNORECASE)
    if match_persona:
        return f"Persona_{int(match_persona.group(1)):02d}"

    # Caso 3: nombre personalizado -> conservar alfanuméricos, guiones y guiones bajos
    limpio = re.sub(r"\s+", "_", limpio)
    limpio = re.sub(r"[^A-Za-z0-9_\-]", "", limpio)
    return limpio.strip("_-")


def inicializar_estructura_persona(directorio_base: str, persona: str) -> str:
    """
    Crea la carpeta de la persona y las subcarpetas de los tres comandos
    obligatorios ('Reconocer', 'Identificar', 'Salir') si no existen.
    Verifica además permisos de escritura.
    """
    ruta_persona = os.path.join(directorio_base, persona)
    try:
        os.makedirs(ruta_persona, exist_ok=True)
        for cmd in COMANDOS_PROYECTO:
            os.makedirs(os.path.join(ruta_persona, cmd), exist_ok=True)
    except OSError as exc:
        raise PermissionError(
            f"No se pudo crear la estructura de carpetas en '{ruta_persona}': {exc}"
        ) from exc

    if not os.access(ruta_persona, os.W_OK):
        raise PermissionError(f"Sin permisos de escritura en '{ruta_persona}'.")

    return ruta_persona


def obtener_ruta_metadata(directorio_base: str) -> str:
    """Devuelve la ruta del archivo metadata.csv dentro del dataset."""
    return os.path.join(directorio_base, ARCHIVO_METADATA_NOMBRE)


def asegurar_archivo_metadata(directorio_base: str) -> str:
    """
    Garantiza que el directorio base y el archivo metadata.csv existan y
    cuenten con la fila de encabezado requerida.
    """
    os.makedirs(directorio_base, exist_ok=True)
    ruta_csv = obtener_ruta_metadata(directorio_base)

    escribir_encabezado = False
    if not os.path.exists(ruta_csv):
        escribir_encabezado = True
    elif os.path.getsize(ruta_csv) == 0:
        escribir_encabezado = True

    if escribir_encabezado:
        with open(ruta_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNAS_METADATA)
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

    return ruta_csv


def leer_registros_metadata(directorio_base: str) -> List[Dict[str, str]]:
    """Lee de forma segura todas las filas existentes en metadata.csv."""
    ruta_csv = obtener_ruta_metadata(directorio_base)
    if not os.path.isfile(ruta_csv):
        return []

    registros: List[Dict[str, str]] = []
    try:
        with open(ruta_csv, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for fila in reader:
                if fila and fila.get("persona") and fila.get("comando"):
                    registros.append(fila)
    except Exception as exc:
        print(f"[!] Advertencia al leer '{ruta_csv}': {exc}")
    return registros


def registrar_en_metadata(
    directorio_base: str,
    ruta_audio_rel: str,
    persona: str,
    comando: str,
    numero_toma: int,
    fecha_hora: str,
    duracion_seg: float,
    frecuencia_muestreo: int,
) -> None:
    """
    Agrega una nueva fila a metadata.csv de forma inmediata y segura (flush + fsync),
    evitando pérdida de datos si la sesión se interrumpe posteriormente.
    """
    ruta_csv = asegurar_archivo_metadata(directorio_base)
    ruta_normalizada = ruta_audio_rel.replace("\\", "/")

    fila = {
        "ruta_audio": ruta_normalizada,
        "persona": persona,
        "comando": comando,
        "numero_toma": int(numero_toma),
        "fecha_hora": fecha_hora,
        "duracion_seg": f"{duracion_seg:.2f}",
        "frecuencia_muestreo": int(frecuencia_muestreo),
    }

    with open(ruta_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS_METADATA)
        writer.writerow(fila)
        f.flush()
        os.fsync(f.fileno())


def listar_personas_existentes(directorio_base: str) -> List[str]:
    """
    Obtiene la lista ordenada de personas registradas tanto en subcarpetas
    de `dataset_voz/` como en `metadata.csv`.
    """
    personas = set()

    if os.path.isdir(directorio_base):
        for entrada in os.listdir(directorio_base):
            ruta_completa = os.path.join(directorio_base, entrada)
            if os.path.isdir(ruta_completa) and not entrada.startswith("."):
                personas.add(entrada)

    for reg in leer_registros_metadata(directorio_base):
        p = reg.get("persona", "").strip()
        if p:
            personas.add(p)

    return sorted(personas)


def sugerir_nombre_nueva_persona(personas_actuales: List[str]) -> str:
    """Sugiere el próximo identificador correlativo del tipo 'Persona_XX'."""
    max_num = 0
    for p in personas_actuales:
        m = re.match(r"^Persona_(\d+)$", p, flags=re.IGNORECASE)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"Persona_{max_num + 1:02d}"


def calcular_siguiente_numero_toma(
    directorio_base: str, persona: str, comando: str
) -> int:
    """
    Determina el siguiente número de toma correlativo inspeccionando tanto
    los archivos WAV presentes en la carpeta como el historial de metadata.csv.
    Garantiza no sobrescribir nunca una toma previa.
    """
    max_toma = 0
    carpeta_comando = os.path.join(directorio_base, persona, comando)

    # 1. Revisar archivos WAV existentes en la carpeta del comando
    patron_toma = re.compile(r"_toma_(\d+)\.wav$", flags=re.IGNORECASE)
    if os.path.isdir(carpeta_comando):
        for nombre in os.listdir(carpeta_comando):
            if nombre.lower().endswith(".wav"):
                match = patron_toma.search(nombre)
                if match:
                    max_toma = max(max_toma, int(match.group(1)))
                else:
                    max_toma += 1

    # 2. Revisar registros en metadata.csv para esa persona y comando
    for reg in leer_registros_metadata(directorio_base):
        if reg.get("persona") == persona and reg.get("comando") == comando:
            try:
                n = int(reg.get("numero_toma", 0))
                max_toma = max(max_toma, n)
            except ValueError:
                pass

    siguiente = max_toma + 1

    # 3. Verificación adicional contra colisión de ruta en disco
    while True:
        nombre_archivo = construir_nombre_archivo(persona, comando, siguiente)
        ruta_candidata = os.path.join(carpeta_comando, nombre_archivo)
        if not os.path.exists(ruta_candidata):
            break
        siguiente += 1

    return siguiente


def construir_nombre_archivo(persona: str, comando: str, numero_toma: int) -> str:
    """
    Genera un nombre de archivo único y descriptivo:
    Ejemplo: Persona_01_Reconocer_toma_001.wav
    """
    return f"{persona}_{comando}_toma_{numero_toma:03d}.wav"


def construir_ruta_metadata(
    directorio_base: str, persona: str, comando: str, nombre_wav: str
) -> str:
    """
    Construye una ruta relativa portable (con barras '/') para guardar en metadata.csv:
    Ejemplo: dataset_voz/Persona_01/Reconocer/Persona_01_Reconocer_toma_001.wav
    """
    nombre_raiz = os.path.basename(os.path.normpath(directorio_base)) or DIRECTORIO_DATASET_DEFAULT
    return f"{nombre_raiz}/{persona}/{comando}/{nombre_wav}"


def resolver_ruta_en_disco(directorio_base: str, ruta_metadata: str) -> str:
    """
    Resuelve una ruta guardada en metadata.csv hacia la ruta real dentro de `directorio_base`,
    funcionando tanto si el script se ejecuta desde la raíz del proyecto como con `--dataset-dir`.
    """
    ruta_norm = ruta_metadata.replace("\\", "/")
    padre_base = os.path.dirname(os.path.abspath(directorio_base))
    candidata_padre = os.path.join(padre_base, ruta_norm)
    if os.path.isfile(candidata_padre):
        return candidata_padre

    partes = ruta_norm.split("/")
    if len(partes) >= 3:
        candidata_interna = os.path.join(directorio_base, *partes[-3:])
        if os.path.isfile(candidata_interna):
            return candidata_interna
        return candidata_interna

    if os.path.isfile(ruta_norm):
        return ruta_norm

    return candidata_padre


def contar_audios_persona(directorio_base: str, persona: str) -> Dict[str, int]:
    """Cuenta cuántos archivos WAV válidos existen por comando para una persona."""
    conteo: Dict[str, int] = {cmd: 0 for cmd in COMANDOS_PROYECTO}
    for cmd in COMANDOS_PROYECTO:
        carpeta = os.path.join(directorio_base, persona, cmd)
        if os.path.isdir(carpeta):
            conteo[cmd] = sum(
                1 for f in os.listdir(carpeta) if f.lower().endswith(".wav")
            )
    return conteo


# ==============================================================================
# CAPTURA Y GUARDADO DE AUDIO (SOUNDDEVICE / SOUNDFILE + RESPALDO SEGURO)
# ==============================================================================

def _arecord_disponible() -> bool:
    """Verifica si la herramienta ALSA `arecord` está disponible en el sistema."""
    return shutil.which("arecord") is not None


def _es_captura_saturada_dc(audio_int16: np.ndarray) -> bool:
    """
    Detecta si un buffer capturado proviene de una entrada analógica desconectada
    o bloqueada en continua (por ejemplo, clavada en -32768 / -1.0 como ocurre en
    algunos jacks analógicos ALSA sin micrófono enchufado).
    """
    if audio_int16.size == 0:
        return True
    norm = audio_int16.astype(np.float32) / 32768.0
    media = float(np.mean(norm))
    desvio_ac = float(np.std(norm - media))
    frac_clavada = float(np.mean(np.abs(norm) >= 0.999))
    # Si más del 50% de las muestras están clavadas al riel o hay gran DC sin variación AC
    if frac_clavada > 0.50:
        return True
    if abs(media) > 0.35 and desvio_ac < 0.01:
        return True
    return False


def _capturar_con_arecord(num_muestras: int, sample_rate: int) -> np.ndarray:
    """Captura audio mono S16_LE usando `arecord` (enrutado por PipeWire/ALSA default)."""
    fd_temp, ruta_temp = tempfile.mkstemp(prefix=".tmp_arecord_", suffix=".wav")
    os.close(fd_temp)
    try:
        cmd = [
            "arecord",
            "-q",
            "-f",
            "S16_LE",
            "-c",
            str(CANALES_DEFAULT),
            "-r",
            str(sample_rate),
            "-t",
            "wav",
            "--samples",
            str(num_muestras),
            ruta_temp,
        ]
        subprocess.run(cmd, check=True)
        with wave.open(ruta_temp, "rb") as wf:
            raw_bytes = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw_bytes, dtype=np.int16).copy()
        return audio[:num_muestras]
    finally:
        if os.path.exists(ruta_temp):
            try:
                os.remove(ruta_temp)
            except OSError:
                pass


def _elegir_mejor_dispositivo_sounddevice() -> Optional[int]:
    """
    En Linux con PipeWire/PulseAudio, PortAudio suele elegir por error el jack
    analógico desconectado (ej. 'SN6140 Analog') en vez de PipeWire o el micrófono
    digital integrado ('acp-pdm-mach' / 'dmic'). Prioriza pipewire -> pulse -> default -> dmic.
    """
    if not SOUNDDEVICE_DISPONIBLE:
        return None
    try:
        todos = sd.query_devices()
        prioridades = ("pipewire", "pulse", "default", "pdm", "dmic")
        for palabra in prioridades:
            for idx, dev in enumerate(todos):
                if int(dev.get("max_input_channels", 0)) > 0:
                    nombre = str(dev.get("name", "")).lower()
                    if palabra in nombre:
                        return idx
    except Exception:
        pass
    return None


def _obtener_info_fuente_pipewire() -> Optional[str]:
    """
    Consulta a PipeWire/WirePlumber (`wpctl`) cuál es exactamente el micrófono
    físico seleccionado actualmente en la configuración de sonido de Ubuntu,
    incluyendo su placa ALSA, volumen y si está silenciado (MUTED).
    """
    if shutil.which("wpctl") is None:
        return None
    try:
        res_insp = subprocess.run(
            ["wpctl", "inspect", "@DEFAULT_AUDIO_SOURCE@"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if res_insp.returncode != 0 or not res_insp.stdout:
            return None

        desc = ""
        alsa_card = ""
        alsa_name = ""
        for linea in res_insp.stdout.splitlines():
            m_desc = re.search(r'node\.description\s*=\s*"([^"]+)"', linea)
            if m_desc:
                desc = m_desc.group(1).strip()
            m_card = re.search(r'alsa\.card_name\s*=\s*"([^"]+)"', linea)
            if m_card:
                alsa_card = m_card.group(1).strip()
            m_aname = re.search(r'alsa\.name\s*=\s*"([^"]+)"', linea)
            if m_aname:
                alsa_name = m_aname.group(1).strip()

        vol_info = ""
        res_vol = subprocess.run(
            ["wpctl", "get-volume", "@DEFAULT_AUDIO_SOURCE@"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if res_vol.returncode == 0 and res_vol.stdout:
            m_v = re.search(r"([0-9]+(?:\.[0-9]+)?)", res_vol.stdout)
            if m_v:
                vol_pct = int(round(float(m_v.group(1)) * 100))
                vol_info = f", Vol: {vol_pct}%"
            if "MUTED" in res_vol.stdout.upper():
                vol_info += " [¡SILENCIADO/MUTED!]"

        etiqueta_hw = alsa_name or alsa_card
        nombre_base = desc or etiqueta_hw or "Fuente PipeWire predeterminada"
        if etiqueta_hw and etiqueta_hw.lower() not in nombre_base.lower():
            texto = f"{nombre_base} [{etiqueta_hw}{vol_info}]"
        else:
            texto = f"{nombre_base} [{vol_info.lstrip(', ') or 'PipeWire'}]"

        if "sn6140" in (alsa_name + " " + desc).lower():
            texto += " [¡OJO: Jack analógico seleccionado!]"
        return texto
    except Exception:
        return None


def listar_fuentes_pipewire() -> List[Tuple[str, str, bool]]:
    """
    Parsea la sección Audio -> Sources de `wpctl status` y devuelve una lista de
    tuplas `(id_wpctl, descripcion, es_predeterminado)`.
    """
    if shutil.which("wpctl") is None:
        return []
    try:
        res = subprocess.run(
            ["wpctl", "status"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode != 0:
            return []

        en_audio = False
        en_sources = False
        fuentes: List[Tuple[str, str, bool]] = []
        for linea in res.stdout.splitlines():
            limpia = linea.strip()
            if limpia.startswith("Audio"):
                en_audio = True
                continue
            if en_audio and limpia.startswith("Video"):
                break
            if en_audio and ("Sources:" in linea):
                en_sources = True
                continue
            if en_sources:
                if any(
                    sec in linea
                    for sec in ("Filters:", "Streams:", "Sinks:", "Devices:", "Settings")
                ):
                    en_sources = False
                    continue
                m = re.search(r"(\*?)\s*(\d+)\.\s+(.+)$", linea)
                if m:
                    activo = m.group(1) == "*"
                    id_num = m.group(2)
                    nombre_f = m.group(3).strip()
                    fuentes.append((id_num, nombre_f, activo))
        return fuentes
    except Exception:
        return []


def menu_cambiar_microfono_sistema() -> None:
    """
    Permite ver qué micrófono está activo en Ubuntu (PipeWire) y cambiarlo
    directamente desde la consola usando `wpctl set-default <ID>`.
    """
    print("\n" + "=" * 72)
    print(" SELECCIÓN DE MICRÓFONO ACTIVO DEL SISTEMA (PIPEWIRE / UBUNTU)")
    print("=" * 72)
    fuentes = listar_fuentes_pipewire()
    if not fuentes:
        info = _obtener_info_fuente_pipewire()
        if info:
            print(f" Micrófono actual: {info}")
        else:
            print(" [!] No se pudo consultar la lista de fuentes de PipeWire (`wpctl status`).")
            disps = listar_dispositivos_entrada()
            if disps:
                print("\n Dispositivos ALSA/PortAudio detectados:")
                for idx, nom, ch in disps:
                    print(f"   - [{idx}] {nom} ({ch} canales)")
        print("=" * 72)
        return

    print(" Micrófonos disponibles en Ubuntu (el '*' indica el activo actualmente):")
    for idx_lista, (id_wp, nombre_f, activo) in enumerate(fuentes, 1):
        marca = " * [ACTIVO AHORA]" if activo else ""
        print(f"   [{idx_lista}] (ID PipeWire {id_wp}) {nombre_f}{marca}")
    print("   [0] Volver sin cambiar")

    op = input("\n Elegí el micrófono que querés activar (1..N) [Default: 0]: ").strip()
    if not op or op == "0":
        return
    if op.isdigit() and 1 <= int(op) <= len(fuentes):
        id_elegido, nom_elegido, _ = fuentes[int(op) - 1]
        res = subprocess.run(
            ["wpctl", "set-default", id_elegido],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            # Asegurar que no esté silenciado (unmute)
            subprocess.run(
                ["wpctl", "set-mute", "@DEFAULT_AUDIO_SOURCE@", "0"],
                capture_output=True,
                check=False,
            )
            print(f"\n [+] Micrófono activo cambiado a: {nom_elegido}")
        else:
            print(f"\n [!] Error al cambiar de micrófono: {res.stderr.strip()}")
    else:
        print(" [!] Opción fuera de rango.")


def verificar_disponibilidad_microfono(
    dispositivo: Optional[int] = None, sample_rate: int = SAMPLE_RATE_DEFAULT
) -> Tuple[bool, str, str]:
    """
    Comprueba qué motor de captura está disponible y devuelve el nombre real del
    micrófono activo en el sistema (consultando PipeWire `wpctl` en Linux o `sounddevice`).
    """
    usar_arecord_preferido = (
        sys.platform.startswith("linux")
        and dispositivo is None
        and _arecord_disponible()
    )

    if usar_arecord_preferido:
        try:
            res = subprocess.run(
                ["arecord", "-l"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if res.returncode == 0 and "card" in res.stdout.lower():
                nombre_pw = _obtener_info_fuente_pipewire()
                if nombre_pw:
                    return True, "arecord", nombre_pw
                return (
                    True,
                    "arecord",
                    "PipeWire / ALSA arecord (micrófono predeterminado del sistema)",
                )
        except Exception:
            pass

    if SOUNDDEVICE_DISPONIBLE:
        disp_efectivo = (
            dispositivo
            if dispositivo is not None
            else _elegir_mejor_dispositivo_sounddevice()
        )
        try:
            sd.check_input_settings(
                device=disp_efectivo, channels=CANALES_DEFAULT, samplerate=sample_rate
            )
            info_dev = sd.query_devices(device=disp_efectivo, kind="input")
            nombre_dev = info_dev.get("name", "Dispositivo predeterminado")
            return True, "sounddevice", f"sounddevice ({nombre_dev})"
        except Exception as exc:
            if not _arecord_disponible():
                return (
                    False,
                    "sounddevice",
                    f"No se pudo acceder al micrófono mediante sounddevice: {exc}",
                )

    if _arecord_disponible():
        try:
            res = subprocess.run(
                ["arecord", "-l"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if res.returncode == 0 and "card" in res.stdout.lower():
                return (
                    True,
                    "arecord",
                    "PipeWire / ALSA arecord (micrófono del sistema)",
                )
            return (
                False,
                "arecord",
                "No se detectaron dispositivos de captura ALSA activos.",
            )
        except Exception as exc:
            return False, "arecord", f"Error al consultar arecord: {exc}"

    return (
        False,
        "ninguno",
        "La biblioteca 'sounddevice' no está instalada. Ejecutá: pip install sounddevice soundfile",
    )


def listar_dispositivos_entrada() -> List[Tuple[int, str, int]]:
    """
    Devuelve una lista de dispositivos de entrada (micrófonos) disponibles
    como tuplas (indice, nombre, canales_entrada).
    """
    dispositivos: List[Tuple[int, str, int]] = []
    if not SOUNDDEVICE_DISPONIBLE:
        return dispositivos

    try:
        todos = sd.query_devices()
        for idx, dev in enumerate(todos):
            max_in = int(dev.get("max_input_channels", 0))
            if max_in > 0:
                dispositivos.append((idx, str(dev.get("name", f"Dev {idx}")), max_in))
    except Exception:
        pass
    return dispositivos


def guardar_wav_seguro(
    ruta_destino: str, audio_mono_int16: np.ndarray, sample_rate: int
) -> None:
    """
    Guarda un arreglo numpy 1D/2D (int16, mono) como archivo WAV PCM de 16 bits
    utilizando escritura atómica mediante archivo temporal en el mismo directorio.
    Usa `soundfile` si está disponible y recurre al módulo estándar `wave` en caso contrario.
    """
    carpeta_destino = os.path.dirname(os.path.abspath(ruta_destino))
    os.makedirs(carpeta_destino, exist_ok=True)

    datos_1d = np.asarray(audio_mono_int16, dtype=np.int16).reshape(-1)

    fd_temp, ruta_temp = tempfile.mkstemp(
        prefix=".tmp_voz_", suffix=".wav", dir=carpeta_destino
    )
    os.close(fd_temp)

    try:
        if SOUNDFILE_DISPONIBLE:
            sf.write(
                ruta_temp,
                datos_1d,
                samplerate=sample_rate,
                subtype="PCM_16",
                format="WAV",
            )
        else:
            with wave.open(ruta_temp, "wb") as wf:
                wf.setnchannels(CANALES_DEFAULT)
                wf.setsampwidth(2)  # 2 bytes = 16 bits (int16)
                wf.setframerate(sample_rate)
                wf.writeframes(datos_1d.tobytes())

        # Reemplazo atómico (nunca deja un archivo a medio escribir)
        os.replace(ruta_temp, ruta_destino)
    except Exception:
        if os.path.exists(ruta_temp):
            try:
                os.remove(ruta_temp)
            except OSError:
                pass
        raise


def capturar_audio_microfono(
    duracion_seg: float,
    sample_rate: int = SAMPLE_RATE_DEFAULT,
    dispositivo: Optional[int] = None,
) -> np.ndarray:
    """
    Captura audio en mono (int16) durante `duracion_seg` segundos.
    - En Linux con PipeWire/ALSA prioriza `arecord` cuando no se fuerza un ID de
      dispositivo específico, asegurando usar el micrófono predeterminado real
      del sistema (ej. micrófono digital interno DMIC) y el mismo motor con el
      que se grabó el dataset, evitando entradas analógicas desconectadas en PortAudio.
    - Si se usa `sounddevice` y se detecta una señal bloqueada en continua (jack
      desconectado clavado en -1.0), recurre automáticamente a `arecord`.
    """
    num_muestras = int(round(duracion_seg * sample_rate))

    usar_arecord_preferido = (
        sys.platform.startswith("linux")
        and dispositivo is None
        and _arecord_disponible()
    )
    if usar_arecord_preferido:
        return _capturar_con_arecord(num_muestras=num_muestras, sample_rate=sample_rate)

    if SOUNDDEVICE_DISPONIBLE:
        disp_efectivo = (
            dispositivo
            if dispositivo is not None
            else _elegir_mejor_dispositivo_sounddevice()
        )
        grabacion = sd.rec(
            frames=num_muestras,
            samplerate=sample_rate,
            channels=CANALES_DEFAULT,
            dtype="int16",
            device=disp_efectivo,
        )
        try:
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            raise
        audio_sd = np.asarray(grabacion, dtype=np.int16).reshape(-1)

        if _es_captura_saturada_dc(audio_sd) and _arecord_disponible():
            print(
                "   [!] Aviso: El dispositivo PortAudio devolvió una señal bloqueada en DC "
                "(entrada analógica sin micrófono). Reintentando captura vía PipeWire/arecord..."
            )
            return _capturar_con_arecord(
                num_muestras=num_muestras, sample_rate=sample_rate
            )
        return audio_sd

    if _arecord_disponible():
        return _capturar_con_arecord(num_muestras=num_muestras, sample_rate=sample_rate)

    raise RuntimeError(
        "No hay ningún backend de captura de audio disponible. "
        "Instalá las dependencias con: pip install sounddevice soundfile"
    )


def calcular_metricas_senal(audio_int16: np.ndarray) -> Tuple[float, float]:
    """
    Calcula el nivel RMS y el pico máximo AC (descontando componente continua DC)
    normalizados en [0.0, 1.0] para informar si el micrófono captó voz real o silencio.
    """
    if audio_int16.size == 0:
        return 0.0, 0.0
    norm = audio_int16.astype(np.float32) / 32768.0
    norm_ac = norm - float(np.mean(norm))
    rms = float(np.sqrt(np.mean(np.square(norm_ac))))
    pico = float(np.max(np.abs(norm_ac)))
    return rms, pico


# ==============================================================================
# FUNCIONES DE INTEGRACIÓN PARA EL AGENTE FINAL Y EL CLASIFICADOR K-NN
# ==============================================================================

def leer_wav_mono(
    ruta_wav: str, normalizar_float: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Lee un archivo WAV del dataset y devuelve (senal_1d, sample_rate).
    - Si `normalizar_float=True`, devuelve un arreglo `float32` en el rango [-1.0, 1.0]
      (formato estándar para calcular MFCC, energía o espectrogramas).
    - Si `normalizar_float=False`, devuelve el arreglo original en `int16`.
    """
    if SOUNDFILE_DISPONIBLE:
        dtype_lectura = "float32" if normalizar_float else "int16"
        datos, sr = sf.read(ruta_wav, dtype=dtype_lectura, always_2d=False)
        if datos.ndim > 1:
            datos = datos[:, 0]
        return np.asarray(datos), int(sr)

    with wave.open(ruta_wav, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)
        audio_i16 = np.frombuffer(raw_bytes, dtype=np.int16).copy()
        if wf.getnchannels() > 1:
            audio_i16 = audio_i16[:: wf.getnchannels()]

    if normalizar_float:
        return audio_i16.astype(np.float32) / 32768.0, int(sr)
    return audio_i16, int(sr)


def cargar_dataset_desde_csv(
    directorio_base: str = DIRECTORIO_DATASET_DEFAULT,
    normalizar_float: bool = True,
) -> Tuple[List[np.ndarray], List[str], List[str], List[Dict[str, str]]]:
    """
    Lee `dataset_voz/metadata.csv` y carga en memoria todos los audios WAV junto con
    sus etiquetas, listos para extraer características acústicas y entrenar el K-NN.

    Retorna una tupla de 4 listas paralelas:
        - audios: lista de arreglos numpy 1D (cada señal de voz).
        - etiquetas_comando: lista con la clase de cada audio ('Reconocer', 'Identificar', 'Salir').
        - personas: lista con el identificador de la persona ('Persona_01', etc.).
        - metadatos: lista con el diccionario completo de cada fila del CSV.
    """
    registros = leer_registros_metadata(directorio_base)
    audios: List[np.ndarray] = []
    etiquetas_comando: List[str] = []
    personas: List[str] = []
    metadatos_validos: List[Dict[str, str]] = []

    for reg in registros:
        ruta_rel = reg.get("ruta_audio", "")
        ruta_real = resolver_ruta_en_disco(directorio_base, ruta_rel)
        if not os.path.isfile(ruta_real):
            continue
        senal, _ = leer_wav_mono(ruta_real, normalizar_float=normalizar_float)
        audios.append(senal)
        etiquetas_comando.append(reg["comando"])
        personas.append(reg["persona"])
        metadatos_validos.append(reg)

    return audios, etiquetas_comando, personas, metadatos_validos


def escuchar_orden_en_vivo(
    duracion_seg: float = DURACION_DEFAULT,
    sample_rate: int = SAMPLE_RATE_DEFAULT,
    dispositivo: Optional[int] = None,
    cuenta_regresiva: int = 0,
    normalizar_float: bool = True,
    guardar_wav_temporal: Optional[str] = None,
    mostrar_mensajes: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Función lista para ser invocada por el Agente en la ejecución final cuando
    necesite escuchar un comando de voz en tiempo real y pasarlo al clasificador K-NN.

    Garantiza capturar el audio exactamente con el mismo formato (Mono, `sample_rate`)
    que se usó al construir el dataset de entrenamiento.

    Retorna:
        (senal_audio, sample_rate)
        donde `senal_audio` es un arreglo numpy 1D (`float32` en [-1, 1] si
        `normalizar_float=True`, o `int16` en caso contrario).
    """
    ok_mic, _, detalle_mic = verificar_disponibilidad_microfono(
        dispositivo=dispositivo, sample_rate=sample_rate
    )
    if not ok_mic:
        raise RuntimeError(f"Micrófono no disponible: {detalle_mic}")

    if mostrar_mensajes:
        print(f"\n[Micrófono activo: {detalle_mic}]", flush=True)

    if cuenta_regresiva > 0:
        ejecutar_cuenta_regresiva(cuenta_regresiva)

    if mostrar_mensajes:
        print(
            f"\n>>> [AGENTE ESCUCHANDO] Diga un comando («Reconocer», «Identificar» o «Salir») "
            f"[{duracion_seg:.1f} s]... <<<",
            flush=True,
        )

    audio_int16 = capturar_audio_microfono(
        duracion_seg=duracion_seg,
        sample_rate=sample_rate,
        dispositivo=dispositivo,
    )

    if mostrar_mensajes:
        print(">>> [FIN DE CAPTURA] Procesando orden de voz... <<<", flush=True)

    if guardar_wav_temporal:
        guardar_wav_seguro(guardar_wav_temporal, audio_int16, sample_rate)

    if normalizar_float:
        return audio_int16.astype(np.float32) / 32768.0, sample_rate
    return audio_int16, sample_rate


# ==============================================================================
# FLUJO DE SESIÓN DE GRABACIÓN
# ==============================================================================

def ejecutar_cuenta_regresiva(segundos: int = CUENTA_REGRESIVA_DEFAULT) -> None:
    """Muestra una cuenta regresiva clara por consola antes de iniciar la captura."""
    for restante in range(segundos, 0, -1):
        print(f"   Preparate... {restante}", flush=True)
        time.sleep(0.85)


def grabar_sesion_comando(
    directorio_base: str,
    persona: str,
    comando: str,
    cantidad_tomas: int,
    duracion_seg: float = DURACION_DEFAULT,
    sample_rate: int = SAMPLE_RATE_DEFAULT,
    dispositivo: Optional[int] = None,
    cuenta_regresiva: int = CUENTA_REGRESIVA_DEFAULT,
    confirmar_entre_tomas: bool = True,
) -> List[Dict[str, object]]:
    """
    Ejecuta una sesión completa de grabación para una persona y un comando.
    - Verifica la disponibilidad del micrófono antes de empezar.
    - Crea la estructura de carpetas de la persona y el archivo metadata.csv.
    - Realiza la cuenta regresiva e indica exactamente cuándo hablar.
    - Guarda cada archivo WAV y actualiza metadata.csv inmediatamente después de cada toma.
    - Permite cancelar la sesión entre tomas o con Ctrl+C sin perder las tomas ya grabadas.
    """
    if comando not in COMANDOS_PROYECTO:
        raise ValueError(
            f"El comando '{comando}' no es válido. Debe ser uno de: {COMANDOS_PROYECTO}"
        )

    ok_mic, backend_nombre, detalle_mic = verificar_disponibilidad_microfono(
        dispositivo=dispositivo, sample_rate=sample_rate
    )
    if not ok_mic:
        print("\n" + "!" * 72)
        print(" [ERROR] MICRÓFONO O DEPENDENCIA DE AUDIO NO DISPONIBLE")
        print("!" * 72)
        print(f" Detalle: {detalle_mic}")
        print("\n Para instalar las bibliotecas recomendadas ejecutá:")
        print("   pip install sounddevice soundfile")
        print("!" * 72)
        return []

    inicializar_estructura_persona(directorio_base, persona)
    asegurar_archivo_metadata(directorio_base)

    print("\n" + "=" * 72)
    print(" INICIO DE SESIÓN DE GRABACIÓN DE COMANDOS DE VOZ")
    print("=" * 72)
    print(f" - Persona seleccionada   : {persona}")
    print(f" - Comando a pronunciar   : «{comando}»")
    print(f" - Tomas programadas      : {cantidad_tomas}")
    print(f" - Duración por toma      : {duracion_seg:.2f} s")
    print(f" - Formato de audio       : WAV Mono (1 canal), {sample_rate} Hz, 16-bit PCM")
    print(f" - Motor de captura activo: {detalle_mic}")
    print("-" * 72)
    print(" Instrucciones:")
    print("  * Esperá la cuenta regresiva y hablá cuando veas '>>> ¡HABLÁ AHORA! <<<'.")
    print("  * En cualquier momento podés presionar Ctrl+C para cancelar la sesión")
    print("    de forma segura (las tomas ya completadas se conservarán intactas).")
    print("=" * 72)

    try:
        resp_inicio = input(
            "\nPresioná [Enter] para comenzar la primera toma (o 'c' para cancelar): "
        ).strip().lower()
        if resp_inicio in ("c", "q", "salir", "cancelar"):
            print("[i] Sesión cancelada antes de iniciar.")
            return []
    except (KeyboardInterrupt, EOFError):
        print("\n[i] Sesión cancelada por el usuario.")
        return []

    grabaciones_sesion: List[Dict[str, object]] = []
    toma_actual_idx = 1

    while toma_actual_idx <= cantidad_tomas:
        numero_toma_archivo = calcular_siguiente_numero_toma(
            directorio_base, persona, comando
        )
        nombre_wav = construir_nombre_archivo(persona, comando, numero_toma_archivo)
        ruta_wav_abs = os.path.abspath(
            os.path.join(directorio_base, persona, comando, nombre_wav)
        )
        ruta_wav_rel = construir_ruta_metadata(
            directorio_base, persona, comando, nombre_wav
        )

        print(
            f"\n--- Toma {toma_actual_idx} de {cantidad_tomas} "
            f"(Archivo #{numero_toma_archivo:03d}: {nombre_wav}) ---"
        )

        try:
            if cuenta_regresiva > 0:
                ejecutar_cuenta_regresiva(cuenta_regresiva)

            print(
                f"\n   >>> ¡HABLÁ AHORA! Pronunciá claramente: «{comando.upper()}» "
                f"({duracion_seg:.1f} s) <<<",
                flush=True,
            )

            audio_capturado = capturar_audio_microfono(
                duracion_seg=duracion_seg,
                sample_rate=sample_rate,
                dispositivo=dispositivo,
            )

            print("   [*] Grabación finalizada. Procesando...", flush=True)

            duracion_real = len(audio_capturado) / float(sample_rate)
            rms, pico = calcular_metricas_senal(audio_capturado)

            # Advertir si la señal es extremadamente baja (posible micrófono muteado)
            if pico < 0.005:
                print(
                    "   [!] Aviso: Se detectó un nivel de volumen muy bajo "
                    f"(pico={pico:.4f}). Verificá que el micrófono no esté silenciado."
                )

            # Guardar WAV de forma atómica y actualizar metadata.csv
            guardar_wav_seguro(ruta_wav_abs, audio_capturado, sample_rate)
            marca_tiempo = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            registrar_en_metadata(
                directorio_base=directorio_base,
                ruta_audio_rel=ruta_wav_rel,
                persona=persona,
                comando=comando,
                numero_toma=numero_toma_archivo,
                fecha_hora=marca_tiempo,
                duracion_seg=duracion_real,
                frecuencia_muestreo=sample_rate,
            )

            info_toma = {
                "ruta_audio": ruta_wav_rel,
                "persona": persona,
                "comando": comando,
                "numero_toma": numero_toma_archivo,
                "fecha_hora": marca_tiempo,
                "duracion_seg": duracion_real,
                "frecuencia_muestreo": sample_rate,
                "pico": pico,
                "rms": rms,
            }
            grabaciones_sesion.append(info_toma)

            print(
                f"   [OK] Guardado: {ruta_wav_rel} "
                f"(duración={duracion_real:.2f}s, nivel pico={pico * 100:.1f}%)"
            )

        except KeyboardInterrupt:
            print(
                "\n\n[!] Captura interrumpida con Ctrl+C. "
                "La toma en curso fue descartada de forma segura."
            )
            break
        except Exception as exc:
            print(f"\n[ERROR] Falló la captura o el guardado de esta toma: {exc}")
            try:
                reintentar = input(
                    "¿Deseás reintentar esta toma? [S/n]: "
                ).strip().lower()
                if reintentar in ("n", "no", "c", "q"):
                    break
                continue
            except (KeyboardInterrupt, EOFError):
                print("\n[i] Sesión cancelada.")
                break

        # Si quedan más tomas por grabar, avisar y permitir cancelar o repetir
        if toma_actual_idx < cantidad_tomas:
            siguiente_idx = toma_actual_idx + 1
            if confirmar_entre_tomas:
                try:
                    accion = input(
                        f"\n   -> Siguiente: Toma {siguiente_idx}/{cantidad_tomas} de «{comando}». "
                        "[Enter] para continuar | 'r' para repetir la toma anterior | 'c' para cancelar sesión: "
                    ).strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print("\n[i] Sesión cancelada por el usuario.")
                    break

                if accion in ("c", "q", "salir", "cancelar"):
                    print("[i] Sesión cancelada a pedido del usuario.")
                    break
                if accion in ("r", "repetir"):
                    # Eliminar la última toma grabada y su fila en metadata para grabarla de nuevo
                    ultima = grabaciones_sesion.pop()
                    _eliminar_ultima_toma(directorio_base, str(ultima["ruta_audio"]))
                    print(
                        f"   [i] Se descartó '{ultima['ruta_audio']}'. "
                        f"Volviendo a grabar la toma {toma_actual_idx}..."
                    )
                    continue
            else:
                print(
                    f"\n   -> Preparando toma {siguiente_idx}/{cantidad_tomas} "
                    f"(presioná Ctrl+C para cancelar)..."
                )
                try:
                    time.sleep(PAUSA_ENTRE_TOMAS_DEFAULT)
                except KeyboardInterrupt:
                    print("\n[i] Sesión cancelada por el usuario.")
                    break

        toma_actual_idx += 1

    mostrar_resumen_sesion(
        directorio_base=directorio_base,
        persona=persona,
        comando=comando,
        tomas_solicitadas=cantidad_tomas,
        grabaciones_sesion=grabaciones_sesion,
    )
    return grabaciones_sesion


def _eliminar_ultima_toma(directorio_base: str, ruta_audio_rel: str) -> None:
    """
    Elimina un archivo WAV recién grabado y su entrada correspondiente en metadata.csv
    cuando el usuario solicita repetir inmediatamente la toma.
    """
    ruta_normalizada = ruta_audio_rel.replace("\\", "/")
    ruta_real = resolver_ruta_en_disco(directorio_base, ruta_normalizada)
    if os.path.isfile(ruta_real):
        try:
            os.remove(ruta_real)
        except OSError:
            pass

    ruta_csv = obtener_ruta_metadata(directorio_base)
    if not os.path.isfile(ruta_csv):
        return

    filas = leer_registros_metadata(directorio_base)
    filas_filtradas = [
        f for f in filas if f.get("ruta_audio", "").replace("\\", "/") != ruta_normalizada
    ]

    with open(ruta_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS_METADATA)
        writer.writeheader()
        writer.writerows(filas_filtradas)
        f.flush()
        os.fsync(f.fileno())


def mostrar_resumen_sesion(
    directorio_base: str,
    persona: str,
    comando: str,
    tomas_solicitadas: int,
    grabaciones_sesion: List[Dict[str, object]],
) -> None:
    """Muestra en consola el detalle de las grabaciones realizadas y el acumulado."""
    print("\n" + "=" * 72)
    print(" RESUMEN DE LA SESIÓN FINALIZADA")
    print("=" * 72)
    print(f" - Persona                 : {persona}")
    print(f" - Comando                 : {comando}")
    print(f" - Tomas guardadas ahora   : {len(grabaciones_sesion)} / {tomas_solicitadas}")

    if grabaciones_sesion:
        print("\n Archivos generados en esta sesión:")
        for item in grabaciones_sesion:
            print(
                f"   * [Toma #{int(item['numero_toma']):03d}] {item['ruta_audio']} "
                f"({float(item['duracion_seg']):.2f}s @ {item['frecuencia_muestreo']} Hz)"
            )

    conteo_actual = contar_audios_persona(directorio_base, persona)
    total_persona = sum(conteo_actual.values())
    print(f"\n Estado acumulado para '{persona}' ({total_persona} audios en total):")
    for cmd in COMANDOS_PROYECTO:
        print(f"   - {cmd:<12}: {conteo_actual[cmd]} grabaciones")
    print(f"\n Metadatos actualizados en : {obtener_ruta_metadata(directorio_base)}")
    print("=" * 72)


# ==============================================================================
# AUDITORÍA Y RESUMEN GENERAL DEL DATASET
# ==============================================================================

def mostrar_estado_dataset(directorio_base: str) -> None:
    """
    Inspecciona el directorio `dataset_voz/` y `metadata.csv`, mostrando una tabla
    con la cantidad de muestras por persona y por comando, y verificando la
    integridad de los archivos WAV registrados.
    """
    print("\n" + "=" * 72)
    print(f" ESTADO ACTUAL DEL DATASET DE VOZ ({directorio_base}/)")
    print("=" * 72)

    personas = listar_personas_existentes(directorio_base)
    if not personas:
        print(" [i] Todavía no hay personas ni grabaciones registradas en el dataset.")
        print("=" * 72)
        return

    encabezado = (
        f" {'Persona':<18} | {'Reconocer':>10} | {'Identificar':>11} | "
        f"{'Salir':>8} | {'Total':>8}"
    )
    print(encabezado)
    print(" " + "-" * (len(encabezado) - 1))

    gran_total = 0
    for p in personas:
        c = contar_audios_persona(directorio_base, p)
        tot_p = sum(c.values())
        gran_total += tot_p
        print(
            f" {p:<18} | {c['Reconocer']:>10} | {c['Identificar']:>11} | "
            f"{c['Salir']:>8} | {tot_p:>8}"
        )

    print(" " + "-" * (len(encabezado) - 1))
    print(f" {'TOTAL GENERAL':<18} | {'':>10} | {'':>11} | {'':>8} | {gran_total:>8}")

    registros = leer_registros_metadata(directorio_base)
    faltantes = 0
    for reg in registros:
        ruta = reg.get("ruta_audio", "")
        if ruta and not os.path.isfile(resolver_ruta_en_disco(directorio_base, ruta)):
            faltantes += 1

    print(f"\n - Filas registradas en metadata.csv : {len(registros)}")
    if faltantes > 0:
        print(f" - [!] Archivos listados en CSV que no están en disco: {faltantes}")
        print("       (Podés usar la opción [5] -> Sincronizar metadata.csv para repararlo)")
    else:
        print(" - Integridad de rutas en metadata.csv: OK (todos los archivos existen)")
    print("=" * 72)


# ==============================================================================
# BORRADO SEGURO DE GRABACIONES Y SINCRONIZACIÓN DE METADATA.CSV
# ==============================================================================

def reescribir_metadata_seguro(
    directorio_base: str, filas: List[Dict[str, object]]
) -> None:
    """
    Reescribe el archivo metadata.csv de manera atómica (usando un archivo temporal
    en el mismo directorio y `os.replace`) para evitar corrupción de datos.
    """
    os.makedirs(directorio_base, exist_ok=True)
    ruta_csv = obtener_ruta_metadata(directorio_base)
    fd_temp, ruta_temp = tempfile.mkstemp(
        prefix=".tmp_meta_", suffix=".csv", dir=directorio_base
    )
    os.close(fd_temp)

    try:
        with open(ruta_temp, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNAS_METADATA)
            writer.writeheader()
            for fila in filas:
                fila_limpia = {col: fila.get(col, "") for col in COLUMNAS_METADATA}
                writer.writerow(fila_limpia)
            f.flush()
            os.fsync(f.fileno())
        os.replace(ruta_temp, ruta_csv)
    except Exception:
        if os.path.exists(ruta_temp):
            try:
                os.remove(ruta_temp)
            except OSError:
                pass
        raise


def sincronizar_metadata_con_disco(directorio_base: str) -> Tuple[int, int, int]:
    """
    Sincroniza `metadata.csv` con los archivos WAV realmente existentes en `dataset_voz/`:
    1. Elimina del CSV cualquier fila cuyo archivo `.wav` haya sido borrado en disco.
    2. Escanea las carpetas `<Persona>/<Comando>/*.wav` y agrega al CSV cualquier
       grabación válida que no estuviera registrada.
    Retorna `(filas_eliminadas, filas_agregadas, total_final)`.
    """
    registros_actuales = leer_registros_metadata(directorio_base)
    filas_validas: List[Dict[str, object]] = []
    rutas_registradas = set()
    eliminadas = 0

    for reg in registros_actuales:
        ruta_rel = reg.get("ruta_audio", "").replace("\\", "/")
        ruta_disco = resolver_ruta_en_disco(directorio_base, ruta_rel)
        if ruta_rel and os.path.isfile(ruta_disco):
            filas_validas.append(reg)
            rutas_registradas.add(os.path.abspath(ruta_disco))
        else:
            eliminadas += 1

    # Buscar archivos WAV en disco que no estén en el CSV
    agregadas = 0
    patron_toma = re.compile(r"_toma_(\d+)\.wav$", flags=re.IGNORECASE)
    if os.path.isdir(directorio_base):
        for persona in sorted(os.listdir(directorio_base)):
            ruta_p = os.path.join(directorio_base, persona)
            if not os.path.isdir(ruta_p) or persona.startswith("."):
                continue
            for cmd in COMANDOS_PROYECTO:
                ruta_c = os.path.join(ruta_p, cmd)
                if not os.path.isdir(ruta_c):
                    continue
                for nombre_wav in sorted(os.listdir(ruta_c)):
                    if not nombre_wav.lower().endswith(".wav") or nombre_wav.startswith("."):
                        continue
                    ruta_abs = os.path.abspath(os.path.join(ruta_c, nombre_wav))
                    if ruta_abs in rutas_registradas:
                        continue
                    # Leer propiedades del WAV para reconstruir la fila
                    try:
                        with wave.open(ruta_abs, "rb") as wf:
                            sr = wf.getframerate()
                            n_frames = wf.getnframes()
                            dur = n_frames / float(sr) if sr > 0 else 0.0
                    except Exception:
                        continue

                    m = patron_toma.search(nombre_wav)
                    num_toma = int(m.group(1)) if m else 1
                    mtime = os.path.getmtime(ruta_abs)
                    fecha_str = datetime.datetime.fromtimestamp(mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    filas_validas.append(
                        {
                            "ruta_audio": construir_ruta_metadata(
                                directorio_base, persona, cmd, nombre_wav
                            ),
                            "persona": persona,
                            "comando": cmd,
                            "numero_toma": num_toma,
                            "fecha_hora": fecha_str,
                            "duracion_seg": f"{dur:.2f}",
                            "frecuencia_muestreo": sr,
                        }
                    )
                    agregadas += 1

    reescribir_metadata_seguro(directorio_base, filas_validas)
    return eliminadas, agregadas, len(filas_validas)


def eliminar_set_comando(
    directorio_base: str, persona: str, comando: str
) -> Tuple[int, int]:
    """
    Borra todos los archivos `.wav` de un comando específico de una persona
    (ej. todo `Persona_02/Identificar`) y limpia sus filas de `metadata.csv`.
    Retorna `(archivos_borrados, filas_csv_eliminadas)`.
    """
    carpeta_cmd = os.path.join(directorio_base, persona, comando)
    archivos_borrados = 0
    if os.path.isdir(carpeta_cmd):
        for f in os.listdir(carpeta_cmd):
            if f.lower().endswith(".wav"):
                try:
                    os.remove(os.path.join(carpeta_cmd, f))
                    archivos_borrados += 1
                except OSError as exc:
                    print(f" [!] No se pudo borrar '{f}': {exc}")

    registros = leer_registros_metadata(directorio_base)
    nuevas_filas = [
        r
        for r in registros
        if not (r.get("persona") == persona and r.get("comando") == comando)
    ]
    filas_quitadas = len(registros) - len(nuevas_filas)
    reescribir_metadata_seguro(directorio_base, nuevas_filas)
    return archivos_borrados, filas_quitadas


def eliminar_tomas_especificas(
    directorio_base: str, persona: str, comando: str, numeros_toma: List[int]
) -> Tuple[int, int]:
    """
    Borra únicamente los números de toma indicados para una persona y comando,
    tanto en disco como en `metadata.csv`.
    Retorna `(archivos_borrados, filas_csv_eliminadas)`.
    """
    conjunto_nums = set(numeros_toma)
    archivos_borrados = 0
    carpeta_cmd = os.path.join(directorio_base, persona, comando)

    for num in sorted(conjunto_nums):
        nombre_wav = construir_nombre_archivo(persona, comando, num)
        ruta_wav = os.path.join(carpeta_cmd, nombre_wav)
        if os.path.isfile(ruta_wav):
            try:
                os.remove(ruta_wav)
                archivos_borrados += 1
            except OSError as exc:
                print(f" [!] Error al borrar '{ruta_wav}': {exc}")

    registros = leer_registros_metadata(directorio_base)
    nuevas_filas = []
    for r in registros:
        es_misma_pc = r.get("persona") == persona and r.get("comando") == comando
        try:
            num_fila = int(r.get("numero_toma", -1))
        except ValueError:
            num_fila = -1
        if es_misma_pc and num_fila in conjunto_nums:
            continue
        nuevas_filas.append(r)

    filas_quitadas = len(registros) - len(nuevas_filas)
    reescribir_metadata_seguro(directorio_base, nuevas_filas)
    return archivos_borrados, filas_quitadas


def eliminar_persona_completa(
    directorio_base: str, persona: str
) -> Tuple[int, int]:
    """
    Elimina la carpeta completa de una persona (`dataset_voz/<Persona>`) junto con
    todos sus audios y borra todas sus entradas en `metadata.csv`.
    Retorna `(archivos_borrados, filas_csv_eliminadas)`.
    """
    conteo = contar_audios_persona(directorio_base, persona)
    archivos_borrados = sum(conteo.values())

    ruta_persona = os.path.join(directorio_base, persona)
    if os.path.isdir(ruta_persona):
        shutil.rmtree(ruta_persona)

    registros = leer_registros_metadata(directorio_base)
    nuevas_filas = [r for r in registros if r.get("persona") != persona]
    filas_quitadas = len(registros) - len(nuevas_filas)
    reescribir_metadata_seguro(directorio_base, nuevas_filas)
    return archivos_borrados, filas_quitadas


# ==============================================================================
# MENÚ INTERACTIVO DE CONSOLA
# ==============================================================================

def _seleccionar_persona_menu(
    directorio_base: str, persona_actual: Optional[str] = None
) -> Optional[str]:
    """Submenú para elegir una persona existente o registrar una nueva."""
    personas = listar_personas_existentes(directorio_base)
    sugerencia_nueva = sugerir_nombre_nueva_persona(personas)

    print("\n" + "-" * 72)
    print(" 1. SELECCIÓN O REGISTRO DE PERSONA")
    print("-" * 72)

    if personas:
        print(" Personas registradas actualmente en el dataset:")
        for idx, p in enumerate(personas, 1):
            c = contar_audios_persona(directorio_base, p)
            resumen = ", ".join(f"{cmd}: {c[cmd]}" for cmd in COMANDOS_PROYECTO)
            marca = " (actual)" if p == persona_actual else ""
            print(f"   [{idx}] {p}{marca} -> ({resumen})")
        print(f"   [N] Registrar una nueva persona (sugerido: {sugerencia_nueva})")
        print("   [0] Volver / Cancelar")

        default_str = "1"
        if persona_actual and persona_actual in personas:
            default_str = str(personas.index(persona_actual) + 1)

        while True:
            entrada = input(
                f"\n Elegí un número de la lista, 'N' para nueva persona o escribí el nombre [Default: {default_str}]: "
            ).strip()
            if not entrada:
                entrada = default_str

            if entrada == "0":
                return None

            if entrada.isdigit():
                indice = int(entrada)
                if 1 <= indice <= len(personas):
                    elegida = personas[indice - 1]
                    inicializar_estructura_persona(directorio_base, elegida)
                    return elegida

            if entrada.lower() in ("n", "nueva", "nuevo"):
                nombre_in = input(
                    f" Ingresá el nombre o número de la nueva persona [Default: {sugerencia_nueva}]: "
                ).strip()
                nombre_final = (
                    sanitizar_nombre_persona(nombre_in)
                    if nombre_in
                    else sugerencia_nueva
                )
                if not nombre_final:
                    print(" [!] El nombre ingresado no es válido. Intentá nuevamente.")
                    continue
                inicializar_estructura_persona(directorio_base, nombre_final)
                print(f" [+] Persona registrada y carpetas listas: {nombre_final}")
                return nombre_final

            # Permitir que escriba directamente el nombre (ej. "Persona_03" o "Maria")
            nombre_directo = sanitizar_nombre_persona(entrada)
            if nombre_directo:
                inicializar_estructura_persona(directorio_base, nombre_directo)
                return nombre_directo

            print(" [!] Entrada no válida. Por favor elegí una opción de la lista.")
    else:
        print(" No hay personas registradas todavía en el dataset.")
        while True:
            nombre_in = input(
                f" Ingresá el nombre o número de la persona a registrar [Default: {sugerencia_nueva}]: "
            ).strip()
            if nombre_in == "0":
                return None
            nombre_final = (
                sanitizar_nombre_persona(nombre_in)
                if nombre_in
                else sugerencia_nueva
            )
            if nombre_final:
                inicializar_estructura_persona(directorio_base, nombre_final)
                print(f" [+] Persona creada con sus 3 carpetas de comandos: {nombre_final}")
                return nombre_final
            print(" [!] Nombre inválido. Intentá nuevamente.")


def _seleccionar_comando_menu(directorio_base: str, persona: str) -> Optional[str]:
    """Submenú para elegir cuál de los tres comandos obligatorios grabar."""
    conteo = contar_audios_persona(directorio_base, persona)
    print("\n" + "-" * 72)
    print(f" 2. SELECCIÓN DEL COMANDO A GRABAR (Persona: {persona})")
    print("-" * 72)
    for idx, cmd in enumerate(COMANDOS_PROYECTO, 1):
        print(f"   [{idx}] «{cmd}» (actualmente: {conteo[cmd]} grabaciones)")
    print("   [0] Volver al menú anterior")

    while True:
        op = input("\n Seleccioná el comando a grabar (1-3) [Default: 1]: ").strip()
        if not op:
            return COMANDOS_PROYECTO[0]
        if op == "0":
            return None
        if op.isdigit() and 1 <= int(op) <= len(COMANDOS_PROYECTO):
            return COMANDOS_PROYECTO[int(op) - 1]
        # También permitir escribir el nombre del comando
        for cmd in COMANDOS_PROYECTO:
            if op.lower() == cmd.lower():
                return cmd
        print(" [!] Opción inválida. Ingresá 1 (Reconocer), 2 (Identificar) o 3 (Salir).")


def _solicitar_cantidad_tomas(default_tomas: int = TOMAS_DEFAULT) -> Optional[int]:
    """Solicita y valida la cantidad de tomas a grabar en la sesión."""
    print("\n" + "-" * 72)
    print(" 3. CANTIDAD DE TOMAS A GRABAR")
    print("-" * 72)
    while True:
        entrada = input(
            f" ¿Cuántas tomas querés grabar de este comando? [Default: {default_tomas}, 0 para cancelar]: "
        ).strip()
        if not entrada:
            return default_tomas
        if entrada == "0":
            return None
        if entrada.isdigit():
            valor = int(entrada)
            if 1 <= valor <= TOMAS_MAX_SESION:
                return valor
        print(
            f" [!] Ingresá un número entero positivo entre 1 y {TOMAS_MAX_SESION}."
        )


def _solicitar_duracion_seg(default_duracion: float = DURACION_DEFAULT) -> float:
    """Solicita y valida la duración en segundos de cada toma."""
    print("\n" + "-" * 72)
    print(" 4. DURACIÓN DE CADA GRABACIÓN")
    print("-" * 72)
    while True:
        entrada = input(
            f" Duración por toma en segundos ({DURACION_MIN_SEG}-{DURACION_MAX_SEG} s) "
            f"[Default: {default_duracion:.1f}]: "
        ).strip().replace(",", ".")
        if not entrada:
            return default_duracion
        try:
            valor = float(entrada)
            if DURACION_MIN_SEG <= valor <= DURACION_MAX_SEG:
                return valor
        except ValueError:
            pass
        print(
            f" [!] Ingresá una duración válida entre {DURACION_MIN_SEG} y {DURACION_MAX_SEG} segundos."
        )


def _configurar_audio_menu(
    sample_rate_actual: int,
    duracion_actual: float,
    dispositivo_actual: Optional[int],
) -> Tuple[int, float, Optional[int]]:
    """Permite ajustar la frecuencia de muestreo, duración predeterminada y micrófono."""
    print("\n" + "=" * 72)
    print(" CONFIGURACIÓN DE PARÁMETROS DE AUDIO")
    print("=" * 72)
    print(f" - Frecuencia de muestreo actual : {sample_rate_actual} Hz")
    print(f" - Duración predeterminada       : {duracion_actual:.2f} s")
    print(f" - Canales                       : 1 (Mono - Fijo por consigna)")

    # 1. Frecuencia de muestreo
    print("\n Frecuencias de muestreo estándar disponibles:")
    for idx, sr in enumerate(SAMPLE_RATES_VALIDOS, 1):
        tag = " (recomendado para voz/MFCC)" if sr == 16000 else ""
        actual_tag = " [ACTUAL]" if sr == sample_rate_actual else ""
        print(f"   [{idx}] {sr} Hz{tag}{actual_tag}")

    op_sr = input(
        f" Elegí frecuencia de muestreo (1-{len(SAMPLE_RATES_VALIDOS)}) o Enter para mantener {sample_rate_actual} Hz: "
    ).strip()
    nuevo_sr = sample_rate_actual
    if op_sr.isdigit():
        val = int(op_sr)
        if 1 <= val <= len(SAMPLE_RATES_VALIDOS):
            nuevo_sr = SAMPLE_RATES_VALIDOS[val - 1]
        elif val in SAMPLE_RATES_VALIDOS:
            nuevo_sr = val

    # 2. Duración predeterminada
    nueva_duracion = _solicitar_duracion_seg(default_duracion=duracion_actual)

    # 3. Selección de dispositivo de micrófono
    nuevo_disp = dispositivo_actual
    if sys.platform.startswith("linux") and shutil.which("wpctl") is not None:
        menu_cambiar_microfono_sistema()
    else:
        dispositivos = listar_dispositivos_entrada()
        if dispositivos:
            print("\n Dispositivos de entrada (micrófono) detectados por sounddevice:")
            print("   [D] Usar micrófono predeterminado del sistema")
            for idx_dev, nombre_dev, ch_in in dispositivos:
                marca = " [ACTUAL]" if dispositivo_actual == idx_dev else ""
                print(f"   [{idx_dev}] {nombre_dev} ({ch_in} ch){marca}")
            op_dev = input(
                " Seleccioná ID de dispositivo, 'D' para predeterminado, o Enter para mantener: "
            ).strip()
            if op_dev.lower() == "d":
                nuevo_disp = None
            elif op_dev.isdigit():
                id_elegido = int(op_dev)
                if any(d[0] == id_elegido for d in dispositivos):
                    nuevo_disp = id_elegido
                else:
                    print(" [!] ID no encontrado en la lista; se mantiene la configuración previa.")

    print(
        f"\n [+] Configuración actualizada: {nuevo_sr} Hz | "
        f"{nueva_duracion:.2f} s | Dispositivo: {nuevo_disp if nuevo_disp is not None else 'Predeterminado'}"
    )
    return nuevo_sr, nueva_duracion, nuevo_disp


def _menu_borrar_y_sincronizar(
    directorio_base: str, persona_activa: Optional[str] = None
) -> Optional[str]:
    """
    Submenú para borrar de forma segura sets de comandos, tomas específicas o
    personas completas manteniendo `metadata.csv` siempre sincronizado.
    Devuelve la `persona_activa` actualizada (o None si fue eliminada).
    """
    while True:
        personas = listar_personas_existentes(directorio_base)
        print("\n" + "=" * 72)
        print(" BORRADO SEGURO Y SINCRONIZACIÓN DE METADATA.CSV")
        print("=" * 72)
        print("   [1] Borrar todo un comando de una persona (ej. rehacer todo «Reconocer» de Persona_01)")
        print("   [2] Borrar tomas puntuales de un comando (ej. borrar toma 2 y 5)")
        print("   [3] Borrar una persona completa (todas sus carpetas y grabaciones)")
        print("   [4] Sincronizar metadata.csv con las carpetas (repara el CSV si borraste archivos a mano)")
        print("   [0] Volver al menú principal")
        print("=" * 72)

        op = input(" Seleccioná una opción [Default: 0]: ").strip()
        if not op or op == "0":
            return persona_activa

        if op == "4":
            elim, agreg, total = sincronizar_metadata_con_disco(directorio_base)
            print(
                f"\n [+] Sincronización completada en metadata.csv:\n"
                f"     - Filas huérfanas eliminadas : {elim}\n"
                f"     - Archivos nuevos agregados  : {agreg}\n"
                f"     - Total de audios válidos    : {total}"
            )
            continue

        if not personas:
            print(" [i] No hay personas registradas en el dataset para borrar.")
            continue

        # Seleccionar persona sobre la cual operar
        print("\n Personas disponibles:")
        for idx, p in enumerate(personas, 1):
            c = contar_audios_persona(directorio_base, p)
            resumen = ", ".join(f"{cmd}: {c[cmd]}" for cmd in COMANDOS_PROYECTO)
            print(f"   [{idx}] {p} ({resumen})")
        print("   [0] Cancelar")

        op_p = input(" Elegí el número de la persona: ").strip()
        if not op_p.isdigit() or not (1 <= int(op_p) <= len(personas)):
            print(" [i] Operación cancelada.")
            continue
        persona_obj = personas[int(op_p) - 1]

        if op == "1":
            # Borrar todo un comando de esa persona
            cmd_obj = _seleccionar_comando_menu(directorio_base, persona_obj)
            if not cmd_obj:
                continue
            conf = input(
                f" ¿Confirmás borrar TODAS las tomas de «{cmd_obj}» para '{persona_obj}'? [s/N]: "
            ).strip().lower()
            if conf in ("s", "si", "sí", "y", "yes"):
                n_wav, n_csv = eliminar_set_comando(
                    directorio_base, persona_obj, cmd_obj
                )
                print(
                    f"\n [+] Set eliminado: {n_wav} archivos WAV borrados y "
                    f"{n_csv} filas quitadas de metadata.csv."
                )
            else:
                print(" [i] Cancelado.")

        elif op == "2":
            # Borrar tomas específicas
            cmd_obj = _seleccionar_comando_menu(directorio_base, persona_obj)
            if not cmd_obj:
                continue
            carpeta_c = os.path.join(directorio_base, persona_obj, cmd_obj)
            archivos = (
                sorted(f for f in os.listdir(carpeta_c) if f.lower().endswith(".wav"))
                if os.path.isdir(carpeta_c)
                else []
            )
            if not archivos:
                print(f" [i] '{persona_obj}' no tiene grabaciones en «{cmd_obj}».")
                continue

            print(f"\n Tomas actuales en {persona_obj} -> «{cmd_obj}»:")
            for f in archivos:
                print(f"   - {f}")
            entrada_nums = input(
                "\n Ingresá los números de toma a borrar separados por coma (ej: 1,3 o 5): "
            ).strip()
            nums = [
                int(x.strip())
                for x in entrada_nums.replace(";", ",").split(",")
                if x.strip().isdigit()
            ]
            if not nums:
                print(" [i] No se indicaron números válidos. Cancelado.")
                continue
            n_wav, n_csv = eliminar_tomas_especificas(
                directorio_base, persona_obj, cmd_obj, nums
            )
            print(
                f"\n [+] Tomas eliminadas: {n_wav} archivos WAV borrados y "
                f"{n_csv} filas quitadas de metadata.csv."
            )

        elif op == "3":
            # Borrar persona completa
            conf = input(
                f" ¿Confirmás borrar TODA la carpeta de '{persona_obj}' y todos sus audios? [s/N]: "
            ).strip().lower()
            if conf in ("s", "si", "sí", "y", "yes"):
                n_wav, n_csv = eliminar_persona_completa(directorio_base, persona_obj)
                if persona_activa == persona_obj:
                    persona_activa = None
                print(
                    f"\n [+] Persona '{persona_obj}' eliminada: {n_wav} archivos WAV borrados y "
                    f"{n_csv} filas quitadas de metadata.csv."
                )
            else:
                print(" [i] Cancelado.")


def menu_interactivo(
    directorio_base: str = DIRECTORIO_DATASET_DEFAULT,
    sample_rate_inicial: int = SAMPLE_RATE_DEFAULT,
    duracion_inicial: float = DURACION_DEFAULT,
) -> None:
    """
    Bucle principal del menú interactivo de consola para gestionar la grabación
    y organización del dataset de comandos de voz.
    """
    asegurar_archivo_metadata(directorio_base)

    sample_rate = sample_rate_inicial
    duracion_predeterminada = duracion_inicial
    dispositivo_mic: Optional[int] = None
    persona_activa: Optional[str] = None

    while True:
        ok_mic, _, desc_mic = verificar_disponibilidad_microfono(
            dispositivo=dispositivo_mic, sample_rate=sample_rate
        )
        estado_mic_str = f"OK ({desc_mic})" if ok_mic else f"NO DISPONIBLE ({desc_mic})"

        print("\n" + "=" * 72)
        print("   GRABADOR Y ORGANIZADOR DE DATASET DE VOZ (IA1 - UNCUYO)")
        print("=" * 72)
        print(f" Carpeta del dataset : {os.path.abspath(directorio_base)}")
        print(f" Persona seleccionada: {persona_activa if persona_activa else '(Ninguna seleccionada aún)'}")
        print(f" Formato de captura  : WAV Mono | {sample_rate} Hz | Duración base: {duracion_predeterminada:.1f} s")
        print(f" Estado del micrófono: {estado_mic_str}")
        print("-" * 72)
        print(" MENÚ PRINCIPAL:")
        print("   [1] Iniciar sesión de grabación (Elegir persona -> comando -> tomas -> grabar)")
        print("   [2] Cambiar o registrar persona activa")
        print("   [3] Grabar otro comando para la persona activa")
        print("   [4] Ver resumen y auditoría del dataset actual")
        print("   [5] Borrar grabaciones o sincronizar metadata.csv")
        print("   [6] Configurar duración, frecuencia de muestreo o micrófono")
        print("   [0] Salir del programa")
        print("=" * 72)

        try:
            opcion = input(" Seleccioná una opción [Default: 1]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[+] Saliendo del programa. ¡Hasta luego!")
            break

        if not opcion:
            opcion = "1"

        if opcion == "0" or opcion.lower() in ("salir", "q", "exit"):
            print("\n[+] Programa finalizado correctamente.")
            break

        elif opcion == "1":
            # Paso 1: Elegir persona (existente o nueva)
            persona_elegida = _seleccionar_persona_menu(
                directorio_base, persona_actual=persona_activa
            )
            if not persona_elegida:
                continue
            persona_activa = persona_elegida

            # Bucle interno para permitir grabar otro comando tras terminar
            while True:
                comando_elegido = _seleccionar_comando_menu(
                    directorio_base, persona_activa
                )
                if not comando_elegido:
                    break

                cantidad_tomas = _solicitar_cantidad_tomas(default_tomas=TOMAS_DEFAULT)
                if cantidad_tomas is None:
                    break

                duracion_toma = _solicitar_duracion_seg(
                    default_duracion=duracion_predeterminada
                )
                duracion_predeterminada = duracion_toma

                grabar_sesion_comando(
                    directorio_base=directorio_base,
                    persona=persona_activa,
                    comando=comando_elegido,
                    cantidad_tomas=cantidad_tomas,
                    duracion_seg=duracion_toma,
                    sample_rate=sample_rate,
                    dispositivo=dispositivo_mic,
                )

                try:
                    post = input(
                        f"\n¿Qué deseás hacer ahora?\n"
                        f"  [1] Grabar otro comando para '{persona_activa}'\n"
                        f"  [2] Cambiar de persona\n"
                        f"  [Enter / 0] Volver al menú principal\n"
                        f"Selección: "
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    post = "0"

                if post == "1":
                    continue
                elif post == "2":
                    nueva_p = _seleccionar_persona_menu(
                        directorio_base, persona_actual=persona_activa
                    )
                    if nueva_p:
                        persona_activa = nueva_p
                        continue
                    break
                else:
                    break

        elif opcion == "2":
            nueva = _seleccionar_persona_menu(
                directorio_base, persona_actual=persona_activa
            )
            if nueva:
                persona_activa = nueva
                print(f"\n[+] Persona activa establecida en: {persona_activa}")

        elif opcion == "3":
            if not persona_activa:
                persona_activa = _seleccionar_persona_menu(directorio_base)
                if not persona_activa:
                    continue

            comando_elegido = _seleccionar_comando_menu(directorio_base, persona_activa)
            if not comando_elegido:
                continue

            cantidad_tomas = _solicitar_cantidad_tomas(default_tomas=TOMAS_DEFAULT)
            if cantidad_tomas is None:
                continue

            duracion_toma = _solicitar_duracion_seg(
                default_duracion=duracion_predeterminada
            )
            duracion_predeterminada = duracion_toma

            grabar_sesion_comando(
                directorio_base=directorio_base,
                persona=persona_activa,
                comando=comando_elegido,
                cantidad_tomas=cantidad_tomas,
                duracion_seg=duracion_toma,
                sample_rate=sample_rate,
                dispositivo=dispositivo_mic,
            )

        elif opcion == "4":
            mostrar_estado_dataset(directorio_base)

        elif opcion == "5":
            persona_activa = _menu_borrar_y_sincronizar(
                directorio_base, persona_activa=persona_activa
            )

        elif opcion == "6":
            sample_rate, duracion_predeterminada, dispositivo_mic = (
                _configurar_audio_menu(
                    sample_rate_actual=sample_rate,
                    duracion_actual=duracion_predeterminada,
                    dispositivo_actual=dispositivo_mic,
                )
            )

        else:
            print(" [!] Opción no reconocida. Elegí un número entre 0 y 6.")


# ==============================================================================
# PUNTO DE ENTRADA CLI
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Programa para grabar y organizar el dataset de comandos de voz "
            "(«Reconocer», «Identificar», «Salir») - IA1 UNCUYO."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DIRECTORIO_DATASET_DEFAULT,
        help=f"Carpeta raíz del dataset (default: {DIRECTORIO_DATASET_DEFAULT})",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE_DEFAULT,
        choices=SAMPLE_RATES_VALIDOS,
        help=f"Frecuencia de muestreo en Hz (default: {SAMPLE_RATE_DEFAULT})",
    )
    parser.add_argument(
        "--duracion",
        type=float,
        default=DURACION_DEFAULT,
        help=f"Duración predeterminada por toma en segundos (default: {DURACION_DEFAULT})",
    )
    parser.add_argument(
        "--resumen",
        action="store_true",
        help="Mostrar únicamente el resumen de auditoría del dataset actual y salir",
    )
    parser.add_argument(
        "--sincronizar",
        action="store_true",
        help="Sincronizar metadata.csv con los archivos reales en dataset_voz/ y salir",
    )

    args = parser.parse_args()

    if args.sincronizar:
        asegurar_archivo_metadata(args.dataset_dir)
        elim, agreg, total = sincronizar_metadata_con_disco(args.dataset_dir)
        print(
            f"[+] Sincronización completada: {elim} filas huérfanas eliminadas, "
            f"{agreg} archivos agregados. Total actual en CSV: {total}."
        )
        mostrar_estado_dataset(args.dataset_dir)
        return

    if args.resumen:
        asegurar_archivo_metadata(args.dataset_dir)
        mostrar_estado_dataset(args.dataset_dir)
        return

    menu_interactivo(
        directorio_base=args.dataset_dir,
        sample_rate_inicial=args.sample_rate,
        duracion_inicial=args.duracion,
    )


if __name__ == "__main__":
    main()
