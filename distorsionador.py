import os
import sys
import glob
import cv2
import numpy as np
import random
import argparse

# ==============================================================================
# FUNCIONES DE TRANSFORMACIÓN
# Todas garantizan mantener exactamente las dimensiones originales (w x h).
# ==============================================================================

def aplicar_rotacion(imagen, angulo_maximo=15):
    """Aplica una rotación afín aleatoria sin alterar las dimensiones en píxeles."""
    h, w = imagen.shape[:2]
    centro = (w // 2, h // 2)
    angulo = random.uniform(-angulo_maximo, angulo_maximo)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    border_val = (255, 255, 255) if len(imagen.shape) == 3 else 255
    res = cv2.warpAffine(imagen, matriz_rotacion, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    return res


def aplicar_traslacion(imagen, desplazamiento_max=15):
    """Aplica una traslación afín aleatoria sin alterar las dimensiones en píxeles."""
    h, w = imagen.shape[:2]
    tx = random.randint(-desplazamiento_max, desplazamiento_max)
    ty = random.randint(-desplazamiento_max, desplazamiento_max)
    matriz_traslacion = np.float32([[1, 0, tx], [0, 1, ty]])
    border_val = (255, 255, 255) if len(imagen.shape) == 3 else 255
    res = cv2.warpAffine(imagen, matriz_traslacion, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    return res


def deformacion_elastica(imagen, alpha_rango=(10, 20), sigma_rango=(3.5, 5.0)):
    """
    Aplica deformación elástica no lineal basada en campos gaussianos aleatorios (Simard et al.).
    Simula la plasticidad de la piel del dedo al presionar el sensor, conservando píxeles exactos.
    """
    h, w = imagen.shape[:2]
    alpha = random.uniform(*alpha_rango)
    sigma = random.uniform(*sigma_rango)

    # Campos de desplazamiento aleatorio [-1, 1]
    dx = (np.random.rand(h, w) * 2 - 1).astype(np.float32)
    dy = (np.random.rand(h, w) * 2 - 1).astype(np.float32)

    ksize = int(sigma * 4) | 1  # impar
    dx = cv2.GaussianBlur(dx, (ksize, ksize), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (ksize, ksize), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    border_val = (255, 255, 255) if len(imagen.shape) == 3 else 255
    res = cv2.remap(imagen, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    return res


def distorsion_ondulatoria(imagen, amplitud_rango=(2.0, 5.0), frecuencia_rango=(0.02, 0.05)):
    """
    Aplica una distorsión armónica no lineal (torsión/ondulación sinusoidal de crestas).
    Conserva rigurosamente el ancho y alto en píxeles.
    """
    h, w = imagen.shape[:2]
    amp_x = random.uniform(*amplitud_rango)
    amp_y = random.uniform(*amplitud_rango)
    freq_x = random.uniform(*frecuencia_rango)
    freq_y = random.uniform(*frecuencia_rango)

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + amp_x * np.sin(2 * np.pi * y * freq_y)).astype(np.float32)
    map_y = (y + amp_y * np.cos(2 * np.pi * x * freq_x)).astype(np.float32)

    border_val = (255, 255, 255) if len(imagen.shape) == 3 else 255
    res = cv2.remap(imagen, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    return res


def simular_variacion_presion(imagen, tipo='aleatorio', max_iter=2):
    """
    Simula variaciones de presión usando transformaciones morfológicas no lineales:
    - Presión fuerte: ensancha las crestas oscuras (Erosión).
    - Presión débil: adelgaza las crestas oscuras (Dilatación).
    """
    if tipo == 'aleatorio':
        tipo = random.choice(['fuerte', 'debil'])
    
    iteraciones = random.randint(1, max_iter)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    if tipo == 'fuerte':
        return cv2.erode(imagen, kernel, iterations=iteraciones)
    else:
        return cv2.dilate(imagen, kernel, iterations=iteraciones)


def agregar_ruido_gaussiano(imagen, media=0, desviacion_rango=(10, 25)):
    """Añade ruido gaussiano con desviación estándar aleatoria."""
    std = random.uniform(*desviacion_rango)
    ruido = np.random.normal(media, std, imagen.shape)
    imagen_ruidosa = np.clip(imagen.astype(np.float32) + ruido, 0, 255)
    return imagen_ruidosa.astype(np.uint8)


def agregar_ruido_sal_pimienta(imagen, probabilidad_rango=(0.01, 0.03)):
    """Añade ruido impulsivo de sal (blanco) y pimienta (negro)."""
    probabilidad = random.uniform(*probabilidad_rango)
    imagen_ruidosa = imagen.copy()
    h, w = imagen.shape[:2]
    rnd = np.random.rand(h, w)

    if len(imagen.shape) == 3:
        mask_pimienta = rnd < (probabilidad / 2.0)
        mask_sal = (rnd >= (probabilidad / 2.0)) & (rnd < probabilidad)
        imagen_ruidosa[mask_pimienta] = [0, 0, 0]
        imagen_ruidosa[mask_sal] = [255, 255, 255]
    else:
        imagen_ruidosa[rnd < (probabilidad / 2.0)] = 0
        imagen_ruidosa[(rnd >= (probabilidad / 2.0)) & (rnd < probabilidad)] = 255

    return imagen_ruidosa


def aplicar_corte_parcial(imagen, porcentaje_max=0.25):
    """
    Simula una lectura incompleta tapando un borde aleatorio.
    NO recorta la imagen: pinta el borde de blanco para preservar el tamaño exacto en píxeles.
    """
    h, w = imagen.shape[:2]
    imagen_cortada = imagen.copy()
    porcentaje_corte = random.uniform(0.08, porcentaje_max)
    borde = random.choice(['arriba', 'abajo', 'izquierda', 'derecha'])

    val = 255 if len(imagen.shape) == 2 else (255, 255, 255)
    if borde == 'arriba':
        imagen_cortada[:int(h * porcentaje_corte), :] = val
    elif borde == 'abajo':
        imagen_cortada[int(h * (1 - porcentaje_corte)):, :] = val
    elif borde == 'izquierda':
        imagen_cortada[:, :int(w * porcentaje_corte)] = val
    elif borde == 'derecha':
        imagen_cortada[:, int(w * (1 - porcentaje_corte)):] = val

    return imagen_cortada


def ajuste_gamma(imagen, gamma_rango=(0.6, 1.5)):
    """Aplica una corrección gamma no lineal de intensidad lumínica y contraste."""
    gamma = random.uniform(*gamma_rango)
    inv_gamma = 1.0 / gamma
    tabla = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(imagen, tabla)


# ==============================================================================
# CATÁLOGO DE TRANSFORMACIONES
# ==============================================================================

TRANSFORMACIONES = {
    1: {
        "nombre": "Deformación Elástica (No Lineal)",
        "desc": "Elasticidad y plasticidad de la piel al apoyarse",
        "funcion": deformacion_elastica
    },
    2: {
        "nombre": "Distorsión Ondulatoria (No Lineal)",
        "desc": "Curvatura/ondulación sinusoidal de las crestas",
        "funcion": distorsion_ondulatoria
    },
    3: {
        "nombre": "Variación de Presión Morfológica (No Lineal)",
        "desc": "Ensanchamiento/adelgazamiento de crestas (Erosión/Dilatación)",
        "funcion": simular_variacion_presion
    },
    4: {
        "nombre": "Corrección Gamma / Contraste (No Lineal)",
        "desc": "Variación no lineal de luminosidad y sensor",
        "funcion": ajuste_gamma
    },
    5: {
        "nombre": "Rotación Aleatoria",
        "desc": "Rotación afín del dedo (-15° a +15°)",
        "funcion": aplicar_rotacion
    },
    6: {
        "nombre": "Traslación Aleatoria",
        "desc": "Desplazamiento horizontal y vertical",
        "funcion": aplicar_traslacion
    },
    7: {
        "nombre": "Ruido Gaussiano",
        "desc": "Ruido electrónico analógico",
        "funcion": agregar_ruido_gaussiano
    },
    8: {
        "nombre": "Ruido Sal y Pimienta",
        "desc": "Suciedad o píxeles defectuosos",
        "funcion": agregar_ruido_sal_pimienta
    },
    9: {
        "nombre": "Corte Parcial / Oclusión",
        "desc": "Lectura incompleta (relleno blanco, sin recortar píxeles)",
        "funcion": aplicar_corte_parcial
    }
}


# ==============================================================================
# PROCESAMIENTO Y CONTROL DE TAMAÑO
# ==============================================================================

def aplicar_transformaciones_aleatorias(imagen, ids_disponibles, min_t=1, max_t=2):
    """
    Aplica una combinación aleatoria de 'k' transformaciones seleccionadas.
    Garantiza que el tamaño en píxeles (w, h) final sea IDÉNTICO al original.
    """
    if not ids_disponibles:
        return imagen, []

    h_orig, w_orig = imagen.shape[:2]
    k = random.randint(min_t, min(max_t, len(ids_disponibles)))
    elegidas = random.sample(ids_disponibles, k)
    
    img_actual = imagen.copy()
    nombres_aplicadas = []
    for tid in elegidas:
        t_info = TRANSFORMACIONES[tid]
        img_actual = t_info["funcion"](img_actual)
        nombres_aplicadas.append(t_info["nombre"])

    # Verificación y garantía estricta de dimensiones
    h_final, w_final = img_actual.shape[:2]
    if (h_final, w_final) != (h_orig, w_orig):
        # Si por alguna razón hubiera una discrepancia, se reajusta exactamente al original
        img_actual = cv2.resize(img_actual, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    return img_actual, nombres_aplicadas


def obtener_fotos_misma_carpeta(carpeta):
    """
    Obtiene exclusivamente las fotos ubicadas directamente en 'carpeta'
    (sin entrar a subdirectorios).
    """
    extensiones = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    if not os.path.isdir(carpeta):
        return []

    archivos = [
        os.path.join(carpeta, f)
        for f in sorted(os.listdir(carpeta))
        if os.path.isfile(os.path.join(carpeta, f))
        and f.lower().endswith(extensiones)
    ]
    return archivos


def procesar_fotos_carpeta(carpeta_objetivo, porcentaje=50.0, ids_transformaciones=None,
                           min_t=1, max_t=2, modo_guardado='sobrescribir'):
    """
    Modifica las fotos de la carpeta especificada según los parámetros indicados.
    """
    if ids_transformaciones is None:
        ids_transformaciones = list(TRANSFORMACIONES.keys())

    archivos = obtener_fotos_misma_carpeta(carpeta_objetivo)
    total_imgs = len(archivos)

    if total_imgs == 0:
        print(f"\n[!] No se encontraron fotos directas en: {carpeta_objetivo}")
        return 0

    # Cantidad y selección aleatoria de fotos a transformar
    cant_a_transformar = max(1, int(round(total_imgs * (porcentaje / 100.0))))
    imagenes_seleccionadas = set(random.sample(archivos, cant_a_transformar))

    print("\n" + "="*65)
    print(" RESUMEN DE PROCESAMIENTO:")
    print(f" - Carpeta de trabajo: {carpeta_objetivo}")
    print(f" - Total de fotos en esta carpeta: {total_imgs}")
    print(f" - Fotos a modificar ({porcentaje:.1f}%): {cant_a_transformar}")
    print(f" - Transformaciones activas: {len(ids_transformaciones)} disponibles")
    print(f" - Combinación por foto: entre {min_t} y {max_t} transformaciones aleatorias")
    print(f" - Modo de guardado: {'Sobrescribir fotos originales' if modo_guardado == 'sobrescribir' else 'Crear copias con sufijo _dist'}")
    print(" - Control de tamaño: Dimensiones de píxeles estrictamente preservadas.")
    print("="*65)

    procesadas = 0
    dimensiones_detectadas = set()

    for ruta_img in archivos:
        if ruta_img not in imagenes_seleccionadas:
            continue

        img = cv2.imread(ruta_img)
        if img is None:
            print(f"[!] Error al abrir la imagen: {ruta_img}")
            continue

        h_orig, w_orig = img.shape[:2]
        dimensiones_detectadas.add(f"{w_orig}x{h_orig} px")

        # Aplicar transformaciones aleatorias con verificación estricta de tamaño
        img_trans, aplicadas = aplicar_transformaciones_aleatorias(
            img, ids_transformaciones, min_t=min_t, max_t=max_t
        )

        h_final, w_final = img_trans.shape[:2]
        assert (h_final, w_final) == (h_orig, w_orig), "Error: discrepancia en tamaño de píxeles"

        procesadas += 1
        nombre_base = os.path.basename(ruta_img)

        if modo_guardado == 'sobrescribir':
            ruta_destino = ruta_img
        else:
            nombre_sin_ext, ext = os.path.splitext(nombre_base)
            ruta_destino = os.path.join(carpeta_objetivo, f"{nombre_sin_ext}_dist{ext}")

        cv2.imwrite(ruta_destino, img_trans)
        print(f"[{procesadas}/{cant_a_transformar}] {nombre_base} ({w_final}x{h_final} px) -> {', '.join(aplicadas)}")

    dims_str = ", ".join(dimensiones_detectadas) if dimensiones_detectadas else "N/A"
    print("\n[+] Proceso finalizado.")
    print(f"    Fotos modificadas: {procesadas} de {total_imgs}")
    print(f"    Tamaño en píxeles verificado: {dims_str} (intacto)")
    return procesadas


# ==============================================================================
# INTERFAZ INTERACTIVA POR TERMINAL
# ==============================================================================

def menu_interactivo():
    print("\n" + "="*65)
    print("      DISTORSIONADOR DE HUELLAS (MISMA CARPETA)")
    print("="*65)

    # Carpeta donde se encuentra este script
    carpeta_script = os.path.dirname(os.path.abspath(__file__))
    fotos_en_carpeta = obtener_fotos_misma_carpeta(carpeta_script)

    print(f"\n1. CARPETA DE TRABAJO:")
    print(f"   Ubicación del script: {carpeta_script}")

    if len(fotos_en_carpeta) > 0:
        print(f"   [+] Se detectaron {len(fotos_en_carpeta)} fotos directamente en esta carpeta.")
        carpeta_objetivo = carpeta_script
    else:
        print("   [i] No hay fotos directas en la carpeta del script.")
        # Buscar carpetas hijas o conocidas con imágenes para facilitar al usuario
        candidatos = []
        for raiz in ['huellas_prueba', 'dataset_huellas', 'huellas_dataset', 'huellas', '.']:
            if os.path.exists(raiz):
                for dirpath, _, filenames in os.walk(raiz):
                    if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) for f in filenames):
                        norm = os.path.normpath(dirpath)
                        if norm not in candidatos and norm != '.':
                            candidatos.append(norm)

        if candidatos:
            print("\n   Subcarpetas con fotos disponibles para trabajar:")
            for idx, c in enumerate(candidatos, 1):
                n_imgs = len(obtener_fotos_misma_carpeta(c))
                print(f"   [{idx}] {c} ({n_imgs} fotos)")
            print("   [0] Escribir otra ruta manualmente")
            
            opcion = input(f"\n   Selecciona una carpeta [Default: 1]: ").strip()
            if not opcion:
                carpeta_objetivo = candidatos[0]
            elif opcion.isdigit() and 1 <= int(opcion) <= len(candidatos):
                carpeta_objetivo = candidatos[int(opcion) - 1]
            elif opcion == '0':
                carpeta_objetivo = input("   Ingresa la ruta de la carpeta: ").strip()
            else:
                carpeta_objetivo = opcion
        else:
            carpeta_objetivo = input("   Ingresa la ruta de la carpeta con las fotos: ").strip()

        fotos_en_carpeta = obtener_fotos_misma_carpeta(carpeta_objetivo)
        if len(fotos_en_carpeta) == 0:
            print(f"[!] No hay imágenes en la carpeta '{carpeta_objetivo}'. Abortando.")
            return

    # 2. Porcentaje de fotos a modificar
    print("\n2. PORCENTAJE DE FOTOS A MODIFICAR:")
    print(f"   Hay {len(fotos_en_carpeta)} fotos. Indica qué porcentaje transformar.")
    entrada_pct = input("   Porcentaje (0-100) [Default: 50]: ").strip()
    try:
        porcentaje = float(entrada_pct) if entrada_pct else 50.0
        porcentaje = max(0.0, min(100.0, porcentaje))
    except ValueError:
        porcentaje = 50.0

    # 3. Selección de transformaciones
    print("\n3. SELECCIÓN DE TRANSFORMACIONES:")
    print("   Elige qué transformaciones pueden aplicarse:")
    for k, v in TRANSFORMACIONES.items():
        tipo_tag = "No Lineal" if "No Lineal" in v['nombre'] else "Lineal/Ruido"
        print(f"   [{k}] ({tipo_tag}) {v['nombre']}: {v['desc']}")

    print("\n   Opciones:")
    print("   - Ingresa los números separados por coma (ej: 1,2,3)")
    print("   - Ingresa 'nl' para usar SOLO transformaciones no lineales (1, 2, 3, 4)")
    print("   - Presiona Enter para usar TODAS las transformaciones (1 a 9)")
    entrada_trans = input("   Selección [Default: TODAS]: ").strip().lower()

    if not entrada_trans:
        ids_elegidos = list(TRANSFORMACIONES.keys())
    elif entrada_trans in ['nl', 'no lineales', 'nolineales']:
        ids_elegidos = [1, 2, 3, 4]
    else:
        ids_elegidos = []
        for parte in entrada_trans.replace(";", ",").split(","):
            parte = parte.strip()
            if parte.isdigit():
                num = int(parte)
                if num in TRANSFORMACIONES:
                    ids_elegidos.append(num)
        if not ids_elegidos:
            ids_elegidos = list(TRANSFORMACIONES.keys())

    print(f"\n   -> Transformaciones activas:")
    for i in ids_elegidos:
        print(f"      * {TRANSFORMACIONES[i]['nombre']}")

    # 4. Cantidad de transformaciones por foto (Diversidad)
    print("\n4. DIVERSIDAD (Para que las fotos no queden con las mismas distorsiones):")
    entrada_k = input("   ¿Cuántas transformaciones combinar por foto? (ej: '1', '1-2', '2-3') [Default: 1-2]: ").strip()
    min_t, max_t = 1, 2
    if entrada_k:
        if '-' in entrada_k:
            partes = entrada_k.split('-')
            try:
                min_t = max(1, int(partes[0]))
                max_t = max(min_t, int(partes[1]))
            except ValueError:
                min_t, max_t = 1, 2
        elif entrada_k.isdigit():
            min_t = max_t = max(1, int(entrada_k))

    # 5. Modo de guardado
    print("\n5. MODO DE GUARDADO:")
    print("   [1] Modificar directamente las fotos originales en la misma carpeta (sobrescribir)")
    print("   [2] Guardar como copias nuevas en la misma carpeta (agregando sufijo _dist)")
    modo_in = input("   Selecciona modo [Default: 1]: ").strip()
    modo_guardado = 'nuevo_archivo' if modo_in == '2' else 'sobrescribir'

    # Ejecutar procesamiento
    procesar_fotos_carpeta(
        carpeta_objetivo=carpeta_objetivo,
        porcentaje=porcentaje,
        ids_transformaciones=ids_elegidos,
        min_t=min_t,
        max_t=max_t,
        modo_guardado=modo_guardado
    )


# ==============================================================================
# MAIN / CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distorsionador de huellas para su misma carpeta")
    parser.add_argument("--carpeta", type=str, default=None, help="Carpeta a procesar (por defecto la carpeta del script)")
    parser.add_argument("--porcentaje", type=float, default=50.0, help="Porcentaje de fotos a modificar (0-100)")
    parser.add_argument("--transformaciones", type=str, default=None, help="IDs separados por coma o 'nl'")
    parser.add_argument("--min_t", type=int, default=1, help="Mínimo de transformaciones por foto")
    parser.add_argument("--max_t", type=int, default=2, help="Máximo de transformaciones por foto")
    parser.add_argument("--modo", choices=['sobrescribir', 'nuevo_archivo'], default='sobrescribir')
    parser.add_argument("--no-interactivo", action="store_true", help="Ejecutar con parámetros sin menú")

    args = parser.parse_args()

    if args.no_interactivo:
        carpeta = args.carpeta if args.carpeta else os.path.dirname(os.path.abspath(__file__))
        if args.transformaciones:
            if args.transformaciones.lower() in ['nl', 'no lineales']:
                ids_t = [1, 2, 3, 4]
            else:
                ids_t = [int(x.strip()) for x in args.transformaciones.split(",") if x.strip().isdigit()]
        else:
            ids_t = list(TRANSFORMACIONES.keys())

        procesar_fotos_carpeta(
            carpeta_objetivo=carpeta,
            porcentaje=args.porcentaje,
            ids_transformaciones=ids_t,
            min_t=args.min_t,
            max_t=args.max_t,
            modo_guardado=args.modo
        )
    else:
        menu_interactivo()
