import os
import glob
import cv2
import numpy as np
import random
import argparse

# ==============================================================================
# FUNCIONES DE TRANSFORMACIÓN
# ==============================================================================

def aplicar_rotacion(imagen, angulo_maximo=15):
    """Aplica una rotación afín aleatoria dentro de un rango especificado."""
    h, w = imagen.shape[:2]
    centro = (w // 2, h // 2)
    angulo = random.uniform(-angulo_maximo, angulo_maximo)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    border_val = (255, 255, 255) if len(imagen.shape) == 3 else 255
    return cv2.warpAffine(imagen, matriz_rotacion, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)


def aplicar_traslacion(imagen, desplazamiento_max=15):
    """Aplica una traslación afín aleatoria en los ejes X e Y."""
    h, w = imagen.shape[:2]
    tx = random.randint(-desplazamiento_max, desplazamiento_max)
    ty = random.randint(-desplazamiento_max, desplazamiento_max)
    matriz_traslacion = np.float32([[1, 0, tx], [0, 1, ty]])
    border_val = (255, 255, 255) if len(imagen.shape) == 3 else 255
    return cv2.warpAffine(imagen, matriz_traslacion, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)


def deformacion_elastica(imagen, alpha_rango=(10, 20), sigma_rango=(3.5, 5.0)):
    """
    Aplica deformación elástica no lineal basada en campos de desplazamiento aleatorios suavizados.
    Simula la elasticidad plástica y no lineal de la piel del dedo al presionar el sensor.
    """
    alpha = random.uniform(*alpha_rango)
    sigma = random.uniform(*sigma_rango)
    h, w = imagen.shape[:2]

    # Campos de desplazamiento aleatorio [-1, 1]
    dx = (np.random.rand(h, w) * 2 - 1).astype(np.float32)
    dy = (np.random.rand(h, w) * 2 - 1).astype(np.float32)

    ksize = int(sigma * 4) | 1  # debe ser impar
    dx = cv2.GaussianBlur(dx, (ksize, ksize), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (ksize, ksize), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    border_val = (255, 255, 255) if len(imagen.shape) == 3 else 255
    return cv2.remap(imagen, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)


def distorsion_ondulatoria(imagen, amplitud_rango=(2.0, 5.0), frecuencia_rango=(0.02, 0.05)):
    """
    Aplica una distorsión armónica no lineal (swirl / ondulación sinusoidal).
    Simula torsión o curvatura del dedo en el sensor óptico.
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
    return cv2.remap(imagen, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)


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
    """Simula una lectura incompleta cortando un borde aleatorio de la huella."""
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
        "desc": "Simula la elasticidad y plasticidad de la piel al apoyarse",
        "funcion": deformacion_elastica
    },
    2: {
        "nombre": "Distorsión Ondulatoria (No Lineal)",
        "desc": "Curvatura/ondulación sinusoidal de las crestas papilares",
        "funcion": distorsion_ondulatoria
    },
    3: {
        "nombre": "Variación de Presión Morfológica (No Lineal)",
        "desc": "Ensanchamiento o adelgazamiento de crestas por presión (Erosión/Dilatación)",
        "funcion": simular_variacion_presion
    },
    4: {
        "nombre": "Corrección Gamma / Contraste (No Lineal)",
        "desc": "Variación no lineal de luminosidad y contraste de escáner",
        "funcion": ajuste_gamma
    },
    5: {
        "nombre": "Rotación Aleatoria",
        "desc": "Rotación afín del dedo (-15° a +15°)",
        "funcion": aplicar_rotacion
    },
    6: {
        "nombre": "Traslación Aleatoria",
        "desc": "Desplazamiento horizontal y vertical en el sensor",
        "funcion": aplicar_traslacion
    },
    7: {
        "nombre": "Ruido Gaussiano",
        "desc": "Ruido de sensor electrónico analógico",
        "funcion": agregar_ruido_gaussiano
    },
    8: {
        "nombre": "Ruido Sal y Pimienta",
        "desc": "Suciedad o píxeles defectuosos en el lector",
        "funcion": agregar_ruido_sal_pimienta
    },
    9: {
        "nombre": "Corte Parcial / Oclusión",
        "desc": "Lectura incompleta por apoyo parcial del dedo",
        "funcion": aplicar_corte_parcial
    }
}


# ==============================================================================
# PROCESAMIENTO Y PIPELINE
# ==============================================================================

def aplicar_transformaciones_aleatorias(imagen, ids_disponibles, min_t=1, max_t=2):
    """
    Aplica una combinación aleatoria de 'k' transformaciones seleccionadas
    del catálogo disponible, con intensidades variables.
    """
    if not ids_disponibles:
        return imagen, []

    k = random.randint(min_t, min(max_t, len(ids_disponibles)))
    elegidas = random.sample(ids_disponibles, k)
    
    img_actual = imagen.copy()
    nombres_aplicadas = []
    for tid in elegidas:
        t_info = TRANSFORMACIONES[tid]
        img_actual = t_info["funcion"](img_actual)
        nombres_aplicadas.append(t_info["nombre"])

    return img_actual, nombres_aplicadas


def buscar_imagenes(ruta_origen):
    """Busca recursivamente archivos de imagen en la ruta indicada."""
    extensiones = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    archivos = []
    for ext in extensiones:
        archivos.extend(glob.glob(os.path.join(ruta_origen, '**', ext), recursive=True))
    archivos = sorted(list(set(archivos)))
    return archivos


def detectar_carpetas_candidatas():
    """Detecta carpetas en el espacio de trabajo que contengan imágenes de huellas."""
    carpetas = []
    bases = ['huellas_prueba', 'huellas_dataset', 'huellas', '.']
    for b in bases:
        if os.path.exists(b):
            for dirpath, _, filenames in os.walk(b):
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) for f in filenames):
                    norm = os.path.normpath(dirpath)
                    if norm not in carpetas and norm != '.':
                        carpetas.append(norm)
    return sorted(carpetas)


def ejecutar_procesamiento(rutas_directorios, porcentaje, ids_transformaciones,
                          min_t=1, max_t=2, modo_guardado='nuevo_archivo',
                          directorio_salida=None):
    """
    Procesa las imágenes aplicando las transformaciones a una fracción del dataset.
    """
    todas_imagenes = []
    for r in rutas_directorios:
        if os.path.isdir(r):
            imgs = buscar_imagenes(r)
            todas_imagenes.extend(imgs)
        elif os.path.isfile(r):
            todas_imagenes.append(r)

    todas_imagenes = sorted(list(set(todas_imagenes)))
    total_imgs = len(todas_imagenes)
    if total_imgs == 0:
        print("\n[!] No se encontraron imágenes en las rutas especificadas.")
        return

    # Determinar cuántas y cuáles imágenes se van a transformar
    cant_a_transformar = max(1, int(round(total_imgs * (porcentaje / 100.0))))
    imagenes_seleccionadas = set(random.sample(todas_imagenes, cant_a_transformar))

    print("\n" + "="*60)
    print(" RESUMEN DEL PROCESAMIENTO:")
    print(f" - Total de imágenes encontradas: {total_imgs}")
    print(f" - Porcentaje a transformar: {porcentaje:.1f}% ({cant_a_transformar} imágenes)")
    print(f" - Transformaciones por imagen: entre {min_t} y {max_t}")
    print(f" - Modo de guardado: {modo_guardado}")
    if directorio_salida:
        print(f" - Directorio de salida: {directorio_salida}")
    print("="*60)

    procesadas = 0
    base_comun = os.path.commonpath(rutas_directorios) if len(rutas_directorios) > 1 else rutas_directorios[0]

    for ruta_img in todas_imagenes:
        img = cv2.imread(ruta_img)
        if img is None:
            print(f"[!] Error al leer: {ruta_img}")
            continue

        if ruta_img in imagenes_seleccionadas:
            # Transformar
            img_trans, transformadas = aplicar_transformaciones_aleatorias(
                img, ids_transformaciones, min_t=min_t, max_t=max_t
            )
            procesadas += 1

            # Determinar ruta destino según modo
            nombre_base = os.path.basename(ruta_img)
            nombre_sin_ext, ext = os.path.splitext(nombre_base)

            if modo_guardado == 'sobrescribir':
                ruta_dest = ruta_img
            elif modo_guardado == 'nuevo_archivo':
                # Guardar en la misma carpeta con sufijo _dist
                dir_padre = os.path.dirname(ruta_img)
                ruta_dest = os.path.join(dir_padre, f"{nombre_sin_ext}_dist{ext}")
            elif modo_guardado == 'carpeta_salida':
                # Replicar subcarpeta en directorio_salida
                rel_path = os.path.relpath(ruta_img, start=base_comun)
                ruta_dest = os.path.join(directorio_salida, rel_path)
                os.makedirs(os.path.dirname(ruta_dest), exist_ok=True)
            else:
                ruta_dest = ruta_img

            cv2.imwrite(ruta_dest, img_trans)
            print(f"[{procesadas}/{cant_a_transformar}] {nombre_base} -> {os.path.basename(ruta_dest)} | {', '.join(transformadas)}")
        else:
            # Si el modo es carpeta_salida, copiamos la original intacta
            if modo_guardado == 'carpeta_salida':
                rel_path = os.path.relpath(ruta_img, start=base_comun)
                ruta_dest = os.path.join(directorio_salida, rel_path)
                os.makedirs(os.path.dirname(ruta_dest), exist_ok=True)
                cv2.imwrite(ruta_dest, img)

    print("\n[+] Proceso finalizado exitosamente.")
    print(f"    Total transformadas: {procesadas} de {total_imgs} imágenes.")


# ==============================================================================
# INTERFAZ INTERACTIVA POR TERMINAL
# ==============================================================================

def menu_interactivo():
    print("\n" + "="*65)
    print("      DISTORSIONADOR DE HUELLAS DACTILARES - TRABAJO FINAL")
    print("="*65)

    # 1. Autodetección de carpetas con huellas
    candidatos = detectar_carpetas_candidatas()
    print("\n1. SELECCIÓN DE CARPETAS / DATASET:")
    if candidatos:
        print("   Carpetas detectadas en el proyecto:")
        for idx, c in enumerate(candidatos, 1):
            n_imgs = len(buscar_imagenes(c))
            print(f"   [{idx}] {c} ({n_imgs} imágenes)")
        print(f"   [0] Ingresar ruta(s) manual(es)")
        print("   [T] Procesar todas las carpetas detectadas arriba")
        
        eleccion_c = input("\n   Elige una opción o escribe rutas [Default: T]: ").strip()
        if not eleccion_c or eleccion_c.upper() == 'T':
            rutas_directorios = candidatos
        elif eleccion_c == '0':
            rutas_manual = input("   Ingresa las rutas separadas por coma: ").strip()
            rutas_directorios = [r.strip() for r in rutas_manual.split(",") if r.strip()]
        elif eleccion_c.isdigit() and 1 <= int(eleccion_c) <= len(candidatos):
            rutas_directorios = [candidatos[int(eleccion_c) - 1]]
        else:
            rutas_directorios = [r.strip() for r in eleccion_c.split(",") if r.strip()]
    else:
        entrada_rutas = input("   Ingresa la(s) ruta(s) a procesar separadas por coma: ").strip()
        rutas_directorios = [r.strip() for r in entrada_rutas.split(",") if r.strip()]

    # Verificar que existan
    rutas_validas = [r for r in rutas_directorios if os.path.exists(r)]
    if not rutas_validas:
        print("[!] Ninguna de las rutas especificadas existe. Abortando.")
        return
    rutas_directorios = rutas_validas

    # 2. Porcentaje a distorsionar
    print("\n2. PORCENTAJE DEL DATASET A DISTORSIONAR:")
    print("   (Indica qué porcentaje de las imágenes recibirán transformaciones)")
    entrada_pct = input("   Porcentaje (0-100) [Default: 50]: ").strip()
    try:
        porcentaje = float(entrada_pct) if entrada_pct else 50.0
        porcentaje = max(0.0, min(100.0, porcentaje))
    except ValueError:
        porcentaje = 50.0
        print(f"   [!] Valor inválido, usando por defecto {porcentaje}%.")

    # 3. Menú de transformaciones
    print("\n3. SELECCIÓN DE TRANSFORMACIONES A UTILIZAR:")
    print("   Elige qué transformaciones pueden aplicarse al dataset:")
    for k, v in TRANSFORMACIONES.items():
        print(f"   [{k}] {v['nombre']}: {v['desc']}")

    print("\n   Opciones de selección:")
    print("   - Escribe los números separados por coma (ej: 1,2,3,5)")
    print("   - Escribe 'nl' o 'no lineales' para usar solo transformaciones no lineales (1, 2, 3, 4)")
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
            print("   [!] No se ingresaron números válidos. Usando todas.")
            ids_elegidos = list(TRANSFORMACIONES.keys())

    print(f"\n   -> Transformaciones activadas ({len(ids_elegidos)}):")
    for i in ids_elegidos:
        print(f"      * {TRANSFORMACIONES[i]['nombre']}")

    # 4. Cantidad de transformaciones por imagen
    print("\n4. DIVERSIDAD DE TRANSFORMACIONES POR FOTO:")
    print("   Para evitar que todas las fotos queden con las mismas distorsiones,")
    print("   se elegirá un subconjunto y parámetros aleatorios para cada foto seleccionada.")
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
    print("   [1] Guardar en la misma carpeta como nuevas imágenes (ej: '1-0_dist.jpg')")
    print("   [2] Guardar en una nueva carpeta (ej: 'huellas_distorsionadas/')")
    print("   [3] Sobrescribir las imágenes originales directamente")
    modo_input = input("   Selecciona modo [Default: 1]: ").strip()

    dir_salida = None
    if modo_input == '2':
        modo_guardado = 'carpeta_salida'
        dir_salida = input("   Ruta de carpeta de salida [Default: huellas_distorsionadas]: ").strip()
        if not dir_salida:
            dir_salida = "huellas_distorsionadas"
    elif modo_input == '3':
        confirmacion = input("   [ADVERTENCIA] ¿Estás seguro de sobrescribir los originales? (s/N): ").strip().lower()
        if confirmacion in ['s', 'si', 'y', 'yes']:
            modo_guardado = 'sobrescribir'
        else:
            print("   Operación cancelada, cambiando a crear nuevos archivos con sufijo.")
            modo_guardado = 'nuevo_archivo'
    else:
        modo_guardado = 'nuevo_archivo'

    # Ejecutar
    ejecutar_procesamiento(
        rutas_directorios=rutas_directorios,
        porcentaje=porcentaje,
        ids_transformaciones=ids_elegidos,
        min_t=min_t,
        max_t=max_t,
        modo_guardado=modo_guardado,
        directorio_salida=dir_salida
    )


# ==============================================================================
# MAIN / CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distorsionador y aumentador de huellas dactilares")
    parser.add_argument("--rutas", nargs="+", help="Rutas a carpetas o imágenes a procesar")
    parser.add_argument("--porcentaje", type=float, default=None, help="Porcentaje de imágenes a distorsionar (0-100)")
    parser.add_argument("--transformaciones", type=str, default=None, help="IDs de transformaciones separadas por coma o 'nl'")
    parser.add_argument("--min_t", type=int, default=1, help="Mínimo de transformaciones por foto")
    parser.add_argument("--max_t", type=int, default=2, help="Máximo de transformaciones por foto")
    parser.add_argument("--modo", choices=['nuevo_archivo', 'carpeta_salida', 'sobrescribir'], default='nuevo_archivo')
    parser.add_argument("--salida", type=str, default="huellas_distorsionadas")
    parser.add_argument("--no-interactivo", action="store_true", help="Ejecutar directamente con parámetros sin menú")

    args = parser.parse_args()

    # Si se pasan parámetros por CLI con --no-interactivo, ejecutamos directamente
    if args.no_interactivo and args.rutas:
        if args.transformaciones:
            if args.transformaciones.lower() in ['nl', 'no lineales']:
                ids_t = [1, 2, 3, 4]
            else:
                ids_t = [int(x.strip()) for x in args.transformaciones.split(",") if x.strip().isdigit()]
        else:
            ids_t = list(TRANSFORMACIONES.keys())

        pct = args.porcentaje if args.porcentaje is not None else 50.0
        ejecutar_procesamiento(
            rutas_directorios=args.rutas,
            porcentaje=pct,
            ids_transformaciones=ids_t,
            min_t=args.min_t,
            max_t=args.max_t,
            modo_guardado=args.modo,
            directorio_salida=args.salida
        )
    else:
        # Modo interactivo por defecto al correr `python3 distorsionador.py`
        menu_interactivo()
