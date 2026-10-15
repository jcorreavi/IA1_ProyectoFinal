import cv2
import numpy as np
import random

def aplicar_rotacion(imagen, angulo_maximo=15):
    """Aplica una rotación aleatoria dentro de un rango especificado."""
    h, w = imagen.shape[:2]
    centro = (w // 2, h // 2)
    angulo = random.uniform(-angulo_maximo, angulo_maximo)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    # Se usa borde blanco (255) asumiendo que el fondo de las huellas generadas es blanco
    return cv2.warpAffine(imagen, matriz_rotacion, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def aplicar_traslacion(imagen, desplazamiento_max=20):
    """Aplica una traslación aleatoria en los ejes X e Y."""
    h, w = imagen.shape[:2]
    tx = random.randint(-desplazamiento_max, desplazamiento_max)
    ty = random.randint(-desplazamiento_max, desplazamiento_max)
    matriz_traslacion = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(imagen, matriz_traslacion, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def agregar_ruido_gaussiano(imagen, media=0, desviacion_estandar=25):
    """Añade ruido gaussiano a la imagen."""
    ruido = np.random.normal(media, desviacion_estandar, imagen.shape).astype(np.float32)
    imagen_ruidosa = cv2.add(imagen.astype(np.float32), ruido)
    return np.clip(imagen_ruidosa, 0, 255).astype(np.uint8)

def agregar_ruido_sal_pimienta(imagen, probabilidad=0.02):
    """Añade ruido de sal (píxeles blancos) y pimienta (píxeles negros)."""
    imagen_ruidosa = np.copy(imagen)
    
    # Sal (Blanco)
    num_sal = np.ceil(probabilidad * imagen.size * 0.5)
    coords_sal = [np.random.randint(0, i - 1, int(num_sal)) for i in imagen.shape]
    imagen_ruidosa[tuple(coords_sal)] = 255
    
    # Pimienta (Negro)
    num_pimienta = np.ceil(probabilidad * imagen.size * 0.5)
    coords_pim = [np.random.randint(0, i - 1, int(num_pimienta)) for i in imagen.shape]
    imagen_ruidosa[tuple(coords_pim)] = 0
    
    return imagen_ruidosa

def simular_variacion_presion(imagen, tipo='fuerte'):
    """Simula variaciones de presión usando transformaciones morfológicas."""
    # Asumiendo que las crestas de la huella son líneas oscuras sobre fondo claro:
    # Una presión fuerte ensancha las crestas oscuras (Erosión de la imagen)
    # Una presión débil adelgaza las crestas oscuras (Dilatación de la imagen)
    kernel = np.ones((3, 3), np.uint8)
    if tipo == 'fuerte':
        return cv2.erode(imagen, kernel, iterations=1)
    else:
        return cv2.dilate(imagen, kernel, iterations=1)

def aplicar_corte_parcial(imagen, porcentaje_corte=0.2):
    """Simula una lectura incompleta cortando un borde de la huella."""
    h, w = imagen.shape[:2]
    imagen_cortada = np.copy(imagen)
    
    # Se decide aleatoriamente qué borde cortar (arriba, abajo, izquierda, derecha)
    borde = random.choice(['arriba', 'abajo', 'izquierda', 'derecha'])
    
    if borde == 'arriba':
        imagen_cortada[:int(h * porcentaje_corte), :] = 255
    elif borde == 'abajo':
        imagen_cortada[int(h * (1 - porcentaje_corte)):, :] = 255
    elif borde == 'izquierda':
        imagen_cortada[:, :int(w * porcentaje_corte)] = 255
    elif borde == 'derecha':
        imagen_cortada[:, int(w * (1 - porcentaje_corte)):] = 255
        
    return imagen_cortada

# === EJEMPLO DE FLUJO PARA GENERAR EL DATASET ===
if __name__ == "__main__":
    # 1. Cargar una de las identidades base generadas (en escala de grises)
    # ruta_imagen = 'identidad_1_base.png'
    # imagen_base = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    # Simulación de carga para que el código no falle si lo pruebas directamente
    imagen_base = np.ones((256, 256), dtype=np.uint8) * 255 
    
    # 2. Generar variantes combinando o aislando transformaciones
    variante_rotada = aplicar_rotacion(imagen_base)
    variante_trasladada = aplicar_traslacion(imagen_base)
    variante_gaussiana = agregar_ruido_gaussiano(imagen_base)
    variante_sp = agregar_ruido_sal_pimienta(imagen_base)
    variante_presion_fuerte = simular_variacion_presion(imagen_base, tipo='fuerte')
    variante_cortada = aplicar_corte_parcial(imagen_base)
    
    # 3. Guardar las nuevas imágenes (ejemplo)
    # cv2.imwrite('variante_rotada.png', variante_rotada)