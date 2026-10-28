import os
import sys

def renombrar_archivos():
    # Obtiene el directorio actual donde se ejecuta el script
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    nombre_script = os.path.basename(__file__)

    # Solicitar valores al usuario
    x = input("Ingresá el valor para X: ").strip()
    
    while True:
        try:
            inicio_y = int(input("Ingresá el número inicial para Y: ").strip())
            break
        except ValueError:
            print("Por favor, ingresá un número entero válido para Y.")

    # Listar solo archivos (excluyendo carpetas y el propio script)
    archivos = [
        f for f in os.listdir(directorio_actual)
        if os.path.isfile(os.path.join(directorio_actual, f)) and f != nombre_script
    ]

    # Ordenar los archivos para que la secuencia sea ordenada
    archivos.sort()

    total_archivos = len(archivos)
    if total_archivos == 0:
        print("No se encontraron archivos para renombrar en esta carpeta.")
        return

    print(f"\nSe encontraron {total_archivos} archivo(s).")
    confirmacion = input(f"¿Deseás renombrarlos con el formato '{x}-Y' empezando en Y={inicio_y}? (s/n): ").strip().lower()
    if confirmacion not in ('s', 'si', 'sí', 'y', 'yes'):
        print("Operación cancelada.")
        return

    # Paso 1: Renombrar a nombres temporales únicos para evitar colisiones
    mapeo_temporal = []
    for idx, archivo in enumerate(archivos):
        ext = os.path.splitext(archivo)[1]
        ruta_orig = os.path.join(directorio_actual, archivo)
        temp_name = f"__temp_rename_{idx}__{ext}"
        ruta_temp = os.path.join(directorio_actual, temp_name)
        os.rename(ruta_orig, ruta_temp)
        mapeo_temporal.append((ruta_temp, ext))

    # Paso 2: Asignar el nombre final X-Y.extension
    y_actual = inicio_y
    for ruta_temp, ext in mapeo_temporal:
        nombre_final = f"{x}-{y_actual}{ext}"
        ruta_final = os.path.join(directorio_actual, nombre_final)
        
        os.rename(ruta_temp, ruta_final)
        print(f"Renombrado a -> {nombre_final}")
        y_actual += 1

    print("\n¡Listo! Todos los archivos fueron renombrados exitosamente.")

if __name__ == "__main__":
    renombrar_archivos()