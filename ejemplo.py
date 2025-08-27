import os
import pandas as pd
import re
from collections import defaultdict

# Ruta base donde están las carpetas de sensores
base_dir = "/Users/cefuentes/Downloads/Datos/Smartphone_Dataset"  # <- reemplaza esto con tu ruta real

# Diccionario: {sensor: {actividad: total_muestras}}
resultados = defaultdict(lambda: defaultdict(int))

# Función para extraer el nombre base de la actividad (sin número)
def extraer_nombre_actividad(nombre_archivo):
    nombre = nombre_archivo.lower().replace(".csv", "")
    # Ej: running1, running_2 => running
    match = re.match(r'([a-z_]+)', nombre)
    return match.group(1) if match else nombre

# Recorrer sensores (carpetas)
for sensor in os.listdir(base_dir):
    sensor_path = os.path.join(base_dir, sensor)

    if os.path.isdir(sensor_path):
        for archivo in os.listdir(sensor_path):
            if archivo.endswith(".csv"):
                actividad = extraer_nombre_actividad(archivo)
                archivo_path = os.path.join(sensor_path, archivo)

                try:
                    df = pd.read_csv(archivo_path)
                    num_muestras = len(df)
                    resultados[sensor][actividad] += num_muestras
                except Exception as e:
                    print(f"Error leyendo {archivo_path}: {e}")

# Convertimos a DataFrame
df_resultado = pd.DataFrame(resultados).fillna(0).astype(int).sort_index()
print(df_resultado)

# Guardar como CSV (opcional)
df_resultado.to_csv("resumen_por_sensor_y_actividad.csv")
