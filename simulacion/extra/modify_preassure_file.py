import datetime
import numpy as np

def procesar_archivo(archivo_entrada, archivo_salida):
    datos_por_intervalo = {}
    contador = 0  # Contador inicializado en 0

    # Leer y procesar el archivo de entrada
    with open(archivo_entrada, 'r') as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue

            if 'T' in linea:  # Detecta líneas con timestamp
                timestamp = linea
            else:  # Detecta líneas con valores
                valores = list(map(float, linea.split(',')))

                # Intentar convertir el timestamp a segundos desde epoch
                try:
                    tiempo = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f")
                except ValueError:
                    tiempo = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")

                tiempo_seg = (tiempo - datetime.datetime(1970, 1, 1)).total_seconds()

                # Agrupar lecturas por intervalos de 0.5 segundos
                intervalo = int(tiempo_seg // 0.5)
                if intervalo not in datos_por_intervalo:
                    datos_por_intervalo[intervalo] = []

                # Añadir los valores al intervalo correspondiente
                datos_por_intervalo[intervalo].append(valores)

    # Calcular promedios y escribir el archivo de salida
    with open(archivo_salida, 'w') as f:
        for intervalo, lecturas in sorted(datos_por_intervalo.items()):
            # Calcular el promedio de los valores
            lecturas_array = np.array(lecturas)
            promedios = np.mean(lecturas_array, axis=0)
            promedios_redondeados = [round(v, 3) for v in promedios]

            # Escribir en el archivo de salida: primer valor es el contador
            linea_salida = f"{contador}, " + ", ".join(map(str, promedios_redondeados))
            f.write(linea_salida + "\n")

            # Incrementar el contador por 500 después de cada intervalo
            contador += 500

# Ejecución del programa
archivo_entrada = "3_acortado.txt"  # Cambiar por el nombre de tu archivo de entrada
archivo_salida = "3_modificado.txt"  # Cambiar por el nombre de tu archivo de salida
procesar_archivo(archivo_entrada, archivo_salida)
