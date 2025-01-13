import matplotlib.pyplot as plt
import pandas as pd

# Función para leer los datos desde un archivo CSV
def leer_datos(nombre_archivo):
    return pd.read_csv(nombre_archivo, header=None)

# Leer los archivos
velocidades = leer_datos("archivo_velocidades.txt")
presiones = leer_datos("archivo_presiones.txt")

# Separar el tiempo y los valores
# El tiempo es la primera columna
# Los valores de los ventiladores son el resto de las columnas
tiempo_vel = velocidades.iloc[:, 0]
valores_vel = velocidades.iloc[:, 1:]

tiempo_pres = presiones.iloc[:, 0]
valores_pres = presiones.iloc[:, 1:]

# Graficar las velocidades
plt.figure(figsize=(12, 6))
for col in valores_vel.columns:
    plt.plot(tiempo_vel, valores_vel[col])
plt.title("Fan speed over time")
plt.ylabel("Speed (%)")
plt.xlim(0, 40000)  # Establecer el límite superior del eje X
plt.grid()
plt.tight_layout()
plt.savefig("grafico_velocidades.png")

# Graficar las presiones
plt.figure(figsize=(12, 6))
for col in valores_pres.columns:
    plt.plot(tiempo_pres, valores_pres[col])
plt.title("Fan pressures over time")
plt.xlabel("Time (ms)")
plt.ylabel("Preassure (Pa)")
plt.xlim(0, 40000)  # Establecer el límite superior del eje X
plt.grid()
plt.tight_layout()
plt.savefig("grafico_presiones.png")
