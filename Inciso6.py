import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
import seaborn as sns

# Definir la función de suma de gaussianas
def gaussian_sum(x, points, sigma=0.5):
    k = len(points)
    result = 0
    for i in range(k):
        dist_sq = np.linalg.norm(x - points[i]) ** 2
        result -= np.exp(-dist_sq / (2 * sigma ** 2))
    return result

# Generar puntos aleatorios en el rectángulo [0,8]x[0,8]
np.random.seed(42)  # Para reproducibilidad
k = 8
points = np.random.uniform(0, 8, size=(k, 2))  # k puntos aleatorios

# Crear una malla de puntos para visualizar la función
x_vals = np.linspace(0, 8, 400)
y_vals = np.linspace(0, 8, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.array([[gaussian_sum(np.array([x, y]), points) for x in x_vals] for y in y_vals])

# Visualizar el mapa de contorno de la función
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=20)
plt.scatter(points[:, 0], points[:, 1], color='green', label='Gaussian Centers')
plt.colorbar()
plt.title("Suma de Gaussianas 2D")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Usar scipy.optimize para minimizar la función desde diferentes inicializaciones
def minimize_from_start(x0):
    res = minimize(gaussian_sum, x0, args=(points,), method='BFGS')
    return res.x, res.fun

# Realizar minimización desde múltiples puntos iniciales
initial_points = np.random.uniform(0, 8, size=(5, 2))  # 5 puntos iniciales
results = [minimize_from_start(p) for p in initial_points]

# Mostrar resultados de optimización
for idx, (min_x, min_val) in enumerate(results):
    print(f"Inicialización {idx+1}: Mínimo encontrado en {min_x} con valor {min_val}")

# Visualizar los puntos de convergencia
# Usar un colormap para asignar diferentes colores a los mínimos
colors = cm.rainbow(np.linspace(0, 1, len(results)))

# Visualizar los puntos de convergencia con colores distintos
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=20)
plt.scatter(points[:, 0], points[:, 1], color='green', label='Gaussian Centers')
for idx, (min_x, _) in enumerate(results):
    plt.scatter(min_x[0], min_x[1], color=colors[idx], label=f'Mínimo {idx+1}')
plt.colorbar()
plt.title("Convergencia hacia los mínimos locales")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
