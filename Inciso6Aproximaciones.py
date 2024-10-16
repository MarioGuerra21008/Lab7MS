import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

# Función para almacenar las aproximaciones intermedias
def minimize_with_trajectory(x0):
    trajectory = []  # Lista para almacenar los puntos intermedios
    
    def callback(xk):
        trajectory.append(xk.copy())  # Guardar cada aproximación como un vector 2D
    
    # Usamos 'Powell' para generar más iteraciones, y aumentar maxiter
    res = minimize(gaussian_sum, x0, args=(points,), method='BFGS', callback=callback, options={'maxiter': 100})
    return res.x, res.fun, trajectory

# Realizar minimización desde múltiples puntos iniciales
initial_points = np.random.uniform(0, 8, size=(5, 2))  # 5 puntos iniciales
results = [minimize_with_trajectory(p) for p in initial_points]

# Visualización de la función con las trayectorias de convergencia
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=20)
plt.scatter(points[:, 0], points[:, 1], color='green', label='Gaussian Centers')

# Usar una paleta de colores para las trayectorias
colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))

# Graficar las trayectorias de convergencia
for idx, (min_x, _, trajectory) in enumerate(results):
    trajectory = np.array(trajectory)  # Convertir la trayectoria en una matriz numpy
    if trajectory.ndim == 2:  # Asegurarse de que sea bidimensional
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color=colors[idx], label=f'Trayectoria {idx+1}')
    plt.scatter(min_x[0], min_x[1], color=colors[idx], s=100, label=f'Mínimo {idx+1}')
    
plt.colorbar()
plt.title("Secuencias de aproximaciones convergiendo a los mínimos locales")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
