import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función base de descenso gradiente
def gradiente(f, df, x0, metodo='naive', alpha=0.01, max_iter=2000, tol=1e-6, ddf=None, funcion_nombre=''):
    xk = x0
    xk_hist = [xk]
    fk_hist = [f(xk)]
    error_hist = []
    norm_grad_hist = []
    convergencia = False
    
    for k in range(max_iter):
        grad = df(xk)
        norm_grad = np.linalg.norm(grad)
        norm_grad_hist.append(norm_grad)
        xk = np.clip(xk, -1e2, 1e2)
        
        if metodo == 'naive':
            xk_new = xk - alpha * grad
        elif metodo == 'aleatorio':
            random_dir = np.random.randn(*xk.shape)
            random_dir = random_dir / np.linalg.norm(random_dir)
            alpha_decay = alpha / (1 + 0.01 * k)
            xk_new = xk - alpha_decay * random_dir * np.linalg.norm(grad)
        elif metodo == 'newton_aprox':
            hess_aprox = np.identity(len(xk))
            xk_new = xk - alpha * np.linalg.inv(hess_aprox) @ grad
        elif metodo == 'newton_exact' and ddf is not None:
            hess = ddf(xk)
            xk_new = xk - alpha * np.linalg.inv(hess) @ grad
        elif metodo == 'newton_exact' and ddf is None:
            raise ValueError("Hessiano no proporcionado para el método Newton exacto")
        else:
            raise ValueError("Método desconocido o Hessiano no proporcionado")
        
        xk_hist.append(xk_new)
        fk_hist.append(f(xk_new))
        
        error = np.linalg.norm(xk_new - xk)
        error_hist.append(error)
        
        if error < tol:
            convergencia = True
            print(f"Convergencia alcanzada con el método {metodo} para la función {funcion_nombre} en {k+1} iteraciones")
            break
        
        xk = xk_new
    
    return {
        "xk_hist": xk_hist,
        "fk_hist": fk_hist,
        "error_hist": error_hist,
        "norm_grad_hist": norm_grad_hist,
        "iteraciones": k+1,
        "convergencia": convergencia,
        "solucion": xk_hist[-1]
    }

# Función f(x, y) = x^4 + y^4 - 4xy + (1/2)y + 1
def f1(x):
    return x[0]**4 + x[1]**4 - 4*x[0]*x[1] + 0.5*x[1] + 1

def df1(x):
    dfdx = 4*x[0]**3 - 4*x[1]
    dfdy = 4*x[1]**3 - 4*x[0] + 0.5
    return np.array([dfdx, dfdy])

def ddf1(x):
    h11 = 12*x[0]**2
    h12 = -4
    h21 = -4
    h22 = 12*x[1]**2
    return np.array([[h11, h12], [h21, h22]])

# Función de Rosembrock en 2D
def f2(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def df2(x):
    dfdx1 = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
    dfdx2 = 200*(x[1] - x[0]**2)
    return np.array([dfdx1, dfdx2])

# Función de Rosembrock en 10D
def f10(x):
    suma = 0
    for i in range(9):
        x = np.clip(x, -1e2, 1e2)
        suma += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return suma

def df10(x):
    grad = np.zeros(10)
    for i in range(9):
        x = np.clip(x, -1e2, 1e2)
        grad[i] = -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
        grad[i+1] = 200*(x[i+1] - x[i]**2)
    return grad

# Función para mostrar las primeras y últimas 3 iteraciones y guardar en txt
def guardar_tablas(resultados, metodo, archivo_txt):
    xk_hist = resultados["xk_hist"]
    fk_hist = resultados["fk_hist"]
    error_hist = resultados["error_hist"]
    norm_grad_hist = resultados["norm_grad_hist"]
    
    iteraciones = len(xk_hist)
    
    primeras_ultimas = pd.DataFrame({
        "Iteración": list(range(1, 4)) + list(range(iteraciones-2, iteraciones+1)),
        "xk": xk_hist[:3] + xk_hist[-3:],
        "f(xk)": fk_hist[:3] + fk_hist[-3:],
        "Error": error_hist[:3] + error_hist[-3:],
        "Norma del gradiente": norm_grad_hist[:3] + norm_grad_hist[-3:]
    })
    
    # Guardar tabla en el archivo txt
    with open(archivo_txt, "a") as file:
        file.write(f"\nTabla de aproximaciones para el método {metodo}:\n")
        file.write(primeras_ultimas.to_string())
        file.write("\n\n")

# Función para graficar las convergencias y guardarlas
def plot_convergencias_metodo(resultados_f1, resultados_2d, resultados_10d, metodo, archivo_img):
    plt.figure(figsize=(8, 6))
    
    plt.plot(resultados_f1["error_hist"], label='f1')
    plt.plot(resultados_2d["error_hist"], label='Rosembrock 2D')
    plt.plot(resultados_10d["error_hist"], label='Rosembrock 10D')
    
    plt.title(f"Convergencia - Método {metodo}")
    plt.xlabel("Iteraciones")
    plt.ylabel("Error de aproximación")
    plt.legend()
    
    # Guardar gráfico en un archivo de imagen
    plt.savefig(archivo_img)
    plt.close()

# Función para graficar los puntos durante la convergencia
def plot_puntos_convergencia(resultados, funcion_nombre, archivo_img):
    xk_hist = np.array(resultados["xk_hist"])
    
    # Solo graficar si es un problema en 2 dimensiones
    if xk_hist.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        
        # Graficar los puntos tomados por el método
        plt.plot(xk_hist[:, 0], xk_hist[:, 1], marker='o', color='b', label=f"Trayectoria de {funcion_nombre}")
        
        plt.title(f"Puntos tomados durante la convergencia - {funcion_nombre}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        
        # Guardar gráfico en un archivo de imagen
        plt.savefig(archivo_img)
        plt.close()

# Parámetros iniciales
x0_f1 = np.array([-3.0, 1.0])  # Punto inicial para f1
x0_2d = np.array([-1.2, 1.0])  # Punto inicial para funciones 2D
x0_10d = np.array([-1.2] + [1.0] * 9)  # Punto inicial para función 10D

# Funciones objetivo (f1, Rosembrock 2D, Rosembrock 10D)
# Parámetros iniciales
x0_f1 = np.array([-3.0, 1.0])  # Punto inicial para f1
x0_2d = np.array([-1.2, 1.0])  # Punto inicial para funciones 2D
x0_10d = np.array([-1.2] + [1.0] * 9)  # Punto inicial para función 10D

# Archivo donde se guardarán las tablas
archivo_txt = "tablas_resultados.txt"

# Pruebas para f1, Rosembrock 2D y 10D
# Descenso gradiente naive
resultado_naive_f1 = gradiente(f1, df1, x0_f1, metodo='naive', alpha=0.001, funcion_nombre="f1")
resultado_naive_2d = gradiente(f2, df2, x0_2d, metodo='naive', alpha=0.001, funcion_nombre="Rosembrock 2D")
resultado_naive_10d = gradiente(f10, df10, x0_10d, metodo='naive', alpha=0.001, funcion_nombre="Rosembrock 10D")

# Descenso con dirección aleatoria
resultado_aleatorio_f1 = gradiente(f1, df1, x0_f1, metodo='aleatorio', alpha=0.001, funcion_nombre="f1")
resultado_aleatorio_2d = gradiente(f2, df2, x0_2d, metodo='aleatorio', alpha=0.001, funcion_nombre="Rosembrock 2D")
resultado_aleatorio_10d = gradiente(f10, df10, x0_10d, metodo='aleatorio', alpha=0.001, funcion_nombre="Rosembrock 10D")

# Descenso de Newton aproximado
resultado_newton_aprox_f1 = gradiente(f1, df1, x0_f1, metodo='newton_aprox', alpha=0.001, funcion_nombre="f1")
resultado_newton_aprox_2d = gradiente(f2, df2, x0_2d, metodo='newton_aprox', alpha=0.001, funcion_nombre="Rosembrock 2D")
resultado_newton_aprox_10d = gradiente(f10, df10, x0_10d, metodo='newton_aprox', alpha=0.001, funcion_nombre="Rosembrock 10D")

# Descenso de Newton exacto
resultado_newton_exacto_f1 = gradiente(f1, df1, x0_f1, metodo='newton_exact', alpha=0.001, ddf=ddf1, funcion_nombre="f1")
# No se usa Newton exacto para Rosembrock 2D y 10D

# Guardar las tablas en un archivo txt
guardar_tablas(resultado_naive_f1, "Naive f1", archivo_txt)
guardar_tablas(resultado_naive_2d, "Naive Rosembrock 2D", archivo_txt)
guardar_tablas(resultado_naive_10d, "Naive Rosembrock 10D", archivo_txt)

guardar_tablas(resultado_aleatorio_f1, "Aleatorio f1", archivo_txt)
guardar_tablas(resultado_aleatorio_2d, "Aleatorio Rosembrock 2D", archivo_txt)
guardar_tablas(resultado_aleatorio_10d, "Aleatorio Rosembrock 10D", archivo_txt)

guardar_tablas(resultado_newton_aprox_f1, "Newton Aproximado f1", archivo_txt)
guardar_tablas(resultado_newton_aprox_2d, "Newton Aproximado Rosembrock 2D", archivo_txt)
guardar_tablas(resultado_newton_aprox_10d, "Newton Aproximado Rosembrock 10D", archivo_txt)

guardar_tablas(resultado_newton_exacto_f1, "Newton Exacto f1", archivo_txt)

# Guardar las gráficas de convergencia
plot_convergencias_metodo(resultado_naive_f1, resultado_naive_2d, resultado_naive_10d, "Naive", "convergencia_naive.png")
plot_convergencias_metodo(resultado_aleatorio_f1, resultado_aleatorio_2d, resultado_aleatorio_10d, "Aleatorio", "convergencia_aleatorio.png")
plot_convergencias_metodo(resultado_newton_aprox_f1, resultado_newton_aprox_2d, resultado_newton_aprox_10d, "Newton Aproximado", "convergencia_newton_aprox.png")
plot_convergencias_metodo(resultado_newton_exacto_f1, resultado_naive_2d, resultado_naive_10d, "Newton Exacto", "convergencia_newton_exacto.png")

plot_puntos_convergencia(resultado_naive_f1, "f1", "trayectoria_f1.png")
plot_puntos_convergencia(resultado_naive_2d, "Rosembrock 2D", "trayectoria_rosem2d.png")
