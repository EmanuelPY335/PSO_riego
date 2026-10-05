"""
Diseño óptimo de un sistema de riego con PSO (Particle Swarm Optimization)
Grupo 3 — Lucas B. y Emanuel B.

Esqueleto del proyecto. Todo lo marcado con TODO es lo que tienen que ajustar
o justificar en el informe. Requiere: numpy, matplotlib.

Estructura:
    1. Parámetros del problema (modelo de riego)
    2. Función objetivo y restricciones
    3. Implementación del PSO
    4. Validación con función de prueba (esfera)
    5. Línea base (búsqueda aleatoria)
    6. Experimentos (varias semillas, distintos parámetros)
    7. Gráficos y tabla de resultados
    8. main
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. PARÁMETROS DEL PROBLEMA
# ============================================================
# TODO: definan el cultivo/parcela y justifiquen cada valor en el informe.

AREA_M2 = 1000.0          # superficie regada (m²)
ET_MM = 5.0               # requerimiento hídrico del cultivo (mm por riego)
REQ_L = AREA_M2 * ET_MM   # litros necesarios por riego (1 mm sobre 1 m² = 1 L)

# Límites de las variables de decisión [Q, P, t]
NOMBRES = ["Caudal Q (L/min)", "Presión P (bar)", "Tiempo t (min)"]
LB = np.array([10.0, 1.0, 10.0])
UB = np.array([100.0, 4.0, 120.0])

# Pesos de penalización
W_DEFICIT = 5.0           # costo por litro de déficit (el cultivo no recibe lo necesario)
W_RESTR = 1e4             # penalización por violar restricciones


# ============================================================
# 2. FUNCIÓN OBJETIVO Y RESTRICCIONES
# ============================================================
def perdida_fraccion(P):
    """Fracción del agua que se pierde (evaporación, deriva, escorrentía).
    TODO: modelo simplificado: mínimo en una presión óptima (2 bar),
    crece cuadráticamente si la presión se aleja. Reemplazar/justificar."""
    return 0.05 + 0.03 * (P - 2.0) ** 2


def restricciones(x):
    """Devuelve la suma de violaciones (0 si cumple todo).
    TODO: agregar o quitar restricciones según su modelo."""
    Q, P, t = x
    viol = 0.0
    # Ejemplo: el caudal máximo que entrega el sistema depende de la presión
    q_max = 50.0 * np.sqrt(P)
    viol += max(0.0, Q - q_max)
    # Ejemplo: límite de consumo total por riego (litros)
    viol += max(0.0, Q * t - 1.5 * REQ_L)
    return viol


def eficiencia(x):
    """Eficiencia hídrica = agua aprovechada / agua aplicada (para reportar)."""
    Q, P, t = x
    aplicada = Q * t
    util = aplicada * (1.0 - perdida_fraccion(P))
    return min(util, REQ_L) / aplicada


def funcion_objetivo(x):
    """A MINIMIZAR: agua desperdiciada + penalización por déficit + restricciones."""
    Q, P, t = x
    aplicada = Q * t                                   # litros aplicados
    util = aplicada * (1.0 - perdida_fraccion(P))      # litros que llegan al cultivo
    desperdicio = aplicada - min(util, REQ_L)          # pérdidas + exceso
    deficit = max(0.0, REQ_L - util)
    return desperdicio + W_DEFICIT * deficit + W_RESTR * restricciones(x)


# ============================================================
# 3. PSO
# ============================================================
def pso(f, lb, ub, n_particulas=30, n_iter=200, w_ini=0.9, w_fin=0.4,
        c1=1.5, c2=1.5, semilla=None, tol_estancamiento=None):
    """Particle Swarm Optimization para minimizar f en el rango [lb, ub].

    Devuelve: (mejor_posicion, mejor_valor, historial_gbest, historial_posiciones)
    """
    rng = np.random.default_rng(semilla)
    dim = len(lb)

    # Inicialización
    x = rng.uniform(lb, ub, size=(n_particulas, dim))
    v_max = 0.2 * (ub - lb)
    v = rng.uniform(-v_max, v_max, size=(n_particulas, dim))

    fx = np.array([f(p) for p in x])
    pbest, pbest_val = x.copy(), fx.copy()
    g_idx = np.argmin(pbest_val)
    gbest, gbest_val = pbest[g_idx].copy(), pbest_val[g_idx]

    hist_gbest = [gbest_val]
    hist_pos = [x.copy()]
    sin_mejora = 0

    for it in range(n_iter):
        w = w_ini - (w_ini - w_fin) * it / max(1, n_iter - 1)   # inercia decreciente
        r1 = rng.random((n_particulas, dim))
        r2 = rng.random((n_particulas, dim))

        # Actualización de velocidad y posición
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = np.clip(v, -v_max, v_max)
        x = np.clip(x + v, lb, ub)          # recorte al borde del rango

        # Evaluación y actualización de pbest / gbest
        fx = np.array([f(p) for p in x])
        mejora = fx < pbest_val
        pbest[mejora] = x[mejora]
        pbest_val[mejora] = fx[mejora]

        g_idx = np.argmin(pbest_val)
        if pbest_val[g_idx] < gbest_val - 1e-12:
            gbest, gbest_val = pbest[g_idx].copy(), pbest_val[g_idx]
            sin_mejora = 0
        else:
            sin_mejora += 1

        hist_gbest.append(gbest_val)
        hist_pos.append(x.copy())

        # Criterio de parada opcional por estancamiento
        if tol_estancamiento and sin_mejora >= tol_estancamiento:
            break

    return gbest, gbest_val, np.array(hist_gbest), hist_pos


# ============================================================
# 4. VALIDACIÓN: función esfera (mínimo en 0 en el origen)
# ============================================================
def validar_con_esfera():
    esfera = lambda x: float(np.sum(x ** 2))
    lb, ub = -5 * np.ones(3), 5 * np.ones(3)
    _, val, _, _ = pso(esfera, lb, ub, semilla=0)
    print(f"[Validación] Esfera: mejor valor = {val:.2e} (debería ser ~0)")


# ============================================================
# 5. LÍNEA BASE: búsqueda aleatoria (mismo nº de evaluaciones)
# ============================================================
def busqueda_aleatoria(f, lb, ub, n_eval, semilla=None):
    rng = np.random.default_rng(semilla)
    mejor_val, mejor_x = np.inf, None
    hist = []
    for _ in range(n_eval):
        x = rng.uniform(lb, ub)
        val = f(x)
        if val < mejor_val:
            mejor_val, mejor_x = val, x
        hist.append(mejor_val)
    return mejor_x, mejor_val, np.array(hist)


# ============================================================
# 6. EXPERIMENTOS
# ============================================================
def experimento_semillas(n_corridas=30, **kwargs):
    """Corre PSO varias veces con distintas semillas."""
    resultados = []
    for s in range(n_corridas):
        x, val, hist, _ = pso(funcion_objetivo, LB, UB, semilla=s, **kwargs)
        resultados.append((x, val, hist))
    return resultados


def experimento_parametros():
    """Compara configuraciones de parámetros. TODO: ampliar la grilla."""
    configs = {
        "w 0.9→0.4, c=1.5": dict(w_ini=0.9, w_fin=0.4, c1=1.5, c2=1.5),
        "w 0.7 fijo, c=2.0": dict(w_ini=0.7, w_fin=0.7, c1=2.0, c2=2.0),
        "10 partículas": dict(n_particulas=10),
        "50 partículas": dict(n_particulas=50),
    }
    salida = {}
    for nombre, cfg in configs.items():
        res = experimento_semillas(n_corridas=20, **cfg)
        vals = np.array([r[1] for r in res])
        salida[nombre] = (vals.mean(), vals.std(), np.mean([r[2] for r in res], axis=0))
    return salida


# ============================================================
# 7. GRÁFICOS Y RESULTADOS
# ============================================================
def graficar_convergencia(res_pso, hist_base):
    plt.figure(figsize=(7, 4))
    plt.plot(np.mean([r[2] for r in res_pso], axis=0), label="PSO (media de corridas)")
    plt.plot(hist_base, label="Búsqueda aleatoria", linestyle="--")
    plt.yscale("log")
    plt.xlabel("Iteración / evaluaciones")
    plt.ylabel("Mejor valor de la función objetivo")
    plt.title("Curva de convergencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergencia.png", dpi=150)
    plt.close()


def graficar_particulas(hist_pos, iteraciones=(0, 10, 50, -1)):
    """Proyección 2D (Caudal vs Tiempo) del enjambre en distintas iteraciones."""
    fig, axs = plt.subplots(1, len(iteraciones), figsize=(14, 3.5), sharey=True)
    for ax, it in zip(axs, iteraciones):
        pos = hist_pos[it]
        ax.scatter(pos[:, 0], pos[:, 2], s=15)
        ax.set_title(f"Iteración {it if it >= 0 else len(hist_pos) - 1}")
        ax.set_xlabel("Caudal (L/min)")
    axs[0].set_ylabel("Tiempo (min)")
    plt.tight_layout()
    plt.savefig("particulas.png", dpi=150)
    plt.close()


def graficar_comparacion_parametros(salida):
    plt.figure(figsize=(7, 4))
    for nombre, (_, _, curva) in salida.items():
        plt.plot(curva, label=nombre)
    plt.yscale("log")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor valor (media)")
    plt.title("Efecto de los parámetros en la convergencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig("parametros.png", dpi=150)
    plt.close()


# ============================================================
# 8. MAIN
# ============================================================
def main():
    N_PART, N_ITER = 30, 200

    # 4. Validar la implementación
    validar_con_esfera()

    # 6. PSO con varias semillas
    res = experimento_semillas(n_corridas=30, n_particulas=N_PART, n_iter=N_ITER)
    vals = np.array([r[1] for r in res])
    mejor = res[int(np.argmin(vals))]
    print(f"\n[PSO] Valor objetivo: media = {vals.mean():.2f}, desv = {vals.std():.2f}")

    x_opt = mejor[0]
    print("\nMejor configuración encontrada:")
    for nombre, valor in zip(NOMBRES, x_opt):
        print(f"  {nombre}: {valor:.2f}")
    print(f"  Eficiencia hídrica: {eficiencia(x_opt) * 100:.1f}%")
    print(f"  Violación de restricciones: {restricciones(x_opt):.2f}")

    # 5. Línea base con el mismo número de evaluaciones
    n_eval = N_PART * (N_ITER + 1)
    x_b, val_b, hist_b = busqueda_aleatoria(funcion_objetivo, LB, UB, n_eval, semilla=0)
    print(f"\n[Búsqueda aleatoria] mejor valor = {val_b:.2f}, eficiencia = {eficiencia(x_b) * 100:.1f}%")

    # Configuración fija "típica" (TODO: definir una razonable)
    x_fija = np.array([50.0, 2.0, 60.0])
    print(f"[Configuración fija] valor = {funcion_objetivo(x_fija):.2f}, eficiencia = {eficiencia(x_fija) * 100:.1f}%")

    # 7. Gráficos
    graficar_convergencia(res, hist_b[::N_PART])
    _, _, _, hist_pos = pso(funcion_objetivo, LB, UB, n_particulas=N_PART, n_iter=N_ITER, semilla=1)
    graficar_particulas(hist_pos)
    graficar_comparacion_parametros(experimento_parametros())
    print("\nGráficos guardados: convergencia.png, particulas.png, parametros.png")


if __name__ == "__main__":
    main()
