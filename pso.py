"""
Algoritmo PSO y línea base (búsqueda aleatoria).
Grupo 3 — Lucas B. y Emanuel B.

Responsable sugerido: Persona B (algoritmo).
Este módulo no depende del modelo de riego: sirve para cualquier f(x).
"""

import numpy as np


def pso(f, lb, ub, n_particulas=30, n_iter=200, w_ini=0.9, w_fin=0.4,
        c1=1.5, c2=1.5, semilla=None, tol_estancamiento=None):
    """Particle Swarm Optimization para minimizar f en el rango [lb, ub].

    Actualización:
        v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        x = x + v
    La velocidad se limita al 20% del rango y la posición se recorta al borde.

    Devuelve: (mejor_posicion, mejor_valor, historial_gbest, historial_posiciones)
    """
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
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


def busqueda_aleatoria(f, lb, ub, n_eval, semilla=None):
    """Línea base: muestrea n_eval puntos uniformes y se queda con el mejor.

    Devuelve: (mejor_posicion, mejor_valor, historial_mejor_por_evaluacion)
    """
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
