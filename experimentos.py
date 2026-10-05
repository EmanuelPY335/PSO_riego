"""
Experimentos: validación, corridas con varias semillas, línea base y
comparación de parámetros.
Grupo 3 — Lucas B. y Emanuel B.
"""

import numpy as np

from modelo import funcion_objetivo, LB, UB
from pso import pso, busqueda_aleatoria


def evaluaciones_pso(n_particulas, n_iter):
    """Cantidad de evaluaciones de f que hace una corrida completa del PSO."""
    return n_particulas * (n_iter + 1)


# ============================================================
# Validación con funciones de prueba (mínimo conocido = 0)
# ============================================================
def esfera(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    """Multimodal: sirve para ver que el PSO no se queda en el primer mínimo local."""
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def validar(n_corridas=10):
    """Corre el PSO sobre funciones de prueba. La esfera debe dar ~0."""
    salida = {}
    for nombre, f, lim in [("Esfera", esfera, 5.0), ("Rastrigin", rastrigin, 5.12)]:
        lb, ub = -lim * np.ones(3), lim * np.ones(3)
        vals = np.array([pso(f, lb, ub, semilla=s)[1] for s in range(n_corridas)])
        salida[nombre] = (vals.mean(), vals.std(), vals.min())
    return salida


# ============================================================
# PSO con varias semillas
# ============================================================
def experimento_semillas(n_corridas=30, **kwargs):
    """Corre el PSO sobre el modelo de riego con semillas 0..n_corridas-1.
    Devuelve una lista de (mejor_x, mejor_valor, historial_gbest)."""
    resultados = []
    for s in range(n_corridas):
        x, val, hist, _ = pso(funcion_objetivo, LB, UB, semilla=s, **kwargs)
        resultados.append((x, val, hist))
    return resultados


def experimento_aleatoria(n_eval, n_corridas=30):
    """Búsqueda aleatoria con el mismo presupuesto de evaluaciones y las mismas semillas."""
    return [busqueda_aleatoria(funcion_objetivo, LB, UB, n_eval, semilla=s)
            for s in range(n_corridas)]


# ============================================================
# Efecto de los parámetros
# ============================================================
CONFIGS_PARAMETROS = {
    # Inercia
    "w 0.9→0.4 (base)":   dict(),
    "w 0.7 fijo":         dict(w_ini=0.7, w_fin=0.7),
    "w 0.4 fijo":         dict(w_ini=0.4, w_fin=0.4),
    # Coeficientes cognitivo / social
    "c1=c2=2.0":          dict(c1=2.0, c2=2.0),
    "c1=2.5, c2=0.5":     dict(c1=2.5, c2=0.5),   # más individualista
    "c1=0.5, c2=2.5":     dict(c1=0.5, c2=2.5),   # más social
    # Tamaño del enjambre
    "10 partículas":      dict(n_particulas=10),
    "50 partículas":      dict(n_particulas=50),
}
# TODO: ampliar/ajustar la grilla. Ojo: al cambiar n_particulas cambia el
#       número de evaluaciones; para una comparación justa se puede igualar
#       el presupuesto ajustando n_iter.


def _rellenar(curva, largo):
    """Extiende una curva con su último valor (por si una corrida paró antes)."""
    return np.concatenate([curva, np.full(largo - len(curva), curva[-1])])


def experimento_parametros(n_corridas=20, configs=CONFIGS_PARAMETROS, **base):
    """Para cada configuración devuelve (media, desv, curva_media_de_convergencia)."""
    salida = {}
    for nombre, cfg in configs.items():
        res = experimento_semillas(n_corridas=n_corridas, **{**base, **cfg})
        vals = np.array([r[1] for r in res])
        largo = max(len(r[2]) for r in res)
        curva = np.mean([_rellenar(r[2], largo) for r in res], axis=0)
        salida[nombre] = (vals.mean(), vals.std(), curva)
    return salida
