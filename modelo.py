"""
Modelo del problema de riego: parámetros, restricciones y función objetivo.
Grupo 3 — Lucas B. y Emanuel B.

Responsable sugerido: Persona A (modelo).
Todo lo marcado con TODO hay que ajustarlo o justificarlo en el informe.
"""

import numpy as np

# ============================================================
# 1. PARÁMETROS DEL PROBLEMA
# ============================================================
# TODO: definan el cultivo/parcela reales y justifiquen cada valor en el informe
#       (p. ej. ET del cultivo según FAO-56 para la zona y época elegidas).

AREA_M2 = 1000.0          # superficie regada (m²)
ET_MM = 5.0               # requerimiento hídrico del cultivo (mm por riego)
REQ_L = AREA_M2 * ET_MM   # litros necesarios por riego (1 mm sobre 1 m² = 1 L)

# Variables de decisión x = [Q, P, t]
NOMBRES = ["Caudal Q (L/min)", "Presión P (bar)", "Tiempo t (min)"]
NOMBRES_CORTOS = ["Q_L_min", "P_bar", "t_min"]
LB = np.array([10.0, 1.0, 10.0])
UB = np.array([100.0, 4.0, 120.0])

# Pesos de penalización
# TODO: revisar escalas. W_RESTR muy bajo -> el PSO ignora las restricciones;
#       muy alto -> el paisaje se vuelve abrupto y puede estancarse.
W_DEFICIT = 5.0           # costo por litro de déficit (el cultivo no recibe lo necesario)
W_RESTR = 1e4             # penalización por unidad de violación de restricciones

# Restricciones (parámetros)
K_CAUDAL = 50.0           # Q_max = K_CAUDAL * sqrt(P)  (curva tipo emisor/bomba)
FACTOR_CONSUMO_MAX = 1.5  # consumo máximo por riego = FACTOR * REQ_L

# Configuración fija "típica" para comparar (línea base del regante)
# TODO: justificar. Supuesto actual: el regante trabaja a 3 bar (presión nominal
#       común en aspersores) y riega "de más" por seguridad (~20% extra).
CONFIG_FIJA = np.array([60.0, 3.0, 100.0])


# ============================================================
# 2. FUNCIÓN OBJETIVO Y RESTRICCIONES
# ============================================================
def perdida_fraccion(P):
    """Fracción del agua que se pierde (evaporación, deriva, escorrentía).

    TODO: modelo simplificado e inventado: mínimo (5%) en una presión óptima
    de 2 bar y crece cuadráticamente al alejarse. Reemplazar por un modelo
    de bibliografía o justificar (baja presión -> mala uniformidad;
    alta presión -> gotas finas, más deriva y evaporación).
    """
    return 0.05 + 0.03 * (P - 2.0) ** 2


def volumenes(x):
    """Devuelve (aplicada, util) en litros para la configuración x = [Q, P, t]."""
    Q, P, t = x
    aplicada = Q * t                                   # litros aplicados
    util = aplicada * (1.0 - perdida_fraccion(P))      # litros que llegan al cultivo
    return aplicada, util


def restricciones(x):
    """Suma de violaciones (0 si cumple todo).
    TODO: agregar o quitar restricciones según su modelo."""
    Q, P, t = x
    viol = 0.0
    # El caudal máximo que entrega el sistema depende de la presión
    viol += max(0.0, Q - K_CAUDAL * np.sqrt(P))
    # Límite de consumo total por riego (litros)
    viol += max(0.0, Q * t - FACTOR_CONSUMO_MAX * REQ_L)
    return viol


def eficiencia(x):
    """Eficiencia hídrica = agua aprovechada / agua aplicada (para reportar)."""
    aplicada, util = volumenes(x)
    return min(util, REQ_L) / aplicada


def funcion_objetivo(x):
    """A MINIMIZAR: agua desperdiciada + penalización por déficit + restricciones."""
    aplicada, util = volumenes(x)
    desperdicio = aplicada - min(util, REQ_L)          # pérdidas + exceso
    deficit = max(0.0, REQ_L - util)
    return desperdicio + W_DEFICIT * deficit + W_RESTR * restricciones(x)


def describir(x):
    """Diccionario con las magnitudes de una configuración (para tablas)."""
    aplicada, util = volumenes(x)
    return {
        **{n: float(v) for n, v in zip(NOMBRES_CORTOS, x)},
        "agua_aplicada_L": aplicada,
        "agua_util_L": util,
        "deficit_L": max(0.0, REQ_L - util),
        "eficiencia_pct": 100.0 * eficiencia(x),
        "violacion": restricciones(x),
        "f_objetivo": funcion_objetivo(x),
    }
