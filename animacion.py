"""
Animación del PSO resolviendo el problema de riego, para explicarlo en vivo.
Grupo 3 — Lucas B. y Emanuel B.

Uso:
    python animacion.py              # abre una ventana animada
    python animacion.py --gif        # además guarda resultados/animacion.gif
    python animacion.py --sin-ventana --gif   # solo guarda el GIF

Qué se ve:
    Izquierda : mapa del problema (Caudal Q vs Tiempo t, con la presión fija en
                la mejor encontrada). Colores = agua desperdiciada f(x): más
                amarillo = mejor. Puntos = partículas; estrella = mejor global.
                La zona rayada es la que viola restricciones.
    Derecha   : mejor valor encontrado en cada iteración (convergencia).
    Abajo     : lo que hace el algoritmo en la iteración actual.
"""

import os
import sys

import numpy as np
import matplotlib

GUARDAR_GIF = "--gif" in sys.argv
SIN_VENTANA = "--sin-ventana" in sys.argv
if SIN_VENTANA:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from modelo import NOMBRES, REQ_L, LB, UB, funcion_objetivo, restricciones
from pso import pso

N_PART, N_ITER, SEMILLA = 30, 120, 1
CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")

# --- Corremos el PSO una vez y guardamos todo el historial para reproducirlo ---
gbest, gbest_val, hist_gbest, hist_pos = pso(
    funcion_objetivo, LB, UB, n_particulas=N_PART, n_iter=N_ITER, semilla=SEMILLA)
P_FIJA = gbest[1]

# Mejor global hasta cada iteración (posición), para dibujar la estrella
mejor_pos = []
mejor_val, mejor_x = np.inf, None
for pos in hist_pos:
    vals = np.array([funcion_objetivo(p) for p in pos])
    i = vals.argmin()
    if vals[i] < mejor_val:
        mejor_val, mejor_x = vals[i], pos[i].copy()
    mejor_pos.append(mejor_x.copy())

# --- Fondo: f(Q, t) con P fija ---
qs = np.linspace(LB[0], UB[0], 160)
ts = np.linspace(LB[2], UB[2], 160)
QQ, TT = np.meshgrid(qs, ts)
F = np.vectorize(lambda q, t: funcion_objetivo([q, P_FIJA, t]))(QQ, TT)
V = np.vectorize(lambda q, t: restricciones([q, P_FIJA, t]))(QQ, TT)
Fvis = np.log10(np.clip(F, 1, None))        # escala log para que se vea el valle

# --- Figura ---
fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[1.2, 1])
ax = fig.add_subplot(gs[0, 0])
axc = fig.add_subplot(gs[0, 1])
axt = fig.add_subplot(gs[1, :])
axt.axis("off")
fig.suptitle("PSO para riego óptimo: cada partícula es una configuración (Q, P, t)",
             fontsize=13, fontweight="bold")

cf = ax.contourf(QQ, TT, Fvis, levels=25, cmap="viridis_r")
ax.contourf(QQ, TT, V, levels=[1e-9, 1e9], colors="none", hatches=["///"], alpha=0)
ax.contour(QQ, TT, V, levels=[1e-9], colors="red", linewidths=1.5)
fig.colorbar(cf, ax=ax, label="log10 agua desperdiciada f(x)  (amarillo = mejor)")
ax.set_xlabel(NOMBRES[0])
ax.set_ylabel(NOMBRES[2])
ax.set_title(f"Mapa del problema (presión fija en {P_FIJA:.2f} bar). "
             "Línea roja: límite de restricciones", fontsize=9)
ax.set_xlim(LB[0], UB[0])
ax.set_ylim(LB[2], UB[2])

sc = ax.scatter([], [], s=45, c="white", edgecolors="black", zorder=3, label="partículas")
estrella, = ax.plot([], [], "*", ms=20, color="gold", mec="black", zorder=4, label="mejor global")
ax.legend(loc="upper right", fontsize=8)

axc.set_xlim(0, N_ITER)
axc.set_ylim(hist_gbest.min() * 0.98, hist_gbest.max() * 1.05)
axc.set_xlabel("Iteración")
axc.set_ylabel("Mejor f encontrado (L desperdiciados)")
axc.set_title("Convergencia")
axc.grid(alpha=0.3)
linea, = axc.plot([], [], "b-", lw=2)
texto = axt.text(0.01, 0.95, "", va="top", fontsize=11, family="monospace",
                 transform=axt.transAxes)


def explicacion(it, w):
    if it == 0:
        return ("Inicio: 30 configuraciones al azar. Cada una se evalúa con la función "
                "objetivo\n(agua desperdiciada + penalizaciones). Se anota la mejor de todas.")
    return (f"Cada partícula ajusta su velocidad mezclando 3 fuerzas:\n"
            f"  inercia (w={w:.2f}: sigue su rumbo) + memoria propia (c1: vuelve a su mejor punto)"
            f" + enjambre (c2: va hacia la estrella).\n"
            f"w baja de 0.9 a 0.4: al principio explora, al final afina.")


def actualizar(it):
    pos = hist_pos[it]
    w = 0.9 - 0.5 * max(it - 1, 0) / (N_ITER - 1)
    sc.set_offsets(pos[:, [0, 2]])
    g = mejor_pos[it]
    estrella.set_data([g[0]], [g[2]])
    linea.set_data(range(it + 1), hist_gbest[:it + 1])
    aplicada = g[0] * g[2]
    texto.set_text(
        f"Iteración {it:3d}/{N_ITER}   mejor f = {hist_gbest[it]:8.2f} L desperdiciados\n"
        f"Mejor configuración: Q={g[0]:5.1f} L/min  P={g[1]:4.2f} bar  t={g[2]:5.1f} min"
        f"  -> aplica {aplicada:6.0f} L (el cultivo necesita {REQ_L:.0f} L)\n"
        + explicacion(it, w))
    return sc, estrella, linea, texto


anim = FuncAnimation(fig, actualizar, frames=len(hist_pos), interval=150, repeat=True)
fig.tight_layout(rect=(0, 0, 1, 0.95))

if GUARDAR_GIF:
    os.makedirs(CARPETA, exist_ok=True)
    ruta = os.path.join(CARPETA, "animacion.gif")
    anim.save(ruta, writer="pillow", fps=8, dpi=80)
    print(f"GIF guardado en {ruta}")

if not SIN_VENTANA:
    plt.show()
