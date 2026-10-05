"""
Figuras para el informe. Todas se guardan en la carpeta de resultados.
Grupo 3 — Lucas B. y Emanuel B.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")      # guardar a archivo sin abrir ventanas
import matplotlib.pyplot as plt

from modelo import NOMBRES, LB, UB


def _guardar(carpeta, nombre):
    ruta = os.path.join(carpeta, nombre)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    return ruta


def graficar_convergencia(res_pso, res_aleatoria, n_particulas, carpeta):
    """Mejor valor vs evaluaciones: PSO vs búsqueda aleatoria (media ± desv)."""
    curvas_pso = np.array([r[2] for r in res_pso])
    # La búsqueda aleatoria se muestrea cada n_particulas evaluaciones para
    # alinearla con las iteraciones del PSO (mismo eje x en evaluaciones).
    curvas_al = np.array([r[2][n_particulas - 1::n_particulas] for r in res_aleatoria])
    evals = n_particulas * np.arange(1, curvas_pso.shape[1] + 1)

    plt.figure(figsize=(7, 4))
    for curvas, etiqueta, estilo in [(curvas_pso, "PSO", "-"),
                                     (curvas_al, "Búsqueda aleatoria", "--")]:
        m, s = curvas.mean(axis=0), curvas.std(axis=0)
        plt.plot(evals[:len(m)], m, estilo, label=f"{etiqueta} (media de {len(curvas)} corridas)")
        plt.fill_between(evals[:len(m)], np.maximum(m - s, 1e-9), m + s, alpha=0.2)
    plt.yscale("log")
    plt.xlabel("Evaluaciones de la función objetivo")
    plt.ylabel("Mejor valor de f (L equivalentes)")
    plt.title("Curva de convergencia")
    plt.legend()
    return _guardar(carpeta, "convergencia.png")


def graficar_particulas(hist_pos, carpeta, iteraciones=(0, 10, 50, -1)):
    """Proyección 2D (Caudal vs Tiempo) del enjambre en distintas iteraciones.
    El color indica la presión de cada partícula."""
    fig, axs = plt.subplots(1, len(iteraciones), figsize=(14, 3.8), sharey=True)
    for ax, it in zip(axs, iteraciones):
        pos = hist_pos[it]
        sc = ax.scatter(pos[:, 0], pos[:, 2], c=pos[:, 1], s=15,
                        cmap="viridis", vmin=LB[1], vmax=UB[1])
        ax.set_title(f"Iteración {it if it >= 0 else len(hist_pos) - 1}")
        ax.set_xlabel(NOMBRES[0])
        ax.set_xlim(LB[0], UB[0])
        ax.set_ylim(LB[2], UB[2])
    axs[0].set_ylabel(NOMBRES[2])
    fig.colorbar(sc, ax=axs, label=NOMBRES[1], shrink=0.9)
    ruta = os.path.join(carpeta, "particulas.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    return ruta


def graficar_parametros_iteracion(hist_pos, carpeta):
    """Evolución de Q, P y t de todas las partículas a lo largo de las iteraciones."""
    pos = np.array(hist_pos)          # (iter, partículas, dim)
    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    for d, ax in enumerate(axs):
        ax.plot(pos[:, :, d], color="tab:blue", alpha=0.15, lw=0.8)
        ax.set_ylabel(NOMBRES[d])
    axs[-1].set_xlabel("Iteración")
    axs[0].set_title("Trayectoria de las partículas por variable")
    return _guardar(carpeta, "trayectorias.png")


def graficar_comparacion_parametros(salida, carpeta):
    plt.figure(figsize=(8, 4.5))
    for nombre, (_, _, curva) in salida.items():
        plt.plot(curva, label=nombre)
    plt.yscale("log")
    plt.xlabel("Iteración")
    plt.ylabel("Mejor valor de f (media)")
    plt.title("Efecto de los parámetros en la convergencia")
    plt.legend(fontsize=8, ncol=2)
    return _guardar(carpeta, "parametros.png")
