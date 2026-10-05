"""
Diseño óptimo de un sistema de riego con PSO — ejecuta todo el trabajo.
Grupo 3 — Lucas B. y Emanuel B.

Uso:
    python main.py            # corrida completa
    python main.py --rapido   # menos corridas, para probar cambios

Genera en resultados/:
    convergencia.png, particulas.png, trayectorias.png, parametros.png
    mejor_configuracion.csv, comparacion.csv, parametros.csv, resumen.txt
"""

import csv
import os
import sys

import numpy as np

from modelo import NOMBRES, REQ_L, CONFIG_FIJA, LB, UB, funcion_objetivo, describir
from pso import pso
from experimentos import (validar, experimento_semillas, experimento_aleatoria,
                          experimento_parametros, evaluaciones_pso)
from graficos import (graficar_convergencia, graficar_particulas,
                      graficar_parametros_iteracion, graficar_comparacion_parametros)

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
RAPIDO = "--rapido" in sys.argv

N_PART, N_ITER = 30, 200
N_CORRIDAS = 5 if RAPIDO else 30
N_CORRIDAS_PARAM = 3 if RAPIDO else 20
SEMILLA_GRAFICO = 1


def guardar_csv(nombre, filas):
    ruta = os.path.join(CARPETA, nombre)
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    return ruta


def main():
    os.makedirs(CARPETA, exist_ok=True)
    lineas = []                      # todo lo que se imprime va también a resumen.txt

    def log(texto=""):
        print(texto)
        lineas.append(texto)

    log(f"Requerimiento por riego: {REQ_L:.0f} L | PSO: {N_PART} partículas, "
        f"{N_ITER} iteraciones, {N_CORRIDAS} corridas")

    # 1) Validación con funciones de prueba
    log("\n[1] Validación del PSO")
    for nombre, (m, s, mn) in validar().items():
        log(f"  {nombre:<10} media = {m:.2e}  desv = {s:.2e}  mínimo = {mn:.2e}")

    # 2) PSO con varias semillas
    res = experimento_semillas(n_corridas=N_CORRIDAS, n_particulas=N_PART, n_iter=N_ITER)
    vals = np.array([r[1] for r in res])
    x_opt = res[int(np.argmin(vals))][0]
    d_opt = describir(x_opt)
    log(f"\n[2] PSO ({N_CORRIDAS} semillas): f media = {vals.mean():.2f}, "
        f"desv = {vals.std():.2f}, mejor = {vals.min():.2f}")
    log("  Mejor configuración encontrada:")
    for nombre, valor in zip(NOMBRES, x_opt):
        log(f"    {nombre:<18} {valor:8.2f}")
    log(f"    Agua aplicada      {d_opt['agua_aplicada_L']:8.0f} L")
    log(f"    Eficiencia hídrica {d_opt['eficiencia_pct']:8.1f} %")
    log(f"    Violación restr.   {d_opt['violacion']:8.2f}")
    guardar_csv("mejor_configuracion.csv", [d_opt])

    # Dispersión de las soluciones entre semillas (¿convergen al mismo punto?)
    xs = np.array([r[0] for r in res])
    log("  Dispersión de la solución entre semillas (media ± desv):")
    for nombre, m, s in zip(NOMBRES, xs.mean(axis=0), xs.std(axis=0)):
        log(f"    {nombre:<18} {m:8.2f} ± {s:.2f}")

    # 3) Línea base: búsqueda aleatoria con el mismo presupuesto
    n_eval = evaluaciones_pso(N_PART, N_ITER)
    res_al = experimento_aleatoria(n_eval, n_corridas=N_CORRIDAS)
    vals_al = np.array([r[1] for r in res_al])
    x_al = res_al[int(np.argmin(vals_al))][0]

    # 4) Configuración fija típica
    d_fija = describir(CONFIG_FIJA)

    log(f"\n[3-4] Comparación ({n_eval} evaluaciones por corrida)")
    comparacion = []
    for metodo, valores, x in [("PSO", vals, x_opt),
                               ("Búsqueda aleatoria", vals_al, x_al),
                               ("Configuración fija", np.array([d_fija["f_objetivo"]]), CONFIG_FIJA)]:
        d = describir(x)
        fila = {"metodo": metodo, "f_media": valores.mean(), "f_desv": valores.std(),
                "f_mejor": valores.min(), **{k: d[k] for k in
                ("Q_L_min", "P_bar", "t_min", "agua_aplicada_L", "eficiencia_pct")}}
        comparacion.append(fila)
        log(f"  {metodo:<20} f = {fila['f_media']:9.2f} ± {fila['f_desv']:7.2f}  "
            f"(mejor {fila['f_mejor']:9.2f})  eficiencia = {fila['eficiencia_pct']:5.1f}%")
    guardar_csv("comparacion.csv", comparacion)

    # 5) Efecto de los parámetros
    log(f"\n[5] Efecto de los parámetros ({N_CORRIDAS_PARAM} corridas c/u)")
    salida_param = experimento_parametros(n_corridas=N_CORRIDAS_PARAM, n_iter=N_ITER)
    filas_param = []
    for nombre, (m, s, curva) in salida_param.items():
        # Primera iteración en la que la curva media queda a <1% del valor final
        it_conv = int(np.argmax(curva <= curva[-1] * 1.01))
        filas_param.append({"configuracion": nombre, "f_media": m, "f_desv": s,
                            "iter_convergencia_1pct": it_conv})
        log(f"  {nombre:<20} f = {m:9.2f} ± {s:7.2f}  converge (1%) en it. {it_conv}")
    guardar_csv("parametros.csv", filas_param)

    # 6) Gráficos
    _, _, _, hist_pos = pso(funcion_objetivo, LB, UB, n_particulas=N_PART,
                            n_iter=N_ITER, semilla=SEMILLA_GRAFICO)
    rutas = [
        graficar_convergencia(res, res_al, N_PART, CARPETA),
        graficar_particulas(hist_pos, CARPETA),
        graficar_parametros_iteracion(hist_pos, CARPETA),
        graficar_comparacion_parametros(salida_param, CARPETA),
    ]
    log("\nArchivos guardados en resultados/: " + ", ".join(os.path.basename(r) for r in rutas)
        + ", mejor_configuracion.csv, comparacion.csv, parametros.csv, resumen.txt")

    with open(os.path.join(CARPETA, "resumen.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")


if __name__ == "__main__":
    main()
