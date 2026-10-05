==============================================================
PSO PARA DISEÑO ÓPTIMO DE RIEGO - Grupo 3 (Lucas B. y Emanuel B.)
==============================================================

Objetivo: encontrar el caudal Q (L/min), la presión P (bar) y el tiempo
t (min) que minimizan el agua desperdiciada cumpliendo el requerimiento
del cultivo, usando Particle Swarm Optimization.

INSTALACIÓN
    pip install -r requirements.txt

EJECUCIÓN
    python main.py            corrida completa (~30-40 s)
    python main.py --rapido   pocas corridas, para probar cambios

    python pso_riego.py       versión original en un solo archivo

ARCHIVOS
    modelo.py        parámetros, función objetivo, restricciones (Persona A)
    pso.py           algoritmo PSO y búsqueda aleatoria (Persona B)
    experimentos.py  validación, semillas, línea base, grilla de parámetros
    graficos.py      todas las figuras
    main.py          ejecuta todo y guarda los resultados
    resultados/      se genera solo: figuras .png, tablas .csv, resumen.txt
    informe/         esqueleto del informe

RESULTADOS GENERADOS (resultados/)
    convergencia.png          PSO vs búsqueda aleatoria (media ± desv)
    particulas.png            enjambre en distintas iteraciones (Q vs t, color = P)
    trayectorias.png          evolución de Q, P y t de cada partícula
    parametros.png            efecto de w, c1, c2 y tamaño del enjambre
    mejor_configuracion.csv   mejor (Q, P, t), agua aplicada, eficiencia
    comparacion.csv           PSO vs búsqueda aleatoria vs configuración fija
    parametros.csv            media, desv e iteración de convergencia por config.
    resumen.txt               salida completa de la consola

Todas las corridas usan semillas fijas (0..N-1): los resultados son
reproducibles.
