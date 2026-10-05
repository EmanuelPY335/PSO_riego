# Diseño óptimo de un sistema de riego mediante PSO
Grupo 3 — Lucas B. y Emanuel B.

## 1. Introducción y objetivo
- Problema: regar desperdiciando la menor cantidad de agua posible.
- Objetivo: encontrar (Q, P, t) óptimos con PSO.

## 2. Marco teórico
- PSO: enjambre, pbest, gbest.
- Ecuaciones: v = w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x);  x = x + v
- Parámetros: w (inercia decreciente 0.9 → 0.4), c1, c2, tamaño del enjambre.

## 3. Descripción del problema y modelo matemático
- Parcela / cultivo: TODO (área, ET, justificación).
- Variables y rangos: Q ∈ [10, 100] L/min, P ∈ [1, 4] bar, t ∈ [10, 120] min.
- Pérdidas: perdida(P) = 0.05 + 0.03·(P − 2)²  — TODO justificar/reemplazar.
- f = desperdicio + W_DEFICIT·déficit + W_RESTR·violaciones.
- Restricciones: Q ≤ 50·√P;  Q·t ≤ 1.5·REQ.
- Supuestos y limitaciones del modelo.

## 4. Implementación
- Pseudocódigo (ver indicaciones, sección 11).
- Decisiones: límite de velocidad 20% del rango, recorte al borde,
  penalización de restricciones, semillas fijas.

## 5. Experimentos y resultados
- Validación: esfera y Rastrigin (resultados/resumen.txt, sección [1]).
- 30 corridas: media y desviación (resumen.txt [2]).
- Tabla mejor configuración (mejor_configuracion.csv).
- Tabla comparativa PSO / aleatoria / fija (comparacion.csv).
- Figuras: convergencia.png, particulas.png, trayectorias.png, parametros.png.
- Efecto de los parámetros (parametros.csv).

## 6. Análisis y discusión
- Óptimo no único: con el modelo actual cualquier Q·t ≈ REQ/(1 − pérdida(2))
  es óptimo, por eso Q y t varían entre semillas pero P siempre da 2 bar.
- Ventajas / desventajas de PSO frente a la búsqueda aleatoria.
- Limitaciones del modelo.

## 7. Conclusiones

## 8. Referencias
- Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization. Proc. IEEE ICNN.
- Shi, Y. & Eberhart, R. (1998). A modified particle swarm optimizer. IEEE ICEC.
- Allen, R. G. et al. (1998). FAO Irrigation and Drainage Paper 56.
