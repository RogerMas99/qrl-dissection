# Paquete para el chat que redacta Resultados y Conclusiones (v3.1)

## Qué contiene

| archivo | para qué |
|---|---|
| `PROMPT_ARRANQUE.txt` | **Mensaje de arranque** para pegar como primer mensaje. Incluye la regla de exclusión de resultados de simulador y la descripción del método del simulador propio (qué es, cómo se construyó, cuánto más rápido es y por qué, límites y uso real) |
| `01_DOSSIER_RESULTADOS_v3.md` | **Fuente única de cifras.** Reglas, mapa de experimentos, tablas (secciones 3–5), lectura de resultados F1–F10 (sección 6), salvedades, verificación del aparato (7b, 25 tests), hipótesis, decisiones abiertas y esquema de capítulos |
| `summary_by_arm.csv` | Una fila por brazo: n, IQM e IC, semillas vivas |
| `per_seed_metrics.csv` | Una fila por semilla (888 runs). Solo si se piden semillas concretas |

## Cómo usarlo
1. Pega `PROMPT_ARRANQUE.txt` como primer mensaje y adjunta el dossier y los dos CSV.
2. El chat tiene acceso al repositorio de GitHub (RogerMas99/qrl-dissection, rama `main`) para verificar código y tests, pero las cifras salen solo del dossier.

## Lo que el chat puede leer pero NO usar
Los resultados de simulador (exp09, brazos `su2_*` y `sv_*`, `docs/EXPERIMENT-09*.md`, `docs/SIMULATED-RESULTS-POLICY.md`). El prompt lo prohíbe explícitamente. La política del proyecto exige tu aprobación explícita para llevarlos al trabajo final, y la tabla de aprobaciones sigue vacía.

## Decisión tuya pendiente sobre el simulador
Mencionar o no el método (sin resultados). Si decides mencionarlo, díselo al chat; el prompt ya le explica cómo debe presentarlo y le prohíbe decir que se usó antes de los experimentos reales, porque no fue así (se hizo después de exp01–exp08).
