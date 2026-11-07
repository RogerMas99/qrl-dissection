# Punto 1. Revalidación de los simuladores (anexo C.7)

**Resultados simulados, uso aprobado solo para las cifras de validación (aprobación del 2026-11-01 en `docs/SIMULATED-RESULTS-POLICY.md`).**

## Qué se hizo

Se volvió a ejecutar, **sobre los manifiestos que hay hoy en disco**, el mismo script y la misma regla de aceptación: `experiments/exp09c_battery_report.py` (lógica idéntica a la del 2026-10-21, mismas parejas, semillas 1-10 en el lado simulador, mismos umbrales, misma semilla del bootstrap = 0). **No se cambió la regla.** Comando:

```
python experiments/exp09c_battery_report.py --results-root <results> --out p1_battery_revalidation.md
```

Salida completa: `p1_battery_revalidation.md` (tabla de las 44 parejas, con las P por métrica). Comparación pareja a pareja con la tabla original: `p1_pairs_old_vs_new.csv` (script `p1_compare_battery.py`).

**Esta reevaluación ya existía documentada** en `docs/EXPERIMENT-09-BATTERY.md`, sección "Re-evaluation with complete real references (2026-11-01)". Regenerada hoy desde cero da **exactamente las mismas cifras** que ese documento. En todos los brazos reales de referencia hay n = 10 semillas (columna "n sim / real" del CSV; el único n distinto es la fila extra `sv_frozen_binary_4q_L5`, 20 frente a 10, que no entra en la regla).

## Veredicto de cada criterio

| criterio (umbral) | evaluación original (referencias incompletas) | evaluación con las 10 semillas reales en todos los brazos (hoy) |
|---|---|---|
| 1. sin sesgo global: IC 95 % bootstrap de la media de d contiene 0 y \|d medio\| ≤ 0,30 | d medio = **+0,119**, IC (+0,010, +0,250) → **NO CUMPLE** | d medio = **+0,090**, IC (**+0,008, +0,175**) → **NO CUMPLE** (el IC excluye el 0 por 0,008; \|d medio\| está dentro del 0,30) |
| 2. como máximo el 25 % de las celdas (pareja informativa × métrica) con P(sim>real) fuera de [0,25, 0,75] | **3 de 63** = 5 % → CUMPLE | **1 de 63** = 2 % → CUMPLE (`sv_frozen_binary_4q_L5 (seeds≤10)` contra `frozen_binary_4q_L5`, `final_perf`, P = 0,21) |
| 3. los dos simuladores coinciden (gemelos): P en [0,25, 0,75] en las 9 comparaciones | **0 de 9** fuera → CUMPLE | **0 de 9** fuera → CUMPLE (las parejas gemelas ya tenían referencias completas, no cambian) |
| **Global** | NO ACEPTADO | **NO ACEPTADO** |

- Parejas informativas: **21 en las dos evaluaciones, las mismas**. Ninguna pasó de informativa a no informativa, ni al revés (comprobado automáticamente en `p1_pairs_old_vs_new.csv`, columna `informative_changed`: vacía en las 44 filas).
- Combinaciones (pareja informativa × 3 métricas) = 63; comparaciones entre simuladores = 3 parejas × 3 métricas = 9.
- Suelo de ruido real-real (`hybrid_fig4` frente a `hybrid_DR5`): d = −0,11, sin cambios (los dos brazos ya tenían n = 10).

## Parejas cuyo d cambió al completarse las referencias (no cambia ninguna bandera)

| pareja (simulador / real) | n real antes → ahora | d antes → ahora |
|---|---|---|
| `sv_paper_skolik_8q_L2` / `paper_skolik_8q_L2` | 4 → 10 | +1,05 → +0,14 |
| `sv_paper_skolik_8q_L5` / `paper_skolik_8q_L5` | 3 → 10 | −0,12 → −0,36 |
| `su2_cartpole_L5_OR16` / `hybrid_or_r16_noent` | 7 → 10 | +0,24 → −0,11 |
| `su2_cartpole_L5_OR32` / `hybrid_or_r32_noent` | 5 → 10 | −0,26 → +0,13 |
| `sv_cartpole_L2_OR16` / `hybrid_DR2_OR16` | 9 → 10 | +0,23 → +0,38 |

(En el resto de las parejas con d, el cambio es de 0,13 o menos; ver el CSV.)

## Lo que mantiene el sesgo medio por encima de 0 (descriptivo, no es un criterio nuevo)

Tres parejas de CartPole entrelazado con Output Reuse alto: `sv_cartpole_L5_OR32` (d = +0,51), `sv_cartpole_L5_OR16` (+0,49) y `sv_cartpole_L2_OR16` (+0,38). Sin ellas, la media de las otras 18 parejas es +0,03 (cifra de `docs/EXPERIMENT-09-BATTERY.md`; no recalculada aquí). El IC t sobre los 21 valores de d es (−0,000, +0,181), es decir, al límite; la regla escrita es la del bootstrap y no se cumple. Cambiar la semilla del bootstrap mueve el límite inferior entre +0,008 y +0,011 (también de ese documento).

## Qué decide el autor

`docs/SIMULATED-RESULTS-POLICY.md` deja al autor la elección de qué conjunto cita el apéndice (la evaluación original, la completa, o ambas); los dos veredictos globales coinciden.
