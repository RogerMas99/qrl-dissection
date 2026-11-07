# Punto 5. FrozenLake, semillas resueltas con el criterio de la tabla 4.4

**Criterio de la tabla 4.4:** una semilla está resuelta si su política greedy llega a la meta en el **último** punto de evaluación (100 000 pasos), es decir, `greedy_return ≥ 0,999` en esa fila de `<ejecución>_eval.csv`. FrozenLake 4x4 sin deslizamiento es determinista y los retornos son 0 o 1, de modo que el retorno medio del último punto es exactamente 1 si llega a la meta y 0 si no.

Script: `scripts/p5_frozenlake_solved_last_checkpoint.py` (lee los `_eval.csv` a través de los manifiestos; semillas con FIX-01 activo y 100 000 pasos). Datos: `p5_frozenlake_solved.csv` (resumen) y `p5_frozenlake_solved_per_seed.csv` (una fila por semilla, con la política del último punto y la media de los tres últimos).

| brazo | semillas | resueltas, criterio de la tabla 4.4 (último punto) | resueltas, criterio del dossier de exp13 (`greedy_final` ≥ 0,999: los tres últimos puntos) |
|---|---|---|---|
| `mlp_bits_w6` (58 parámetros) | 30 | **0 de 30** | 0 de 30 |
| `mlp_bits_w12` (112 parámetros) | 30 | **19 de 30** | 15 de 30 |
| circuito binario entrelazado L = 1 (`frozen_binary_4q_L1`, exp04) | 10 | **0 de 10** | 0 de 10 |
| circuito binario entrelazado L = 5 (`frozen_binary_4q_L5`, exp04) | 10 | **9 de 10** | 7 de 10 |

- **Comprobación:** el circuito binario entrelazado L = 5 da **9 de 10**, igual que la tabla 4.4; L = 1 da 0 de 10.
- En todos los brazos el último punto de evaluación es el de 100 000 pasos (comprobado: mínimo = máximo = 100 000).
- La diferencia entre los dos criterios (19 frente a 15 en `w12`; 9 frente a 7 en L = 5) viene de semillas cuya política llega a la meta en el último punto pero no en los tres últimos. Para comparar `mlp_bits_w12` con el circuito hay que usar el **mismo criterio en los dos**: último punto, 19/30 frente a 9/10; tres últimos puntos, 15/30 frente a 7/10.
- Los resultados de `mlp_bits_*` son clásicos (red sobre los 4 bits de entrada), no simulados, y exploratorios por diseño (bloque G de exp13, con estos hiperparámetros).
