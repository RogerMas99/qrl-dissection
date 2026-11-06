# Dossier del experimento 11: control clásico que replica su entrada

Entorno real (CartPole-v1), solo brazos clásicos. **No interviene ningún simulador**, así que estos resultados pueden entrar en la memoria cuando el autor lo apruebe. Diseño, contrastes y regla de lectura, escritos antes de lanzar: `docs/EXPERIMENT-11-REPLICATION-CONTROL.md`.

## 1. Qué se probó y por qué

La conclusión "el Output Reuse (OR) ayuda al agente híbrido y no a los controles clásicos" se apoyaba en dos controles que no aíslan el efecto de la replicación en una red clásica capaz de aprender: `classical_OR{R}` (lineal, sin capa oculta, sin la posición del carro, en el suelo ≈ 10) y `classical_or_matched_r{R}` (presupuesto igualado al híbrido, capa oculta ReLU y observación completa, pero con la replicación **fija en 4**; solo cambia el ancho oculto). Faltaba el control que junta las cuatro cosas. Brazo nuevo: `classical_or_matched_rep_r{R}`: replicación R, una capa oculta ReLU, observación completa y presupuesto igual al total medido de `hybrid_OR{R}`.

**Anchos y parámetros medidos** (se construyó cada brazo y se contaron los parámetros entrenables):

| R | híbrido `hybrid_OR{R}` | control con replicación fija en 4 (ancho / parámetros) | control nuevo, replicación R (ancho / parámetros) |
|---|---|---|---|
| 4 | 222 | 12 / 230 | 12 / 230 (idéntico; no se repite) |
| 8 | 350 | 18 / 344 | 10 / 352 |
| 16 | 606 | 32 / 610 | 9 / 605 |
| 32 | 1118 | 59 / 1123 | 9 / 1181 |

Replicar la entrada R veces hace que la primera capa gaste más presupuesto, así que la capa oculta del control nuevo queda mucho más estrecha (10 / 9 / 9 neuronas frente a 18 / 32 / 59). Es parte de la lectura. El brazo de R=32 supera en un 5,6 % el presupuesto del híbrido (1181 frente a 1118): se acepta porque va en la dirección que favorece al control, la misma regla que el control de FrozenLake. `match_hidden_width` devuelve el ancho más cercano.

**R=4 no se repitió**: con R=4 el brazo nuevo tiene exactamente la especificación de `classical_or_matched_r4` (mismos índices, replicación 4, ancho 12, ReLU, presupuesto 222; comprobado con un test y por el ejecutor antes de empezar). Donde el diseño dice rep_r4 se usa `classical_or_matched_r4` de exp02. Se ejecutaron R = 8, 16 y 32.

**Protocolo.** Idéntico a exp02: DQN, CartPole-v1, 100 000 pasos, `batch_size` 128, `buffer_size` 10 000, `train_frequency` 10, resto por defecto de la librería, FIX-01 activado, semillas 1-10. Evaluación greedy cada 10 000 pasos **con 5 episodios**, que es lo que usó exp02 (no se le pasó número y se aplicó el valor por defecto): así `greedy_final` se compara entre mediciones con el mismo ruido. `final_perf` no depende de esto. No se ajustó ningún hiperparámetro: los resultados negativos son "con estos hiperparámetros". 30 celdas, ~42 s cada una, sin incidencias (un solo proceso por celda).

## 2. IQM por brazo (IC bootstrap percentil al 95 %, n = 10 por brazo)

| brazo | experimento | parámetros | `final_perf` IQM (IC) | `greedy_final` IQM (IC) |
|---|---|---|---|---|
| `hybrid_OR4` | exp02 | 222 | 94,7 (81,0-111,5) | 112,0 (91,3-152,4) |
| `hybrid_OR8` | exp02 | 350 | 109,8 (93,8-125,1) | 124,3 (107,2-189,9) |
| `hybrid_OR16` | exp02 | 606 | 104,2 (76,7-123,6) | 146,2 (109,7-180,0) |
| `hybrid_OR32` | exp02 | 1118 | 137,6 (106,3-179,7) | 161,1 (120,0-224,0) |
| `hybrid_or_r4_noent` | exp08 | | 60,4 (46,6-74,3) | 65,4 (50,4-83,7) |
| `hybrid_or_r8_noent` | exp08 | | 69,1 (55,3-82,0) | 86,9 (66,3-106,5) |
| `hybrid_or_r16_noent` | exp08 | | 89,5 (72,6-104,0) | 103,9 (83,8-123,8) |
| `hybrid_or_r32_noent` | exp08 | | 77,9 (64,3-91,5) | 96,6 (80,2-106,3) |
| `classical_or_matched_r4` (= rep_r4) | exp02 | 230 | 12,4 (11,5-15,5) | 13,8 (10,9-17,5) |
| `classical_or_matched_r8` | exp02 | 344 | 23,8 (20,5-27,8) | 24,8 (18,0-36,9) |
| `classical_or_matched_r16` | exp02 | 610 | 36,3 (28,4-43,5) | 43,0 (32,5-59,3) |
| `classical_or_matched_r32` | exp02 | 1123 | 61,2 (46,0-71,5) | 88,5 (62,3-117,4) |
| **`classical_or_matched_rep_r8`** | **exp11** | 352 | **10,0 (9,7-11,1)** | **9,5 (9,3-10,6)** |
| **`classical_or_matched_rep_r16`** | **exp11** | 605 | **10,6 (9,8-14,7)** | **9,8 (9,4-13,3)** |
| **`classical_or_matched_rep_r32`** | **exp11** | 1181 | **13,0 (9,8-17,3)** | **15,0 (9,8-22,4)** |
| `classical_OR4 / 8 / 16 / 32` | exp02 | | 9,6 / 9,6 / 9,6 / 10,2 | 9,5 / 9,5 / 9,6 / 10,0 |

Semillas del control nuevo por encima de 12 en `final_perf`: 1 de 10 (R=8, máximo 14,4), 3 de 10 (R=16, máximo 30,4) y 5 de 10 (R=32, máximo 22,7). La referencia de R=4: 6 de 10 (máximo 20,1). El control con replicación fija en 4 y ancho creciente: 10 de 10 en R=8, 16 y 32.

## 3. Contrastes (P(a>b) no emparejada, empates = 1/2; sin corrección por comparaciones múltiples)

"Fuerte" = P ≥ 0,95 en las dos métricas. Intermedios: "indica una dirección", con la dirección; "mixto" si las dos métricas apuntan en sentidos opuestos.

**C1. Efecto de la replicación en un control clásico a igual presupuesto** (replicación R frente a replicación 4):

| R | IQM final a / b | P final | IQM greedy a / b | P greedy | lectura |
|---|---|---|---|---|---|
| 8 | 10,0 / 23,8 | 0,00 | 9,5 / 24,8 | 0,02 | fuerte, sentido inverso |
| 16 | 10,6 / 36,3 | 0,03 | 9,8 / 43,0 | 0,00 | fuerte, sentido inverso |
| 32 | 13,0 / 61,2 | 0,00 | 15,0 / 88,5 | 0,00 | fuerte, sentido inverso |

**C2. El circuito frente al control con la misma replicación** (`hybrid_OR{R}` frente a `classical_or_matched_rep_r{R}`):

| R | IQM final a / b | P final | IQM greedy a / b | P greedy | lectura |
|---|---|---|---|---|---|
| 4 | 94,7 / 12,4 | 1,00 | 112,0 / 13,8 | 1,00 | fuerte |
| 8 | 109,8 / 10,0 | 1,00 | 124,3 / 9,5 | 1,00 | fuerte |
| 16 | 104,2 / 10,6 | 1,00 | 146,2 / 9,8 | 1,00 | fuerte |
| 32 | 137,6 / 13,0 | 1,00 | 161,1 / 15,0 | 1,00 | fuerte |

**C3. Tendencia con R dentro del control nuevo:**

| contraste | IQM final a / b | P final | IQM greedy a / b | P greedy | lectura |
|---|---|---|---|---|---|
| rep_r32 > rep_r4 | 13,0 / 12,4 | 0,43 | 15,0 / 13,8 | 0,49 | cerca de 0,5 |
| rep_r16 > rep_r4 | 10,6 / 12,4 | 0,24 | 9,8 / 13,8 | 0,17 | indica una dirección (r4 por encima) |

**C4 (contexto). El circuito sin entrelazar frente al control con la misma replicación:** `hybrid_or_r{R}_noent` frente a `rep_r{R}`: P = 1,00 en las dos métricas en R = 4, 8, 16 y 32 (IQM final 60-90 frente a 10-13): fuerte en los cuatro.

## 4. Aplicación de la regla de lectura (tal como se fijó antes de lanzar)

- **Rama "mejora con la replicación" (C1 y C3 fuertes a favor del control): no se cumple.** C1 sale fuerte pero en sentido **inverso** (replicar R veces es peor que replicar 4 a igual presupuesto) y C3 no muestra mejora con R.
- **Rama "no mejora (C1 y C3 cerca de 0,5) y el híbrido gana (C2)": se cumple solo la parte del híbrido.** C2 es fuerte en los cuatro R. Pero C1 **no** está cerca de 0,5 (es fuerte en sentido inverso) y C3 a R=16 indica dirección contraria. Por la regla escrita, la condición no se cumple.
- **Rama "el control no aprende en absoluto" (IQM ≤ 12 en las dos métricas en todos los R): se cumple casi, no del todo.** R=8 y R=16 están por debajo de 12 (10,0 / 9,5 y 10,6 / 9,8). Pero R=4 (12,4 / 13,8, la referencia, justo en el límite) y R=32 (13,0 / 15,0) lo superan por poco. Lo decimos expresamente en lugar de clasificarlos en silencio. Todos los IQM del control con replicación están por debajo de 15.

**Lectura honesta:** ninguna de las tres ramas se cumple de forma limpia. Lo que sí se puede decir con estos datos es (1) que el control que replica R veces **no mejora** con R, y a igual presupuesto queda peor que el que replica 4 veces y tiene más neuronas; (2) que el híbrido lo supera con P = 1,00 en los cuatro R; y (3) que, como el control está en el suelo o muy cerca (1, 3 y 5 de 10 semillas lo superan 12), el contraste C2 se parece al de "circuito frente a control casi muerto" y **informa poco** sobre si la replicación podría ayudar a una red clásica que sí aprende. Este experimento no confirma ni refuta que el OR sea específico del circuito: no apoya que sea un mecanismo genérico en redes clásicas, pero el control no llega a aprender lo bastante como para descartarlo.

**Limitación de diseño que hay que decir:** el control nuevo mezcla dos cosas a la vez, más replicación y menos neuronas ocultas (9-10 frente a 18-59), porque el presupuesto está fijado. La caída de C1 puede deberse a la replicación, al ancho o a las dos; con este diseño no se separa. Un control que las desconfunda (por ejemplo dejando que el presupuesto del control supere al del híbrido para mantener un ancho oculto mayor) no se ha ejecutado y no estaba en el encargo.

n = 10 por brazo, sin corrección por comparaciones múltiples, una sola configuración de hiperparámetros.

## 5. Tarea 2: parámetros de `frozen_onehot_mlp`

**El valor correcto es 5 508, no 4 548.**

- `frozen_onehot_mlp` (`net_arch` [64, 64], ReLU) entrena sobre su entorno real, `FrozenLake4x4OneHot-v0` (`FROZEN_ARM_ENV`); los manifiestos de exp04 registran ese `env_id`.
- Construido sobre ese entorno cuenta **5 508** parámetros entrenables, igual que a mano: 16·64+64 + 64·64+64 + 64·4+4 = 5 508.
- Construido sobre el entorno escalar cuenta 4 548 (1·64+64 + 64·64+64 + 64·4+4). Pero ese no es el que se usó en el entrenamiento.
- `docs/EXPERIMENT-04.md` (línea 128), `docs/RESULTS-LOG.md` (línea 664) y `docs/EXPERIMENT-09-WAVE2.md` (línea 134) ya dicen 5 508.
- **De dónde salía el 4 548:** del dossier v3. Mi generador de tablas contaba los parámetros de todos los brazos `frozen_*` sobre el entorno escalar, también `frozen_onehot_mlp`. 4 548 es, de hecho, el valor correcto de `frozen_scalar_mlp_large`. Afecta a las dos filas de `frozen_onehot_mlp` de la tabla de FrozenLake del dossier v3 (líneas 148 y 149) y a cualquier texto que se haya copiado de ahí; ningún resultado de rendimiento cambia. `frozen_onehot_mlp` es el único brazo `frozen_*` que no entrena sobre el entorno escalar. Fe de erratas: `handoff_memoria/DOSSIER_v3_ERRATA.md`.

## 6. Ficheros

`per_seed_exp11.csv` (una fila por ejecución, mismas columnas que `per_seed_metrics.csv` del dossier; incluye las referencias de exp02 y exp08), `curves_exp11_train_ma50.csv`, `curves_exp11_eval_greedy.csv`, `summary_exp11.csv`, `tables_exp11.md` (tablas completas con la aplicación mecánica de la regla), `design_widths_params.json`.
