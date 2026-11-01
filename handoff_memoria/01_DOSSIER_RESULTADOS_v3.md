# Dossier de resultados — TFM `qrl-dissection` (versión 3, datos finales)

**Para quién es este documento:** es el material de entrada para redactar los capítulos **Resultados** y **Conclusiones** de la memoria del TFM. El alcance y el estado del arte ya están escritos. Las cifras de las secciones 3–5 salen de los manifiestos y CSV de los experimentos mediante un script (ninguna está transcrita a mano). Las secciones 1, 2, 6, 7, 8, 9 y 10 son texto escrito a mano a partir de esas cifras.

**Estado de los datos:** **completos**. Todos los brazos de exp01 a exp08 tienen n = 10 semillas (excepto el piloto de 60k pasos de exp01, 3 semillas, que se incluye solo por trazabilidad). Instantánea: 2026-11-01. A diferencia de la versión 2, **ya no hay resultados parciales**: las lecturas F6, F8 y F9 se han reescrito con n = 10.

## Reglas obligatorias para quien redacte (no negociables)

1. **Todo lo propio es DQN.** No hay ningún experimento propio con PPO. Los números de PPO son los del paper de referencia (arXiv:2511.17112, `data/paper_ppo_summary.csv`). Una diferencia con el paper mezcla algoritmo y circuito: **no es por sí sola una "no-replicación"**.
2. **Métricas principales:** `final_perf` (media del retorno en el último 10 % de episodios de entrenamiento) y `greedy_final` (media de los últimos 3 puntos de evaluación greedy), con **IQM e IC bootstrap al 95 %**, y **P(mejora)** para los contrastes. `best_ma50` (la métrica del paper) solo para comparar con el paper: está inflado en los brazos muertos.
3. **Los hiperparámetros de DQN nunca se ajustaron** (tasa de aprendizaje 2,5e-4 sin tocar; ver 2b). Todo resultado negativo se redacta "**con estos hiperparámetros**", nunca "el circuito no puede" ni "no funciona".
4. **Resultados de simulador: excluidos de este paquete a propósito.** Cualquier brazo `su2_*` o `sv_*` y todo el experimento exp09 son exploratorios y **no pueden ir al trabajo final en ninguna forma sin aprobación explícita del autor**. Aquí no hay ninguna de sus cifras. **No las busques, no las cites y no pidas que te las den.** Si el autor te las da por otra vía, indícale que contradice esta regla.
5. **No convertir un resultado inconcluyente en uno positivo.** Etiquetas de evidencia usadas en la sección 6: **[FUERTE]** (P ≥ 0,95 en las dos métricas con n = 10), **[MODERADA]**, **[DÉBIL]**, **[NEGATIVO]** (resultado nulo o contrario a la hipótesis de partida).
6. Donde un dato dependa de una decisión del autor, marcar **DECISIÓN A TOMAR**; donde no se pueda saber con los datos actuales, **NO DETERMINABLE**. No inventar cifras ni mecanismos: si falta un dato, pregúntalo.
7. Las hipótesis pre-declaradas (sección 8) se citan **en inglés y literalmente**.
8. Los contrastes P(a>b) son **no pareados y sin corrección por comparaciones múltiples** (se han mirado decenas). Tratar P entre 0,05 y 0,95 como indicativo, no como prueba.

---

## 1. Qué es el proyecto

Extiende "Dissecting Quantum Reinforcement Learning" (arXiv:2511.17112), que estudia circuitos cuánticos variacionales (PQC) como aproximadores de la función Q o de la política con **PPO sobre CartPole**, a **DQN** y a **FrozenLake 4×4 determinista**. El diseño usa el mismo presupuesto que el paper (100 000 pasos, 10 semillas) y audita la implementación de partida (lista de correcciones FIX-xx / NEW-xx en `docs/CORRECTIONS.md`).

Hiperparámetros DQN comunes: CartPole `batch_size=128, buffer_size=10 000, train_frequency=10`; FrozenLake `batch_size=128, buffer_size=50 000, train_frequency=1, learning_starts=1 000`. 100 000 pasos por run; evaluación greedy cada 10 000 pasos (20 episodios en CartPole). Semillas 1–10. FIX-01 (transición fantasma del autoreset) activado en todo salvo en las filas marcadas `off` (contraste explícito).

**Circuitos.** Todos los núcleos usan la plantilla de Skolik (re-uploading de datos). *Hybrid* = circuito cuántico + cabeza lineal clásica. `ent=False` = misma arquitectura sin puertas entrelazadoras (CZ). **Output Reuse (OR)** = repetir R veces la salida del circuito antes de la cabeza lineal (es una reparametrización: no añade una clase de funciones nueva, pero sí parámetros). **DR** = profundidad L (capas) de re-uploading.

## 2. Mapa de experimentos

| exp | entorno | pregunta | brazos principales | n |
|---|---|---|---|---|
| exp01 | CartPole | ¿Es el híbrido mejor que baselines clásicos de capacidad comparable? Efecto de FIX-01 | `hybrid_fig4` (Skolik 8q, L=5, 126 params), `matched_classical`, `paper_linear` (26), `oversized_mlp` (10 934) | 10 (FIX-01 on y off) |
| exp02 | CartPole | Output Reuse | `hybrid_OR{4,8,16,32}`, `classical_OR*`, `classical_or_matched_r*` | 10 |
| exp03 / exp03b | CartPole | Profundidad de DR; entrelazado frente a no | `hybrid_DR{1,2,5}` y `_noent` | 10 |
| exp04 | FrozenLake | Embedding (escalar frente a binario), profundidad, entrelazamiento | Config A: 1 qubit escalar, L∈{1,5,10,15}; Config B: 4 qubits binario, L∈{1,5}, ent/noent; controles `frozen_onehot_mlp`, `frozen_matched_scalar`, `frozen_scalar_mlp_large` | 10 |
| exp05 | FrozenLake | Techo clásico aditivo (Fourier) de Config B | `frozen_binary_4q_fourier_ceiling` (20 params) | 10 (en el directorio de exp04) |
| exp06 | CartPole | Techo clásico aditivo de la red de exp01 | `cartpole_fourier_ceiling_L5` (162 params) | 10 (en el directorio de exp01) |
| exp07 | CartPole | Réplica literal de los 24 brazos del paper bajo DQN | `paper_*` | 10 |
| exp08 | CartPole + FrozenLake | Cuatro huecos: ablación sin entrelazar de exp01/02; cruce DR×OR; techos Fourier de Config A; completar Hsiao | `hybrid_fig4_noent`, `hybrid_or_r*_noent`, `hybrid_DR{1,2}_OR*`, `frozen_scalar_1q_fourier_ceiling_L*` | 10 |

Los resultados de exp05 y de exp08 (nivel 0) se guardan en el directorio de exp04; los de exp06, en el de exp01. En total son **888 runs** (excluidos los de simulador), todos de 100 000 pasos salvo el piloto de 60k.

**Nota sobre exp06.** Su diseño original comparaba el techo clásico contra un emulador del circuito sin entrelazar. Esa comparación se hace aquí contra el **circuito real** `hybrid_fig4_noent` (exp08), que la sustituye, y el emulador queda fuera del paquete (regla 4).


## 2b. Procedencia de los hiperparámetros DQN (cómo se eligieron las configuraciones)

**Resumen para el documento final:** no hay ninguna búsqueda ni ajuste de hiperparámetros documentado en el repositorio. En CartPole se heredaron los valores por defecto de la librería; en FrozenLake se razonaron a priori, con una única ejecución de humo previa. Dentro de cada experimento todos los brazos comparten los mismos hiperparámetros; los desvíos solo ocurren entre entornos.

| entorno / experimentos | `batch_size` | `buffer_size` | `train_frequency` | `learning_starts` | origen |
|---|---|---|---|---|---|
| CartPole: exp01, 02, 03, 03b, 07, 08 | 128 | 10 000 | 10 | 10 000 (por defecto, no se pasa) | valores por defecto del DQN de SimplyQRL (`src/simplyqrl/dqn.py:67-77`); exp01 los repite como defaults de argparse (`experiments/exp01_dqn_cartpole_capacity.py:78-80`); exp02/exp03 los fijan a mano (`exp02_...py:119`, `exp03_...py:90`) |
| FrozenLake: exp04, exp05 | 128 | 50 000 | 1 | 1 000 | decisión deliberada documentada en comentarios (`experiments/exp04_dqn_frozenlake_embeddings.py:79-91`) |

**Valores que nunca se han variado** (por defecto de la librería): tasa de aprendizaje 2,5e-4, γ=0,99, `target_network_frequency`=500, τ=1,0, ε de 1,0 a 0,05 durante el 50 % del entrenamiento.

**Razones declaradas para los desvíos de FrozenLake** (literal, `exp04_...py:79-91`):
- `buffer_size` 50k rather than 10k: "with one bit of reward, a buffer that forgets its rare successful trajectories starves the only signal there is."
- `train_frequency` 1 rather than 10: "exp01 established that train_frequency=10 gradient-starves small networks enough to mask every other effect under study."
- `learning_starts` 1k rather than 10k: "at these episode lengths 10k steps is a large fraction of the run spent not learning."

**Única prueba previa a esa decisión:** una ejecución de humo (1 semilla, 5 000 pasos) que solo comprobó que el control one-hot estaba vivo (`exp04_...py:98-104`); no se ajustó nada por brazo.

**Lo único explorado en CartPole** (`docs/EXPERIMENT-01.md:53-60`, 3 semillas): MLP 120-84 con batch 16 y 128, una de 60k pasos, y el brazo lineal del paper con `train_frequency` 10 y 1 (muerto en ambos). El objetivo era comprobar si FIX-01 se podía medir sobre un agente vivo, no elegir hiperparámetros. `docs/EXPERIMENT-01.md:203` deja como pendiente reproducir la configuración numpy que funcionaba (buffer 50k, `train_frequency` 1, batch 16, Double DQN, recorte de gradiente) y compararla con `dqn.py`; no he encontrado evidencia de que se hiciera.

**Dos puntos que el documento final debe señalar:**
1. **NO DETERMINABLE:** no se encuentra en el repositorio la evidencia de que "exp01 established that train_frequency=10 gradient-starves small networks". La tabla de exp01 solo muestra el brazo lineal muerto con `train_frequency` 10 y también con 1, lo que no demuestra esa afirmación.
2. **Posible inconsistencia:** si esa razón fuera cierta, también aplicaría a CartPole, pero los barridos de CartPole con redes pequeñas (exp02, 03, 07, 08) siguen con `train_frequency=10`. Podría explicar parte de los brazos muertos de exp07; es una hipótesis, no un resultado.

**Consecuencia para las conclusiones:** un brazo muerto puede ser un brazo mal ajustado, no un brazo incapaz. Por eso los resultados negativos (en especial exp07, F8) deben redactarse como "con estos hiperparámetros", no como "el circuito no puede". **DECISIÓN A TOMAR:** informar sin ajuste (con esa limitación explícita) o dedicar un barrido de hiperparámetros (incluida la tasa de aprendizaje, nunca variada).

**Contraste accidental, único dato de sensibilidad a hiperparámetros que existe (FIX-13, resuelto):** los techos Fourier de Config A se entrenaron por error con los hiperparámetros de CartPole (buffer 10 000, `train_frequency` 10, `learning_starts` 10 000) y se han reentrenado con los de FrozenLake (etiqueta `flkw`), con las mismas arquitecturas y semillas. Resultado: a L=10 y L=15 los dos ajustes son equivalentes (≈0,95 de éxito, todas las semillas vivas); **solo a L=5 importaron**: con los de CartPole 7 de 10 semillas vivas en greedy (`final_perf` 0,666), con los de FrozenLake 0 de 10 (0,103). Los resultados antiguos se **conservan** como pista de una configuración prometedora para un posible barrido sobre el modelo clásico (techo Fourier), con tres límites: selección a posteriori (tratarla como exploratoria), válida solo para el modelo clásico (no probada en el circuito) y no se sabe cuál de los tres ajustes pesa (buffer, frecuencia de entrenamiento o arranque tardío).


## 3. Per-experiment tables

Columns: `final_perf` = mean return over the last 10 % of training episodes (unbiased, includes epsilon-greedy noise); `greedy_final` = mean of the last 3 greedy-evaluation checkpoints (epsilon=0; 20 episodes each on CartPole, 1 effective sample each on deterministic FrozenLake); `best_ma50` and `final_ma50` = the paper's metrics (max / last value of the 50-episode trailing mean). IQM = interquartile mean over seeds; CI = 95 % bootstrap. `params` = trainable parameters of the Q-network built for that arm (`-` = not recomputable from the registry, see docs).

### 3.1 exp01 — capacity ladder on CartPole (100k steps; FIX-01 on and off)

| arm | FIX-01 | params | n | final_perf IQM (CI) | greedy_final IQM (CI) | best_ma50 mean ± sd | final_ma50 mean ± sd | h/seed |
|---|---|---|---|---|---|---|---|---|
| `cartpole_fourier_ceiling_L5` | on | 162 | 10 | 76.5 (69.6-94.0) | 84.7 (77.1-104.7) | 104.4 ± 16.9 | 103.6 ± 16.3 | 0.00 |
| `hybrid_fig4` | off | 126 | 10 | 19.9 (13.3-32.7) | 28.5 (17.8-46.5) | 45.5 ± 17.2 | 30.7 ± 18.2 | 9.08 |
| `hybrid_fig4` | on | 126 | 10 | 35.8 (24.0-48.1) | 47.3 (31.2-67.8) | 69.6 ± 23.1 | 42.6 ± 14.1 | 9.40 |
| `matched_classical` | off | - | 10 | 9.9 (9.7-10.3) | 9.6 (9.5-10.0) | 26.8 ± 3.7 | 10.7 ± 2.0 | 0.02 |
| `matched_classical` | on | - | 10 | 11.8 (10.7-12.6) | 11.1 (10.1-12.5) | 29.2 ± 6.1 | 13.0 ± 3.8 | 0.02 |
| `oversized_mlp` | off | 10934 | 10 | 198.3 (166.4-231.5) | 212.0 (165.8-292.6) | 301.8 ± 37.5 | 210.9 ± 44.7 | 0.01 |
| `oversized_mlp` | on | 10934 | 10 | 252.0 (234.8-279.4) | 266.0 (245.3-314.8) | 320.8 ± 27.2 | 272.7 ± 27.0 | 0.01 |
| `paper_linear` | off | 26 | 10 | 9.6 (9.6-9.7) | 9.5 (9.3-9.7) | 26.7 ± 5.8 | 9.6 ± 0.3 | 0.01 |
| `paper_linear` | on | 26 | 10 | 9.6 (9.6-9.7) | 9.5 (9.3-9.7) | 26.3 ± 3.2 | 9.7 ± 0.2 | 0.01 |

exp01 60k-step pilot (3 seeds, superseded by the 100k table above, kept for traceability):

| arm | FIX-01 | params | n | final_perf IQM (CI) | greedy_final IQM (CI) | best_ma50 mean ± sd | final_ma50 mean ± sd | h/seed |
|---|---|---|---|---|---|---|---|---|
| `matched_classical` | off | - | 3 | 9.6 (9.6-9.6) | 9.4 (9.0-9.6) | 36.3 ± 2.0 | 9.6 ± 0.1 | 0.01 |
| `matched_classical` | on | - | 3 | 9.7 (9.6-9.7) | 9.4 (9.0-9.6) | 38.0 ± 0.8 | 9.7 ± 0.1 | 0.01 |
| `oversized_mlp` | off | 10934 | 3 | 247.6 (174.4-326.8) | 269.6 (183.0-383.4) | 323.1 ± 131.0 | 200.1 ± 26.7 | 0.01 |
| `oversized_mlp` | on | 10934 | 3 | 182.1 (110.4-310.9) | 267.9 (229.4-318.4) | 261.4 ± 80.4 | 201.7 ± 97.1 | 0.01 |
| `paper_linear` | off | 26 | 3 | 9.8 (9.6-10.1) | 9.8 (9.6-10.2) | 23.8 ± 1.7 | 9.8 ± 0.2 | 0.01 |
| `paper_linear` | on | 26 | 3 | 9.8 (9.7-10.0) | 9.8 (9.6-10.2) | 24.9 ± 0.9 | 9.8 ± 0.2 | 0.01 |

### 3.2 exp02 — Output Reuse (OR) on the 8-qubit Skolik circuit, CartPole

| arm | FIX-01 | params | n | final_perf IQM (CI) | greedy_final IQM (CI) | best_ma50 mean ± sd | final_ma50 mean ± sd | h/seed |
|---|---|---|---|---|---|---|---|---|
| `classical_OR16` | on | - | 10 | 9.6 (9.6-27.6) | 9.6 (9.4-28.6) | 56.0 ± 95.5 | 19.4 ± 30.6 | 0.01 |
| `classical_OR32` | on | - | 10 | 10.2 (10.2-10.2) | 10.0 (10.0-10.1) | 26.3 ± 4.6 | 10.2 ± 0.1 | 0.01 |
| `classical_OR4` | on | - | 10 | 9.6 (9.6-9.7) | 9.5 (9.3-9.7) | 26.6 ± 3.9 | 9.7 ± 0.2 | 0.01 |
| `classical_OR8` | on | - | 10 | 9.6 (9.6-9.6) | 9.5 (9.3-9.6) | 28.7 ± 7.5 | 9.6 ± 0.2 | 0.01 |
| `classical_or_matched_r16` | on | - | 10 | 36.3 (28.6-43.8) | 43.0 (32.6-59.9) | 68.1 ± 14.6 | 51.3 ± 26.6 | 0.02 |
| `classical_or_matched_r32` | on | - | 10 | 61.2 (46.1-71.7) | 88.5 (62.5-116.9) | 121.8 ± 36.2 | 93.1 ± 34.9 | 0.01 |
| `classical_or_matched_r4` | on | - | 10 | 12.4 (11.5-15.6) | 13.8 (10.8-17.5) | 33.2 ± 9.6 | 19.3 ± 11.4 | 0.02 |
| `classical_or_matched_r8` | on | - | 10 | 23.8 (20.5-27.8) | 24.8 (17.9-37.0) | 54.2 ± 14.2 | 27.3 ± 10.8 | 0.02 |
| `hybrid_OR16` | on | 606 | 10 | 104.2 (77.1-123.1) | 146.2 (109.8-180.0) | 167.6 ± 79.9 | 102.3 ± 33.1 | 9.60 |
| `hybrid_OR32` | on | 1118 | 10 | 137.6 (106.0-179.6) | 161.1 (119.6-223.9) | 214.1 ± 56.4 | 160.7 ± 77.9 | 8.10 |
| `hybrid_OR4` | on | 222 | 10 | 94.7 (81.0-111.9) | 112.0 (91.4-153.7) | 150.2 ± 49.0 | 102.9 ± 52.7 | 9.08 |
| `hybrid_OR8` | on | 350 | 10 | 109.8 (93.9-124.9) | 124.3 (107.2-189.7) | 138.2 ± 39.0 | 110.8 ± 19.9 | 8.85 |

### 3.3 exp03 / exp03b — Data Reuploading depth, entangled vs unentangled, CartPole

| arm | FIX-01 | params | n | final_perf IQM (CI) | greedy_final IQM (CI) | best_ma50 mean ± sd | final_ma50 mean ± sd | h/seed |
|---|---|---|---|---|---|---|---|---|
| `hybrid_DR1` | on | 62 | 10 | 10.1 (9.9-10.5) | 9.8 (9.3-10.7) | 25.3 ± 1.2 | 11.1 ± 1.9 | 2.41 |
| `hybrid_DR2` | on | 78 | 10 | 30.2 (23.8-37.2) | 40.2 (29.9-51.7) | 47.4 ± 14.4 | 32.7 ± 11.1 | 3.63 |
| `hybrid_DR5` | on | 126 | 10 | 37.7 (24.6-49.7) | 73.7 (38.4-108.3) | 68.9 ± 22.9 | 45.1 ± 22.0 | 9.08 |

| arm | FIX-01 | params | n | final_perf IQM (CI) | greedy_final IQM (CI) | best_ma50 mean ± sd | final_ma50 mean ± sd | h/seed |
|---|---|---|---|---|---|---|---|---|
| `hybrid_DR1_noent` | on | 62 | 10 | 10.0 (9.7-11.7) | 9.8 (9.6-12.2) | 26.8 ± 1.7 | 10.9 ± 2.9 | 2.28 |
| `hybrid_DR2_noent` | on | 78 | 10 | 34.5 (28.7-41.1) | 35.0 (29.5-54.6) | 52.7 ± 9.8 | 44.8 ± 13.0 | 3.80 |
| `hybrid_DR5_noent` | on | 126 | 10 | 41.8 (38.2-50.6) | 47.3 (41.6-104.5) | 69.0 ± 31.9 | 49.4 ± 27.3 | 7.67 |

### 3.4 exp04 / exp05 / exp08(tier 0) — FrozenLake 4x4 (deterministic), metric = success rate

Rows ending in `__flkw` are the CONTROLLED reruns of the Config A ceilings (FrozenLake DQN kwargs, n=10); the same arm without the suffix is the original run, which actually used CartPole kwargs although its manifest declares FrozenLake's (FIX-13).

| arm | FIX-01 | params | n | final_perf IQM (CI) | greedy_final IQM (CI) | best_ma50 mean ± sd | final_ma50 mean ± sd | h/seed |
|---|---|---|---|---|---|---|---|---|
| `frozen_binary_4q_L1` | on | 28 | 10 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0.060 ± 0.021 | 0.000 ± 0.000 | 7.13 |
| `frozen_binary_4q_L5` | on | 60 | 10 | 0.952 (0.806-0.957) | 0.944 (0.667-1.000) | 0.952 ± 0.152 | 0.894 ± 0.195 | 25.49 |
| `frozen_binary_4q_fourier_ceiling` | on | 20 | 10 | 0.203 (0.021-0.377) | 0.000 (0.000-0.000) | 0.526 ± 0.380 | 0.142 ± 0.148 | 0.02 |
| `frozen_binary_4q_noent_L1` | on | 28 | 10 | 0.000 (0.000-0.001) | 0.000 (0.000-0.000) | 0.060 ± 0.016 | 0.000 ± 0.000 | 7.15 |
| `frozen_binary_4q_noent_L5` | on | 60 | 10 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0.062 ± 0.015 | 0.000 ± 0.000 | 22.76 |
| `frozen_matched_scalar` | off | - | 10 | 0.000 (0.000-0.003) | 0.000 (0.000-0.000) | 0.080 ± 0.034 | 0.002 ± 0.006 | 0.06 |
| `frozen_matched_scalar` | on | - | 10 | 0.001 (0.000-0.013) | 0.000 (0.000-0.000) | 0.096 ± 0.062 | 0.002 ± 0.006 | 0.06 |
| `frozen_onehot_mlp` | off | 4548 | 10 | 0.952 (0.948-0.957) | 1.000 (1.000-1.000) | 1.000 ± 0.000 | 0.944 ± 0.026 | 0.09 |
| `frozen_onehot_mlp` | on | 4548 | 10 | 0.949 (0.946-0.955) | 1.000 (1.000-1.000) | 1.000 ± 0.000 | 0.948 ± 0.038 | 0.10 |
| `frozen_scalar_1q_L1` | on | 10 | 10 | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0.078 ± 0.020 | 0.000 ± 0.000 | 6.06 |
| `frozen_scalar_1q_L10` | on | 28 | 10 | 0.081 (0.007-0.228) | 0.000 (0.000-0.056) | 0.452 ± 0.350 | 0.122 ± 0.306 | 12.66 |
| `frozen_scalar_1q_L15` | on | 38 | 10 | 0.011 (0.006-0.078) | 0.000 (0.000-0.000) | 0.190 ± 0.135 | 0.048 ± 0.063 | 17.63 |
| `frozen_scalar_1q_L5` | on | 18 | 10 | 0.125 (0.024-0.253) | 0.000 (0.000-0.000) | 0.420 ± 0.315 | 0.262 ± 0.235 | 7.83 |
| `frozen_scalar_1q_fourier_ceiling_L1` | on | 12 | 10 | 0.000 (0.000-0.001) | 0.000 (0.000-0.000) | 0.106 ± 0.063 | 0.000 ± 0.000 | 0.02 |
| `frozen_scalar_1q_fourier_ceiling_L10` | on | 84 | 10 | 0.949 (0.788-0.957) | 0.944 (0.722-1.000) | 0.906 ± 0.297 | 0.848 ± 0.299 | 0.02 |
| `frozen_scalar_1q_fourier_ceiling_L10__flkw` | on | 84 | 10 | 0.949 (0.944-0.954) | 1.000 (1.000-1.000) | 1.000 ± 0.000 | 0.952 ± 0.025 | 0.02 |
| `frozen_scalar_1q_fourier_ceiling_L15` | on | 124 | 10 | 0.949 (0.943-0.955) | 1.000 (0.944-1.000) | 1.000 ± 0.000 | 0.956 ± 0.018 | 0.02 |
| `frozen_scalar_1q_fourier_ceiling_L15__flkw` | on | 124 | 10 | 0.950 (0.946-0.955) | 1.000 (1.000-1.000) | 1.000 ± 0.000 | 0.948 ± 0.029 | 0.02 |
| `frozen_scalar_1q_fourier_ceiling_L1__flkw` | on | 12 | 10 | 0.004 (0.003-0.009) | 0.000 (0.000-0.000) | 0.122 ± 0.024 | 0.008 ± 0.014 | 0.02 |
| `frozen_scalar_1q_fourier_ceiling_L5` | on | 44 | 10 | 0.666 (0.245-0.933) | 0.389 (0.111-0.722) | 0.726 ± 0.441 | 0.642 ± 0.433 | 0.02 |
| `frozen_scalar_1q_fourier_ceiling_L5__flkw` | on | 44 | 10 | 0.103 (0.081-0.124) | 0.000 (0.000-0.000) | 0.768 ± 0.060 | 0.000 ± 0.000 | 0.02 |
| `frozen_scalar_mlp_large` | off | 4548 | 10 | 0.067 (0.034-0.137) | 0.000 (0.000-0.000) | 0.530 ± 0.174 | 0.072 ± 0.078 | 0.09 |
| `frozen_scalar_mlp_large` | on | 4548 | 10 | 0.050 (0.023-0.081) | 0.000 (0.000-0.000) | 0.444 ± 0.108 | 0.038 ± 0.062 | 0.09 |

### 3.5 exp06 — additive Fourier ceiling on CartPole (rows `cartpole_fourier_ceiling_L5` in table 3.1; compared against the real circuit `hybrid_fig4_noent` of exp08)

### 3.6 exp07 — the paper's literal replication arms under DQN (CartPole; n=10)

| arm | FIX-01 | params | n | final_perf IQM (CI) | greedy_final IQM (CI) | best_ma50 mean ± sd | final_ma50 mean ± sd | h/seed |
|---|---|---|---|---|---|---|---|---|
| `paper_classical_or_r16` | on | 98 | 10 | 9.6 (9.6-25.3) | 9.6 (9.5-28.4) | 44.4 ± 60.9 | 19.0 ± 29.3 | 0.01 |
| `paper_classical_or_r32` | on | 194 | 10 | 10.3 (10.2-10.3) | 10.0 (10.0-10.0) | 25.7 ± 2.8 | 10.3 ± 0.1 | 0.02 |
| `paper_classical_or_r4` | on | 26 | 10 | 9.6 (9.6-9.7) | 9.5 (9.4-9.6) | 27.4 ± 5.5 | 9.6 ± 0.2 | 0.02 |
| `paper_classical_or_r8` | on | 50 | 10 | 9.6 (9.6-9.6) | 9.5 (9.4-9.5) | 27.8 ± 5.2 | 9.6 ± 0.1 | 0.02 |
| `paper_hsiao_or_r16` | on | 134 | 10 | 10.0 (9.9-10.9) | 9.8 (9.7-10.6) | 31.7 ± 11.9 | 10.9 ± 3.3 | 1.55 |
| `paper_hsiao_or_r16_ent` | on | 134 | 10 | 9.6 (9.6-9.6) | 9.5 (9.4-9.6) | 25.4 ± 1.4 | 9.6 ± 0.2 | 1.78 |
| `paper_hsiao_or_r32` | on | 262 | 10 | 10.0 (10.0-29.6) | 9.9 (9.9-33.9) | 56.9 ± 52.5 | 17.9 ± 16.7 | 1.60 |
| `paper_hsiao_or_r4` | on | 38 | 10 | 9.6 (9.6-9.7) | 9.5 (9.4-9.6) | 24.8 ± 0.9 | 9.6 ± 0.2 | 1.62 |
| `paper_hsiao_or_r4_ent` | on | 38 | 10 | 9.6 (9.6-9.6) | 9.5 (9.4-9.5) | 25.7 ± 1.3 | 9.6 ± 0.1 | 1.73 |
| `paper_hsiao_or_r8` | on | 70 | 10 | 9.8 (9.7-9.9) | 9.7 (9.6-9.8) | 64.0 ± 115.6 | 9.8 ± 0.2 | 1.50 |
| `paper_salinas_1q_L1` | on | 7 | 10 | 9.6 (9.6-9.6) | 9.5 (9.4-9.6) | 28.3 ± 9.3 | 9.6 ± 0.1 | 0.98 |
| `paper_salinas_1q_L2` | on | 10 | 10 | 9.6 (9.6-9.6) | 9.5 (9.4-9.6) | 25.8 ± 0.8 | 9.7 ± 0.1 | 1.43 |
| `paper_salinas_1q_L5` | on | 19 | 10 | 9.6 (9.6-9.7) | 9.5 (9.5-9.6) | 26.7 ± 2.5 | 9.6 ± 0.2 | 2.80 |
| `paper_salinas_2q_L1` | on | 12 | 10 | 9.6 (9.6-9.6) | 9.5 (9.4-9.5) | 28.2 ± 10.7 | 9.5 ± 0.2 | 1.19 |
| `paper_salinas_2q_L2` | on | 18 | 10 | 9.6 (9.6-9.7) | 9.5 (9.4-9.6) | 28.3 ± 5.7 | 9.6 ± 0.1 | 2.11 |
| `paper_salinas_2q_L5` | on | 36 | 10 | 9.6 (9.6-9.6) | 9.5 (9.4-9.5) | 26.1 ± 2.4 | 9.6 ± 0.1 | 4.09 |
| `paper_skolik_4q_L1` | on | 18 | 10 | 9.6 (9.6-10.7) | 9.5 (9.4-10.4) | 27.3 ± 4.0 | 10.4 ± 2.3 | 1.43 |
| `paper_skolik_4q_L2` | on | 26 | 10 | 9.6 (9.6-9.7) | 9.4 (9.4-9.5) | 25.2 ± 1.1 | 9.7 ± 0.2 | 2.22 |
| `paper_skolik_4q_L2_noent` | on | 26 | 10 | 9.7 (9.6-10.3) | 9.5 (9.4-9.8) | 28.0 ± 7.8 | 10.2 ± 0.9 | 2.05 |
| `paper_skolik_4q_L5` | on | 50 | 10 | 9.6 (9.6-9.6) | 9.5 (9.4-9.6) | 25.6 ± 2.1 | 9.6 ± 0.3 | 4.85 |
| `paper_skolik_4q_L5_noent` | on | 50 | 10 | 9.7 (9.6-9.9) | 9.5 (9.5-9.9) | 25.7 ± 0.9 | 10.2 ± 1.3 | 3.93 |
| `paper_skolik_8q_L1` | on | 34 | 10 | 10.9 (10.2-12.1) | 10.4 (9.8-13.5) | 30.6 ± 5.6 | 12.3 ± 3.8 | 2.40 |
| `paper_skolik_8q_L2` | on | 50 | 10 | 12.9 (10.8-16.1) | 12.4 (10.5-17.9) | 33.2 ± 10.4 | 13.9 ± 7.1 | 4.15 |
| `paper_skolik_8q_L5` | on | 98 | 10 | 19.8 (14.5-29.3) | 24.8 (15.0-42.7) | 53.9 ± 22.0 | 28.4 ± 19.4 | 10.04 |

### 3.7 exp08 — four follow-up gaps (CartPole; n=10)

| arm | FIX-01 | params | n | final_perf IQM (CI) | greedy_final IQM (CI) | best_ma50 mean ± sd | final_ma50 mean ± sd | h/seed |
|---|---|---|---|---|---|---|---|---|
| `hybrid_DR1_OR16` | on | 542 | 10 | 11.6 (11.2-12.1) | 12.4 (9.5-15.8) | 25.3 ± 1.3 | 11.3 ± 1.6 | 2.51 |
| `hybrid_DR1_OR32` | on | 1054 | 10 | 10.8 (10.4-11.2) | 10.7 (9.6-13.8) | 25.8 ± 1.2 | 11.0 ± 1.7 | 2.38 |
| `hybrid_DR1_OR4` | on | 158 | 10 | 10.6 (10.2-11.0) | 9.9 (9.5-13.2) | 25.9 ± 1.8 | 9.9 ± 0.7 | 2.49 |
| `hybrid_DR1_OR8` | on | 286 | 10 | 11.2 (10.9-11.5) | 11.5 (9.8-15.0) | 26.0 ± 0.5 | 11.8 ± 2.4 | 2.54 |
| `hybrid_DR2_OR16` | on | 558 | 10 | 67.9 (52.4-79.2) | 75.8 (53.5-91.3) | 88.5 ± 25.4 | 70.8 ± 23.0 | 3.81 |
| `hybrid_DR2_OR32` | on | 1070 | 10 | 77.1 (68.7-80.5) | 91.2 (81.1-97.7) | 93.6 ± 19.4 | 75.8 ± 11.3 | 4.04 |
| `hybrid_DR2_OR4` | on | 174 | 10 | 44.6 (35.6-54.6) | 50.1 (40.0-66.2) | 63.2 ± 20.6 | 44.0 ± 16.6 | 4.19 |
| `hybrid_DR2_OR8` | on | 302 | 10 | 54.6 (43.5-65.3) | 69.5 (46.8-88.9) | 72.3 ± 14.8 | 55.5 ± 18.3 | 4.31 |
| `hybrid_fig4_noent` | on | 126 | 10 | 43.5 (39.7-51.2) | 46.4 (38.2-81.1) | 65.8 ± 21.2 | 50.6 ± 21.4 | 8.53 |
| `hybrid_or_r16_noent` | on | 606 | 10 | 89.5 (72.6-103.5) | 103.9 (83.9-123.2) | 119.7 ± 23.9 | 97.6 ± 33.9 | 8.22 |
| `hybrid_or_r32_noent` | on | 1118 | 10 | 77.9 (64.7-91.6) | 96.6 (80.0-106.3) | 111.4 ± 29.3 | 93.4 ± 30.7 | 7.32 |
| `hybrid_or_r4_noent` | on | 222 | 10 | 60.4 (46.5-74.5) | 65.4 (50.7-83.8) | 90.9 ± 16.1 | 66.0 ± 28.3 | 8.54 |
| `hybrid_or_r8_noent` | on | 350 | 10 | 69.1 (55.0-82.6) | 86.9 (64.3-106.8) | 110.5 ± 35.4 | 76.4 ± 31.5 | 8.23 |

## 4. Pre-specified contrasts (P(a>b) = probability of improvement, unpaired, over seeds)

P(a>b) = fraction of (seed_a, seed_b) pairs with a larger than b (ties count half). 0.5 = no information; values near 0 or 1 = near-complete separation. n is per arm.

### 4.1 CartPole — is the quantum circuit better than classical baselines?

| contrast (a vs b) | n_a / n_b | metric | IQM a | IQM b | P(a>b) |
|---|---|---|---|---|---|
| `hybrid_fig4` (exp01, FIX-01 on) vs `matched_classical` (exp01, on) | 10 / 10 | final_perf | 35.8 | 11.8 | 1.00 |
| `hybrid_fig4` (exp01, FIX-01 on) vs `matched_classical` (exp01, on) | 10 / 10 | greedy_final | 47.3 | 11.1 | 1.00 |
| `hybrid_fig4` (exp01, FIX-01 on) vs `paper_linear` (exp01, on) | 10 / 10 | final_perf | 35.8 | 9.6 | 1.00 |
| `hybrid_fig4` (exp01, FIX-01 on) vs `paper_linear` (exp01, on) | 10 / 10 | greedy_final | 47.3 | 9.5 | 1.00 |
| `oversized_mlp` (exp01, FIX-01 on) vs `hybrid_fig4` (exp01, on) | 10 / 10 | final_perf | 252.0 | 35.8 | 1.00 |
| `oversized_mlp` (exp01, FIX-01 on) vs `hybrid_fig4` (exp01, on) | 10 / 10 | greedy_final | 266.0 | 47.3 | 1.00 |
| `cartpole_fourier_ceiling_L5` (exp01, FIX-01 on) vs `hybrid_fig4` (exp01, on) | 10 / 10 | final_perf | 76.5 | 35.8 | 1.00 |
| `cartpole_fourier_ceiling_L5` (exp01, FIX-01 on) vs `hybrid_fig4` (exp01, on) | 10 / 10 | greedy_final | 84.7 | 47.3 | 0.87 |

### 4.2 CartPole — does entanglement matter? (entangled vs the SAME circuit with ent=False)

| contrast (a vs b) | n_a / n_b | metric | IQM a | IQM b | P(a>b) |
|---|---|---|---|---|---|
| `hybrid_fig4` (exp01, FIX-01 on) vs `hybrid_fig4_noent` (exp08, on) | 10 / 10 | final_perf | 35.8 | 43.5 | 0.33 |
| `hybrid_fig4` (exp01, FIX-01 on) vs `hybrid_fig4_noent` (exp08, on) | 10 / 10 | greedy_final | 47.3 | 46.4 | 0.46 |
| `hybrid_OR4` (exp02, FIX-01 on) vs `hybrid_or_r4_noent` (exp08, on) | 10 / 10 | final_perf | 94.7 | 60.4 | 0.90 |
| `hybrid_OR4` (exp02, FIX-01 on) vs `hybrid_or_r4_noent` (exp08, on) | 10 / 10 | greedy_final | 112.0 | 65.4 | 0.89 |
| `hybrid_OR8` (exp02, FIX-01 on) vs `hybrid_or_r8_noent` (exp08, on) | 10 / 10 | final_perf | 109.8 | 69.1 | 0.90 |
| `hybrid_OR8` (exp02, FIX-01 on) vs `hybrid_or_r8_noent` (exp08, on) | 10 / 10 | greedy_final | 124.3 | 86.9 | 0.86 |
| `hybrid_OR16` (exp02, FIX-01 on) vs `hybrid_or_r16_noent` (exp08, on) | 10 / 10 | final_perf | 104.2 | 89.5 | 0.64 |
| `hybrid_OR16` (exp02, FIX-01 on) vs `hybrid_or_r16_noent` (exp08, on) | 10 / 10 | greedy_final | 146.2 | 103.9 | 0.78 |
| `hybrid_OR32` (exp02, FIX-01 on) vs `hybrid_or_r32_noent` (exp08, on) | 10 / 10 | final_perf | 137.6 | 77.9 | 0.92 |
| `hybrid_OR32` (exp02, FIX-01 on) vs `hybrid_or_r32_noent` (exp08, on) | 10 / 10 | greedy_final | 161.1 | 96.6 | 0.89 |
| `hybrid_DR1` (exp03, FIX-01 on) vs `hybrid_DR1_noent` (exp03b, on) | 10 / 10 | final_perf | 10.1 | 10.0 | 0.60 |
| `hybrid_DR1` (exp03, FIX-01 on) vs `hybrid_DR1_noent` (exp03b, on) | 10 / 10 | greedy_final | 9.8 | 9.8 | 0.38 |
| `hybrid_DR2` (exp03, FIX-01 on) vs `hybrid_DR2_noent` (exp03b, on) | 10 / 10 | final_perf | 30.2 | 34.5 | 0.35 |
| `hybrid_DR2` (exp03, FIX-01 on) vs `hybrid_DR2_noent` (exp03b, on) | 10 / 10 | greedy_final | 40.2 | 35.0 | 0.57 |
| `hybrid_DR5` (exp03, FIX-01 on) vs `hybrid_DR5_noent` (exp03b, on) | 10 / 10 | final_perf | 37.7 | 41.8 | 0.45 |
| `hybrid_DR5` (exp03, FIX-01 on) vs `hybrid_DR5_noent` (exp03b, on) | 10 / 10 | greedy_final | 73.7 | 47.3 | 0.53 |

### 4.3 CartPole — Output Reuse and Data Reuploading effects

| contrast (a vs b) | n_a / n_b | metric | IQM a | IQM b | P(a>b) |
|---|---|---|---|---|---|
| `hybrid_OR4` (exp02, FIX-01 on) vs `hybrid_fig4` (exp01, on) | 10 / 10 | final_perf | 94.7 | 35.8 | 0.97 |
| `hybrid_OR4` (exp02, FIX-01 on) vs `hybrid_fig4` (exp01, on) | 10 / 10 | greedy_final | 112.0 | 47.3 | 0.96 |
| `hybrid_OR8` (exp02, FIX-01 on) vs `hybrid_fig4` (exp01, on) | 10 / 10 | final_perf | 109.8 | 35.8 | 1.00 |
| `hybrid_OR8` (exp02, FIX-01 on) vs `hybrid_fig4` (exp01, on) | 10 / 10 | greedy_final | 124.3 | 47.3 | 0.99 |
| `hybrid_OR16` (exp02, FIX-01 on) vs `hybrid_fig4` (exp01, on) | 10 / 10 | final_perf | 104.2 | 35.8 | 0.96 |
| `hybrid_OR16` (exp02, FIX-01 on) vs `hybrid_fig4` (exp01, on) | 10 / 10 | greedy_final | 146.2 | 47.3 | 0.96 |
| `hybrid_OR32` (exp02, FIX-01 on) vs `hybrid_fig4` (exp01, on) | 10 / 10 | final_perf | 137.6 | 35.8 | 1.00 |
| `hybrid_OR32` (exp02, FIX-01 on) vs `hybrid_fig4` (exp01, on) | 10 / 10 | greedy_final | 161.1 | 47.3 | 0.99 |
| `hybrid_OR4` (exp02, FIX-01 on) vs `classical_or_matched_r4` (exp02, on) | 10 / 10 | final_perf | 94.7 | 12.4 | 1.00 |
| `hybrid_OR4` (exp02, FIX-01 on) vs `classical_or_matched_r4` (exp02, on) | 10 / 10 | greedy_final | 112.0 | 13.8 | 1.00 |
| `hybrid_OR16` (exp02, FIX-01 on) vs `classical_or_matched_r16` (exp02, on) | 10 / 10 | final_perf | 104.2 | 36.3 | 0.95 |
| `hybrid_OR16` (exp02, FIX-01 on) vs `classical_or_matched_r16` (exp02, on) | 10 / 10 | greedy_final | 146.2 | 43.0 | 0.98 |
| `hybrid_OR32` (exp02, FIX-01 on) vs `classical_or_matched_r32` (exp02, on) | 10 / 10 | final_perf | 137.6 | 61.2 | 0.99 |
| `hybrid_OR32` (exp02, FIX-01 on) vs `classical_or_matched_r32` (exp02, on) | 10 / 10 | greedy_final | 161.1 | 88.5 | 0.88 |
| `hybrid_DR2` (exp03, FIX-01 on) vs `hybrid_DR1` (exp03, on) | 10 / 10 | final_perf | 30.2 | 10.1 | 1.00 |
| `hybrid_DR2` (exp03, FIX-01 on) vs `hybrid_DR1` (exp03, on) | 10 / 10 | greedy_final | 40.2 | 9.8 | 1.00 |
| `hybrid_DR5` (exp03, FIX-01 on) vs `hybrid_DR2` (exp03, on) | 10 / 10 | final_perf | 37.7 | 30.2 | 0.63 |
| `hybrid_DR5` (exp03, FIX-01 on) vs `hybrid_DR2` (exp03, on) | 10 / 10 | greedy_final | 73.7 | 40.2 | 0.70 |
| `hybrid_DR2_OR4` (exp08, FIX-01 on) vs `hybrid_DR2` (exp03, on) | 10 / 10 | final_perf | 44.6 | 30.2 | 0.78 |
| `hybrid_DR2_OR4` (exp08, FIX-01 on) vs `hybrid_DR2` (exp03, on) | 10 / 10 | greedy_final | 50.1 | 40.2 | 0.71 |
| `hybrid_DR2_OR32` (exp08, FIX-01 on) vs `hybrid_DR2` (exp03, on) | 10 / 10 | final_perf | 77.1 | 30.2 | 0.99 |
| `hybrid_DR2_OR32` (exp08, FIX-01 on) vs `hybrid_DR2` (exp03, on) | 10 / 10 | greedy_final | 91.2 | 40.2 | 0.96 |
| `hybrid_DR2_OR32` (exp08, FIX-01 on) vs `hybrid_OR32` (exp02, on) | 10 / 10 | final_perf | 77.1 | 137.6 | 0.02 |
| `hybrid_DR2_OR32` (exp08, FIX-01 on) vs `hybrid_OR32` (exp02, on) | 10 / 10 | greedy_final | 91.2 | 161.1 | 0.05 |
| `hybrid_DR1_OR32` (exp08, FIX-01 on) vs `hybrid_DR1` (exp03, on) | 10 / 10 | final_perf | 10.8 | 10.1 | 0.81 |
| `hybrid_DR1_OR32` (exp08, FIX-01 on) vs `hybrid_DR1` (exp03, on) | 10 / 10 | greedy_final | 10.7 | 9.8 | 0.68 |

### 4.4 FIX-01 (autoreset phantom transition): fix on vs off, same arm and seeds

| contrast (a vs b) | n_a / n_b | metric | IQM a | IQM b | P(a>b) |
|---|---|---|---|---|---|
| `hybrid_fig4` (exp01, FIX-01 on) vs `hybrid_fig4` (exp01, off) | 10 / 10 | final_perf | 35.8 | 19.9 | 0.75 |
| `hybrid_fig4` (exp01, FIX-01 on) vs `hybrid_fig4` (exp01, off) | 10 / 10 | greedy_final | 47.3 | 28.5 | 0.71 |
| `oversized_mlp` (exp01, FIX-01 on) vs `oversized_mlp` (exp01, off) | 10 / 10 | final_perf | 252.0 | 198.3 | 0.85 |
| `oversized_mlp` (exp01, FIX-01 on) vs `oversized_mlp` (exp01, off) | 10 / 10 | greedy_final | 266.0 | 212.0 | 0.70 |
| `matched_classical` (exp01, FIX-01 on) vs `matched_classical` (exp01, off) | 10 / 10 | final_perf | 11.8 | 9.9 | 0.80 |
| `matched_classical` (exp01, FIX-01 on) vs `matched_classical` (exp01, off) | 10 / 10 | greedy_final | 11.1 | 9.6 | 0.78 |
| `frozen_onehot_mlp` (exp04, FIX-01 on) vs `frozen_onehot_mlp` (exp04, off) | 10 / 10 | final_perf | 0.949 | 0.952 | 0.40 |
| `frozen_onehot_mlp` (exp04, FIX-01 on) vs `frozen_onehot_mlp` (exp04, off) | 10 / 10 | greedy_final | 1.000 | 1.000 | 0.50 |
| `frozen_matched_scalar` (exp04, FIX-01 on) vs `frozen_matched_scalar` (exp04, off) | 10 / 10 | final_perf | 0.001 | 0.000 | 0.58 |
| `frozen_matched_scalar` (exp04, FIX-01 on) vs `frozen_matched_scalar` (exp04, off) | 10 / 10 | greedy_final | 0.000 | 0.000 | 0.50 |

### 4.5 FrozenLake — embedding, entanglement and the classical ceilings

| contrast (a vs b) | n_a / n_b | metric | IQM a | IQM b | P(a>b) |
|---|---|---|---|---|---|
| `frozen_binary_4q_L5` (exp04, FIX-01 on) vs `frozen_binary_4q_noent_L5` (exp04, on) | 10 / 10 | final_perf | 0.952 | 0.000 | 1.00 |
| `frozen_binary_4q_L5` (exp04, FIX-01 on) vs `frozen_binary_4q_noent_L5` (exp04, on) | 10 / 10 | greedy_final | 0.944 | 0.000 | 0.95 |
| `frozen_binary_4q_L5` (exp04, FIX-01 on) vs `frozen_binary_4q_fourier_ceiling` (exp04, on) | 10 / 10 | final_perf | 0.952 | 0.203 | 0.94 |
| `frozen_binary_4q_L5` (exp04, FIX-01 on) vs `frozen_binary_4q_fourier_ceiling` (exp04, on) | 10 / 10 | greedy_final | 0.944 | 0.000 | 0.95 |
| `frozen_binary_4q_L5` (exp04, FIX-01 on) vs `frozen_matched_scalar` (exp04, on) | 10 / 10 | final_perf | 0.952 | 0.001 | 1.00 |
| `frozen_binary_4q_L5` (exp04, FIX-01 on) vs `frozen_matched_scalar` (exp04, on) | 10 / 10 | greedy_final | 0.944 | 0.000 | 0.95 |
| `frozen_binary_4q_L5` (exp04, FIX-01 on) vs `frozen_binary_4q_L1` (exp04, on) | 10 / 10 | final_perf | 0.952 | 0.000 | 1.00 |
| `frozen_binary_4q_L5` (exp04, FIX-01 on) vs `frozen_binary_4q_L1` (exp04, on) | 10 / 10 | greedy_final | 0.944 | 0.000 | 0.95 |
| `frozen_scalar_1q_L5` (exp04, FIX-01 on) vs `frozen_matched_scalar` (exp04, on) | 10 / 10 | final_perf | 0.125 | 0.001 | 0.88 |
| `frozen_scalar_1q_L5` (exp04, FIX-01 on) vs `frozen_matched_scalar` (exp04, on) | 10 / 10 | greedy_final | 0.000 | 0.000 | 0.50 |
| `frozen_scalar_1q_fourier_ceiling_L5__flkw` (exp04, FIX-01 on) vs `frozen_scalar_1q_L5` (exp04, on) | 10 / 10 | final_perf | 0.103 | 0.125 | 0.52 |
| `frozen_scalar_1q_fourier_ceiling_L5__flkw` (exp04, FIX-01 on) vs `frozen_scalar_1q_L5` (exp04, on) | 10 / 10 | greedy_final | 0.000 | 0.000 | 0.50 |
| `frozen_scalar_1q_fourier_ceiling_L10__flkw` (exp04, FIX-01 on) vs `frozen_scalar_1q_L10` (exp04, on) | 10 / 10 | final_perf | 0.949 | 0.081 | 1.00 |
| `frozen_scalar_1q_fourier_ceiling_L10__flkw` (exp04, FIX-01 on) vs `frozen_scalar_1q_L10` (exp04, on) | 10 / 10 | greedy_final | 1.000 | 0.000 | 1.00 |
| `frozen_scalar_1q_fourier_ceiling_L15__flkw` (exp04, FIX-01 on) vs `frozen_scalar_1q_L15` (exp04, on) | 10 / 10 | final_perf | 0.950 | 0.011 | 1.00 |
| `frozen_scalar_1q_fourier_ceiling_L15__flkw` (exp04, FIX-01 on) vs `frozen_scalar_1q_L15` (exp04, on) | 10 / 10 | greedy_final | 1.000 | 0.000 | 1.00 |
| `frozen_scalar_1q_fourier_ceiling_L10__flkw` (exp04, FIX-01 on) vs `frozen_scalar_mlp_large` (exp04, on) | 10 / 10 | final_perf | 0.949 | 0.050 | 1.00 |
| `frozen_scalar_1q_fourier_ceiling_L10__flkw` (exp04, FIX-01 on) vs `frozen_scalar_mlp_large` (exp04, on) | 10 / 10 | greedy_final | 1.000 | 0.000 | 1.00 |
| `frozen_scalar_1q_fourier_ceiling_L5__flkw` (exp04, FIX-01 on) vs `frozen_scalar_1q_fourier_ceiling_L5` (exp04, on) | 10 / 10 | final_perf | 0.103 | 0.666 | 0.30 |
| `frozen_scalar_1q_fourier_ceiling_L5__flkw` (exp04, FIX-01 on) vs `frozen_scalar_1q_fourier_ceiling_L5` (exp04, on) | 10 / 10 | greedy_final | 0.000 | 0.389 | 0.15 |
| `frozen_scalar_1q_fourier_ceiling_L10__flkw` (exp04, FIX-01 on) vs `frozen_scalar_1q_fourier_ceiling_L10` (exp04, on) | 10 / 10 | final_perf | 0.949 | 0.949 | 0.51 |
| `frozen_scalar_1q_fourier_ceiling_L10__flkw` (exp04, FIX-01 on) vs `frozen_scalar_1q_fourier_ceiling_L10` (exp04, on) | 10 / 10 | greedy_final | 1.000 | 0.944 | 0.65 |
| `frozen_scalar_1q_fourier_ceiling_L15__flkw` (exp04, FIX-01 on) vs `frozen_scalar_1q_fourier_ceiling_L15` (exp04, on) | 10 / 10 | final_perf | 0.950 | 0.949 | 0.52 |
| `frozen_scalar_1q_fourier_ceiling_L15__flkw` (exp04, FIX-01 on) vs `frozen_scalar_1q_fourier_ceiling_L15` (exp04, on) | 10 / 10 | greedy_final | 1.000 | 1.000 | 0.55 |

### 4.6 Additional contrasts at n=10 (Output Reuse without entanglement, scaling in R, DR×OR)

| bloque | contraste | n_a / n_b | metrica | IQM a | IQM b | P(a>b) |
|---|---|---|---|---|---|---|
| Output Reuse sin entrelazar frente a sin OR (sin entrelazar) | `hybrid_or_r4_noent vs hybrid_fig4_noent` | 10 / 10 | final_perf | 60.4 | 43.5 | 0.74 |
| Output Reuse sin entrelazar frente a sin OR (sin entrelazar) | `hybrid_or_r4_noent vs hybrid_fig4_noent` | 10 / 10 | greedy_final | 65.4 | 46.4 | 0.70 |
| Output Reuse sin entrelazar frente a sin OR (sin entrelazar) | `hybrid_or_r8_noent vs hybrid_fig4_noent` | 10 / 10 | final_perf | 69.1 | 43.5 | 0.83 |
| Output Reuse sin entrelazar frente a sin OR (sin entrelazar) | `hybrid_or_r8_noent vs hybrid_fig4_noent` | 10 / 10 | greedy_final | 86.9 | 46.4 | 0.72 |
| Output Reuse sin entrelazar frente a sin OR (sin entrelazar) | `hybrid_or_r16_noent vs hybrid_fig4_noent` | 10 / 10 | final_perf | 89.5 | 43.5 | 0.97 |
| Output Reuse sin entrelazar frente a sin OR (sin entrelazar) | `hybrid_or_r16_noent vs hybrid_fig4_noent` | 10 / 10 | greedy_final | 103.9 | 46.4 | 0.86 |
| Output Reuse sin entrelazar frente a sin OR (sin entrelazar) | `hybrid_or_r32_noent vs hybrid_fig4_noent` | 10 / 10 | final_perf | 77.9 | 43.5 | 0.95 |
| Output Reuse sin entrelazar frente a sin OR (sin entrelazar) | `hybrid_or_r32_noent vs hybrid_fig4_noent` | 10 / 10 | greedy_final | 96.6 | 46.4 | 0.81 |
| Circuito OR sin entrelazar frente a clasico igualado en parametros | `hybrid_or_r4_noent vs classical_or_matched_r4` | 10 / 10 | final_perf | 60.4 | 12.4 | 1.00 |
| Circuito OR sin entrelazar frente a clasico igualado en parametros | `hybrid_or_r4_noent vs classical_or_matched_r4` | 10 / 10 | greedy_final | 65.4 | 13.8 | 1.00 |
| Circuito OR sin entrelazar frente a clasico igualado en parametros | `hybrid_or_r8_noent vs classical_or_matched_r8` | 10 / 10 | final_perf | 69.1 | 23.8 | 0.97 |
| Circuito OR sin entrelazar frente a clasico igualado en parametros | `hybrid_or_r8_noent vs classical_or_matched_r8` | 10 / 10 | greedy_final | 86.9 | 24.8 | 0.96 |
| Circuito OR sin entrelazar frente a clasico igualado en parametros | `hybrid_or_r16_noent vs classical_or_matched_r16` | 10 / 10 | final_perf | 89.5 | 36.3 | 1.00 |
| Circuito OR sin entrelazar frente a clasico igualado en parametros | `hybrid_or_r16_noent vs classical_or_matched_r16` | 10 / 10 | greedy_final | 103.9 | 43.0 | 0.97 |
| Circuito OR sin entrelazar frente a clasico igualado en parametros | `hybrid_or_r32_noent vs classical_or_matched_r32` | 10 / 10 | final_perf | 77.9 | 61.2 | 0.75 |
| Circuito OR sin entrelazar frente a clasico igualado en parametros | `hybrid_or_r32_noent vs classical_or_matched_r32` | 10 / 10 | greedy_final | 96.6 | 88.5 | 0.55 |
| Escalado en R, entrelazado | `hybrid_OR8 vs hybrid_OR4` | 10 / 10 | final_perf | 109.8 | 94.7 | 0.66 |
| Escalado en R, entrelazado | `hybrid_OR8 vs hybrid_OR4` | 10 / 10 | greedy_final | 124.3 | 112.0 | 0.64 |
| Escalado en R, entrelazado | `hybrid_OR16 vs hybrid_OR8` | 10 / 10 | final_perf | 104.2 | 109.8 | 0.44 |
| Escalado en R, entrelazado | `hybrid_OR16 vs hybrid_OR8` | 10 / 10 | greedy_final | 146.2 | 124.3 | 0.61 |
| Escalado en R, entrelazado | `hybrid_OR32 vs hybrid_OR16` | 10 / 10 | final_perf | 137.6 | 104.2 | 0.72 |
| Escalado en R, entrelazado | `hybrid_OR32 vs hybrid_OR16` | 10 / 10 | greedy_final | 161.1 | 146.2 | 0.59 |
| Escalado en R, sin entrelazar | `hybrid_or_r8_noent vs hybrid_or_r4_noent` | 10 / 10 | final_perf | 69.1 | 60.4 | 0.60 |
| Escalado en R, sin entrelazar | `hybrid_or_r8_noent vs hybrid_or_r4_noent` | 10 / 10 | greedy_final | 86.9 | 65.4 | 0.65 |
| Escalado en R, sin entrelazar | `hybrid_or_r16_noent vs hybrid_or_r8_noent` | 10 / 10 | final_perf | 89.5 | 69.1 | 0.73 |
| Escalado en R, sin entrelazar | `hybrid_or_r16_noent vs hybrid_or_r8_noent` | 10 / 10 | greedy_final | 103.9 | 86.9 | 0.65 |
| Escalado en R, sin entrelazar | `hybrid_or_r32_noent vs hybrid_or_r16_noent` | 10 / 10 | final_perf | 77.9 | 89.5 | 0.35 |
| Escalado en R, sin entrelazar | `hybrid_or_r32_noent vs hybrid_or_r16_noent` | 10 / 10 | greedy_final | 96.6 | 103.9 | 0.37 |
| DR2 con OR frente a DR2 sin OR | `hybrid_DR2_OR4 vs hybrid_DR2` | 10 / 10 | final_perf | 44.6 | 30.2 | 0.78 |
| DR2 con OR frente a DR2 sin OR | `hybrid_DR2_OR4 vs hybrid_DR2` | 10 / 10 | greedy_final | 50.1 | 40.2 | 0.71 |
| DR2 con OR frente a DR2 sin OR | `hybrid_DR2_OR8 vs hybrid_DR2` | 10 / 10 | final_perf | 54.6 | 30.2 | 0.89 |
| DR2 con OR frente a DR2 sin OR | `hybrid_DR2_OR8 vs hybrid_DR2` | 10 / 10 | greedy_final | 69.5 | 40.2 | 0.77 |
| DR2 con OR frente a DR2 sin OR | `hybrid_DR2_OR16 vs hybrid_DR2` | 10 / 10 | final_perf | 67.9 | 30.2 | 0.97 |
| DR2 con OR frente a DR2 sin OR | `hybrid_DR2_OR16 vs hybrid_DR2` | 10 / 10 | greedy_final | 75.8 | 40.2 | 0.81 |
| DR2 con OR frente a DR2 sin OR | `hybrid_DR2_OR32 vs hybrid_DR2` | 10 / 10 | final_perf | 77.1 | 30.2 | 0.99 |
| DR2 con OR frente a DR2 sin OR | `hybrid_DR2_OR32 vs hybrid_DR2` | 10 / 10 | greedy_final | 91.2 | 40.2 | 0.96 |

## 5. Comparison with the dissection paper (PPO, n=10) — NOT like for like

The paper trains with **PPO**; every run here is **DQN**. Same environment, 100k steps, same circuit templates, same metric (`best_ma50`), different learning algorithm — so a gap is algorithm × circuit, not a replication failure by itself. Paper numbers from `data/paper_ppo_summary.csv` (mean ± sd over 10 seeds); ours: mean ± sd over our n seeds.

| arm (ours, DQN) | paper config | paper PPO best_ma50 | ours best_ma50 (n) | paper PPO final_ma50 | ours final_ma50 | ours greedy_final IQM |
|---|---|---|---|---|---|---|
| `paper_classical_or_r16` | Classical_r16 | 381.7 ± 37.3 | 44.4 ± 60.9 (10) | 246.2 ± 54.0 | 19.0 ± 29.3 | 9.6 |
| `paper_classical_or_r32` | Classical_r32 | 330.2 ± 63.4 | 25.7 ± 2.8 (10) | 188.0 ± 81.6 | 10.3 ± 0.1 | 10.0 |
| `paper_classical_or_r4` | Classical_r4 | 263.9 ± 18.4 | 27.4 ± 5.5 (10) | 242.7 ± 19.2 | 9.6 ± 0.2 | 9.5 |
| `paper_classical_or_r8` | Classical_r8 | 320.0 ± 36.3 | 27.8 ± 5.2 (10) | 274.4 ± 41.1 | 9.6 ± 0.1 | 9.5 |
| `paper_hsiao_or_r16` | Quantum_r16 | 354.8 ± 124.3 | 31.7 ± 11.9 (10) | 346.3 ± 123.8 | 10.9 ± 3.3 | 9.8 |
| `paper_hsiao_or_r16_ent` | Hsiao_OR_r16_Entangled | 35.4 ± 2.4 | 25.4 ± 1.4 (10) | 27.8 ± 2.9 | 9.6 ± 0.2 | 9.5 |
| `paper_hsiao_or_r32` | Quantum_r32 | 430.5 ± 115.6 | 56.9 ± 52.5 (10) | 365.9 ± 155.2 | 17.9 ± 16.7 | 9.9 |
| `paper_hsiao_or_r4` | Quantum_r4 | 116.5 ± 28.9 | 24.8 ± 0.9 (10) | 103.6 ± 25.1 | 9.6 ± 0.2 | 9.5 |
| `paper_hsiao_or_r4_ent` | Hsiao_OR_r4_Entangled | 31.9 ± 1.9 | 25.7 ± 1.3 (10) | 24.9 ± 2.6 | 9.6 ± 0.1 | 9.5 |
| `paper_hsiao_or_r8` | Quantum_r8 | 230.0 ± 88.1 | 64.0 ± 115.6 (10) | 210.2 ± 84.5 | 9.8 ± 0.2 | 9.7 |
| `paper_salinas_1q_L1` | Salinas_1Q_L1 | 35.4 ± 9.6 | 28.3 ± 9.3 (10) | 27.4 ± 6.4 | 9.6 ± 0.1 | 9.5 |
| `paper_salinas_1q_L2` | Salinas_1Q_L2 | 53.6 ± 9.3 | 25.8 ± 0.8 (10) | 46.2 ± 8.5 | 9.7 ± 0.1 | 9.5 |
| `paper_salinas_1q_L5` | Salinas_1Q_L5 | 95.5 ± 62.7 | 26.7 ± 2.5 (10) | 76.7 ± 59.5 | 9.6 ± 0.2 | 9.5 |
| `paper_salinas_2q_L1` | Salinas_2Q_L1 | 48.6 ± 17.5 | 28.2 ± 10.7 (10) | 37.4 ± 13.6 | 9.5 ± 0.2 | 9.5 |
| `paper_salinas_2q_L2` | Salinas_2Q_L2 | 51.0 ± 10.6 | 28.3 ± 5.7 (10) | 41.5 ± 10.3 | 9.6 ± 0.1 | 9.5 |
| `paper_salinas_2q_L5` | Salinas_2Q_L5 | 107.1 ± 47.5 | 26.1 ± 2.4 (10) | 94.2 ± 47.5 | 9.6 ± 0.1 | 9.5 |
| `paper_skolik_4q_L1` | Skolik_4Q_L1 | 28.2 ± 1.3 | 27.3 ± 4.0 (10) | 22.7 ± 1.9 | 10.4 ± 2.3 | 9.5 |
| `paper_skolik_4q_L2` | Skolik_4Q_L2 | 74.5 ± 50.9 | 25.2 ± 1.1 (10) | 64.4 ± 46.5 | 9.7 ± 0.2 | 9.4 |
| `paper_skolik_4q_L2_noent` | Skolik_DR_L2_Unentangled | 176.6 ± 83.2 | 28.0 ± 7.8 (10) | 165.2 ± 76.6 | 10.2 ± 0.9 | 9.5 |
| `paper_skolik_4q_L5` | Skolik_4Q_L5 | 175.4 ± 68.5 | 25.6 ± 2.1 (10) | 162.2 ± 70.1 | 9.6 ± 0.3 | 9.5 |
| `paper_skolik_4q_L5_noent` | Skolik_DR_L5_Unentangled | 147.2 ± 49.9 | 25.7 ± 0.9 (10) | 133.0 ± 46.8 | 10.2 ± 1.3 | 9.5 |
| `paper_skolik_8q_L1` | Skolik_8Q_L1 | 32.2 ± 2.6 | 30.6 ± 5.6 (10) | 22.9 ± 2.4 | 12.3 ± 3.8 | 10.4 |
| `paper_skolik_8q_L2` | Skolik_8Q_L2 | 144.0 ± 37.9 | 33.2 ± 10.4 (10) | 131.8 ± 33.4 | 13.9 ± 7.1 | 12.4 |
| `paper_skolik_8q_L5` | Skolik_8Q_L5 | 199.3 ± 60.1 | 53.9 ± 22.0 (10) | 188.2 ± 60.8 | 28.4 ± 19.4 | 24.8 |

exp03 (hybrid_fig4 with n_layers_q = L) is the repo's direct comparator for the paper's Skolik 8-qubit DR sweep:

| ours (DQN, exp03) | paper Skolik_8Q | paper PPO best_ma50 | ours best_ma50 (n) |
|---|---|---|---|
| `hybrid_DR1` | L1 | 32.2 ± 2.6 | 25.3 ± 1.2 (10) |
| `hybrid_DR2` | L2 | 144.0 ± 37.9 | 47.4 ± 14.4 (10) |
| `hybrid_DR5` | L5 | 199.3 ± 60.1 | 68.9 ± 22.9 (10) |

## 6. Lectura de los resultados (escrita a mano; cada punto remite a las tablas 3–5)

Etiquetas de evidencia: ver regla 5. Siempre `final_perf` y `greedy_final`, nunca `best_ma50`. Todas las cifras son IQM con IC 95 % bootstrap entre paréntesis salvo que se diga otra cosa.

**F1. CartPole: ningún brazo cuántico se acerca a una red clásica sin restricción de capacidad, y el techo clásico aditivo supera al circuito de igual estructura. [FUERTE]**
`oversized_mlp` (10 934 params) 252,0 (234,8–279,4) frente al mejor híbrido, `hybrid_OR32` (1 118 params) 137,6 (106,0–179,6). El techo Fourier clásico de 162 params (76,5; 69,6–94,0) supera a `hybrid_fig4` (126 params, 35,8; 24,0–48,1): P(techo > híbrido) = 1,00 en `final_perf` y 0,87 en greedy. En cambio `hybrid_fig4` sí supera a `matched_classical` (11,8) y a `paper_linear` (9,6, 26 params): P = 1,00 en ambas métricas. Es decir: el híbrido vence a clásicos pequeños y de capacidad igualada, pero no a un clásico mayor ni al techo aditivo. Salvedad de capacidad: el techo usa 162 parámetros frente a 126 (no está igualado; `docs/CORRECTIONS.md#new-06`).

**F2. Output Reuse es la palanca cuántica más fuerte en CartPole. [FUERTE, con salvedad de capacidad]**
`hybrid_OR{4,8,16,32}` = 94,7 / 109,8 / 104,2 / 137,6 frente a `hybrid_fig4` 35,8: P entre 0,96 y 1,00 en `final_perf`. Frente a los clásicos con capacidad igualada (`classical_or_matched_r*` = 12,4 / 23,8 / 36,3 / 61,2): P = 1,00 (r4), 0,95 (r16), 0,99 (r32) en `final_perf` y 1,00 / 0,98 / 0,88 en greedy. **No es monótono en R**: OR8 ≥ OR16 en `final_perf`, y los saltos entre R consecutivos tienen P entre 0,44 y 0,72 (no separables). Salvedad: OR aumenta los parámetros (222 → 1 118 frente a 126); los controles `classical_or_matched_*` los igualan y quedan muy por debajo de `oversized_mlp` (252,0).
**El efecto de OR no depende del entrelazamiento:** sin entrelazar, `hybrid_or_r{4,8,16,32}_noent` = 60,4 / 69,1 / 89,5 / 77,9 frente a `hybrid_fig4_noent` 43,5; P(OR > sin OR) = 0,74 / 0,83 / 0,97 / 0,95 en `final_perf` (greedy 0,70 / 0,72 / 0,86 / 0,81). Y el circuito sin entrelazar con OR supera al clásico igualado en parámetros a R = 4, 8 y 16 (P = 1,00 / 0,97 / 1,00 en `final_perf`), pero **no a R = 32** (P = 0,75 y 0,55 en greedy).

**F3. Profundidad de Data Reuploading: sube y satura. [FUERTE para L1→L2; DÉBIL para L2→L5]**
`hybrid_DR1` 10,1 (suelo, muerto) → `hybrid_DR2` 30,2 → `hybrid_DR5` 37,7. P(DR2 > DR1) = 1,00; P(DR5 > DR2) = 0,63 (`final_perf`) / 0,70 (greedy), con IC amplios (DR5 greedy 73,7: 38,4–108,3). Comparado con el PPO del paper (Skolik 8Q L1/L2/L5, `best_ma50` 32,2 / 144,0 / 199,3), nuestro DQN da 25,3 / 47,4 / 68,9: mismo orden creciente, magnitud 3–4 veces menor (algoritmo distinto, sección 5).

**F4. FIX-01 (transición fantasma del autoreset): en CartPole mejora de forma consistente pero modesta; en FrozenLake es nulo. [MODERADA / NEGATIVO]**
CartPole on frente a off: `hybrid_fig4` 35,8 frente a 19,9 (P = 0,75), `oversized_mlp` 252,0 frente a 198,3 (0,85), `matched_classical` 11,8 frente a 9,9 (0,80); los IC se solapan en cada brazo, pero la dirección es la misma en los tres. FrozenLake: `frozen_onehot_mlp` 0,949 frente a 0,952 (P = 0,40) y `frozen_matched_scalar` ≈ 0 en ambos: sin efecto, pese a que la fracción de transiciones fantasma es ~3–13 % de los pasos. La hipótesis H3 de `docs/EXPERIMENT-04.md` **no se cumple** en FrozenLake.

**F5. FrozenLake Config B: solo sale del suelo el circuito entrelazado a L=5, y necesita ambas cosas. [FUERTE]**
`frozen_binary_4q_L5` 0,952 (greedy 0,944; 9 de 10 semillas vivas, la semilla 4 está muerta). Muertos: `frozen_binary_4q_noent_L5` (0,000), `frozen_binary_4q_L1` y `_noent_L1` (0,000), el techo afín en los 4 bits `frozen_binary_4q_fourier_ceiling` (aprende algo en entrenamiento, 0,203, pero su política greedy es 0,000 en las 10 semillas) y `frozen_matched_scalar` (≈0). P(entrelazado L5 > noent L5) = 1,00 / 0,95. Control de vida: `frozen_onehot_mlp` 0,949 (greedy 1,000), así que el entorno es resoluble. Conclusión acotada: *frente a la clase afín en los bits y frente al circuito sin entrelazar*, el entrelazamiento a profundidad suficiente es necesario. **No se ha probado ninguna MLP clásica con capa oculta sobre los 4 bits con presupuesto igualado**, así que no se puede decir "el circuito supera a un clásico equivalente" (DECISIÓN A TOMAR si se añade ese brazo; ver sección 9).

**F6. CartPole: el entrelazamiento solo aporta algo cuando se combina con Output Reuse, y aun así sin separación completa. [MODERADA]**
Se compara siempre el circuito real con `ent=False` (misma arquitectura, n = 10 en los dos lados).
- *Sin OR:* `hybrid_fig4` 35,8 frente a `hybrid_fig4_noent` 43,5: P(ent > noent) = 0,33 (`final_perf`) / 0,46 (greedy). El no entrelazado es igual o algo mejor.
- *Profundidad (exp03 frente a exp03b):* P(ent > noent) entre 0,35 y 0,60 en `final_perf` y entre 0,38 y 0,57 en greedy, para L = 1, 2, 5: sin diferencia.
- *Con OR (exp02 frente a exp08):* entrelazado 94,7 / 109,8 / 104,2 / 137,6 frente a sin entrelazar 60,4 / 69,1 / 89,5 / 77,9 para R = 4, 8, 16, 32. P(ent > noent) = 0,90 / 0,90 / 0,64 / 0,92 en `final_perf` y 0,89 / 0,86 / 0,78 / 0,89 en greedy. La dirección es la misma en los cuatro valores de R, pero **ninguno llega a P ≥ 0,95** y R = 16 es débil (0,64).
Lectura prudente: la ventaja del entrelazamiento aparece solo junto a OR, es consistente en dirección y de tamaño medio (unos 15–60 puntos de retorno según R), pero no se puede calificar de fuerte. Esta lectura sustituye a la provisional de la versión 2 (n = 5–9); con n = 10 se confirma el patrón y no cambia de signo.

**F7. FrozenLake Config A: ningún circuito aprende a ninguna profundidad, pero un techo clásico sobre la MISMA entrada escalar sí, a L=10 y L=15. [FUERTE en L=10 y L=15; NULO en L=1 y L=5]**
Circuitos `frozen_scalar_1q_L{1,5,10,15}`: `final_perf` 0,000 / 0,125 / 0,081 / 0,011 y `greedy_final` 0,000 en todos (una única semilla viva: L10, semilla 8, `greedy_final` 0,33). Techos Fourier clásicos reentrenados con los hiperparámetros de FrozenLake (`..._fourier_ceiling_L{1,5,10,15}__flkw`, 12 / 44 / 84 / 124 params frente a 10 / 18 / 28 / 38 del circuito): 0,004 / 0,103 / 0,949 / 0,950, con `greedy_final` 0,000 / 0,000 / 1,000 / 1,000 y todas las semillas vivas a L=10 y L=15 (10/10). P(techo > circuito) = 1,00 en ambas métricas a L=10 y L=15, y 0,52 / 0,50 a L=5 (los dos muertos). Una MLP genérica de 4 548 params sobre la misma entrada escalar está muerta (0,050, greedy 0): P(techo L10 > MLP) = 1,00.
Lectura: la clase de funciones que el circuito podría expresar *sí contiene* una política resolutoria, pero el circuito real no la encuentra (optimización, sesgo inductivo o coeficientes restringidos por unitariedad: **NO DETERMINABLE** cuál), y lo que resuelve la tarea son los rasgos periódicos de frecuencia suficiente, no una MLP genérica. Confusión que queda: el techo usa 3–4× más parámetros que el circuito.
**Verificación tras un fallo de control (FIX-13):** los techos originales se habían entrenado por error con hiperparámetros de CartPole. Reentrenados con los de FrozenLake, el resultado a L=10 y L=15 es idéntico (P = 0,51 y 0,52 en `final_perf` entre ambas versiones), de modo que el fallo no sostenía el hallazgo. A L=5 sí importó: con los ajustes de CartPole, 7 de 10 semillas vivas en greedy (`final_perf` 0,666; greedy 0,389), frente a 0 de 10 con los de FrozenLake (`final_perf` 0,103; greedy 0,000). Ver 2b y la salvedad 5.

**F8. Réplica del paper bajo DQN (exp07, n = 10): prácticamente todos los brazos están muertos. [NEGATIVO, con la salvedad de hiperparámetros]**
21 de los 24 brazos terminan con `final_perf` ≤ 10,3 (suelo ≈ 9,5–9,6, compatible con una política de acción constante). Solo los tres de 8 qubits superan ese suelo: `paper_skolik_8q_L1` 10,9 (10,2–12,1; prácticamente muerto), `_L2` 12,9 (10,8–16,1) y `_L5` 19,8 (14,5–29,3). Incluye los controles clásicos `paper_classical_or_r{4,8,16,32}` (9,6–10,3), que con PPO alcanzan 264–382 de `best_ma50`, y los Hsiao OR (con PPO: r16 354,8; r32 430,5; aquí ≈ 10). Hay outliers aislados (`paper_classical_or_r16` y `paper_hsiao_or_r32`, con IC superior 25,3 y 29,6 y `best_ma50` con desviación grande): atribuibles a una o pocas semillas, no verificado semilla a semilla (están en `per_seed_metrics.csv`).
Contraste interno: los brazos de exp02 con OR sobre el circuito de 8 qubits L=5 sí aprenden (95–138), así que **el DQN no es incapaz de entrenar híbridos**. Lo que no se transfiere de PPO a DQN, con los hiperparámetros actuales y sin ajuste, son *estas* configuraciones (4 qubits poco profundos, 1–2 qubits). No se puede distinguir "no se transfiere" de "DQN mal ajustado para estos circuitos" (sección 2b, DECISIÓN A TOMAR). Redactar siempre como "con estos hiperparámetros".

**F9. Cruce DR×OR: OR solo ayuda cuando ya hay profundidad, y la profundidad pesa más que OR. [MODERADA a FUERTE]**
`hybrid_DR1_OR{4,8,16,32}` = 10,6 / 11,2 / 11,6 / 10,8 (muertos, igual que DR1 = 10,1; P(DR1_OR32 > DR1) = 0,81, irrelevante en magnitud). `hybrid_DR2_OR{4,8,16,32}` = 44,6 / 54,6 / 67,9 / 77,1 frente a DR2 = 30,2: P(DR2_OR > DR2) = 0,78 / 0,89 / 0,97 / 0,99 en `final_perf`, creciente con R. Pero la profundidad L=5 con OR32 (`hybrid_OR32`, 137,6) queda muy por encima de L=2 con OR32 (77,1): P(DR2_OR32 > OR32) = 0,02 (`final_perf`) / 0,05 (greedy). Con OR a R = 32 el parámetro cuenta casi igual en ambos (1 070 frente a 1 118), así que la diferencia no es de capacidad bruta.

**F10. Resumen de lo que apoya y de lo que no apoya la hipótesis "el circuito cuántico aporta algo" bajo DQN.**
- *Lo apoyan:* el circuito supera a clásicos pequeños o igualados en parámetros (F1, F2); OR es una palanca robusta en CartPole (F2); el circuito entrelazado a L=5 resuelve FrozenLake Config B, que los controles afines y el circuito sin entrelazar no resuelven (F5); la profundidad ayuda de L=1 a L=2 (F3).
- *No lo apoyan:* una MLP grande gana por mucho en CartPole (F1) y un techo clásico aditivo de 162 parámetros también; el techo clásico resuelve Config A donde ningún circuito lo hace (F7); sin OR, el entrelazamiento no ayuda en CartPole (F6); casi toda la réplica del paper queda muerta bajo DQN (F8).
- *Qué no se puede afirmar:* "el circuito supera a un clásico equivalente" en FrozenLake Config B (falta el control, F5); ninguna afirmación general del tipo "ventaja cuántica", porque el único resultado donde un circuito resuelve algo que un modelo clásico de la misma entrada no resuelve (F5) no tiene control clásico oculto igualado.


---

## 7. Salvedades y calidad de datos (el texto final debe mencionar las que apliquen)

1. **PPO frente a DQN.** El paper usa PPO; todo lo propio es DQN. Una diferencia con el paper mezcla algoritmo y circuito (sección 5).
2. **`best_ma50` está inflado en los brazos muertos.** Un brazo con `final_perf` ≈ 9,6 puede tener `best_ma50` ≈ 25–30 (pico de la fase inicial de exploración). Por eso la lectura usa `final_perf` y `greedy_final`; `best_ma50` solo para comparar con el paper.
3. **FrozenLake determinista: el greedy tiene una muestra efectiva por checkpoint.** `greedy_final` solo toma los valores 0, 0,33, 0,67, 1. Hay **disociación entre entrenamiento y greedy** (el techo afín de Config B: 0,203 en entrenamiento, 0,000 en greedy): se aprende valor pero la política greedy entra en bucles. No interpretar el entrenamiento aislado como "aprende".
4. **Hiperparámetros nunca ajustados** (sección 2b). Un brazo muerto puede ser un brazo mal ajustado, no un brazo incapaz. Esto afecta sobre todo a F8.
5. **Fallo de control resuelto (FIX-13).** Los techos Fourier originales de Config A se entrenaron con hiperparámetros de CartPole (`buffer_size=10 000`, `train_frequency=10`, `learning_starts` 10 000) en vez de los de FrozenLake (50 000 / 1 / 1 000). Evidencia independiente de los manifiestos: la primera pérdida registrada del techo L15 aparece en el paso 10 100, la del circuito en el 1 100. Los manifiestos de esos techos declaran los valores de FrozenLake (no reflejan lo entrenado; causa exacta **NO DETERMINABLE**, probable `scripts/migrate_manifests.py`). Se reentrenaron los 4 techos × 10 semillas con los hiperparámetros correctos (etiqueta `flkw`). **Para cualquier afirmación sobre estos techos, usar solo las filas `__flkw`.** Las originales se conservan en las tablas por trazabilidad.
6. **Presupuestos de parámetros distintos.** OR añade parámetros; el techo Fourier de CartPole (162) y los de Config A (12–124) usan más que el circuito comparado; `oversized_mlp` (10 934) es el límite superior de capacidad. Los parámetros de `classical_OR*` y `classical_or_matched_*` no se pueden recalcular desde el registro (columna `-`).
7. **Contrastes.** P(a>b) no pareada sobre semillas; sin corrección por comparaciones múltiples (regla 8). Con n = 10 por brazo y IC de bootstrap anchos, las diferencias moderadas no son concluyentes.
8. **Manifiestos antiguos de exp02/exp03.** Las semillas 1–3 no traen el campo `fix_autoreset`; se ha inferido FIX-01 activo a partir de `probe.fix01_active=True` (verificado en todas) y se han unido a las semillas 4–10, dando n = 10 por brazo.
9. **Otras correcciones relevantes** (`docs/CORRECTIONS.md`): FIX-02 (`OutputScale`) no está activo en ningún brazo; FIX-07 (en la plantilla Skolik, `L` capas llevan `L-1` bloques entrelazadores efectivos, y a L=1 el circuito entrelazado y el no entrelazado son el mismo, de ahí que DR1 ent y noent sean indistinguibles); FIX-10 (muestreo no semillado de `action_space.sample()`: la reproducibilidad es estadística, no bit a bit).
10. **Hardware mixto.** Parte de los runs son locales y parte de Colab: los tiempos `h/seed` no son comparables entre experimentos.
11. **Reproducibilidad de las cifras.** Las tablas de las secciones 3–5 se han generado con un script a partir de `per_seed_metrics.csv`; los números de la sección 6 se han copiado de ellas. Si una cifra de la sección 6 no coincide con una tabla, **manda la tabla**. Los IC bootstrap proceden de un remuestreo aleatorio: `summary_by_arm.csv` puede diferir del dossier en una o dos décimas en los extremos del IC (los IQM y las P coinciden). **Citar siempre los IC del dossier.**

---

## 8. Hipótesis pre-declaradas (literal, en inglés) y qué dicen los datos

Estado frente a los datos: exp04 **H1** (DR depth) → F3/F5; **H2** (embedding; `noent` decides embedding vs entanglement) → F5; **H3** (FIX-01) → F4 (no se cumple en FrozenLake); **H4** (control) → `frozen_matched_scalar` está muerto: la comparación circuito-frente-a-clásico de Config A **no es contestable** con ese control, pero el techo Fourier (F7) sí está vivo; exp05 **H2** → F5 (cumplida en lo esencial: sin entrelazamiento y el techo afín están a nivel de azar en greedy; matiz: el techo aprende algo en entrenamiento, 0,203, aunque no en greedy).


### exp04 (`docs/EXPERIMENT-04.md` §5)

> **H1 — DR depth (the exp03 replication).** Rising then saturating success rate
> with `n_layers_q` replicates exp03's CartPole finding in a second environment and
> generalises the thesis's one clean positive. Flat or falling bounds that claim to
> CartPole — also a result, and a more interesting one than a second confirmation.
> Falling with depth indicates optimisation difficulty, not expressivity, and
> should not be called a barren plateau without gradient-norm evidence.
> 
> **H2 — embedding.** Binary-basis above scalar-to-phase at equal depth. Note this
> is *not* a transfer claim: the chapter's single-seed Fig. 6 is too weak a
> reference to transfer from. It is a first controlled measurement. The `noent`
> ablation decides whether any gap is embedding or entanglement.
> 
> **H3 — FIX-01.** From stage 1. A large effect where the phantom fraction is ~10×
> CartPole's, against the CartPole null, is the mechanism result. A null here too
> means the bug is real but practically inconsequential — publishable, honest, and
> it would substantially reduce how much weight the correction can carry.
> 
> **H4 — control, and the caveat that may swallow it.** If `frozen_matched_scalar`
> reaches Config A, the circuit buys nothing over an equal-budget classical net on
> the same information, and the honest claim concerns the *embedding*, not the
> circuit. That would echo exp01.
> 
> But the smoke run (1 seed, 5k steps) suggests something more awkward: both scalar
> arms sit at success 0.03-0.07 against a random floor of 0.015, while the one-hot
> arm reaches 0.86-0.96. A network fed the raw state **index** has almost nothing to
> work with - the ordering is row-major and carries no usable metric, so it must fit
> a near-arbitrary 16-point function through one input.
> 
> If `frozen_matched_scalar` stays at chance, **H4 is unanswerable**: a hybrid
> beating a dead control proves nothing, and claiming otherwise would repeat exactly
> the error exp01 caught. Config A runs on that same scalar input and may be dead
> for the same reason. In that case the informative comparison is Config B against
> the one-hot MLP, and it must be reported as a result about the *encoding* rather
> than dressed up as circuit-vs-classical.

### exp05 (`docs/EXPERIMENT-05.md` §4)

> Fixed before any exp05 cell runs, so the analysis cannot become "find a story
> that fits."
> 
> *(H1 of exp05, on the equivalence of a simulator, is omitted on purpose: rule 4.)*
> 
> **H2.** `noent` ≈ ceiling ≈ chance at every depth; the entangled arm sits
> above both. This is the interesting outcome — it would mean entanglement
> (not depth, not the embedding) is what lets Config B escape the dead floor
> `frozen_matched_scalar` and `frozen_scalar_1q` already showed at low depth
> (see `docs/RESULTS-LOG.md`, exp04). **If the entangled arm is ALSO at
> chance**, the contrast is null in this environment — report it as a bound on
> the claim, not as an inconclusive footnote to a positive story.
> 
> **H3.** Not the primary question here (CartPole is exp06's job), but worth
> watching: if `noent` or the ceiling ever beat the entangled arm, that is not
> evidence against entanglement mattering elsewhere — it would say this
> particular depth/qubit-count combination does not need it, a narrower and
> still-useful claim.


---

## 9. Decisiones abiertas del autor (DECISIÓN A TOMAR)

Estas decisiones no las toma quien redacta: hay que redactar el texto de forma que sea válido en cualquiera de los dos casos, o dejar el marcador visible.

1. **Hiperparámetros y réplica bajo DQN (F8, sección 2b).** 21 de 24 brazos de la réplica literal quedan muertos. ¿Se informa tal cual (con la limitación explícita "sin ajuste de hiperparámetros") o se dedica un ajuste específico de DQN (tasa de aprendizaje, `train_frequency`, `learning_starts`, ε) antes de concluir? Sin ese ajuste no se distingue "no se transfiere" de "DQN mal ajustado".
2. **Control clásico para Config B (F5).** No existe una MLP con capa oculta sobre los 4 bits con presupuesto igualado; sin ella "el circuito supera a un clásico equivalente" no es afirmable. Hoy el texto debe decir que ese control **no se ha hecho**.
3. **Lectura sobre el entrelazamiento en CartPole (F6).** Se redacta como [MODERADA], solo con OR, sin separación completa.
4. **Qué se afirma sobre el aporte del trabajo.** Candidatos, no excluyentes: (a) la auditoría y corrección del código de partida (FIX-xx/NEW-xx); (b) la evidencia de que el beneficio cuántico depende del algoritmo (PPO frente a DQN) y de la configuración (OR; profundidad + entrelazamiento en FrozenLake); (c) los techos clásicos como control, más que un "efecto cuántico" genérico. El autor decide el énfasis.
5. **Ampliaciones fuera del plan.** Hay ideas anotadas sin compromiso (barrido de tasa de aprendizaje, MLP oculta sobre bits en FrozenLake, OR sobre el circuito entrelazado real de FrozenLake). No describirlas como hechas ni como planificadas; en todo caso, como "trabajo futuro".
6. **Resultados de simulador (exp09).** Siguen excluidos (regla 4). Solo el autor puede levantar la restricción.

## 10. Esquema sugerido para los capítulos Resultados y Conclusiones

*(Propuesta; el autor y el tutor deciden el orden final.)*

**Resultados**
1. Protocolo y métricas (IQM, IC bootstrap, P(mejora), `final_perf` frente a `greedy_final`; por qué no `best_ma50`).
2. CartPole: ladder de capacidad (F1) → Output Reuse (F2) → profundidad (F3) → efecto del entrelazamiento (F6) → cruce DR×OR (F9) → efecto de FIX-01 (F4).
3. FrozenLake: Config B (F5) → Config A y techos clásicos (F7).
4. Réplica del paper bajo DQN (F8) y comparación con PPO (sección 5, con la advertencia de la regla 1).
5. Correcciones y controles que cambiaron conclusiones (FIX-13; salvedad 5), como muestra de rigor metodológico.

**Conclusiones**
- Qué se confirma (F1, F2, F3, F5, F7), qué se matiza (F6, F9, F4) y qué es negativo (F8).
- Limitaciones explícitas: hiperparámetros nunca ajustados; un único entorno determinista en FrozenLake; sin hardware cuántico real (todo es simulación clásica del circuito con la librería de referencia, sin ruido); sin control oculto igualado en FrozenLake; contrastes sin corrección múltiple.
- Trabajo futuro: ajuste de hiperparámetros (sobre todo tasa de aprendizaje), control clásico oculto en FrozenLake, OR sobre el circuito entrelazado de FrozenLake, ruido.

## 11. Procedencia

- Datos: `Mi unidad/tfm_qrl/results/{exp01_dqn_cartpole_capacity, exp01_dqn_cartpole_capacity_60k_20260805, exp02_dqn_cartpole_output_reuse, exp03_dqn_cartpole_data_reuploading, exp03b_dqn_cartpole_dr_unentangled, exp04_dqn_frozenlake_embeddings, exp07_dqn_cartpole_paper_replication, exp08_dqn_cartpole_frozenlake_followups}` (manifiestos `*.manifest.json` + `runs/*.csv` + `runs/*_eval.csv`).
- Cifras por semilla: `per_seed_metrics.csv` (columnas `exp, arm, fix, tag, steps, seed, final, greedy, best_ma50, final_ma50, wall_h, phantom`). Resumen por brazo: `summary_by_arm.csv`.
- Estadística: `src/qrl_dissection/core/stats.py` (IQM, IC bootstrap al 95 %, `probability_of_improvement`, `final_performance`, `greedy_final`); definiciones en `docs/STATISTICS.md`.
- Paper: `data/paper_ppo_summary.csv` (PPO, 10 semillas, `best_ma50`/`final_ma50`; `scripts/extract_paper_baselines.py`).
- Hipótesis y diseño: `docs/EXPERIMENT-01.md`, `-04.md`, `-05.md`, `-06.md`; correcciones: `docs/CORRECTIONS.md`.
- Dossier versión 3 generado el 2026-11-01. Sustituye por completo a las versiones 1 y 2: **no mezclar cifras de versiones anteriores** (tenían n < 10 en exp07 y exp08).
- Cómputo aproximado: ~3 100 horas de reloj de pared sumadas entre todos los runs de este dossier (hardware mixto, Colab y local; las sesiones se solapan, no es tiempo calendario).
