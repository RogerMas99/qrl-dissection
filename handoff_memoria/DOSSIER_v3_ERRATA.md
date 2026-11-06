# Fe de erratas del dossier de resultados v3

Fecha 2026-11-06. El fichero `01_DOSSIER_RESULTADOS_v3.md` **no se ha editado**: esta nota recoge lo que hay que corregir y por qué, para que quede trazado qué cambió. Si el texto de la memoria se copió de esas líneas, hay que corregirlo allí.

## E1. Parámetros de `frozen_onehot_mlp`: 5 508, no 4 548

- **Dónde:** tabla de FrozenLake del dossier, líneas 148 y 149 (las dos filas de `frozen_onehot_mlp`, FIX-01 `off` y `on`), columna `params`.
- **Valor correcto: 5 508.** `frozen_onehot_mlp` (`net_arch` [64, 64], ReLU) entrena sobre su entorno real, `FrozenLake4x4OneHot-v0`: los manifiestos de exp04 registran ese `env_id`. Construido sobre ese entorno cuenta 5 508 parámetros entrenables, y a mano da 16·64+64 + 64·64+64 + 64·4+4 = 5 508. `docs/EXPERIMENT-04.md` (línea 128), `docs/RESULTS-LOG.md` (línea 664) y `docs/EXPERIMENT-09-WAVE2.md` (línea 134) ya dicen 5 508.
- **Origen del error:** el generador de las tablas del dossier contaba los parámetros de todos los brazos `frozen_*` sobre el entorno escalar. Con entrada de 1 componente la misma red tiene 4 548 (1·64+64 + 64·64+64 + 64·4+4), que es el valor correcto de `frozen_scalar_mlp_large`. `frozen_onehot_mlp` es el único brazo `frozen_*` que no entrena sobre el entorno escalar (`FROZEN_ARM_ENV`), así que no hay otra fila afectada.
- **Qué no cambia:** ningún resultado de rendimiento. La frase de la línea 435 ("una MLP genérica de 4 548 params sobre la misma entrada escalar está muerta") habla de `frozen_scalar_mlp_large` y es correcta. Las filas de `frozen_scalar_mlp_large` (líneas 162-163) también.
- **Corrección:** donde la memoria dé el tamaño de la MLP de referencia one-hot, 5 508.

## E2. Episodios de la evaluación greedy: 20 no vale para todo CartPole

- **Dónde:** línea 24 ("evaluación greedy cada 10 000 pasos (20 episodios en CartPole)") y línea 77 ("epsilon=0; 20 episodes each on CartPole").
- **Realidad:** `GreedyEvalConfig.n_episodes` vale 5 por defecto y varios experimentos no pasaron ningún número. Resumen (tabla completa por experimento, por brazo de exp01 y por grupo en `TABLA_EPISODIOS_GREEDY.md` y `TABLA_EPISODIOS_GREEDY_por_grupo.csv`):

| experimento | episodios por punto greedy |
|---|---|
| exp01: `hybrid_fig4`, `matched_classical`, `oversized_mlp`, `paper_linear` | **5** |
| exp01: `cartpole_fourier_ceiling_L5` (y `su2_cartpole_L5`, simulador) | 20 |
| exp02, exp03, exp03b | **5** |
| exp04, exp05, exp06 (FrozenLake y techos clásicos) | 20 (por el script; en exp04 los datos no deciden: retornos 0/1) |
| exp07, exp08 | 20 (compatible con los datos; no hay script en el repositorio) |
| exp09, exp10 (simuladores) | 20 |
| exp11 (nuevo) | **5**, para igualar a las referencias de exp02 |

- **Cómo se sabe:** una puntuación greedy es la media de retornos enteros; con 5 episodios es siempre múltiplo de 0,2 y con 20 de 0,05. En exp02, exp03 y exp03b, las 180 ejecuciones tienen todas las puntuaciones en múltiplos de 0,2, con valores no enteros distintos de sobra para descartar que sea azar; en exp07 y exp08 todos los grupos tienen puntuaciones que un 5 episodios no puede dar. Método y límites en el preámbulo de `TABLA_EPISODIOS_GREEDY.md`.
- **Qué cambia:** el ruido de `greedy_final`. Con 5 episodios es más ruidoso que con 20. No cambia su media esperada, ni `final_perf`, ni `best_ma50`.
- **Efecto sobre los contrastes:** los que comparan un brazo de exp01/exp02/exp03/exp03b (5 episodios) con uno de exp07/exp08 (20) miden `greedy_final` con distinto ruido. Es el caso, entre otros, de `hybrid_OR{R}` (exp02) frente a `hybrid_or_r{R}_noent` (exp08), de `hybrid_fig4` (exp01) frente a `hybrid_fig4_noent` (exp08) y del techo de Fourier de exp01 (20) frente a `hybrid_fig4` (5). Con el brazo más ruidoso la P(mejora) tiende hacia 0,5 sin cambiar su dirección esperada. `final_perf` no tiene este problema y sirve de comprobación.
- **Corrección:** donde la memoria diga que la evaluación greedy usa 20 episodios en CartPole, decir 5 en exp01 (salvo el techo de Fourier), exp02, exp03 y exp03b, y 20 en exp07 y exp08; y añadir una nota en las comparaciones que cruzan ambos grupos.
