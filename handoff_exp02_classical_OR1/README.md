# Brazo añadido: `classical_OR1` (control clásico de referencia con R = 1, DQN, CartPole)

**Brazo añadido fuera del plan registrado**, para completar el eje R de la figura 4.2 (R = 4, 8, 16, 32 ya existían). Ninguna celda ni resultado existente se ha tocado: las ejecuciones están en una carpeta nueva, `exp02_classical_OR1/`, de `tfm_qrl/results`. Se calculó en este equipo (AMD Ryzen AI 7 350, CPU, torch 2.13.0+cpu).

## Qué es el brazo

| | |
|---|---|
| Red | `Linear(3 → 2)`, **8 parámetros** (medido, ver abajo) |
| Entrada | `[x_dot, theta, theta_dot]` (componentes 1, 2, 3 de la observación; se descarta la posición del carro) |
| Nombre de celda | `classical_OR1__s{semilla}`, semillas 1 a 10 |
| Protocolo | el de `classical_OR{4,8,16,32}` de exp02: `agent_type` "classic", 100 000 pasos, `fix_autoreset=True` (FIX-01), `batch_size=128`, `buffer_size=10_000`, `train_frequency=10`, resto por defecto; evaluación greedy cada 10 000 pasos, 5 episodios, mismas semillas de evaluación |

**Por qué la selección está en el entorno.** El `simplyqrl` incluido en el repositorio solo aplica `reuse_indices` cuando `n_repeats > 1` (`src/simplyqrl/agents.py`, líneas 137-143). Con `paper_classical_or_config(1)` el agente recibiría las 4 componentes (`Linear(4 → 2)`, 10 parámetros). Para que el brazo sea el mismo control con una sola copia, las componentes [1, 2, 3] se seleccionan en el entorno con el wrapper `SelectObs` (`make_selected_env`, el mecanismo de exp13 PPO, `src/qrl_dissection/ppo/lr_ppo.py`) y la configuración del agente es `{"n_repeats": 1, "net_arch": []}`. La evaluación greedy usa el mismo wrapper. `simplyqrl` no se ha modificado; `SafeDQN` y `GreedyEvalConfig` han ganado un parámetro opcional `env_factory` (por defecto `None`, comportamiento anterior intacto; los 34 tests de DQN/seguro/runner/autoreset pasan). Cada manifiesto lleva un campo `note` y `obs_selection` que lo dicen.

## Comprobaciones previas al lanzamiento (`experiments/exp02_classical_or1.py check`)

- (a) **Parámetros:** el agente construido tiene **8** parámetros entrenables (capa `Linear` con pesos de forma (2, 3) y sesgo de 2).
- (b) **Observación que llega a la red:** dimensión **3**. El wrapper devuelve exactamente `obs[1:]` del entorno sin envolver, durante un episodio completo con la misma semilla; la primera observación del DQN con semilla 1 es `[0.04505, -0.03558, 0.04486]` = `[x_dot, theta, theta_dot]` de la observación cruda `[0.00118, 0.04505, -0.03558, 0.04486]`; en una ejecución corta, la red solo vio entradas de forma (3,); la evaluación greedy también devuelve observaciones de dimensión 3. Resultado: `CHECK OK`.

## Resultados (10 semillas)

| métrica | IQM | IC bootstrap 95 % |
|---|---|---|
| final (último 10 % de retornos de entrenamiento) | **9,61** | (9,58, 9,63) |
| greedy (media de los 3 últimos checkpoints) | **9,50** | (9,30, 9,67) |

(`stats.iqm` y `stats.bootstrap_ci`, semilla 0, 10 000 remuestreos: las mismas funciones que `summary_by_arm.csv`.) Para situarlo, las filas ya entregadas de `classical_OR{4,8,16,32}` (`handoff_memoria/summary_by_arm.csv`) tienen IQM final 9,60 / 9,63 / 9,64 / 10,22 e IQM greedy 9,47 / 9,47 / 9,60 / 10,02. Con estos hiperparámetros, el control de 8 parámetros no aprende CartPole (retornos cercanos al mínimo, ≈ 9,5); los valores se parecen a los de R = 4…16.

Las métricas por semilla usan las mismas definiciones que `scripts/build_dossier_v3.py`. Se comprobó que la función que las calcula reproduce con diferencia 0 las 20 filas de `classical_OR4` y `classical_OR8` de `per_seed_metrics.csv`.

## Comprobación de reproducibilidad (`classical_OR4`, semilla 1, fix = on)

**Los valores no coinciden exactamente; la comparación es estadística, no exacta.** El original se corrió en Colab (las rutas del manifiesto lo indican). Aquí, en este equipo y con el mismo código y kwargs:

| | final | greedy | episodios |
|---|---|---|---|
| guardado | 9,5829 | 9,6 | 8 316 |
| repetición 1 | 9,5793 | 9,6 | 8 321 |
| repetición 2 (diagnóstico) | 9,5798 | 9,6 | 8 330 |
| repetición 3 (diagnóstico, 1 hilo) | 9,5911 | 9,6 | 8 293 |

En esta máquina no hay determinismo bit a bit: dos repeticiones idénticas difieren entre sí en el tercer decimal de `final`, y el greedy coincide (9,6). No se investigó más la causa. Detalle de la repetición 1 en `repro_check.json`. Las trayectorias de `classical_OR1` heredan esa limitación: no se puede garantizar que otra ejecución dé exactamente las mismas cifras, sí cifras equivalentes.

## Cómputo

- 10 ejecuciones, **18,9–20,8 s cada una, 197,4 s en total (0,055 h)**; 204 s de reloj con el arranque.
- Para comparar: `classical_OR4` semilla 1 tardó 45,2 s (guardado, Colab) y 28,5 s aquí. Los tiempos no son comparables entre equipos.
- Fracción de transiciones fantasma descartadas por FIX-01 (`probe`): 7,8–8,3 %, igual que en R = 4…32.

## Contenido

| fichero | contenido |
|---|---|
| `per_seed_classical_OR1.csv` | una fila por semilla; mismas columnas que `per_seed_metrics.csv`; `exp = exp02`, `arm = classical_OR1`, `fix = on` |
| `curves_classical_OR1_train_ma50.csv` | media móvil (50 episodios) del retorno de entrenamiento cada 1 000 pasos; mismo formato que `curves_train_ma50.csv` |
| `curves_classical_OR1_eval_greedy.csv` | retorno greedy cada 10 000 pasos; mismo formato que `curves_eval_greedy.csv` |
| `summary_classical_OR1.json` | IQM e IC, tiempos |
| `repro_check.json` | comprobación de `classical_OR4` semilla 1 |
| `manifests/` | los 10 manifiestos de `classical_OR1` (con `note` y `obs_selection`) |
| `experiments/exp02_classical_or1.py` | el script (`check`, `run`, `repro`, `tables`, `validate`, `params`) |

Regenerar las tablas: `python experiments/exp02_classical_or1.py --results-root <results> tables --out <dir>`.
