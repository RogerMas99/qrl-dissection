# Datos para cerrar la memoria (sin entrenamientos nuevos)

No se ha lanzado ningún entrenamiento. Todo sale de código del repositorio, de los manifiestos y de los CSV de `tfm_qrl/results`. Cada cifra lleva el fichero y el script de donde sale. **Lo que no se ha podido obtener está en la última sección.** No se ha cambiado ningún número del circuito real ni se afirma nada sobre cuándo se hizo cada experimento respecto a otro. No se ha hecho commit ni tag.

| punto | documento | datos | script (en `scripts/`) |
|---|---|---|---|
| 1. Revalidación de los simuladores | `p1/P1_revalidacion_simuladores.md` | `p1/p1_battery_revalidation.md`, `p1/p1_pairs_old_vs_new.csv`, copia de `docs/EXPERIMENT-09-BATTERY.md` | `exp09c_battery_report.py` (la regla, sin cambios), `p1_compare_battery.py` |
| 2. Recuentos de parámetros | `p2/P2_recuentos_de_parametros.md` | `p2/p2a_paper_configs_params.csv`, `p2/p2b_frozenlake_su2_and_linear_ceiling_params.csv` | `p2_param_counts.py` |
| 3. Hiperparámetros | `p3/P3_hiperparametros.md` | `p3/p3_effective_kwargs_by_experiment.csv`, `p3/p3_library_defaults_{dqn,ppo}.csv` | `p3_effective_hparams.py` |
| 4. Coste y huella | `p4/P4_coste_y_huella.md` | `p4/p4_*.csv` (la tabla base, `p4_all_manifests.csv`, tiene una fila por ejecución) | `p4_collect_manifests.py`, `p4_summarize.py`, `p4c_reused_runs.py` |
| 5. FrozenLake, semillas resueltas | `p5/P5_frozenlake_semillas_resueltas.md` | `p5/p5_frozenlake_solved*.csv` | `p5_frozenlake_solved_last_checkpoint.py` |
| 6. Repositorio | `p6/P6_repositorio.md` | — (inventario) | — |

## Respuestas cortas

1. **Revalidación:** con las 10 semillas reales en todos los brazos, d medio = **+0,090**, IC 95 % (**+0,008, +0,175**) → criterio 1 **no se cumple** (fallo por 0,008); **1 de 63** combinaciones fuera de [0,25, 0,75] → criterio 2 se cumple; **0 de 9** comparaciones entre simuladores fuera → criterio 3 se cumple; veredicto global **no aceptado**. Evaluación original: +0,119 (+0,010, +0,250), 3 de 63, 0 de 9, no aceptado. Las 21 parejas informativas son las mismas; ninguna cambió de bandera. Esta reevaluación ya estaba hecha en `docs/EXPERIMENT-09-BATTERY.md` (sección del 2026-11-01); regenerada hoy da las mismas cifras.
2. **Parámetros:** las 24 configuraciones están medidas (tabla en `p2`). Confirmados **18, 28 y 60** para el emulador SU(2) de FrozenLake (Config A L=5, Config B sin entrelazamiento L=1 y L=5) y **20** para el techo lineal del circuito binario.
3. **Hiperparámetros:** tablas de DQN y de PPO (bloque D) en `p3`. FrozenLake difiere del defecto de la librería en `buffer_size` (50 000), `learning_starts` (1 000) y `train_frequency` (1). El PPO usa `num_minibatches = 16` (defecto de la librería: 4; es el valor del repositorio de referencia).
4. **Coste:** **4 127 h** de reloj sumadas (6 358 ejecuciones): 3 103 h circuito real, 968 h emulador, 56 h clásico. La cifra «3 100 h» del dossier v3 cubre **solo exp01–exp08** (3 112 h) y **no incluye** exp09–exp13 (1 014,5 h más). Los manifiestos **no registran el hardware**; CodeCarbon **no se usó** (sin datos de emisiones). La diferencia 9,40 h frente a 8,10 h está dentro de la dispersión entre semillas del propio brazo (Welch p = 0,067) y no se puede atribuir a un equipo concreto. Reutilización medible: 200 ejecuciones de exp10, 48,2 h.
5. **FrozenLake, último checkpoint:** `mlp_bits_w6` **0/30**, `mlp_bits_w12` **19/30**; circuito binario entrelazado L=1 **0/10** y L=5 **9/10** (coincide con la tabla 4.4). Con el criterio del dossier de exp13 (tres últimos puntos) salen 15/30 y 7/10.
6. **Repositorio:** nada comprometido de exp13 ni del addendum; hay que añadir el aviso de copyright de SimplyQRL (MIT, J. Lázaro González) y localizar o reescribir el generador de las tablas v3. Detalle y tamaños en `p6`.

## Cosas que conviene saber antes de redactar

- **Errata menor del dossier de exp13 ya entregado (`handoff_exp13/DOSSIER_EXP13.md`, no se ha editado):** su sección 6 lista «PPO: 16 minibatches» entre las desviaciones. En realidad 16 es el valor del repositorio de referencia (`PAPER_PPO_CONFIG`); lo que difiere es el valor por defecto de la librería (4). No es una desviación respecto al estudio de referencia.
- **«Ningún hiperparámetro se ajustó» necesita matiz:** la batería exploratoria exp09 (simulador) varió el lr (1e-3 y 4e-3; 70 ejecuciones) y, en FrozenLake, `buffer_size`, `learning_starts` y `train_frequency` (48 ejecuciones). Los resultados principales no se eligieron con eso, pero la frase literal «el lr nunca se varió» no es exacta si se refiere a todo el trabajo. Está en `p3`.
- **Clasificación de horas:** 76 manifiestos antiguos de exp02/03/03b no traen `agent_type`; se han contado como circuito real por su nombre (`hybrid_*`). Si se clasificaran por defecto como clásicos, las horas de circuito real y de clásico cambiarían en 421,3 h (76 ejecuciones); está marcado en `p4_all_manifests.csv` (columna `agent_type_inferred_from_run_name`).

## Lo que NO se ha podido obtener

- **Hardware de las ejecuciones de Colab** (CPU, GPU, modelo, sesión): no consta en ningún sitio. Solo la pista de la ruta de los CSV (Colab frente a este equipo). Del equipo local sí: AMD Ryzen AI 7 350, 8 núcleos y 16 hilos, 31 GB, PyTorch solo CPU.
- **Asignación de cada semilla a un equipo** (necesaria para demostrar la explicación de 9,40 h frente a 8,10 h).
- **Emisiones de CO2** (nunca se midieron).
- **Horas de ejecuciones abortadas, repetidas o sobrescritas, de tests y de análisis**, y cuántas celdas se saltaron al reanudar cuadrículas: no dejan registro y no se estiman. Las trazas float64 de exp13 (7,5 h) existen en los JSON, pero no son manifiestos y no entran en la suma de 4 127 h.
- **El script que generó las tablas v3 de exp01–exp08** y **el origen de las figuras de la memoria**: no están en el repositorio (ver `p6`).
- **Si exp05 y exp06 tienen resultados utilizables**: no tienen carpeta propia.
