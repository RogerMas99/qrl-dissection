# Punto 4. Coste y huella (sección 4.7 y anexo E)

Todo sale de los **manifiestos** (`outcome.wall_seconds` de cada ejecución; 6 366 manifiestos leídos, 0 ilegibles) con `scripts/p4_collect_manifests.py` → `p4_all_manifests.csv`, y de `scripts/p4_summarize.py`. Una ejecución = una semilla de un brazo. Las horas son **tiempo de reloj por ejecución, sumadas**: las sesiones se solapan, no es tiempo de calendario. Se excluyen 8 ejecuciones de la carpeta `_quarantine_dup_writers` de exp10 (1,18 h; lista en `p4_excluded_quarantine_runs.csv`).

Clasificación: **circuito real** = `agent_type = hybrid` (PennyLane); **emulador** = `su2` o `sv`; **clásico** = el resto. 76 manifiestos antiguos de exp02, exp03 y exp03b no traen `agent_type`; todos se llaman `hybrid_*` y se han clasificado como circuito real (columna `agent_type_inferred_from_run_name`).

## 1. Horas de reloj por experimento

| experimento | circuito real | emulador | clásico | total (h) | ejecuciones |
|---|---|---|---|---|---|
| exp01 (CartPole, capacidad) | 184,85 | 0,55 | 0,93 | 186,33 | 110 |
| exp01, carpeta de 60 000 pasos | 0 | 0 | 0,16 | 0,16 | 18 |
| exp02 (Output Reuse) | 356,28 | 0 | 1,19 | 357,47 | 121 |
| exp03 (Data Reuploading) | 151,26 | 0 | 0 | 151,26 | 31 |
| exp03b (DR sin entrelazamiento) | 137,43 | 0 | 0 | 137,43 | 30 |
| exp04 (FrozenLake) | 1 067,16 | 0 | 6,66 | 1 073,82 | 230 |
| exp07 (réplica del artículo) | 534,57 | 0 | 0,59 | 535,17 | 240 |
| exp08 (seguimientos) | 671,13 | 0 | 0 | 671,13 | 130 |
| exp09 (batería de simuladores) | 0 | 170,60 | 0 | 170,60 | 888 |
| exp10 (adicionales de 100 semillas, simuladores) | 0 | 580,14 | 0 | 580,14 | 2 300 |
| exp11 | 0 | 0 | 0,31 | 0,31 | 30 |
| exp12 | 0 | 0 | 0,86 | 0,86 | 100 |
| exp13, bloques E, F, G | 0 | 0 | 16,10 | 16,10 | 330 |
| exp13, bloques A, B, C (T clásico y H emulador) | 0 | 217,03 | 9,98 | 227,01 | 1 200 |
| exp13, bloque D (PPO) | 0 | 0 | 19,46 | 19,46 | 600 |
| **TOTAL** | **3 102,69** | **968,32** | **56,23** | **4 127,24** | **6 358** |

Fichero: `p4_hours_by_experiment_and_class.csv`. (exp05 y exp06 no tienen carpeta de resultados propia, ver `TABLA_EPISODIOS_GREEDY.md`, así que no aparecen.)

## 2. La cifra «unas 3 100 horas»

- **Es la cifra del dossier v3 y cubre solo exp01–exp08**: la suma de `wall_h` de su `per_seed_metrics.csv` (888 filas) es **3 112,2 h**; la suma de todos los manifiestos de exp01–exp08 es **3 112,8 h** (910 ejecuciones; la diferencia de 0,6 h (22 ejecuciones) corresponde a ejecuciones que el dossier no lleva, por ejemplo las del emulador de exp01, 0,55 h).
- **No incluye** exp09, exp10 (los adicionales de 100 semillas), exp11, exp12 ni exp13: **1 014,5 h más (5 448 ejecuciones)**.
- **Cifra actualizada: 4 127 h** de reloj sumadas en 6 358 ejecuciones. Por clase: 3 103 h de circuito real (PennyLane), 968 h de emulador, 56 h de redes clásicas. Si la memoria cuenta solo lo que no es simulador, son 3 159 h (3 103 + 56).
- **Qué no entra en ninguna de estas cifras** (no hay manifiesto o no se puede recuperar): las 8 ejecuciones en cuarentena (1,2 h); los cálculos float64 de comprobación de exp13 (trazas, 7,5 h según los JSON de `exp13_traces`: híbrido 5,3 h, PPO 1,5 h, gemelo y E/F 0,6 h); los tests y las ejecuciones de humo; los análisis; las ejecuciones abortadas o repetidas que sobrescribieron un manifiesto (no dejan rastro). No se estiman.

## 3. Rendimiento medido y hardware

Mediana de pasos por segundo (percentiles 10–90 entre paréntesis), calculada como `total_timesteps / wall_seconds` por ejecución; tabla completa en `p4_throughput_by_agent_type_and_recorded_path.csv`:

| tipo de brazo | entorno | dónde (según la ruta guardada en el manifiesto) | ejecuciones | pasos/s mediana (p10–p90) | h por ejecución, mediana |
|---|---|---|---|---|---|
| circuito real (PennyLane) | CartPole | Colab | 448 | 8,5 (2,9–19,9) | 3,25 |
| circuito real | CartPole | este equipo | 4 | 5,5 (5,0–5,9) | 5,12 |
| circuito real | FrozenLake | este equipo | 70 | 2,3 (1,1–4,0) | 12,13 |
| circuito real | FrozenLake | Colab | 10 | 4,3 (4,0–5,8) | 6,52 |
| emulador `sv` (statevector) | CartPole | este equipo | 2 440 | 78 (71–244) | 0,35 |
| emulador `sv` | FrozenLake | este equipo | 320 | 60 (57–152) | 0,46 |
| emulador `su2` | CartPole | este equipo | 610 | 432 (400–497) | 0,06 |
| emulador `su2` | FrozenLake | este equipo | 378 | 167 (82–473) | 0,17 |
| clásico (MLP, gemelo, techo de Fourier…) | CartPole | este equipo | 1 682 | 870–1 970 según el tipo | 0,01–0,03 |
| clásico (`classic`) | CartPole | Colab | 186 | 1 887 (1 548–2 315) | 0,01 |
| MLP sobre bits (exp13, G) | FrozenLake | este equipo | 60 | 217 (199–231) | 0,13 |

**Hardware, lo que se sabe y lo que no:**

- **Los manifiestos NO registran el equipo** (ni CPU, ni GPU, ni sesión de Colab). Lo único que permite una pista es la ruta donde la ejecución escribió sus CSV: `/content/drive/MyDrive/...` indica un entorno Colab con Drive montado; `C:\Users\...\Mi unidad\...` indica este equipo con Drive de escritorio. Se ha usado como pista (columna `recorded_path_kind`), **no** como registro de hardware.
- **Este equipo (el único con hardware comprobable hoy):** Windows 11, AMD Ryzen AI 7 350 (8 núcleos, 16 hilos), 31 GB de RAM, PyTorch 2.13 solo CPU (`torch.cuda.is_available() = False`); 8 hilos de PyTorch por defecto. Las ejecuciones largas se lanzaron con varios procesos en paralelo, así que los pasos/s dependen de la carga (exp13: `OMP_NUM_THREADS=1`; sin ello las redes pequeñas bajo carga iban entre 25 y 60 veces más lentas).
- **Colab:** el tipo de entorno de ejecución (CPU/GPU/TPU, modelo) **no está registrado en ningún sitio del repositorio ni de los manifiestos**; no se puede recuperar ni se estima. El propio `docs/REUSE.md` avisa de que la velocidad del circuito varía varias veces entre una CPU de Colab, una GPU de Colab y un portátil.
- La mayor parte de las horas de circuito real son de Colab según la ruta (exp01–03b, exp07, la mayor parte de exp02 y exp08); casi todo exp04 (FrozenLake, 1 007,6 de 1 073,8 h) se ejecutó en este equipo; la batería exploratoria y los experimentos 9–13 se ejecutaron en este equipo.
- Versiones de Python anotadas en los manifiestos: 3.11 (exp09–exp13 y parte de exp01/02/04), 3.12 (la mayor parte de exp01 y la carpeta de 60 000 pasos) y 3.13 (exp07/08 y algo de exp04). No identifican el equipo.

## 4. La diferencia 9,40 h frente a 8,10 h por semilla (hybrid sin reuse frente a R = 32)

Datos por semilla: `p4_per_seed_hours_hybrid_noreuse_vs_OR.csv`; resumen: `p4_hybrid_hours_per_seed_summary.csv`. Las dos cifras salen de FIX-01 activo, n = 10, 100 000 pasos, mismo circuito (8 cúbits, L=5, entrelazado), todas con ruta de Colab.

| brazo | n | media (h) | desv. típica | mínimo | máximo | semillas por debajo de 7 h |
|---|---|---|---|---|---|---|
| `hybrid_fig4` (sin reuse) | 10 | **9,40** | 1,24 | 6,17 | 10,82 | 1 |
| `hybrid_OR4` | 10 | 9,08 | 1,52 | 6,21 | 10,61 | 2 |
| `hybrid_OR8` | 10 | 8,85 | 1,46 | 6,18 | 10,83 | 2 |
| `hybrid_OR16` | 10 | 9,60 | 1,12 | 6,75 | 11,12 | 1 |
| `hybrid_OR32` | 10 | **8,10** | 1,70 | 6,07 | 10,64 | 3 |

**Lectura:** la diferencia (1,30 h) es menor que la dispersión entre semillas **dentro de cada brazo** (desv. típica 1,2–1,7 h; en todos los brazos hay semillas de 6–7 h y otras de 9–11 h). Welch p = 0,067; Mann-Whitney p = 0,076, o sea, no es una diferencia detectable. La mezcla de tiempos de cada brazo se explica por cuántas de sus semillas cayeron en una ejecución rápida (3 de 10 en R = 32; 1 de 10 sin reuse). Esto es coherente con un reparto desigual entre sesiones/equipos de Colab, pero **los manifiestos no permiten asignar cada semilla a un equipo**, así que no se puede demostrar; lo que sí se puede decir es que, en este circuito, el coste lo domina la simulación del circuito y la cabeza más ancha de R = 32 no lo cambia de forma medible. Los tiempos por semilla de la tabla 4.7 no son comparables entre experimentos (ya lo dice el dossier v3: hardware mixto).

## 5. CodeCarbon / emisiones

**No se registraron emisiones.** CodeCarbon no está instalado en el entorno, no aparece en `requirements.txt`, `pyproject.toml`, scripts ni cuadernos, y no hay ningún fichero de emisiones en el repositorio ni en la carpeta de resultados de Drive (búsqueda por `emissions*` y `*carbon*`). No hay kg de CO2-eq medidos, ni parciales, y **no se extrapola ninguna cifra**. Si la memoria necesita una estimación, tendría que presentarse como cálculo explícito a partir de las horas de la sección 1 y de una potencia y una intensidad de red asumidas por el autor, sin mezclarlo con medidas.

## 6. Cómputo evitado por reutilización

- **Medible: exp10.** 200 ejecuciones de los simuladores (la carpeta de exp10 tiene 2 300 ejecuciones nuevas; estas 200 se suman a ellas sin repetirse) se reutilizaron de carpetas anteriores en lugar de repetirse (`reused_seeds.json`; lista y horas de cada una en `p4c_reused_runs_exp10.csv`, script `p4c_reused_runs.py`): **48,2 horas ahorradas** (suma del reloj de las 200 ejecuciones fuente). Son brazos de emulador (`su2_cartpole_L5` de exp01 y otros de exp09).
- **No medible:** la reanudación de cuadrículas (una celda con manifiesto no se vuelve a ejecutar) y la adopción de resultados previos al migrar el diseño de carpetas no dejan registro de cuántas celdas se saltaron; **no se estima**. En exp13 las referencias reales (10 semillas de exp01/exp04/exp11/exp12) se usaron solo en el análisis; ningún cálculo de ellas se repitió.
- Las ejecuciones duplicadas por dos sesiones a la vez (el sistema de reserva de celdas es «de mejor esfuerzo») están en `_quarantine_dup_writers` (8 ejecuciones, 1,18 h) y no se cuentan en ninguna cifra.
