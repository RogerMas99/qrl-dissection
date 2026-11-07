# Calibración del consumo con CodeCarbon y estimación de emisiones de la campaña

**No cambia ningún resultado de la memoria.** La campaña se ejecutó sin CodeCarbon; aquí se mide ahora el consumo de ejecuciones representativas en este equipo y se extrapola a las horas registradas en los manifiestos (`handoff_cierre_memoria/p4/p4_hours_by_experiment_and_class.csv`). Las ejecuciones de calibración (etiqueta `calib`) están en `results/_calibration_codecarbon/` (ignorado por git), fuera de cualquier cuadrícula, y no se analizan como resultados.

Scripts: `scripts/codecarbon/` (`worker.py`, `calibrate.py`, `run_all.py`, `local_concurrency.py`, `summarize.py`). Tablas derivadas: `docs/codecarbon/`.

## 1. Cómo mide CodeCarbon en este equipo

CodeCarbon 3.3.2, `OfflineEmissionsTracker(country_iso_code="ESP", measure_power_secs=5)`, modo máquina (mide el equipo entero). Equipo: Windows 11, AMD Ryzen AI 7 350 (8 núcleos, 16 hilos), 31 GB de RAM, **más una GPU NVIDIA GeForce RTX 5060 Laptop** (no es «sin GPU»: PyTorch es solo CPU y no la usa, pero CodeCarbon la ve y mide su consumo en reposo). Plan de energía «Equilibrado», con la batería conectada a la red. Lo que dice su log de arranque:

| componente | método | ¿medición o estimación? |
|---|---|---|
| CPU | Windows Energy Meter Interface (contadores tipo RAPL del paquete) | **medición** (no usa TDP) |
| GPU | `pynvml` (NVML) | **medición** (≈ 9 W en reposo) |
| RAM | modelo de estimación: 4 módulos × 5 W (17–64 GB → 4 módulos) = 20 W fijos | **estimación** |

Intensidad de carbono de España: **0,17405 kg CO2-eq/kWh** (174,05 g/kWh), del fichero `global_energy_mix.json` que trae CodeCarbon (año 2023); según su documentación sale de Our World in Data / Ember (ciclo de vida). Fuente de la documentación: <https://docs.codecarbon.io/latest/explanation/methodology/>.

## 2. Diseño de las mediciones

Seis tipos de ejecución, con la configuración de la campaña (ejecuciones consecutivas de semillas distintas hasta agotar el tiempo; sin tocar los 100k pasos):

| tipo | brazo | entorno |
|---|---|---|
| `real_cartpole` | `hybrid_fig4` (circuito real, 8 cúbits, L=5, entrelazado) | CartPole |
| `real_frozen` | `frozen_binary_4q_L5` (circuito real binario entrelazado) | FrozenLake |
| `sv_cartpole` | `hybrid_emu_reuse_r1` (simulador por lotes, como exp13 H) | CartPole |
| `su2_cartpole` | `su2_cartpole_L5` (SU(2), híbrido sin entrelazamiento) | CartPole |
| `classic_dqn` | `classical_twin_reuse_r8` (gemelo clásico, exp13 T) | CartPole |
| `ppo` | `ppo_reuse_clip_r4` (control de referencia, exp13 D) | CartPole |

Dos condiciones por tipo, 12 minutos cada una: (a) **un proceso**, `OMP_NUM_THREADS=1`; (b) **N procesos en paralelo** con un único tracker para la máquina entera y reparto entre N: 14 para el emulador, el clásico y PPO (la campaña usó 14 procesos por bloque de exp13), **8 para los dos circuitos reales** (decisión mía: el número de procesos de la campaña local del circuito real no está registrado). Además, 2 minutos con la máquina en reposo. Ninguna medición salió anómala, así que ninguna se ha repetido; el único ajuste fue de comprobación previa (una prueba de humo de 40 s por tipo, descartada, hecha mientras corría otro análisis).

## 3. Resultados de la calibración

Duración medida por CodeCarbon (≈ 12,1 min por medición). kWh = máquina entera. «kWh/h·proc» = kWh totales / (procesos × horas).

| tipo | condición | procesos | kWh total | CPU / RAM / GPU (kWh) | g CO2-eq | kWh por hora de proceso | potencia media de la máquina (W) | pasos/s por proceso |
|---|---|---|---|---|---|---|---|---|
| reposo | — | 0 | 0,0012 (2,0 min) | 0,0002 / 0,0007 / 0,0003 | 0,21 | — | **35,0** | — |
| `real_cartpole` | un proceso | 1 | 0,0086 | 0,0029 / 0,0040 / 0,0017 | 1,51 | 0,0431 | 43,1 | 18,4 |
| `real_cartpole` | paralelo | 8 | 0,0151 | 0,0094 / 0,0040 / 0,0017 | 2,63 | 0,0094 | 75,2 | 17,4 |
| `real_frozen` | un proceso | 1 | 0,0083 | 0,0026 / 0,0040 / 0,0017 | 1,44 | 0,0413 | 41,3 | 2,5 |
| `real_frozen` | paralelo | 8 | 0,0145 | 0,0088 / 0,0040 / 0,0017 | 2,53 | 0,0090 | 72,2 | 2,1 |
| `sv_cartpole` | un proceso | 1 | 0,0085 | 0,0028 / 0,0040 / 0,0017 | 1,47 | 0,0422 | 42,2 | 160,4 |
| `sv_cartpole` | paralelo | 14 | 0,0146 | 0,0089 / 0,0040 / 0,0017 | 2,54 | 0,0052 | 72,5 | 84,7 |
| `su2_cartpole` | un proceso | 1 | 0,0108 | 0,0051 / 0,0040 / 0,0017 | 1,88 | 0,0537 | 53,7 | 1 327 |
| `su2_cartpole` | paralelo | 14 | 0,0166 | 0,0108 / 0,0040 / 0,0017 | 2,89 | 0,0059 | 82,4 | 595 |
| `classic_dqn` | un proceso | 1 | 0,0114 | 0,0057 / 0,0040 / 0,0017 | 1,98 | 0,0566 | 56,6 | 4 836 |
| `classic_dqn` | paralelo | 14 | 0,0166 | 0,0109 / 0,0040 / 0,0017 | 2,89 | 0,0059 | 82,4 | 2 115 |
| `ppo` | un proceso | 1 | 0,0092 | 0,0034 / 0,0040 / 0,0017 | 1,59 | 0,0456 | 45,6 | 1 730 |
| `ppo` | paralelo | 14 | 0,0166 | 0,0109 / 0,0040 / 0,0017 | 2,89 | 0,0059 | 82,4 | 1 110 |

Fichero: `docs/codecarbon/calibration_table.csv`. Lecturas:

- Con la máquina en reposo consume **35 W** (RAM 20 W, GPU ≈ 9 W, CPU ≈ 8 W); un proceso añade solo 6–22 W; con 14 procesos la CPU llega a ≈ 43–53 W (la máquina entera, ≈ 72–82 W). Por eso el consumo **por hora de proceso** es entre 4,6 y 9,6 veces menor en paralelo que con un proceso: el consumo base se reparte.
- Las tres condiciones de 14 procesos de `su2`, `classic_dqn` y `ppo` dan el mismo total (0,0166 kWh, 53,5 W de CPU): la CPU está saturada y el consumo se estabiliza.
- **Comparación de ritmos con la campaña** (`p4_throughput_by_agent_type_and_recorded_path.csv`, mediana en este equipo): `sv` 84,7 frente a 78; `classic_dqn` 2 115 frente a 1 970 (gemelo); `real_frozen` 2,1–2,5 frente a 2,3. Son comparables. `ppo` (1 110 frente a 873) y `su2` (595 frente a 432) van entre un 27 % y un 38 % más rápidos que en la campaña, que corrió con más procesos que hilos (el bloque H y los bloques T y D a la vez). `real_cartpole` marca 17–18 pasos/s porque los primeros 10 000 pasos (sin entrenar) van a más de 250 pasos/s; el ritmo de entrenamiento de la campaña es 5,5–8,5 pasos/s. El consumo por hora no depende de esto, porque el proceso trabaja todo el tiempo.

## 4. Extrapolación a la campaña

Horas = suma de `wall_seconds` de los manifiestos (6 358 ejecuciones, 4 127,2 h), separadas por dónde escribió cada ejecución sus ficheros: este equipo (2 043,4 h) o Colab (2 083,8 h). Tablas: `docs/codecarbon/extrapolation_by_type.csv` y `extrapolation_totals.csv`.

### 4.1 Horas en este equipo: medición calibrada

Para elegir entre un proceso y paralelo se ha calculado cuántas ejecuciones coincidían en el tiempo (a partir de la fecha de modificación de los CSV de episodios; `local_concurrency.py`). Resultado:

| tipo | horas en este equipo | ejecuciones simultáneas (media ponderada por horas) | condición usada |
|---|---|---|---|
| circuito real (FrozenLake 1 006,5 h + CartPole 20,5 h) | 1 027,1 | 1,0 (1 006,5 h con una sola ejecución viva) | **un proceso** |
| emulador `sv` | 861,6 | 13,7 | paralelo (14) |
| emulador `su2` | 106,7 | 12,3 | paralelo (14) |
| clásico DQN | 28,6 | 17,8 | paralelo (14) |
| PPO | 19,5 | 15,0 | paralelo (14) |

(Solo se cuentan ejecuciones con manifiesto: no entran pruebas, trazas ni análisis.)

| tipo | horas | kWh/h·proc usado | kWh | kg CO2-eq (0,17405) |
|---|---|---|---|---|
| circuito real FrozenLake | 1 006,5 | 0,0413 | 41,55 | 7,23 |
| circuito real CartPole | 20,5 | 0,0431 | 0,88 | 0,15 |
| `sv` | 861,6 | 0,0052 | 4,46 | 0,78 |
| `su2` | 106,7 | 0,0059 | 0,63 | 0,11 |
| clásico DQN | 28,6 | 0,0059 | 0,17 | 0,03 |
| PPO | 19,5 | 0,0059 | 0,11 | 0,02 |
| **Total este equipo** | **2 043,4** | | **47,80** | **8,32** |

El 89 % de esa cifra es el circuito real, que se ejecutó de una en una: el equipo estuvo encendido con una sola ejecución, y casi todo su consumo es el consumo base (RAM, GPU en reposo, CPU en reposo).

Variantes de esta parte (todas con la medición calibrada):
- **Solo la energía por encima del equipo en reposo** (35 W, restando el consumo base): 9,3 kWh, **1,62 kg**. Cota inferior, si el equipo habría estado encendido igualmente.
- **Solo CPU + RAM, sin la GPU en reposo**: 38,5 kWh, **6,70 kg**.
- **Todo lo medido por CodeCarbon (máquina entera, incluida la GPU en reposo)**: 47,8 kWh, **8,32 kg**.

La RAM es una estimación de CodeCarbon (20 W fijos), no una medida; pesa 0,004 kWh de 0,0085–0,017 kWh por medición.

### 4.2 Horas en Colab: supuesto, no medición

El hardware de Colab no consta. Se calcula con un **supuesto explícito**: potencia por proceso igual a la medida aquí para el mismo tipo (rango: condición paralelo a un proceso, 0,009–0,043 kWh/h·proc para el circuito real) y una intensidad de red entre **0,10 y 0,45 kg CO2-eq/kWh**.

| tipo | horas en Colab | kWh (bajo – alto) | kg CO2-eq (bajo – alto) |
|---|---|---|---|
| circuito real CartPole | 2 015,0 | 18,95 – 86,76 | 1,90 – 39,04 |
| circuito real FrozenLake | 60,6 | 0,55 – 2,50 | 0,05 – 1,13 |
| clásico DQN | 8,2 | 0,05 – 0,46 | 0,00 – 0,21 |
| **Total Colab** | **2 083,8** | **19,55 – 89,73** | **1,95 – 40,38** |

Fuentes de los extremos de la intensidad: *Google Cloud, «Carbon data across Google Cloud regions»*, datos de 2024 (<https://github.com/GoogleCloudPlatform/region-carbon-info>; datos horarios de Electricity Maps, sin auditar). Extremo bajo ≈ 0,10: `europe-west1` (Bélgica) 103,4 g/kWh y `us-west1` (Oregón) 79,2 g/kWh. Extremo alto ≈ 0,45: `asia-northeast1` (Tokio) 452,9 g/kWh y `us-central1` (Iowa) 412,7 g/kWh. Colab no publica la región. Fuera de ese rango están los extremos del mundo (2,7 g/kWh en `europe-north2`, 678,8 g/kWh en `asia-south1`). Los extremos de potencia (paralelo y un proceso) abarcan, de forma aproximada, desde un servidor compartido por muchas sesiones hasta una máquina virtual dedicada a una sola; no se sabe cuál fue.

### 4.3 Rango total

| parte | origen | kWh | kg CO2-eq |
|---|---|---|---|
| este equipo (2 043 h) | **medición calibrada** (CodeCarbon, intensidad de España) | 9,3 – 47,8 | 1,6 – 8,3 |
| Colab (2 084 h) | **supuesto** (potencia de aquí, intensidad 0,10–0,45) | 19,6 – 89,7 | 2,0 – 40,4 |
| **Total campaña** (4 127 h) | | **28,9 – 137,5** | **3,6 – 48,7** |

(Mínimo = energía sobre el equipo en reposo en este equipo + extremo bajo en Colab. Máximo = todo lo medido en este equipo + extremo alto en Colab.) Lo calibrado aporta entre 1,6 y 8,3 kg; el resto del rango (hasta 40 kg) es el supuesto sobre Colab. Si en Colab la potencia por proceso fuera la medida aquí en paralelo y la red de 0,10, serían ≈ 2 kg; con el extremo contrario, ≈ 40 kg.

## 5. Equivalencia cotidiana

Un coche nuevo medio de la UE emitió 96,7 g CO2/km en 2025 (Agencia Europea de Medio Ambiente, ensayo WLTP; <https://www.eea.europa.eu/en/newsroom/news/average-co2-emissions-from-new-cars-and-vans-significantly-decreased-in-2025>; es la cifra de los coches nuevos, no del parque en circulación). Con ella:

- parte calibrada de este equipo (1,6 – 8,3 kg): **17 – 86 km** en coche;
- campaña completa (3,6 – 48,7 kg): **37 – 504 km** en coche.

## 6. Límites

- La RAM es una estimación del modelo de CodeCarbon, no una medida; la GPU en reposo (≈ 9 W) está en el total aunque no trabaje.
- El plan de energía era «Equilibrado», con corriente; la campaña pudo ejecutarse con otro. Es un ajuste del sistema que no se ha tocado.
- Las 12 min de cada medición son una ventana: no cubren las variaciones de carga de cien mil pasos (por ejemplo, la fase de recogida inicial de datos es más corta en las ejecuciones largas).
- La concurrencia se deduce de fechas de ficheros; no hay un registro de procesos.
- Las trazas float64 (7,5 h), las pruebas, los análisis y las ejecuciones abortadas no entran: no hay manifiesto o no están registradas.
- La intensidad de España es una media anual de 2023; la real depende de la hora y del año.
