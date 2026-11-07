# Punto 6. Repositorio: qué falta para congelar la versión de entrega

**No se ha creado ningún tag ni se ha hecho ningún commit.** Esto es solo el inventario; cada paso de la lista de abajo necesita tu confirmación.

## 1. Estado del código (git)

- Repositorio `RogerMas99/qrl-dissection`; rama de trabajo **`session/exp10-exp11`** (HEAD `339f2a4`).
- `main` está en `359aa5b` (= base común). La rama de trabajo lleva **8 commits que `main` no tiene**, y **3 de ellos no están subidos** a su rama remota (`origin/session/exp10-exp11` va 3 commits por detrás: los de exp12).
- Otras ramas locales y remotas de sesión (`session/exp09-*`, `session/handoff-v3-1`, `session/statevector-simulator-wave2`, `session/nb12-n10-multisession`, `session/controlled-reruns-…`); `drive-notebooks` va 3 por delante y 49 por detrás de `origin/main`. Ningún tag existe (`git tag` vacío).
- **Cambios sin confirmar en el árbol de trabajo (todo exp13, el addendum y los paquetes de entrega):**
  - modificados (5): `docs/SIMULATED-RESULTS-POLICY.md`, `src/qrl_dissection/core/compat.py`, `core/configs.py`, `dqn/runner.py`, `dqn/safe.py`;
  - sin seguimiento: `docs/EXPERIMENT-13-OR-VS-LR.md`; `experiments/exp13_*.py` (9 ficheros); `src/qrl_dissection/core/lr_efg.py`, `core/lr_equivalence.py`, `ppo/lr_ppo.py`; `tests/test_exp13_{dqn,ppo,efg}.py`; `scripts/closing_data/` (los scripts de este paquete); `handoff_exp13/`, `handoff_exp13_parcial/`, `handoff_cierre_memoria/`; `handoff_exp11/LEEME_PARA_EL_OTRO_CHAT.md`; `exp09_logs/` (1,7 MB de registros).
  - Los `*.zip` y `results/` están en `.gitignore` (los zip de entrega no se versionan).

## 2. Qué falta para poder regenerar todas las tablas y figuras

| pieza | estado |
|---|---|
| Código (`src/`, `experiments/`, `scripts/`, `tests/`, `notebooks/`) y documentos de diseño (`docs/`) | versionado salvo lo de la sección 1 |
| Configuraciones | están en `core/configs.py` (versionado, con las modificaciones de exp13 sin confirmar) |
| Manifiestos y CSV de cada ejecución | **fuera del repositorio** (`.gitignore`): 6 366 manifiestos, 14 889 CSV, en Drive (`tfm_qrl/results`) |
| Scripts de análisis exp09 (batería), exp10, exp11, exp12, exp13 (A–D y E/F/G), auditoría de episodios greedy | están y se ejecutan contra los resultados |
| **Script que genera las tablas del dossier v3 (exp01–exp08): `per_seed_metrics.csv`, `summary_by_arm.csv`, curvas v3.3** | **no está en el repositorio.** Solo están los ficheros de salida (`handoff_memoria/`), añadidos en los commits `df7ee45` y `07c0446`. Los scripts de exp10–exp13 repiten sus columnas, pero no son ese script. Hay que escribirlo (o recuperarlo de donde se ejecutó) y comprobar que reproduce los CSV entregados |
| Figuras de la memoria | no se ha podido localizar ningún script del repositorio que genere las figuras de la memoria (solo hay gráficos exploratorios en `notebooks/20_dqn_results.ipynb` y en `core/analysis.py`); no se sabe de dónde salen. **Pregunta pendiente al autor** |
| Entorno | `requirements.txt` y `pyproject.toml` fijan `pennylane==0.41.1`, `gymnasium==1.1.1`, `autoray==0.7.1`; el entorno que ejecutó exp13 tiene `torch 2.13.0+cpu`, `numpy 2.4.6`, `pandas 3.0.5`, `scipy 1.17.1`, Python 3.11 (rango admitido por los pins). No hay un `pip freeze`/lock del entorno exacto; conviene guardarlo |
| exp05, exp06 | sus resultados no tienen carpeta propia en los manifiestos (ver `TABLA_EPISODIOS_GREEDY.md`); hay que decidir de dónde se rehacen las cifras que la memoria use de ellos |
| Versiones de código por ejecución | cada manifiesto lleva el hash de git con que se ejecutó; las ejecuciones antiguas migradas (`_migrated_by`) no lo traen en todos los casos. Los hashes deben existir en el historial publicado |

## 3. Tamaño aproximado de lo que habría que publicar

- **Repositorio de código:** `git` ocupa 3,7 MiB (paquete de 0,66 MiB); el árbol de trabajo sin `.git` pesa 98 MiB, casi todo paquetes de entrega y registros locales no versionados (`handoff_*` 28 + 26 + 14 MiB, `logs/` 5,7 MiB).
- **Resultados (en Drive, 4 787 MiB en total):**

| contenido | tamaño | ¿hace falta para regenerar tablas y figuras? |
|---|---|---|
| manifiestos (`*.manifest.json`, 6 366) | 11 MiB | sí |
| CSV por ejecución (episodios, evaluación greedy, diagnósticos; 14 889) | 391 MiB | sí |
| trazas `.npy` (6 366) | 636 MiB | no las lee ningún script de análisis que se haya podido localizar; útiles como dato bruto |
| pesos `.pt` (5 403) | 24 MiB | no para las tablas; sí para reevaluar políticas |
| registros de TensorBoard (6 738) | 3 724 MiB | no |
| copias `.bak` y varios | 1 MiB | no |

  Lo mínimo para regenerar: manifiestos + CSV ≈ **400 MiB**; con trazas y pesos ≈ 1,06 GiB; todo ≈ 4,7 GiB.
- **No caben en el repositorio de GitHub (código):** los CSV (400 MiB) y menos aún los 4,7 GiB. Sitio razonable: un depósito de datos con DOI aparte (p. ej. Zenodo, que admite decenas de GiB por registro), enlazado desde el README del repositorio y desde la memoria; o una GitHub Release para el paquete mínimo (≈400 MiB repartidos en archivos de menos de 2 GiB). Los registros de TensorBoard no hace falta publicarlos.
- Dos carpetas de la raíz de resultados quedan fuera del recuento por ser copias: `OtherDriveAccounts` (1 766 manifiestos de otras cuentas de Drive; **todos están ya en `results/`**, comprobado por experimento y nombre de fichero) y tres carpetas de la raíz de `tfm_qrl` que no son resultados de los experimentos numerados: `exp01` (25 MiB: sondeos y dos figuras `.png`), `autoreset_regime` (172 MiB) y `bridge_capacity` (177 MiB), estas dos con figuras, salidas de diagnóstico y su propia copia del repositorio. No se han inspeccionado a fondo; si la memoria usa alguna de sus figuras, habrá que decidir si entran en el paquete.

## 4. Licencia y dependencias

- **Licencia actual del código:** MIT, «Copyright (c) 2026 Roger» (`LICENSE`; `pyproject.toml` declara `MIT`).
- **Código de terceros incluido:** `src/simplyqrl/` es una copia **sin modificar** de SimplyQRL (revisión `b534cc9`) de Javier Lázaro González. El repositorio del artículo que la contiene tiene licencia **MIT, Copyright (c) 2025 Javier Lázaro González**. **Punto a resolver:** la licencia MIT exige conservar el aviso de copyright y el texto de la licencia en las copias; `src/simplyqrl/` no lleva su propio fichero de licencia y `LICENSE` solo nombra a «Roger». Hace falta añadir el aviso de ese autor (por ejemplo `src/simplyqrl/LICENSE` con el texto del original) antes de publicar. No lo he hecho. Comprobado también en el `LICENSE` del repositorio upstream de SimplyQRL (`javier-lazaro/SimplyQRL`, rama `main`): MIT, «Copyright (c) 2025 Javier Lázaro González» (lectura de la web, sin modificar nada).
- **Dependencias** (licencias leídas de los metadatos del entorno instalado): gymnasium 1.1.1 MIT; PyTorch 2.13 BSD-3 / Apache-2.0 (metadatos compuestos); PennyLane 0.41.1 y pennylane-lightning 0.41.1 Apache-2.0; autoray 0.7.1 Apache; numpy BSD-3 (con componentes 0BSD, MIT, Zlib, CC0); pandas BSD-3; scipy BSD-3; matplotlib licencia propia permisiva (PSF); tensorboard Apache-2.0; cloudpickle BSD-3; tqdm MPL-2.0 y MIT; pytest MIT. **Ninguna es copyleft fuerte que condicione la licencia MIT del código**; tqdm (MPL-2.0) tiene copyleft de fichero, que solo afecta si se modificaran sus propios ficheros. Los datos de `data/` (curvas de PPO del artículo) provienen del repositorio del artículo (MIT); conviene citar su origen.

## 5. Lista de pasos para congelar (nada hecho sin tu confirmación)

1. Decidir qué ramas entran: probablemente subir `session/exp10-exp11`, integrar en `main` (exp10–exp13) y descartar o archivar las ramas de sesión.
2. Confirmar y hacer commit de todo lo de la sección 1 (código, tests, scripts de análisis, documento de diseño de exp13).
3. Añadir el aviso de copyright/licencia de SimplyQRL.
4. Localizar o reescribir el generador de las tablas v3 (exp01–exp08) y los scripts de las figuras; comprobar que reproducen lo entregado.
5. Guardar el entorno exacto (`pip freeze`) y las versiones.
6. Preparar el paquete de datos mínimo (manifiestos + CSV, ≈400 MiB) y decidir dónde se publica (Zenodo o Release).
7. Crear el tag (por ejemplo `v1.0-memoria`) y, si procede, la Release, con el DOI de los datos en el README.
