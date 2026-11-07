# Punto 3. Hiperparámetros (anexo B)

Fuentes, en este orden de prioridad: (1) la **configuración efectiva** que guardan los manifiestos (`spec.dqn_kwargs` / `spec.ppo_kwargs`, leída con `scripts/p3_effective_hparams.py`; tabla completa por experimento en `p3_effective_kwargs_by_experiment.csv`); (2) los **valores por defecto de la librería vendida**, leídos de las firmas de `src/simplyqrl/dqn.py` y `ppo.py` (`p3_library_defaults_dqn.csv`, `p3_library_defaults_ppo.csv`); (3) el código del bucle de entrenamiento (`src/simplyqrl/dqn.py`, `ppo.py`). Lo que un manifiesto no lista es el valor por defecto de la librería. Ningún valor está escrito de memoria.

## A. DQN

| hiperparámetro | CartPole | FrozenLake | descripción | experimentos | ¿difiere del defecto de la librería? |
|---|---|---|---|---|---|
| `learning_rate` | 2,5e-4 | 2,5e-4 | tasa de aprendizaje de Adam | todos los de DQN | No. **Excepciones:** (i) batería exploratoria exp09 (simulador): 40 ejecuciones de CartPole y 30 de FrozenLake con 1e-3 y 4e-3 (etiquetas `lr0.001`, `lr0.004`); (ii) exp13 multiplica **a propósito** el lr del peso de la cabeza (o el global, bloque C) por k, como parte del diseño |
| optimizador | Adam, β = (0,9; 0,999), ε = 1e-8, sin AMSGrad | igual | `optim.Adam(q_network.parameters(), lr=...)` (`dqn.py:136`) | todos | Es la construcción de la librería sin cambios; β y ε son los valores por defecto de `torch.optim.Adam` (comprobados en la versión instalada) |
| weight decay | 0 | 0 | no se pasa a Adam | todos | No |
| recorte de gradiente | ninguno | ninguno | el bucle de DQN no llama a ningún `clip_grad_*` | todos | No (esto es una condición de la proposición de exp13) |
| planificador de lr | ninguno | ninguno | el lr no se modifica durante el entrenamiento | todos | No |
| función de pérdida | MSE | MSE | `F.mse_loss(td_target, q_old)`, media sobre el lote; sin Huber; objetivo `r + γ·max_a' Q_obj(s',a')·(1−done)`, sin Double DQN | todos | No |
| `gamma` | 0,99 | 0,99 | factor de descuento | todos | No |
| `tau` | 1,0 | 1,0 | actualización de la red objetivo: copia dura | todos | No |
| `target_network_frequency` | 500 | 500 | pasos entre copias de la red objetivo | todos | No |
| `batch_size` | 128 | 128 | tamaño del lote | todos | No (se pasa explícito, igual que el defecto) |
| `buffer_size` | 10 000 | 50 000 | capacidad del buffer de repetición | CartPole: exp01, 02, 03, 03b, 07, 08, 09–13; FrozenLake: exp04, exp09, exp10, exp13 (bloque G) | CartPole: no (defecto 10 000). **FrozenLake: sí**, 50 000 |
| `learning_starts` | 10 000 (defecto) | 1 000 | pasos de exploración antes del primer paso de gradiente (se entrena si `global_step > learning_starts`) | ídem | CartPole: no. **FrozenLake: sí**, 1 000 |
| `train_frequency` | 10 | 1 | un paso de gradiente cada tantos pasos de entorno | ídem | CartPole: no (defecto 10). **FrozenLake: sí**, 1 |
| `start_e` / `end_e` | 1,0 / 0,05 | 1,0 / 0,05 | ε-greedy lineal | todos | No |
| `exploration_fraction` | 0,5 | 0,5 | ε decae durante el 50 % de los pasos (50 000 de 100 000) | todos | No |
| `total_timesteps` | 100 000 | 100 000 | pasos de entrenamiento por semilla | exp01–exp13 salvo: carpeta de 60 000 pasos de exp01, sondeos de 1 500 pasos | Lo fija cada script |
| entornos en paralelo | 1 | 1 | `num_envs = 1` en `dqn.py` | todos | No |
| FIX-01 (reinicio automático) | activo en la mayoría de las ejecuciones; desactivado en 79 (40 de exp01, 9 de la carpeta de 60 000 pasos de exp01, 30 de exp04); el campo no consta en 25 manifiestos antiguos (16 de exp02, 9 de exp03) | idem | descarta la transición «fantasma» que el reinicio automático de gymnasium mete en el buffer (`docs/CORRECTIONS.md#fix-01`) | ver `fix_autoreset` en `p3_effective_kwargs_by_experiment.csv` | Corrección nuestra aplicada en tiempo de ejecución; la librería vendida no se edita |
| evaluación greedy | cada 10 000 pasos | cada 10 000 pasos | `GreedyEvalConfig`; episodios por evaluación: **5 o 20 según el experimento** (tabla ya entregada: `handoff_memoria/TABLA_EPISODIOS_GREEDY.md`; exp13 usa 20) | todos | Defecto de la librería = 5 |
| inicialización de pesos cuánticos | U[0, 2π) | U[0, 2π) | por defecto de `qml.qnn.TorchLayer` | brazos con circuito | No (el emulador tuvo un fallo de inicialización a cero, corregido: FIX-12) |

Por qué FrozenLake difiere (de `docs/EXPERIMENT-04.md`, tabla de protocolo): `train_frequency = 1` porque el valor de 10 «deja sin gradientes» a redes pequeñas hasta enmascarar cualquier otro efecto; `buffer_size = 50 000` porque con una sola recompensa de 1 bit un buffer que olvida los éxitos raros priva de la única señal; `learning_starts = 1 000` porque 10 000 es una fracción grande de una ejecución con episodios cortos. Son los valores de exp04; exp10 y exp13 (bloque G) los repiten.

**Aviso sobre «no se ajustó ningún hiperparámetro»:** el lr se varió en la batería exploratoria exp09 (simulador, 70 ejecuciones, CartPole y FrozenLake) y se multiplica de forma deliberada en exp13. La misma batería exploratoria exp09 (simulador) también varió `buffer_size`, `learning_starts` y `train_frequency` en FrozenLake (48 ejecuciones, etiquetas `p_tf*_ls*_buf*`). Fuera de eso, ningún valor de la tabla se probó con otros valores para elegir el mejor.

## B. PPO del bloque D de exp13 (control clásico del estudio de referencia)

Configuración efectiva: `spec.ppo_kwargs` de los 600 manifiestos de `exp13_ppo` (idéntica en todos). Es `PAPER_PPO_CONFIG` de `experiments/exp13_ppo.py`, copiado literalmente de `src/experiments/*.py` del repositorio de referencia.

| hiperparámetro | valor | descripción | ¿difiere del defecto de la librería? |
|---|---|---|---|
| `learning_rate` | 2,5e-4 | lr inicial de Adam | No |
| `anneal_lr` | `True` | el lr baja linealmente a 0; se escala **cada grupo de parámetros** desde su valor base (en los brazos «lrx» el peso de la cabeza queda en R veces el resto en cada actualización; tests en `tests/test_exp13_ppo.py`) | No |
| optimizador | Adam con ε = 1e-5, β = (0,9; 0,999), sin weight decay | `optim.Adam(..., eps=1e-5)` (`ppo.py:126`) | Es la construcción de la librería |
| `num_envs` | 1 | entornos en paralelo | No |
| `num_steps` | 512 | pasos por despliegue (lote de 512 transiciones por actualización) | No |
| `gamma` | 0,99 | descuento | No |
| `gae_lambda` | 0,95 | GAE | No |
| `num_minibatches` | **16** (minilotes de 32) | minilotes por época | **Sí:** defecto 4. El valor 16 es el del repositorio de referencia |
| `update_epochs` | 4 | épocas por actualización | No |
| `norm_adv` | `True` | normaliza ventajas | No |
| `clip_coef` | 0,2 | recorte de la razón de política | No |
| `clip_vloss` | `True` | recorte de la pérdida del crítico | No |
| `ent_coef` | 0,01 | coeficiente de entropía | No |
| `vf_coef` | 0,5 | coeficiente del crítico | No |
| `max_grad_norm` | 0,5 (recorte de norma global) | en los brazos «noclip» se desactiva; en «modclip» la norma se calcula como √(R‖g_V‖² + ‖g_resto‖²) (código nuestro, `ppo/lr_ppo.py`) | Defecto 0,5; los brazos «noclip» y «modclip» son del diseño |
| `target_kl` | `None` | sin parada anticipada | No |
| `total_timesteps` | 100 000 (se ejecutan 100 352 por el tamaño del despliegue) | pasos de entrenamiento | Lo fija el script |
| evaluación greedy | cada 10 000 pasos, 20 episodios (argmax del actor) | definida en `exp13_ppo.py` | — |
| red | actor y crítico lineales sobre la entrada replicada R veces (`net_arch = []`) | arquitectura, no hiperparámetro | — |

Medición adicional de exp13 (no es hiperparámetro): en los 20 brazos de PPO el recorte de norma global dispara en el 100 % de las actualizaciones (norma mediana 5,6–10 frente al umbral 0,5).
