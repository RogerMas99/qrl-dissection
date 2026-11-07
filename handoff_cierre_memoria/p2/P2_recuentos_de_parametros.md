# Punto 2. Recuentos de parámetros (anexo D)

Todos los recuentos son **medidos**: se construye el agente con `build_arm_config` + `build_agent_for` (`core/capacity.py`) y se cuentan los parámetros con `requires_grad`. Script: `scripts/p2_param_counts.py`. Datos completos (con tipo de agente y entorno): `p2a_paper_configs_params.csv`, `p2b_frozenlake_su2_and_linear_ceiling_params.csv`.

## a) Las 24 configuraciones del artículo transcritas en `core/configs.py`

Son los brazos de `PAPER_ARM_TO_RESULTS`: 6 Skolik (4 y 8 cúbits × L 1, 2, 5) + 6 Salinas (1 y 2 cúbits × L 1, 2, 5) + 4 Hsiao con Output Reuse (R 4, 8, 16, 32) + 4 control clásico con Output Reuse (R 4, 8, 16, 32) + 2 Skolik sin entrelazamiento (4 cúbits, L 2 y 5) + 2 Hsiao entrelazado (R 4 y 16) = **24**. CartPole-v1.

| identificador | plantilla | cúbits | L | R | entrelazamiento | transformación de entrada | índices de entrada | parámetros (medidos) |
|---|---|---|---|---|---|---|---|---|
| paper_skolik_4q_L1 | hybrid / skolik | 4 | 1 |  | sí | CartPoleNormalizationTransformer |  | 18 |
| paper_skolik_4q_L2 | hybrid / skolik | 4 | 2 |  | sí | CartPoleNormalizationTransformer |  | 26 |
| paper_skolik_4q_L5 | hybrid / skolik | 4 | 5 |  | sí | CartPoleNormalizationTransformer |  | 50 |
| paper_skolik_8q_L1 | hybrid / skolik | 8 | 1 |  | sí | CartPoleNormalizationTransformer |  | 34 |
| paper_skolik_8q_L2 | hybrid / skolik | 8 | 2 |  | sí | CartPoleNormalizationTransformer |  | 50 |
| paper_skolik_8q_L5 | hybrid / skolik | 8 | 5 |  | sí | CartPoleNormalizationTransformer |  | 98 |
| paper_salinas_1q_L1 | hybrid / dr | 1 | 1 |  | sí | CartPoleNormalizationTransformer | [1, 2, 3] | 7 |
| paper_salinas_1q_L2 | hybrid / dr | 1 | 2 |  | sí | CartPoleNormalizationTransformer | [1, 2, 3] | 10 |
| paper_salinas_1q_L5 | hybrid / dr | 1 | 5 |  | sí | CartPoleNormalizationTransformer | [1, 2, 3] | 19 |
| paper_salinas_2q_L1 | hybrid / dr | 2 | 1 |  | sí | CartPoleNormalizationTransformer | [1, 2, 3] | 12 |
| paper_salinas_2q_L2 | hybrid / dr | 2 | 2 |  | sí | CartPoleNormalizationTransformer | [1, 2, 3] | 18 |
| paper_salinas_2q_L5 | hybrid / dr | 2 | 5 |  | sí | CartPoleNormalizationTransformer | [1, 2, 3] | 36 |
| paper_hsiao_or_r4 | hybrid / hsiao | 4 | 1 | 4 | no | ArctanTransformer | [1, 2, 3] | 38 |
| paper_classical_or_r4 | classic (Linear sobre entrada replicada) |  |  | 4 | — (clásico) | — (observación sin transformar; ver nota) | [1, 2, 3] | 26 |
| paper_hsiao_or_r8 | hybrid / hsiao | 4 | 1 | 8 | no | ArctanTransformer | [1, 2, 3] | 70 |
| paper_classical_or_r8 | classic (Linear sobre entrada replicada) |  |  | 8 | — (clásico) | — (observación sin transformar; ver nota) | [1, 2, 3] | 50 |
| paper_hsiao_or_r16 | hybrid / hsiao | 4 | 1 | 16 | no | ArctanTransformer | [1, 2, 3] | 134 |
| paper_classical_or_r16 | classic (Linear sobre entrada replicada) |  |  | 16 | — (clásico) | — (observación sin transformar; ver nota) | [1, 2, 3] | 98 |
| paper_hsiao_or_r32 | hybrid / hsiao | 4 | 1 | 32 | no | ArctanTransformer | [1, 2, 3] | 262 |
| paper_classical_or_r32 | classic (Linear sobre entrada replicada) |  |  | 32 | — (clásico) | — (observación sin transformar; ver nota) | [1, 2, 3] | 194 |
| paper_skolik_4q_L2_noent | hybrid / skolik | 4 | 2 |  | no | CartPoleNormalizationTransformer |  | 26 |
| paper_skolik_4q_L5_noent | hybrid / skolik | 4 | 5 |  | no | CartPoleNormalizationTransformer |  | 50 |
| paper_hsiao_or_r4_ent | hybrid / hsiao | 4 | 1 | 4 | sí | ArctanTransformer | [1, 2, 3] | 38 |
| paper_hsiao_or_r16_ent | hybrid / hsiao | 4 | 1 | 16 | sí | ArctanTransformer | [1, 2, 3] | 134 |

Notas: «R» es `reuse_repetitions` (Hsiao) o `n_repeats` (control clásico); vacío = sin Output Reuse. En Salinas el entrelazamiento figura `True` pero no tiene efecto con 1 cúbit (lo dice el propio código); con 2 cúbits sí. En Skolik L=1 el entrelazamiento no afecta al resultado (FIX-07: el último anillo CZ no puede cambiar una lectura PauliZ). Los controles clásicos son `Linear` sobre las observaciones [1, 2, 3] replicadas R veces (`net_arch = []`, sin capa oculta); no usan transformación de entrada. La columna «transformación de entrada» muestra la clase del código (`CartPoleNormalizationTransformer`, `ArctanTransformer`).

## b) Filas que faltan

| identificador | config | parámetros (medidos) |
|---|---|---|
| su2_frozen_scalar_1q_L5 | Config A (escalar a fase, 1 cúbit), L=5, emula `frozen_scalar_1q_L5` | 18 |
| su2_frozen_binary_4q_L1 | Config B (base binaria, 4 cúbits) sin entrelazamiento, L=1, emula `frozen_binary_4q_noent_L1` | 28 |
| su2_frozen_binary_4q_L5 | Config B sin entrelazamiento, L=5, emula `frozen_binary_4q_noent_L5` | 60 |
| frozen_binary_4q_fourier_ceiling | techo lineal sobre los 4 bits (`linear_on_bits`); un solo brazo vale para todo L | 20 |

- **Confirmado:** el emulador SU(2) de FrozenLake tiene **18, 28 y 60** parámetros, que corresponden, en ese orden, a `su2_frozen_scalar_1q_L5` (Config A, L=5), `su2_frozen_binary_4q_L1` (Config B sin entrelazamiento, L=1) y `su2_frozen_binary_4q_L5` (Config B sin entrelazamiento, L=5).
- **Confirmado:** el techo lineal del circuito binario (`frozen_binary_4q_fourier_ceiling`) tiene **20** parámetros.
- El emulador solo existe sin entrelazamiento (`SU2HybridAgent` lanza error con `ent=True`), por eso no hay brazo SU(2) que emule los circuitos entrelazados.
- Los demás brazos `su2_frozen_*` registrados (otros L, con Output Reuse x4 y x16, y variantes `__head16`, `__init*`, `__tis`) pertenecen a la batería exploratoria; sus recuentos están en `p2b_frozenlake_su2_and_linear_ceiling_params.csv` por si hacen falta, pero no son los tres de la memoria.
