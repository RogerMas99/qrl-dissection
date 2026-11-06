# Dossier del experimento 10 (100 semillas en el emulador)

> **AVISO DE POLÍTICA (léelo antes que nada).** Todo lo de este paquete sale de **simuladores** (agentes `su2` y `sv`), no del circuito real de PennyLane. Por regla del autor, los resultados de simulador no entran en la memoria salvo aprobación explícita. Aprobación vigente para este material (2026-11-03): se te entrega **para que lo revises y valores si conviene ajustar alguna parte del trabajo**. Si algo de esto se incluye en la memoria, es **obligatorio** decir que son *resultados adicionales calculados con el emulador para corroborar o acabar de cerrar hipótesis*. Qué entra y dónde lo decide el autor. Registro: `docs/SIMULATED-RESULTS-POLICY.md`.

## 1. Qué es

Los contrastes principales del trabajo se midieron con 10 semillas sobre el circuito real. Con 10 semillas varias probabilidades de mejora quedan en zona dudosa (por ejemplo P(entrelazado > sin entrelazar) entre 0,64 y 0,92 según Output Reuse). El experimento 10 repite esos contrastes con **100 semillas** en los emuladores, que son mucho más rápidos que la librería base, para ver cómo se comportan con más datos.

- **Emuladores.** `sv`: simulador de vector de estado por lotes (acepta entrelazamiento o no). `su2`: emulador exacto del caso sin entrelazar. En una misma comparación se usa **siempre la misma herramienta en los dos lados**. Los brazos `sv` sin entrelazar (`ent=False`) son gemelos exactos de los `su2` (misma configuración y mismo número de parámetros).
- **Protocolo.** Idéntico al de los experimentos reales de CartPole (100 000 pasos, FIX-01 activo, DQN, mismos hiperparámetros) y al de FrozenLake Config B (mismos hiperparámetros primarios). Los hiperparámetros **no se han ajustado** nunca: cualquier resultado negativo es "con estos hiperparámetros".
- **Semillas.** 1-100 por brazo. De las 2 500 ejecuciones, 200 son semillas que ya existían con exactamente la misma especificación y se reutilizan (listadas en `reused_seeds.json`); 2 300 se han ejecutado nuevas.
- **Métricas.** `final_perf` (media del último 10 % de episodios de entrenamiento) y `greedy_final` (media de los 3 últimos controles greedy). IQM con IC bootstrap al 95 %. P(a>b) = probabilidad de mejora, sin emparejar, con empates contados como medio. **No hay corrección por comparaciones múltiples.** Un contraste es "fuerte" solo si P ≥ 0,95 en **las dos** métricas (regla fijada antes de lanzar, en `docs/EXPERIMENT-10-X100.md`).

## 2. Tablas (n = 100 por brazo, salvo indicación)

**C1. Entrelazamiento × Output Reuse (CartPole, L=5), `sv` entrelazado frente a `sv` sin entrelazar.**

| R | IQM final ent. / sin ent. | P(ent>sin) final | IQM greedy ent. / sin ent. | P greedy | fuerte |
|---|---|---|---|---|---|
| 1 (sin OR) | 42,3 / 45,6 | 0,43 | 54,0 / 55,5 | 0,48 | no |
| 4 | 81,0 / 60,7 | 0,70 | 95,2 / 70,5 | 0,69 | no |
| 8 | 106,1 / 64,8 | 0,85 | 119,4 / 71,7 | 0,82 | no |
| 16 | 115,4 / 72,5 | 0,83 | 134,5 / 82,3 | 0,82 | no |
| 32 | 133,8 / 75,6 | 0,84 | 154,9 / 85,2 | 0,85 | no |

Concordancia entre herramientas (mismo circuito sin entrelazar, `sv ent=False` frente a `su2`): P(sv>su2) final = 0,50 / 0,46 / 0,51 / 0,52 / 0,50 para R = 1, 4, 8, 16, 32. Las dos herramientas dan lo mismo.

**C2. Profundidad (CartPole, R=1).**

| contraste | IQM final a / b | P final | IQM greedy a / b | P greedy | fuerte |
|---|---|---|---|---|---|
| entrelazado: L2 > L1 | 26,2 / 10,3 | 0,97 | 28,6 / 9,9 | 0,92 | no |
| entrelazado: L5 > L2 | 42,3 / 26,2 | 0,76 | 54,0 / 28,6 | 0,79 | no |
| sin entrelazar: L2 > L1 | 35,0 / 10,2 | 1,00 | 37,2 / 9,6 | 0,99 | **sí** |
| sin entrelazar: L5 > L2 | 45,6 / 35,0 | 0,74 | 55,5 / 37,2 | 0,74 | no |

**C3. Profundidad × Output Reuse (CartPole).**

| contraste | IQM final a / b | P final | IQM greedy a / b | P greedy | fuerte |
|---|---|---|---|---|---|
| L2 con OR4 > L2 | 52,1 / 26,2 | 0,91 | 55,5 / 28,6 | 0,85 | no |
| L2 con OR8 > L2 | 63,4 / 26,2 | 0,96 | 66,3 / 28,6 | 0,92 | no |
| L2 con OR16 > L2 | 71,2 / 26,2 | 0,98 | 75,0 / 28,6 | 0,96 | **sí** |
| L2 con OR32 > L2 | 71,6 / 26,2 | 0,97 | 76,8 / 28,6 | 0,95 | **sí** |
| L5_OR32 > L2_OR32 | 133,8 / 71,6 | 0,88 | 154,9 / 76,8 | 0,88 | no |

**C4. FrozenLake Config B, L=5.**

| brazo | semillas vivas (greedy_final > 0) | final_perf IQM (IC) | greedy_final IQM (IC) |
|---|---|---|---|
| entrelazado | 82 de 100 | 0,946 (0,917-0,948) | 0,933 (0,813-0,993) |
| sin entrelazar | 0 de 100 | 0,000 | 0,000 |

P(entrelazado > sin entrelazar) = 1,00 (final) y 0,91 (greedy): "no fuerte" por la regla, **solo porque** en la métrica greedy las 18 semillas entrelazadas con valor 0 empatan con las 100 del otro brazo y el empate cuenta medio. El dato informativo es 82 de 100 frente a 0 de 100.

Todos los IQM, IC y el resto de brazos (25 en total) están en `summary_x100.csv`; las comprobaciones de coherencia, en `tables_x100.md`.

## 3. Lectura (qué dicen y qué no)

1. **El entrelazamiento no ayuda sin Output Reuse y ayuda con él.** Sin OR no hay diferencia (P 0,43 / 0,48). Con OR, el entrelazado queda por encima en los cuatro valores de R (IQM final +20 a +58 pasos), con P ≈ 0,70-0,85. Ninguno llega a "fuerte": con 100 semillas la P se estabiliza alrededor de 0,8 en vez de subir hacia 1, lo que indica solapamiento amplio entre las distribuciones. La dirección es consistente; la magnitud es moderada.
2. **Posible cierre de la caída en R=16.** Con el circuito real y n=10, P(ent>sin) fue 0,90 / 0,90 / 0,64 / 0,92 para R = 4, 8, 16, 32 (0,33 sin OR). En el emulador con n=100 es 0,70 / 0,85 / 0,83 / 0,84 (0,43 sin OR): el hueco de R=16 desaparece. Es coherente con que el valor real sin entrelazar en R=16 (IQM 89,5, n=10) fuera una muestra favorable: el emulador da 72,5 con n=100 (P(sim>real) = 0,29) y, en tres brazos sin entrelazar con OR, las semillas 1-10 salen por encima de las 11-100. Es una hipótesis apoyada, no una demostración.
3. **Las dos herramientas coinciden** en el circuito sin entrelazar (P entre 0,46 y 0,52 en los cinco R).
4. **Profundidad.** L=1 no aprende (≈10 pasos, sea o no con entrelazado). L=2 aprende algo (26 con entrelazado, 35 sin) y L=5 algo más (42 / 46), con P ≈ 0,75 entre L2 y L5, no fuerte. Con Output Reuse, L=2 pasa de 52 (R=4) a 71 (R=16 y R=32): saturación entre 16 y 32. Fuerte: L2 sin entrelazar > L1, y L2 con OR16 y con OR32 > L2 sin OR. L5_OR32 > L2_OR32 da 0,88, no fuerte.
5. **FrozenLake Config B.** El entrelazado aprende en 82 de 100 semillas; sin entrelazar, en ninguna. Coincide con el resultado real (0,952 frente a 0,000, n=10).
6. **Coherencia.** (a) Emulador frente a circuito real (n=10): 0 de 25 brazos fuera de [0,25; 0,75] en P(sim>real). (b) Semillas 1-10 frente a 11-100 del mismo brazo: 3 de 25 fuera de [0,25; 0,75], los tres con las semillas 1-10 por encima (`su2_cartpole_L5_OR16`, `sv_cartpole_L5_OR16_noent`, `sv_cartpole_L5_OR32_noent`).

Resumen de "fuerte": 3 de los contrastes registrados (1 en C2 y 2 en C3); ninguno en C1 ni en C4.

## 4. Límites

- Es simulación, no el circuito real. La batería de validación de los emuladores (Apéndice C) **no queda aceptada** con la regla fijada antes de evaluar; el emulador y el circuito real pueden diferir y aquí no se ha medido en cuánto más allá de la comparación n=10 de la coherencia (a).
- Hiperparámetros no ajustados; no hay corrección por comparaciones múltiples; P(a>b) no emparejada.
- Los emuladores no sustituyen ningún número del trabajo basado en el circuito real. Sirven para ver si la dirección y el orden de magnitud se sostienen con más semillas.
- Incidencias de ejecución (reinicio del equipo y 8 celdas repetidas por doble escritura) y su tratamiento, en `docs/EXPERIMENT-10-X100.md`, sección "Amendments". Una auditoría de las 2 300 celdas no encuentra otro problema.

## 5. Ficheros

`summary_x100.csv` (por brazo), `per_seed_x100.csv` (una fila por ejecución), `curves_x100_train_ma50.csv` y `curves_x100_eval_greedy.csv` (curvas por semilla), `tables_x100.md` (contrastes y coherencias), `reused_seeds.json` (semillas reutilizadas). El diseño, escrito antes de lanzar, está en `docs/EXPERIMENT-10-X100.md` del repositorio.
