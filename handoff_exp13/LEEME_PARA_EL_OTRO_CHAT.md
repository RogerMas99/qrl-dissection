# Para el chat que redacta la memoria: experimento 13 (Output Reuse frente al learning rate de la cabeza) - ENTREGA FINAL

Este paquete **sustituye al paquete parcial** `handoff_exp13_parcial.zip` (si lo tienes, descártalo: tenía H sin terminar). No cambia ningún fichero entregado antes.

## Aviso sobre resultados simulados

Los brazos `hybrid_emu_*` (agente H) están calculados con el **emulador**. El autor aprobó incluir los brazos de emulador de exp13 con la misma condición que exp10: **si se incluyen en la memoria, es obligatorio decir que son resultados adicionales calculados con el emulador.** Los demás brazos (T, D, E, F, G) son redes clásicas sobre el entorno real.

## Qué leer primero

`DOSSIER_EXP13.md`: brazos con id descriptivo, protocolo, resultado por predicción (P1-P8, E, F, G) con cifras, lo que no encaja y su diagnóstico, qué se repitió, pruebas, tiempos y lo que no se puede afirmar.

## Contenido

| ruta | qué es |
|---|---|
| `DOSSIER_EXP13.md` | el informe |
| `docs/EXPERIMENT-13-OR-VS-LR.md` | diseño y predicciones escritos antes de ver resultados, con las enmiendas fechadas |
| `docs/SIMULATED-RESULTS-POLICY.md` | política de resultados simulados |
| `exp13_A-D/` | `summary_exp13.csv`, `contrasts_exp13.csv`, `per_seed_exp13.csv`, `tables_exp13.md` (todas las tablas), curvas, diagnósticos de escala y clipping, trazas |
| `exp13_EFG/` | `summary_exp13_efg.csv`, `contrasts_exp13_efg.csv`, `per_seed_exp13_efg.csv`, `tables_exp13_efg.md`, diagnósticos y trazas float64 |
| `exp13_traces/` | trazas float64 completas, incluido el control de sensibilidad de PPO |
| `code/`, `RESULTADO_TESTS.txt` | código usado y salida de los tests (83 pasan) |

## Resumen para decidir qué entra (propón, la decisión es del autor)

La proposición se cumple: con Adam, sin clipping global y sin weight decay, Output Reuse con R copias se comporta como R=1 con el lr del peso de la cabeza ×R. 30 semillas por brazo.

- **Colapsada = reuse** (P1) y **gradiente ×R no imita al reuse** (P3): se cumplen en el gemelo clásico (T) y en el híbrido en emulador (H).
- **Lr de la cabeza ×R con inicialización fresca ≈ reuse** (P2): equivalente en los 4 R de T y de H; la variante colapsada coincide más que la fresca, pero con n=30 no se puede atribuir la diferencia a la inicialización.
- **Más allá de R=32:** el lr de la cabeza ×128 supera a reuse R=32 en T (241 frente a 190) y en H (137 frente a 127); equivalentes. Sin degradación hasta ×128.
- **PPO:** reuse y lr ×R equivalentes sin clipping (4/4), con clipping (3/4, 1 inconcluso) y con la regla modificada (2/2).
- **E:** el techo de Fourier con columnas duplicadas es reuse con R=2: `ceiling_unique_lrx2` ≡ `ceiling_dup`.
- **F:** `matched_norep_w7_lrx4` ≡ `matched_rep4_w7`, pero ninguno aprende con estos hiperparámetros: la equivalencia se verifica sobre un control que no aprende. El híbrido real sigue por encima de los cuatro brazos.
- **G (exploratorio):** un MLP de 112 parámetros sobre los 4 bits resuelve 15 de 30 semillas (éxito 0,75); el de 58 no aprende.

## Qué no encaja (detalle en el dossier, sec. 5)

1. El clipping de PPO dispara en el 100 % de las actualizaciones, también en R=1; esto vuelve inevaluable la parte de P7 que depende de cuánto dispare y hace que "con" y "sin" clipping difieran sobre todo en escala.
2. A R=32 las trazas float64 de PPO se separan aunque las redes son equivalentes; un control con 1e-14 de ruido se separa igual, así que es amplificación de redondeo y no diferencia entre redes.
3. En PPO, R=16 y 32 rinden menos que R=8 en `final_perf` y `greedy_final`; el lr ×R reproduce la misma caída.
4. Reuse − fresco no tiene signo estable entre T y H, con IC que incluyen el 0.

## Reglas de siempre

- Resultados negativos: "con estos hiperparámetros" (nunca se ajustaron).
- n = 30 por brazo en exp13 (referencias reales con 10 semillas, otro montaje: muestreador de exploración sin sembrar; episodios greedy según la fe de erratas); P(a>b) no emparejada, **sin corrección por comparaciones múltiples**; "fuerte" solo si P ≥ 0,95 en las dos métricas.
- No cambies ni sustituyas ningún número del circuito real.
- No afirmes nada sobre cuándo se hizo cada experimento respecto a otros. Si hace falta, pregunta al autor.
- Si se incluyen los brazos del emulador, di que son resultados adicionales calculados con el emulador.
- Ante cualquier duda de especificación, pregunta antes de redactar.
