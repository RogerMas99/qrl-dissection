# Dossier del experimento 13: Output Reuse frente al learning rate de la cabeza

Versión final. Cubre los bloques A, B, C(T), D, E, F, G y el agente H (emulador), todos con 30 semillas. Diseño, predicciones y criterios: `docs/EXPERIMENT-13-OR-VS-LR.md` (escritos antes de ver resultados; las enmiendas posteriores están fechadas al final).

## 0. Aviso sobre resultados simulados

Los 540 cálculos del agente H (`hybrid_emu_*`) usan el **emulador** de statevector por lotes: son resultados simulados. El autor aprobó incluir los brazos de emulador de exp13 con la misma condición que exp10: **si se incluyen, es obligatorio decir que son resultados adicionales calculados con el emulador.** El resto (T, D, E, F, G) son redes clásicas en el entorno real.

## 1. Qué se probó

Proposición (sec. 3.4.2 de la memoria): con Adam, sin clipping global de gradiente y sin weight decay, Output Reuse con R copias equivale a R=1 con el learning rate del **peso** de la cabeza multiplicado por R (no el del sesgo), partiendo de V0 = suma de los R bloques. Con el clipping global de PPO la equivalencia se mantiene si la norma se calcula como sqrt(R·‖g_V‖² + ‖g_resto‖²).

## 2. Brazos (id descriptivo, 30 semillas cada uno)

**Agentes T (gemelo clásico, 132 parámetros sin reuse) y H (híbrido Fig. 4, emulador, 126 sin reuse).**

| bloque | ids | qué es |
|---|---|---|
| A | `*_reuse_r{1,4,8,16,32}` | Output Reuse con R copias |
| A | `*_collapsed_r{4,8,16,32}` | modelo con R copias colapsado a una cabeza con V0 = Σ W_r y lr de V ×R |
| A | `*_gradx{8,32}` | R=1 con el gradiente de V multiplicado por R (lr normal) |
| B | `*_headlr_x{2,4,8,16,32,64,128}` | R=1, inicialización fresca, lr del peso de la cabeza ×k |
| C (solo T) | `classical_twin_globallr_x{0p5,2,4,8}` | R=1, lr de toda la red ×k (exploratorio, sin predicción) |

El bloque C del agente H no se programó (decisión recogida en el documento de diseño).

**Bloque D, PPO clásico (control del estudio de referencia):** `ppo_reuse_{clip,noclip}_r{1,4,8,16,32}`, `ppo_lrx_{clip,noclip}_r{4,8,16,32}`, `ppo_lrx_modclip_r{8,32}` (600 celdas).

**Bloques E, F, G (addendum), todos DQN:**

| bloque | id | parámetros | qué es |
|---|---|---|---|
| E | `ceiling_dup` | 162 | techo aditivo de Fourier CartPole; los cables i e i+4 reciben la misma variable, así que Linear(80→2) es reuse con R=2 sobre Linear(40→2) |
| E | `ceiling_unique` | 82 | 40 columnas (sin duplicar) |
| E | `ceiling_unique_lrx2` | 82 | 40 columnas, lr del peso ×2 (inicialización fresca) |
| E | `ceiling_unique_lrx2_collapsed` | 162→82 | `ceiling_dup` colapsado (V0 = suma de los dos bloques), lr ×2 |
| F | `matched_rep4_w7` | 135 | control igualado de RQ2: entrada ×4 → Linear(16→7) |
| F | `matched_norep_w7_lrx4` | 51 | Linear(4→7), lr del peso de la primera capa ×4 (inicialización fresca) |
| F | `matched_norep_w7_lrx4_collapsed` | 135→51 | `matched_rep4_w7` colapsado, lr ×4 |
| F | `matched_norep_w18` / `matched_norep_w18_lrx4` | 128 | sin reuse, ancho 18, sin y con lr ×4 |
| G | `mlp_bits_w6` / `mlp_bits_w12` | 58 / 112 | MLP sobre los 4 bits {0,1} del estado, FrozenLake 4x4 determinista (exploratorio) |

Se corrieron las dos inicializaciones (fresca y colapsada) porque la inicialización por defecto de la red pequeña tiene la misma varianza que la suma de los bloques pero distinta distribución (triangular frente a uniforme). Así, la prueba de equivalencia y la de inicialización quedan separadas.

## 3. Protocolo y estadística

Semillas 1-30; 100 000 pasos; evaluación greedy cada 10 000 pasos con 20 episodios; métricas `final_perf` (media del último 10 % de episodios de entrenamiento), `greedy_final` (media de los 3 últimos puntos greedy) y, en PPO, `best_ma50`. IQM con IC bootstrap al 95 %, P(a>b) no emparejada, **sin corrección por comparaciones múltiples**. Equivalencia: P en [0,35; 0,65] en las dos métricas **y** IC de los IQM solapados. "Fuerte": P ≥ 0,95 en las dos. "No aprende": IQM ≤ 12 en las dos. **Los hiperparámetros no se ajustaron:** los resultados negativos son "con estos hiperparámetros".

## 4. Resultados por predicción

(`P` = P(a>b) en `final_perf` / `greedy_final`.)

**P1, colapsada = reuse.** Se cumple en T y H, en R = 4, 8, 16, 32 (T: P entre 0,48 y 0,56; H: entre 0,49 y 0,51). En float64 y con la misma semilla, T sigue la trayectoria del reuse durante 100 000 pasos sin separarse (diferencias de parámetros ~1e-14, R=8 y R=32, 3 semillas); H igual (4e-9 a 4e-8 en el emulador en float64).

**P2, lr de la cabeza ×R con inicialización fresca ≈ reuse R.** Equivalente en los 4 R, en T (P 0,38-0,49) y en H (P final 0,35-0,63; P greedy 0,38-0,59). La diferencia reuse − fresco no tiene signo estable: en T es positiva (+26, +17, +8, +4 de IQM final en R=4, 8, 16, 32) y en H cambia de signo (−19 en R=4, +8 en R=8, y reuse por encima de fresco en R=16 y 32, con P_final 0,35 en R=16, justo en el borde). Todos los IC incluyen el 0. **Lo que sí está claro es que la variante colapsada (V0 = Σ W_r) coincide con reuse mucho más que la fresca** (P ≈ 0,50 frente a 0,35-0,63): parte de la diferencia fresca-frente-a-reuse puede deberse a la inicialización, pero con n=30 no se puede separar de ruido.

**P3, gradiente ×R no imita al reuse (Adam es invariante a la escala).** Se cumple en T y H: `gradx8` y `gradx32` quedan al nivel de R=1 (T 15,0 / 15,7 frente a 16,0; H 43,7 / 43,8 frente a 43,9; P entre 0,45 y 0,51).

**P4, la curva en k reproduce la curva en R; se extendió más allá de 32.**
- T: lr ×128 llega a 241 (reuse R=32: 190); mejor fresco frente a mejor reuse equivalentes (P 0,60 / 0,54).
- H: lr ×128 llega a 137 (reuse R=32: 127); equivalentes (P 0,57 / 0,55).
- **Dónde se degrada:** no se encontró degradación hasta ×128 en ninguno de los dos agentes; la curva crece de forma sublineal y se aplana entre ×16 y ×64 en T (180, 185, 217) y en H (108, 114, 116). El punto de degradación, si existe, queda por encima de ×128.

**P5, P1-P3 valen igual para el híbrido en el emulador.** Sí (ver P1-P3 arriba, agente H).

**P6, lr global ×k (exploratorio, solo T).** ×0,5: 12,9; ×2: 30,2; ×4: 138; ×8: 207. Subir el lr de toda la red también da ganancia (×4: 138 frente a 112 del reuse R=4; ×8: 207 frente a 151 del reuse R=8), así que la mejora no es específica del lr de la cabeza.

**P7, PPO sin clipping: reuse R ≡ lr ×R.** 4 de 4 pares equivalentes (R = 4, 8, 16, 32). Con clipping 0,5: 3 equivalentes y 1 inconcluso (R=8: lrx 284 frente a reuse 315, P_final 0,34, P_greedy 0,45). Trazas float64 (R=8, 3 semillas): sin separación en 100 000 pasos (diferencias de parámetros 2e-11 a 7e-8).

**P8, clipping modificado reproduce reuse con clipping.** 2 de 2 pares equivalentes (R=8, R=32). Trazas float64: R=8 sin separación (diferencias 5e-12 a 9e-11); R=32 se separa (ver diagnóstico 2).

**E.** `ceiling_unique_lrx2` ≡ `ceiling_dup` (final 83,1 frente a 78,6; P 0,57 / 0,55): la predicción se cumple. La variante colapsada es idéntica a `ceiling_dup` (P 0,500; trazas float64 sin separación, ~1e-14). `ceiling_unique` sin el ×2 queda en 62,6 / 59,8 (P frente a `ceiling_dup` 0,14 / 0,11, "indica una dirección"); no había predicción para este brazo. Frente a los híbridos reales (10 semillas): `ceiling_dup` > híbrido sin entrelazamiento P 0,96 / 0,83 y > híbrido entrelazado P 1,00 / 0,91. El rerun de `ceiling_dup` es consistente con el techo de exp01 (78,6 / 92,9 frente a 76,5 / 84,7; P 0,51 / 0,57).

**F.** `matched_norep_w7_lrx4` ≡ `matched_rep4_w7` (P 0,54 / 0,53; colapsada 0,50 / 0,51): se cumple, pero **ninguno aprende** (IQM ~ 11, por debajo del umbral de 12 en las dos métricas, con estos hiperparámetros): la equivalencia se verifica sobre un control que no aprende. `matched_norep_w18`: 14,1 / 17,9; `matched_norep_w18_lrx4`: 21,5 / 24,9. Frente a `matched_rep4_w7`, `w18_lrx4` queda por encima (P 0,97 / 0,95 de que sea mayor; «fuerte (reverse)» en el CSV). P(híbrido real sin reuse > brazo), 10 semillas: contra los dos brazos w7, 0,99 / 0,99-1,00 (fuerte); contra `w18` 0,88 / 0,87; contra `w18_lrx4` 0,74 / 0,79 (el híbrido sigue por delante, sin llegar a "fuerte"). Con el híbrido sin entrelazamiento: 0,96-1,00.

**G (exploratorio).** `mlp_bits_w12`: éxito 0,75 (IQM final), 15 de 30 semillas resueltas; `mlp_bits_w6`: éxito 0,04, 0 semillas, no aprende con estos hiperparámetros. Frente al circuito binario entrelazado real (10 semillas): L=5 (éxito 0,95; 7 de 10 semillas resueltas): `w12` inconcluso (P 0,31 / 0,36), `w6` por debajo (P 0,03 / 0,05). L=1 (éxito 0): `w12` por encima (P 1,00 / 0,82), `w6` (0,92 / 0,50).

## 5. Lo que no encaja del todo, y diagnóstico

1. **El clipping de PPO dispara en el 100 % de las actualizaciones**, también en R=1 (norma mediana 5,6-10 frente al umbral 0,5; factor mediano 0,05-0,09). Consecuencias: (a) "con clipping" y "sin clipping" difieren sobre todo en un factor de escala, lo que explica que los contrastes se parezcan; (b) la parte de P7 que dice «la separación será menor cuanto menos dispare el clipping» **no se puede evaluar**, porque dispara siempre; (c) con n=30 el IQM no distingue la regla usual de la modificada (lrx_clip frente a lrx_modclip: equivalentes en R=8 y R=32), aunque en float64 la regla usual sí se separa muy pronto (573-2 600 pasos) y la modificada no a R=8. La diferencia entre reglas existe en trayectoria pero no se detecta en rendimiento con estos datos.
2. **Trazas float64 de PPO a R=32 se separan** para los pares algebraicamente equivalentes (primera acción distinta entre 2,7e4 y 5,1e4 pasos; diferencias de parámetros 2e-4 a 9e-3 antes de ella). **Control de sensibilidad:** el mismo brazo con 1e-14 de ruido se separa igual (pasos 26 145, 26 688 y 40 944; diferencias 2e-3 a 2e-2), y a R=8 ni el perturbado ni los equivalentes se separan (≤ 2,4e-6). Conclusión: amplificación del error de redondeo por la dinámica de entrenamiento, no diferencia entre redes. La evidencia de P7/P8 a R=32 descansa en la equivalencia de rendimiento, con la traza float64 como apoyo solo a R=8.
3. **Reuse en PPO no es monótono en R.** `best_ma50` sube de 72 (R=1) a 268-358 (R=4-32), dentro del rango del estudio de referencia (264-382). Con `final_perf` y `greedy_final`, R=16 y R=32 quedan por debajo de R=8 (p. ej. reuse clip: 315 / 500 en R=8; 218 / 450 en R=16; 171 / 223 en R=32). El lr ×R reproduce esa misma caída, así que no es un fallo de la equivalencia. La escala del crítico crece con R (V crítico mediano a 100k: 1,0 en R=4, 34 en R=32), lo que apunta a inestabilidad del crítico a R grande; no se probó una causa.
4. **El efecto de reuse en T y H se aplana a partir de R=16.** T: 188 (R=16), 190 (R=32); H: 126 (R=16), 128 (R=32).
5. **Pipelines distintos en las referencias.** Las referencias reales (10 semillas) se calcularon con el muestreador de exploración sin sembrar; aquí está sembrado con la semilla del run (desviación documentada). El rerun de `ceiling_dup` y de `matched_rep4_w7` frente a sus referencias (P 0,51 / 0,57 y 0,35 / 0,39) es consistente con que el cambio no altera los resultados.
6. **Parámetros.** Los brazos con reuse tienen más parámetros que sus equivalentes con lr (T: 612 con R=16 frente a 132), igual que en exp12. Las tablas lo recogen.

## 6. Qué se repitió y desviaciones

- **No se repitió ninguna celda.** El addendum pedía comprobar que el annealing de lr de PPO escala todos los grupos de parámetros; el código ya lo hacía desde el primer lanzamiento (`for g, base in zip(param_groups, base_lrs): g["lr"] = frac*base`). Se añadió un test explícito de que el lr de la cabeza es R veces el de los demás en cada actualización (R=8 con clipping, R=32 sin él, R=8 con clipping modificado), y otro de que los brazos de reuse anillan su único grupo. El DQN no tiene scheduler ni asignaciones de lr (comprobado en la copia de la librería).
- Muestreador de acciones aleatorias de ε-greedy sembrado con la semilla del run (en el código original no lo estaba).
- PPO: 16 minibatches; la evaluación greedy la definió quien ejecutó el experimento (decisión de implementación recogida en el documento de diseño); el brazo lrx a R=1 es un alias de reuse R=1, de ahí 600 celdas de PPO.
- Semillas 1-30 en todo (acordado), frente a las 10 semillas de las referencias.
- El bloque C del agente H no se programó.
- El control de sensibilidad de PPO (`exp13_trace64_ppo_perturb.py`) se añadió después de ver las trazas a R=32; no estaba en el diseño original.

## 7. Pruebas

Tres ficheros de tests (`tests/test_exp13_dqn.py`, `test_exp13_ppo.py`, `test_exp13_efg.py`): 83 pasan, 0 fallan (última ejecución sobre el código entregado; salida en `RESULTADO_TESTS.txt`). Cubren: recuento de parámetros y que los brazos del emulador están marcados como simulados; que la colapsada en el paso 0 coincide con la replicada (< 1e-12); que el flujo de números aleatorios no se toca; los grupos de parámetros de Adam; que una ejecución real en float64 sigue la replicada; los diagnósticos; la reproducibilidad por semilla; el lr de la cabeza R veces el resto en cada actualización de PPO; y que `post_build` de los brazos antiguos no cambia al añadir E/F/G.

## 8. Tiempos

Calendario de pared (6-7 nov.): T y PPO terminaron en unas 2 h desde el lanzamiento; E/F/G (330 celdas, 8 procesos en paralelo con H) en unas 2 h; H (540 celdas, 14 procesos, ~30 min por celda) en unas 17,7 h por compartir CPU con lo anterior. Coste de cómputo aproximado: E+F ≈ 50 s por celda; G ≈ 5-6 min por celda; T ≈ minutos por celda; H ≈ 30 min por celda en pared. Las trazas float64 del híbrido tardaron ~40-60 min por semilla.

## 9. Qué NO se puede afirmar con estos datos

- Que el efecto de reuse sea específico del circuito, ni que desaparezca con un control: el control F que replica la entrada no aprende con estos hiperparámetros, así que no sirve para comparar en nivel absoluto.
- Que lr ×k sea mejor que reuse (las curvas se solapan) ni lo contrario.
- Que la regla de clipping modificada rinda mejor que la usual en PPO (no se distinguen).
- Que la inicialización explique la diferencia reuse − fresco; solo que la colapsada coincide más.
