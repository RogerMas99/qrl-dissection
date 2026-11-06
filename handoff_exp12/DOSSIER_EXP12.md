# Dossier del experimento 12: gemelo clásico del agente híbrido

Entorno real (CartPole-v1), solo brazos clásicos. **No interviene ningún simulador**, así que estos resultados pueden entrar en la memoria cuando el autor lo apruebe. Diseño, contrastes y regla de lectura, escritos antes de lanzar: `docs/EXPERIMENT-12-CLASSICAL-TWIN.md` (incluido en este paquete).

## 1. Qué se probó y por qué

En el agente híbrido el Output Reuse (OR) actúa solo en la frontera entre el circuito y la cabeza lineal: el circuito queda fijo en 80 parámetros y lo único que crece con R es la cabeza. El control fiel es un **gemelo clásico**: la misma cabeza y el mismo OR en la misma posición, con el circuito sustituido por un extractor clásico pequeño con las mismas 8 salidas. Así, lo único que distingue gemelo e híbrido es de dónde salen las 8 características.

**Arquitectura del híbrido, verificada en el código** (`hybrid_fig4` / `hybrid_OR{R}`, `HybridAgent` con `is_qnet=True`):
`obs (4, sin transformar) → PQC Skolik, 8 cúbits, L=5, entrelazado (80 pesos) → 8 valores ⟨Z_i⟩ en [-1, 1] → OutputReuse(R) → Linear(8R→4) → Identity → Linear(4→2)`; capas lineales con la inicialización por defecto de `nn.Linear`; un solo Adam. Cabeza: 32R + 14 parámetros.

**Brazos.** R ∈ {1, 4, 8, 16, 32} (con R=1 no hay OutputReuse, como `hybrid_fig4`):

- `classical_twin_r{R}` (variante principal, salidas acotadas): `obs (4) → Linear(4→6) → ReLU → Linear(6→8) → Tanh → OutputReuse(R) → Linear(8R→4) → Identity → Linear(4→2)`. Extractor: 86 parámetros.
- `classical_twin_unbounded_r{R}` (control del rango): igual **sin** la Tanh final.

**Equivalencia estructural (verificada con 33 tests, `tests/test_exp12_arms.py`).** El gemelo *es* un `HybridAgent` de la librería: se construye con la configuración de `hybrid_OR{R}` (más dos claves del gemelo) y después se cambia el primer módulo de la red, la capa PennyLane, por el extractor clásico. Por tanto OutputReuse, las capas de la cabeza, su inicialización, la activación y el `forward` son el código de la librería, no una copia; la librería vendorizada no se toca; bajo una misma semilla la cabeza del gemelo arranca con exactamente los mismos pesos que la del híbrido; no queda ninguna capa cuántica ni parámetros duplicados; el optimizador y el bucle de DQN ven un `HybridAgent` ordinario. Las salidas del extractor acotado están en [-1, 1] (probado incluso con entradas 50 veces mayores que las reales) y las del no acotado no.

**Parámetros medidos** (se construyó cada brazo):

| R | híbrido | gemelo (acotado y no acotado) | exceso | cabeza | extractor |
|---|---|---|---|---|---|
| 1 | 126 | 132 | +6 | 46 | 86 |
| 4 | 222 | 228 | +6 | 142 | 86 |
| 8 | 350 | 356 | +6 | 270 | 86 |
| 16 | 606 | 612 | +6 | 526 | 86 |
| 32 | 1118 | 1124 | +6 | 1038 | 86 |

El gemelo tiene 6 parámetros más que el híbrido en todos los R (extractor 86 frente a circuito 80): va en la dirección que favorece al control; se acepta y se anota. Las capas del extractor usan la inicialización por defecto de `nn.Linear`, como la cabeza (el enunciado no fijaba otra).

**Protocolo.** Idéntico a exp02 y a `hybrid_fig4` de exp01: DQN, CartPole-v1, 100 000 pasos, `batch_size` 128, `buffer_size` 10 000, `train_frequency` 10, resto por defecto de la librería, FIX-01 activado, semillas 1-10. Evaluación greedy cada 10 000 pasos **con 5 episodios**, igual que las referencias (`hybrid_fig4`, `hybrid_OR{R}`), de modo que `greedy_final` tiene el mismo ruido en todas las comparaciones de este dossier. No se ajustó ningún hiperparámetro: los resultados negativos son "con estos hiperparámetros". 100 celdas (2 familias × 5 R × 10 semillas), cinco procesos en paralelo, un solo proceso por celda, sin incidencias (ningún error en los registros, 10 semillas en cada brazo).

## 2. IQM por brazo (IC bootstrap percentil al 95 %, n = 10 por brazo)

| brazo | experimento | parámetros | `final_perf` IQM (IC) | `greedy_final` IQM (IC) |
|---|---|---|---|---|
| `hybrid_fig4` | exp01 | 126 | 35,8 (24,0-48,2) | 47,3 (31,3-68,4) |
| `hybrid_OR4` | exp02 | 222 | 94,7 (81,0-111,5) | 112,0 (91,3-152,4) |
| `hybrid_OR8` | exp02 | 350 | 109,8 (93,8-125,1) | 124,3 (107,2-189,9) |
| `hybrid_OR16` | exp02 | 606 | 104,2 (76,7-123,6) | 146,2 (109,7-180,0) |
| `hybrid_OR32` | exp02 | 1118 | 137,6 (106,3-179,7) | 161,1 (120,0-224,0) |
| **`classical_twin_r1`** | exp12 | 132 | **13,9 (11,4-30,2)** | **29,8 (14,8-79,9)** |
| **`classical_twin_r4`** | exp12 | 228 | **120,6 (79,5-166,7)** | **212,2 (154,2-299,7)** |
| **`classical_twin_r8`** | exp12 | 356 | **171,0 (91,6-262,5)** | **292,1 (152,8-425,2)** |
| **`classical_twin_r16`** | exp12 | 612 | **212,3 (152,5-288,6)** | **276,6 (214,7-395,2)** |
| **`classical_twin_r32`** | exp12 | 1124 | **198,6 (157,6-235,4)** | **230,7 (159,4-299,4)** |
| `classical_twin_unbounded_r1` | exp12 | 132 | 37,4 (17,0-63,2) | 70,7 (22,2-117,9) |
| `classical_twin_unbounded_r4` | exp12 | 228 | 120,0 (72,0-163,5) | 179,3 (138,2-248,3) |
| `classical_twin_unbounded_r8` | exp12 | 356 | 112,2 (72,5-146,6) | 175,1 (119,0-209,0) |
| `classical_twin_unbounded_r16` | exp12 | 612 | 173,7 (127,2-235,5) | 210,3 (149,7-297,1) |
| `classical_twin_unbounded_r32` | exp12 | 1124 | 184,5 (130,0-279,1) | 275,7 (204,6-337,8) |

`classical_twin_r1` está **en el límite**: su IQM de `final_perf` (13,9) supera el umbral de 12 por poco; 3 de sus 10 semillas están en el suelo (9,6-10,6) y 7 por encima de 12 (máximo 75,4). El IC de ambas métricas es ancho en todos los brazos con R ≥ 4 (n = 10).

## 3. Contrastes (P(a>b) no emparejada, empates = 1/2; sin corrección por comparaciones múltiples)

"Fuerte" = P ≥ 0,95 en las dos métricas; "cerca de 0,5" = P en [0,25; 0,75] en las dos; intermedios = "indica una dirección"; "mixto" si las métricas apuntan en sentidos opuestos. Entre paréntesis, P de `final_perf` / P de `greedy_final`.

**G1. Efecto del OR en el gemelo acotado** (`twin_r{R}` frente a `twin_r1`):

| R | P final / P greedy | lectura |
|---|---|---|
| 4 | 0,97 / 0,93 | indica una dirección (a > b) |
| 8 | 0,96 / 0,94 | indica una dirección (a > b) |
| 16 | 1,00 / 0,96 | fuerte |
| 32 | 0,99 / 0,95 | fuerte |

**G2. Efecto del OR en el híbrido** (`hybrid_OR{R}` frente a `hybrid_fig4`, mismos scripts):

| R | P final / P greedy | lectura |
|---|---|---|
| 4 | 0,97 / 0,96 | fuerte |
| 8 | 1,00 / 0,99 | fuerte |
| 16 | 0,96 / 0,96 | fuerte |
| 32 | 1,00 / 0,99 | fuerte |

**G3. Circuito frente a gemelo a igual R** (`hybrid_OR{R}` frente a `twin_r{R}`; `hybrid_fig4` en R=1):

| R | IQM final híbrido / gemelo | P final / P greedy | lectura |
|---|---|---|---|
| 1 | 35,8 / 13,9 | 0,79 / 0,65 | indica una dirección (híbrido > gemelo) |
| 4 | 94,7 / 120,6 | 0,38 / 0,20 | indica una dirección (gemelo > híbrido) |
| 8 | 109,8 / 171,0 | 0,36 / 0,27 | cerca de 0,5 |
| 16 | 104,2 / 212,3 | 0,09 / 0,06 | indica una dirección (gemelo > híbrido); no llega a "fuerte inverso" (0,09 > 0,05) |
| 32 | 137,6 / 198,6 | 0,26 / 0,35 | cerca de 0,5 |

En ningún R el híbrido supera al gemelo de forma fuerte; a partir de R = 4 el gemelo queda igual o por encima.

**G4. Papel del rango acotado.** Gemelo acotado frente a no acotado (`twin_r{R}` frente a `twin_unbounded_r{R}`):

| R | P final / P greedy | lectura |
|---|---|---|
| 1 | 0,34 / 0,40 | cerca de 0,5 |
| 4 | 0,57 / 0,58 | cerca de 0,5 |
| 8 | 0,65 / 0,72 | cerca de 0,5 |
| 16 | 0,62 / 0,69 | cerca de 0,5 |
| 32 | 0,55 / 0,40 | cerca de 0,5 |

G1 repetido en el gemelo no acotado (`twin_unbounded_r{R}` frente a `twin_unbounded_r1`): R=4: 0,84 / 0,91; R=8: 0,88 / 0,86; R=16: 1,00 / 0,92 (los tres, "indica una dirección"); R=32: 0,97 / 0,95 (fuerte).

**Descriptivo: cociente IQM(R) / IQM(R=1)** (IC bootstrap al 95 %, remuestreando ambos grupos):

| familia | R | cociente `final_perf` | cociente `greedy_final` |
|---|---|---|---|
| gemelo acotado | 4 | 8,66 (3,52-12,62) | 7,13 (2,56-15,49) |
| gemelo acotado | 8 | 12,28 (4,49-19,56) | 9,82 (2,99-21,32) |
| gemelo acotado | 16 | 15,25 (6,31-21,83) | 9,30 (3,34-20,25) |
| gemelo acotado | 32 | 14,26 (6,13-18,47) | 7,75 (2,66-16,03) |
| gemelo no acotado | 4 / 8 / 16 / 32 | 3,21 / 3,00 / 4,65 / 4,94 | 2,54 / 2,48 / 2,97 / 3,90 |
| híbrido | 4 | 2,64 (1,88-4,06) | 2,37 (1,53-3,98) |
| híbrido | 8 | 3,07 (2,21-4,68) | 2,63 (1,77-4,73) |
| híbrido | 16 | 2,91 (1,91-4,47) | 3,09 (1,95-4,86) |
| híbrido | 32 | 3,84 (2,61-6,28) | 3,41 (2,13-5,80) |

(Los ICs de la familia no acotada están en `tables_exp12.md`; todos superponen ampliamente con los del acotado.) **Precaución con los cocientes:** el denominador del gemelo acotado (13,9) está casi en el suelo, así que su cociente se infla; el del híbrido parte de 35,8 y el del gemelo no acotado de 37,4. En niveles absolutos, a R ≥ 4 el gemelo (acotado y no acotado) llega a valores similares o mayores que el híbrido.

## 4. Aplicación de la regla de lectura (fijada antes de lanzar; cuantificadores en la sección 6 del documento de diseño)

1. **Rama 1 (OR específico de las características del circuito): no se cumple.** El gemelo acotado aprende sin OR por poco (R=1: 13,9 / 29,8, por encima de 12 en el límite), pero G1 no está cerca de 0,5 en ningún R (va de 0,93 a 1,00) y el híbrido sí gana con el OR (G2 fuerte en los cuatro R).
2. **Rama 2 (el OR da al gemelo una ganancia fuerte): se cumple.** G1 es fuerte en R=16 y R=32 y "indica una dirección" (P entre 0,93 y 0,97) en R=4 y R=8. El OR actúa, en este montaje, como un mecanismo de optimización genérico sobre características seguido de una cabeza lineal, no específico del circuito. Comparación del tamaño relativo de la ganancia: el cociente del gemelo acotado (7-15) es mayor que el del híbrido (2,4-3,8), pero con IC muy anchos (inferiores entre 2,6 y 6,3) y con el denominador del gemelo casi en el suelo; en nivel absoluto el gemelo no queda por debajo del híbrido (G3). No se afirma más.
3. **Rama 3 (G4, rango acotado): no se cumple de forma limpia, y se informa de que la etiqueta mecánica engaña.** La aplicación mecánica de los criterios fijados da "ligada al rango acotado" en R=16 (G1 fuerte en el acotado y G1 del no acotado no fuerte, 1,00 / 0,92), "no depende del rango" en R=32, y "no evaluable" en R=4 y R=8 (G1 no llega a fuerte). Esa etiqueta de R=16 depende de que 0,92 quede por debajo de 0,95 en una sola métrica; la comparación directa entre acotado y no acotado (G4) está **cerca de 0,5 en todos los R** y los ICs de los cocientes se superponen. Con estos datos no se puede decir que el efecto del OR dependa del rango acotado ni que no dependa: el contraste directo no distingue las dos variantes. Descriptivamente, el gemelo acotado parte de un punto más bajo en R=1 (13,9 frente a 37,4) y los dos llegan a niveles parecidos con OR.
4. **Rama 4 (el gemelo no aprende en ningún R): no se cumple.** Con R ≥ 4 ambos gemelos aprenden claramente (IQM 112-212 en `final_perf`).

Ramas que se cumplen limpiamente: solo la 2. La rama 3 queda como se describe arriba; la 1 y la 4 no se cumplen.

## 5. Relación con exp11 (sin extrapolar)

En exp11 el control que replica la **entrada** antes de toda la red no mejoraba con R (en el suelo o cerca). Aquí el gemelo, que replica la salida de un extractor de 8 características justo antes de la cabeza lineal, mejora mucho con R. Las dos arquitecturas difieren en más de una cosa (dónde se replica, capa de características previa, anchos), así que no se separa cuál causa la diferencia; lo que sí muestran juntos es que, con estos hiperparámetros, el OR ayuda a una red clásica cuando actúa en la frontera extractor-cabeza, como en el híbrido. La conclusión de la memoria de que "el OR ayuda al híbrido y no a los controles clásicos" necesita, por tanto, una matización: depende del control, y el gemelo es el que reproduce la posición del OR del híbrido.

## 6. Limitaciones

- n = 10 por brazo, P no emparejada, **sin corrección por comparaciones múltiples** (27 comparaciones en las tablas, 22 de ellas en la regla de lectura); ICs muy anchos en los brazos con R ≥ 4. Las lecturas "indica una dirección" no son pruebas.
- Un solo extractor (4→6→8, ReLU), con la inicialización por defecto, y los hiperparámetros del protocolo sin ajustar: ninguna conclusión negativa sobre el híbrido es general. Que el gemelo iguale o supere al híbrido aquí no dice que el circuito no pueda aportar nada en otras condiciones.
- El gemelo tiene 6 parámetros más que el híbrido (favorece al control).
- `greedy_final` con 5 episodios es más ruidoso que con 20; todas las referencias de este dossier usan 5, así que la comparación es homogénea. `final_perf` no depende de esto.
- El gemelo acotado en R=1 casi no aprende (13,9), y eso condiciona los cocientes de G1 y la comparación G4 en R=1.
- No se afirma nada sobre cuándo se hizo cada experimento respecto a otros.

## 7. Ficheros

| fichero | contenido |
|---|---|
| `per_seed_exp12.csv` | una fila por ejecución (mismas columnas que `per_seed_metrics.csv`), incluidas las referencias `hybrid_fig4` y `hybrid_OR{R}` |
| `curves_exp12_train_ma50.csv`, `curves_exp12_eval_greedy.csv` | curvas por semilla |
| `summary_exp12.csv` | resumen por brazo con parámetros medidos |
| `tables_exp12.md` | todas las tablas, incluida la aplicación mecánica de la regla |
| `design_params.json` | parámetros medidos (híbrido, gemelo, extractor, cabeza) |
| `EXPERIMENT-12-CLASSICAL-TWIN.md` | diseño, contrastes y regla de lectura, escritos antes de lanzar |
