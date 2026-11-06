# Para el chat que redacta la memoria: experimento 12 (gemelo clásico del agente híbrido)

Este paquete añade un experimento a lo que ya tienes. **No sustituye ningún fichero entregado.** Es CartPole real con brazos clásicos, sin simulador: puede entrar en la memoria cuando el autor lo apruebe.

## Qué contiene

| fichero | para qué |
|---|---|
| `DOSSIER_EXP12.md` | todo el experimento: diseño, equivalencia estructural, parámetros, IQM por brazo, contrastes G1-G4, cocientes, regla de lectura aplicada, limitaciones |
| `EXPERIMENT-12-CLASSICAL-TWIN.md` | diseño, contrastes y regla de lectura, escritos antes de lanzar |
| `tables_exp12.md` | todas las tablas, incluida la aplicación mecánica de la regla |
| `per_seed_exp12.csv`, `curves_exp12_*.csv`, `summary_exp12.csv`, `design_params.json` | datos |

## Qué se hizo, en dos frases

En el híbrido el Output Reuse (OR) solo actúa entre el circuito y la cabeza lineal. El **gemelo clásico** conserva la cabeza y el OR del híbrido (el mismo código de la librería) y sustituye únicamente el circuito por un extractor clásico 4→6→8 (con Tanh final, salidas acotadas como el circuito, o sin ella). 100 celdas: 2 familias × R ∈ {1, 4, 8, 16, 32} × 10 semillas, mismo protocolo que exp02, 5 episodios por evaluación greedy.

## Qué dice, para que lo redactes con cuidado

- **El OR también ayuda al gemelo clásico.** Ganancia fuerte en R=16 y 32 (P entre 0,95 y 1,00 en las dos métricas) y "indica una dirección" en R=4 y 8 (P entre 0,93 y 0,97). `final_perf` IQM pasa de 13,9 (R=1) a 121-212. Es la rama 2 de la regla: en este montaje el OR actúa como mecanismo genérico sobre características seguidas de una cabeza lineal, no específico del circuito.
- **El híbrido no supera al gemelo** en ningún R: en R=4 y 16 el gemelo queda por encima ("indica una dirección"), en R=8 y 32 la P está cerca de 0,5. Solo en R=1 el híbrido va por delante.
- **Matiz sobre la conclusión actual de la memoria** ("el OR ayuda al híbrido y no a los controles clásicos"): depende del control. El control que replica la entrada (exp11) no mejoraba; el gemelo, que replica en la frontera extractor-cabeza como el híbrido, sí. Los dos montajes difieren en más de una cosa, así que no se separa la causa.
- **Rango acotado (G4): no concluyente.** La etiqueta mecánica da "ligado al rango" en R=16, pero depende de un 0,92 frente al umbral 0,95; la comparación directa acotado/no acotado está cerca de 0,5 en todos los R. No escribas que el efecto dependa del rango ni que no dependa.
- **Cuidado con los cocientes IQM(R)/IQM(1)** (7-15 en el gemelo, 2,4-3,8 en el híbrido): el gemelo acotado en R=1 casi no aprende (13,9, en el límite del umbral de 12), lo que infla su cociente, y los IC son muy anchos. En nivel absoluto el gemelo no queda por debajo del híbrido.
- El gemelo tiene 6 parámetros más que el híbrido (86 frente a 80 del circuito), a favor del control.

## Qué te pedimos

1. Revisar `DOSSIER_EXP12.md` y decir si conviene ajustar la parte de la memoria sobre Output Reuse, **antes de escribir nada**. Propón primero los ajustes; la decisión de qué entra y dónde es del autor.
2. Si algo se incluye, redactarlo según las reglas de abajo.

## Reglas de siempre

- Resultados negativos: "con estos hiperparámetros" (nunca se ajustaron).
- n = 10 por brazo, P(a>b) no emparejada, **sin corrección por comparaciones múltiples**; "fuerte" solo si P ≥ 0,95 en las dos métricas; los intermedios, "indica una dirección".
- No cambies ni sustituyas ningún número del circuito real.
- No afirmes nada sobre cuándo se hizo cada experimento respecto a otros. Si hace falta, pregunta al autor.
