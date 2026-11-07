# Para el chat que redacta la memoria: experimento 11 y dos erratas

Este paquete añade tres cosas al dossier v3 que ya tienes. **No sustituye ningún fichero que te entregamos**: las correcciones van en un fichero aparte (`erratas/DOSSIER_v3_ERRATA.md`).

## Qué contiene

| carpeta / fichero | para qué |
|---|---|
| `erratas/DOSSIER_v3_ERRATA.md` | dos errores del dossier v3: (E1) parámetros de `frozen_onehot_mlp`, 5 508 y no 4 548; (E2) episodios de la evaluación greedy, no son 20 en todo CartPole |
| `erratas/TABLA_EPISODIOS_GREEDY.md` y `..._por_grupo.csv` | episodios greedy por experimento y por brazo de exp01, con el método y sus límites |
| `exp11/DOSSIER_EXP11.md` | experimento nuevo: control clásico que replica su entrada (diseño, anchos, tablas, contrastes C1-C4, regla de lectura aplicada) |
| `exp11/*.csv`, `tables_exp11.md`, `design_widths_params.json` | datos: una fila por ejecución, curvas por semilla, resumen por brazo |
| `exp11/EXPERIMENT-11-REPLICATION-CONTROL.md` | diseño y regla de lectura, escritos antes de lanzar |

## Orden de lectura

1. `erratas/DOSSIER_v3_ERRATA.md` y la tabla de episodios.
2. `exp11/DOSSIER_EXP11.md`.

## Qué te pedimos

1. **Corregir en la memoria** el tamaño de la MLP one-hot de referencia (5 508) y la frase de la evaluación greedy: 5 episodios en exp01 (salvo el techo de Fourier), exp02, exp03 y exp03b, y 20 en exp07 y exp08. Añade una nota en las comparaciones que cruzan los dos grupos (por ejemplo `hybrid_OR{R}` de exp02 frente a `hybrid_or_r{R}_noent` de exp08): `greedy_final` se mide con distinto ruido; `final_perf` no tiene ese problema.
2. **Revisar exp11 y decir si conviene ajustar** la parte de la memoria sobre Output Reuse, antes de escribir nada. Propón primero los ajustes.

## Lo que dice exp11, para que lo redactes con cuidado

- Es **el circuito real y brazos clásicos**, no un simulador. Puede entrar en la memoria cuando el autor lo apruebe; la decisión es suya.
- El nuevo control (replicación R + capa oculta ReLU + observación completa + presupuesto igual al del híbrido) **no mejora con R**: está en el suelo o muy cerca (IQM 10,0 / 10,6 / 13,0 en R = 8 / 16 / 32), y a igual presupuesto queda por debajo del control que replica 4 veces (P ≤ 0,03).
- El híbrido lo supera con P = 1,00 en las dos métricas en R = 4, 8, 16 y 32.
- **No escribas que esto confirma que el Output Reuse sea específico del circuito.** Como el control casi no aprende, el contraste informa poco sobre si la replicación podría ayudar a una red clásica que sí aprende. Además el control mezcla más replicación con menos neuronas ocultas (9-10 frente a 18-59), y no se puede separar cuál de las dos cosas causa la caída. La regla de lectura fijada antes de lanzar no se cumple limpia en ninguna de sus tres ramas; el dossier lo explica.

## Reglas de siempre

- Resultados negativos: "con estos hiperparámetros" (nunca se ajustaron).
- n = 10 por brazo, P(a>b) no emparejada, **sin corrección por comparaciones múltiples**; "fuerte" solo si P ≥ 0,95 en las dos métricas; los valores intermedios se redactan como "indica una dirección".
- No cambies ni sustituyas ningún número del circuito real.
- No afirmes nada sobre cuándo se hizo cada experimento respecto a otros. Si hace falta, pregunta al autor.
