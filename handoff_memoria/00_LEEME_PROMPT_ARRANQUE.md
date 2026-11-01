# Paquete para el chat que redacta Resultados y Conclusiones

## Qué subir y en qué orden

| archivo | para qué | cómo darlo |
|---|---|---|
| `01_DOSSIER_RESULTADOS_v3.md` | **Fuente principal.** Reglas, mapa de experimentos, tablas (secciones 3–5), lectura de resultados F1–F10 (sección 6), salvedades, hipótesis, decisiones abiertas, esquema de capítulos. ~67 000 caracteres | adjuntar como archivo (o pegar entero si el chat no admite adjuntos) |
| `summary_by_arm.csv` | Una fila por brazo: n, IQM e IC de `final_perf` y `greedy_final`, semillas vivas. Para consultar una cifra sin buscarla en el dossier | adjuntar |
| `per_seed_metrics.csv` | Una fila por semilla (888 runs). Solo si hay que mirar semillas concretas o hacer gráficos | adjuntar solo si lo pide (es grande) |

El dossier ya contiene todo lo necesario para escribir; los CSV son de consulta.

## Mensaje de arranque (copiar y pegar como primer mensaje, con los archivos adjuntos)

```
Eres el redactor de la memoria de mi Trabajo Final de Máster (Ciencia de Datos, UOC) sobre aprendizaje por refuerzo cuántico: un estudio que extiende "Dissecting Quantum Reinforcement Learning" (arXiv:2511.17112) de PPO a DQN y a FrozenLake. Ya tienes escritos conmigo los dos primeros capítulos (alcance y estado del arte). Ahora te toca redactar los capítulos de RESULTADOS y CONCLUSIONES.

Te adjunto:
- 01_DOSSIER_RESULTADOS_v3.md: es tu fuente única de datos y de lectura. Léelo entero antes de escribir nada. Empieza por las "Reglas obligatorias" y respétalas todas.
- summary_by_arm.csv (y, si lo pides, per_seed_metrics.csv): cifras de consulta.

Cómo trabajar:
1. Lee el dossier y respóndeme solo con: (a) un resumen de 8 líneas de lo que has entendido, (b) las dudas o datos que te falten. No escribas capítulos todavía.
2. Cuando yo lo confirme, propón un índice detallado de los dos capítulos basándote en la sección 10 del dossier. Espera mi visto bueno.
3. Redacta después capítulo a capítulo, una sección cada vez, en español académico, en prosa (no listas largas), con tablas solo donde aporten. Cada afirmación numérica debe salir del dossier; indica entre corchetes la referencia (por ejemplo [F6], [tabla 3.2]) para que yo pueda verificarla.

Reglas que no puedes romper:
- Todo lo propio es DQN; el PPO es solo del paper. No hables de "no-replicación" por comparar con el paper.
- Resultados negativos siempre como "con estos hiperparámetros" (nunca se ajustaron).
- No uses nada de simulador/emulador; no hay cifras de él en el paquete y no debes pedirlas ni inventarlas.
- No conviertas un resultado inconcluyente en uno positivo; respeta las etiquetas de evidencia (FUERTE / MODERADA / DÉBIL / NEGATIVO) y la advertencia de contrastes sin corrección múltiple.
- Si te falta un dato o una cifra, pregúntame. No lo inventes ni lo deduzcas.
- Donde ponga "DECISIÓN A TOMAR", deja un marcador visible [DECISIÓN DEL AUTOR: ...]. Las decisiones las tomo yo.
- Si una cifra del texto de lectura (sección 6) no coincide con una tabla, manda la tabla: avísame.
```

## Por si el chat tiene poca memoria o pide trocear

Se puede pasar el dossier por partes sin perder nada:
1. Secciones 1–2b (contexto, reglas, hiperparámetros).
2. Secciones 3–5 (tablas).
3. Secciones 6–11 (lectura, salvedades, hipótesis, decisiones, esquema).

Y pedir que redacte primero Resultados (CartPole → FrozenLake → réplica) y después Conclusiones.

## Qué NO está en el paquete (a propósito)

- Todo lo de simulador (exp09, brazos `su2_*` y `sv_*`): requiere aprobación explícita del autor (`docs/SIMULATED-RESULTS-POLICY.md`).
- Los dossiers anteriores (v1, v2): tenían n < 10 en exp07 y exp08. No mezclar.
