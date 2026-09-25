# hackatón · capacidad, calidad y costo de subtítulos en vivo (experimental, 2026-09-25)

Encargo de la principal (`registro/pendientes.md`, 25-09). Todo corrido FUERA
del clúster de casa, en un venv propio (`hackaton/.venv`: Python 3.12,
CTranslate2 4.8.2, faster-whisper 1.2.1, cuDNN 9 y cuBLAS 12 por pip). La RTX
5070 (Blackwell, driver 595.84) funciona con esa combinación sin tocar nada.
Guiones aquí mismo: `bench.py` (salas en vivo), `salas_procesos.py` (una sala
por proceso), `calidad.py` (WER, recall de glosario, chrF), `gemini_bench.py`
(Vertex). Crudos en `crudos/`. Ninguna credencial en ficheros.

**Material:** los dos fragmentos del repo Subtitula (`flordelcastillo/subtitula`
`edc5c89`, Apache 2.0; audio de Nerdearla 2025): Thor Schaeff (inglés, 130 s) y
midudev (español, 130 s). Las salas alternan en→es y es→en, en bucle, a ritmo
real: cada tramo existe recién cuando terminó de decirse.

## Resumen para el README

| backend | modelo | salas medidas | latencia de proceso p95 | llega detrás de la voz | USD por sala-hora |
|---|---|---|---|---|---|
| **RTX 5070 (casa)** | large-v3-turbo fp16 + opus-mt, tramos 4 s | **32** (40 no) | 0,44 s | ~4,5 s | **0,0024** (luz: ver §4) |
| **CPU 8 núcleos (5950X)** | small int8 + opus-mt, tramos 10 s | **6** (8 no) | 6,3 s | ~16 s | — (hardware propio) |
| CPU Vultr vhp-12c-24gb | igual | ~4 (estimado desde el 5950X) | — | — | **~0,05** (estimado) |
| **Gemini 2.5 Flash-Lite** | por tramo de 4 s | sin límite propio (cuota) | 2,94 s | ~6 s | **0,057** (medido) |
| Gemini 2.5 Flash | por tramo de 4 s | sin límite propio | 2,50 s | ~6 s | 0,228 (medido) |
| Gemini Live | audio nativo | — | — | — | ~0,37 (calculado) |
| RTX 4090 / 5090 (RunPod) | como la 5070 | ≥32 (piso: son más potentes) | — | — | ≤0,011 / ≤0,022 (estimado) |

**Calidad** (contra un borrador de Gemini 2.5 Flash, NO humano): con el audio
entero, WER 5,8–8,8 % (precisión ~91–94 %). En vivo con large-v3-turbo y
tramos de 10 s, WER 8,5 % (en) y 8,8 % (es); con tramos de 4 s el español cae
a 20 %. Traducción opus-mt, chrF 54–61 contra la traducción de Gemini.

## 1. GPU: RTX 5070 12 GB (`crudos/gpu-*.jsonl`)
Un proceso, el modelo compartido por todas las salas (CTranslate2
`num_workers` = salas), beam 1, VAD, tramos de 4 s, traducción en la misma
GPU. CPU de apoyo: 8 núcleos (0-7,16-23). AGUANTA = p95 ≤ 3 s y sin atraso.

| modelo | 8 | 16 | 24 | 32 | 40 | 48 |
|---|---|---|---|---|---|---|
| large-v3-turbo fp16 | p95 0,17 s | 0,27 | 0,36 | **0,44 (10,8 GB)** | NO (VRAM llena) | — |
| large-v3-turbo int8_fp16 | 0,16 | 0,24 | 0,32 | **0,35 (7,8 GB)** | NO (CPU de apoyo al 88 %) | NO |
| small fp16 | 0,15 | 0,20 | — | **0,27 (7,7 GB)** | — | NO (VRAM llena) |

- **32 salas es el techo medido en las tres configuraciones.** Lo marcan la
  VRAM (fp16) y los núcleos que preparan el audio (~0,3 núcleos por sala: el
  espectrograma de Whisper se calcula en CPU).
- Consumo con 32 salas: **180 W de media** (máx. 190 W, uso de GPU 90 %),
  10 W en reposo (`crudos/potencia-5070-32salas-*.csv`, nvidia-smi cada 1 s).

## 2. CPU: Ryzen 9 5950X (`crudos/cpu-*.jsonl`)
Fijado a 8 núcleos físicos / 16 hilos (CPUs 8-15,24-31; el otro bloque lo
usaba el operador). int8. Primero salas como hilos de un proceso; después
**una sala por proceso** (como un pod por sala), que es lo que vale: los
hilos se pisaban en el intérprete de Python y ocultaban la CPU libre.

- **large-v3-turbo int8 NO es tiempo real en CPU**: 1 sala, 8 hilos, RTF 1,43.
- **El tramo de 4 s es lo que encarece la CPU**: Whisper procesa una ventana
  fija de 30 s por llamada. small, 4 hilos: RTF 0,35 con tramos de 4 s,
  0,16 con 8 s, 0,14 con 10 s.
- **small int8, tramos de 10 s, 2 hilos, una sala por proceso:**

| salas | p95 | atraso | hilos ocupados | veredicto |
|---|---|---|---|---|
| 2 | 2,76 s | +0,04 s | 1,3 | AGUANTA |
| 4 | 4,44 s | +0,17 s | 3,8 | AGUANTA (sin atraso) |
| **6** | **6,28 s** | **+0,01 s** | 6,8 | **AGUANTA: el techo** |
| 8 | 8,81 s | +0,58 s | 11,8 | NO (el atraso crece) |

  **6 salas en 8 núcleos físicos: ~1,3 núcleos físicos (2,7 hilos) por sala.**
  Coincide con la hipótesis de ~1,5 núcleos (condiciones §8), pero con ~6 s de
  proceso encima de los 10 s del tramo.
- **A vCPU de una VPS:** una vCPU de Vultr es un hilo compartido, no un
  núcleo. 12 vCPU ≈ 12 hilos → 12 / 2,7 ≈ 4 salas, menos lo que consume la
  propia plataforma (k3s, ArgoCD, Jenkins, observabilidad): **3–4 salas por
  vhp-12c-24gb, estimado, no medido en la VPS** (regla de la principal).

## 3. Calidad (`crudos/calidad-*.jsonl`)
Referencia: **borrador de Gemini 2.5 Flash** (transcripción literal del
fragmento entero + traducción), `referencias-borrador/`. YouTube no tiene
subtítulos humanos de estas charlas. WER sin muletillas (um, uh, eh). El
glosario (faster-whisper `hotwords`) tiene 10 términos por charla que
APARECEN en la referencia (ElevenLabs, HDMI, Midu.dev, Javier Tebas, la Liga…).

| modelo (GPU) | entero en / es | vivo 4 s en / es | vivo 10 s en / es |
|---|---|---|---|
| small fp16 | 6,4 % / 8,8 % | 12,9 % / 17,5 % | 21,0 % / 11,2 % |
| large-v3-turbo fp16 | 5,8 % / 6,1 % | 9,2 % / 20,2 % | **8,5 % / 8,8 %** |
| large-v3-turbo int8 | 11,5 % / 5,8 % | 9,5 % / 20,7 % | — |

- **Glosario:** sube el recall de términos (small en vivo 4 s: en 71 %→100 %,
  es 63 %→84 %; turbo 10 s es: 74 %→95 %) pero EMPEORA el WER general en
  tramos cortos (el modelo mete los términos donde no van: small en vivo 4 s en
  12,9 %→20,0 %). Con el audio entero apenas mueve el WER. Recomendación: sí
  al glosario, con tramos largos o sólo para nombres propios, y medirlo.
- **Traducción** opus-mt (CTranslate2 int8) sobre la transcripción del
  modelo: chrF 54–61 en vivo, 57–70 con el audio entero, contra la traducción
  de Gemini (mide «distancia a Gemini», no calidad absoluta).
- Si alguien corrige a mano los 4 min 20 s de `referencias-borrador/`,
  `calidad.py` da los mismos números contra humano.

## 4. Costo por sala-hora
- **Gemini (medido):** tokens de `usageMetadata` × precio de
  ai.google.dev/gemini-api/docs/pricing (actualizada 2026-09-24; extracto en crudos).
  Tramos de 4 s = 900 llamadas por hora. Flash-Lite 125 tokens de audio + 63 de
  texto + 49 de salida por llamada → 0,057 USD; Flash → 0,228 USD. Nuestro
  prompt es corto: un glosario de ~500 tokens por llamada lleva Flash-Lite a
  ~0,10. Live: 3,00 USD/1M de audio de entrada → 115 k tokens/h = 0,345 + ~0,024
  de texto = ~0,37 (calculado, no medido). Gasto total en Vertex: US$0,044.
- **5070 de casa:** GPU 180 W medidos con 32 salas + CPU de apoyo y resto del
  equipo ~130 W (estimado, no medido: 10 núcleos del 5950X ocupados, placa,
  disco) ≈ 0,31 kW → **0,0097 kWh por sala-hora**. A 0,25 USD/kWh (SUPUESTO,
  cambiar por la tarifa real) = **0,0024 USD por sala-hora**. Sin amortizar el
  hardware.
- **VM de CPU Vultr** `vhp-12c-24gb-amd`: US$0,197/h (API pública, 25-09,
  disponible en Santiago) ÷ 3–4 salas (estimado, §2) = **0,05–0,066 USD**.
- **GPU de alquiler (no medidas):** Vultr no tiene GPU en Santiago y pedía un
  depósito de US$50 en dinero real para desplegar una (descartado por el
  operador). RunPod (runpod.io/pricing, 2026-09-13): RTX 4090 US$0,34/h, RTX
  5090 US$0,69/h (nube comunitaria). Con ≥32 salas (la 5070 es el piso) →
  ≤0,011 y ≤0,022 USD por sala-hora, si el pod trae ~10 núcleos de CPU para
  preparar el audio de 32 salas.

## Límites de estas mediciones
- La referencia de calidad es de un modelo, no humana.
- 2 charlas × 130 s: poco audio. Los números de capacidad son de régimen
  (80 s por sala y nivel); los de calidad, indicativos.
- CPU medida en 8 núcleos de un 5950X de escritorio (hasta 4,9 GHz), con el
  otro bloque ocupado por el operador (juego): una VPS rinde menos por hilo.
- La GPU corre un solo proceso con el modelo compartido; batching real entre
  salas podría subir el techo de 32.
