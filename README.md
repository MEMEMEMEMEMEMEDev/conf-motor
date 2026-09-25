# conf-motor — transcribe y traduce todas las salas

Parte de [conf](https://github.com/MEMEMEMEMEMEMEDev/conf-hub): subtítulos y
traducción en vivo para conferencias. Este servicio lee los tramos de audio que
el hub deja en redis, los transcribe y los traduce, y devuelve el texto.

**Un proceso para todas las salas**, con el modelo compartido: es la forma que
midió 32 salas en una RTX 5070 (`mediciones/`). El tope de salas es
`MAX_SALAS` (la demo usa 28).

## Backends, elegibles por sala

| Backend | Qué corre | Cuándo |
|---|---|---|
| `gpu` | faster-whisper large-v3-turbo fp16 + opus-mt (CTranslate2) | por defecto |
| `cpu` | faster-whisper small int8 + opus-mt | respaldo sin GPU ni credencial; ~6 s de proceso por tramo con 6 salas |
| `gemini` | Gemini 2.5 Flash-Lite en Vertex AI, transcripción y traducción en una llamada | por sala, o automático si la GPU falla |

Reglas del despacho (`motor/nucleo.py`, cada una con su test):

- un tramo con más de `ATRASO_MAX_S` (25 s) de atraso se salta y se cuenta;
- los finales antes que los parciales; con el motor cargado, los parciales se descartan;
- un error de CUDA marca la GPU caída y todo lo que pedía `gpu` va al respaldo;
- un motor que arrancó sin GPU no la carga al vuelo porque una sala la pida;
- si redis se reinicia (no tiene disco), el grupo de consumo se recrea solo.

opus-mt traduce **por oración**: con dos o tres juntas traduce una y descarta
el resto sin avisar (visto en la primera corrida local).

## Variables

| Variable | Qué | Defecto |
|---|---|---|
| `REDIS_URL` | el bus | `redis://localhost:6379/0` |
| `BACKEND` | backend local que se carga al arrancar: `gpu`, `cpu`, `falso`, `ninguno` | `gpu` |
| `MAX_SALAS` | hilos del pool y `num_workers` de CTranslate2 | `28` |
| `MODELOS` | pesos: `whisper-large-v3-turbo/`, `whisper-small/`, `opus-mt-en-es-ct2/`, `opus-mt-es-en-ct2/` | `/modelos` |
| `ATRASO_MAX_S` | atraso a partir del cual se salta un tramo | `25` |
| `VERTEX_PROYECTO`, `VERTEX_REGION`, `GEMINI_MODELO` | Gemini | —, `us-central1`, `gemini-2.5-flash-lite` |
| `GOOGLE_APPLICATION_CREDENTIALS` | cuenta de servicio montada como archivo; la lee `google.auth`, el motor nunca la abre | — |

## Los pesos

No viajan en la imagen. Los traductores se convierten una vez:

```bash
pip install ctranslate2==4.8.2 "transformers<5" sentencepiece torch
for par in en-es es-en; do
  ct2-transformers-converter --model Helsinki-NLP/opus-mt-$par --quantization int8 \
    --output_dir modelos/opus-mt-$par-ct2 --copy_files source.spm target.spm vocab.json tokenizer_config.json
done
```

Whisper: `huggingface-cli download mobiuslabsgmbh/faster-whisper-large-v3-turbo --local-dir modelos/whisper-large-v3-turbo`.

## En una RTX 50xx (Blackwell)

CTranslate2 4.8.2 con `nvidia-cublas-cu12` y `nvidia-cudnn-cu12==9.*` por pip
anda sin instalar CUDA en el sistema; sólo hay que poner las `lib/` de esos
paquetes en `LD_LIBRARY_PATH` (el `Containerfile` lo hace). Esas dos
bibliotecas son la mayor parte del peso de la imagen (~2,5 GB en total).

## Tests

```bash
pip install --require-hashes -r requirements.txt -r requirements-test.txt
python -m unittest discover -s tests -t .
```

Sin GPU, sin modelos y sin redis: backends dobles y fakeredis. Es la puerta
del build: un test rojo y la imagen no existe.
