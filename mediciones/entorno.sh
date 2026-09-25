# entorno.sh — cargar con `source entorno.sh` desde la carpeta del banco.
# Activa un venv (./.venv) y pone en LD_LIBRARY_PATH las bibliotecas CUDA que
# llegan por pip (nvidia-cublas-cu12, nvidia-cudnn-cu12==9.*): así CTranslate2
# 4.8.2 corre en una RTX 50xx (Blackwell) sin instalar CUDA en el sistema.
# Preparar una vez:
#   python3.12 -m venv .venv && . .venv/bin/activate
#   pip install "ctranslate2==4.8.2" "faster-whisper==1.2.1" nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" \
#               sentencepiece psutil pynvml numpy jiwer sacrebleu
# Los guiones esperan, relativo a esta carpeta: fuentes/subtitula/samples/*.mp3
# (repo flordelcastillo/subtitula) y modelos/opus-mt-{en-es,es-en}-ct2
# (ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-es --quantization int8
#  --copy_files source.spm target.spm vocab.json tokenizer_config.json).
. "$(pwd)/.venv/bin/activate"
export HF_HOME="$(pwd)/modelos/hf" HF_HUB_DISABLE_TELEMETRY=1
SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(ls -d "$SP"/nvidia/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}
