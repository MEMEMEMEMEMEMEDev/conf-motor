# conf-motor — transcribe y traduce los tramos de todas las salas en la GPU.
#
# EL FROM LO ESCRIBE `aegis app new` (plantilla service-python), resuelto
# contra el registro interno y fijado por digest. No se copia de otro repo:
# plataforma lo acaba de actualizar por CVE (pendientes.md, 06:45).
#
# DOS ETAPAS, como pide la plantilla el día que hay dependencias:
#   prueba   instala TODO (con las de test), corre la suite; si falla,
#            no hay imagen
#   final    sólo el venv de runtime y el código
#
# LA IMAGEN NO LLEVA PESOS. Los modelos (large-v3-turbo, opus-mt) vienen del
# volumen que plataforma siembra en /modelos. Sí lleva las bibliotecas CUDA
# por pip (cuBLAS 12, cuDNN 9): es la combinación que midió experimental en
# la RTX 5070 sin tocar el sistema (mediciones/entorno.sh), y pesan ~1,5 GB.
# El driver lo pone el nodo; el acceso a la placa, el contrato.
FROM registry.registry-system.svc.cluster.local:5000/python:3.12-slim@sha256:2d62568c3174136030ac1da4e534cfbfe709aec913daeb2a3d49328e27438093 AS prueba
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN python -m venv /opt/venv
COPY requirements.txt requirements-test.txt ./
RUN /opt/venv/bin/pip install --require-hashes -r requirements.txt \
 && python -m venv /opt/venv-test \
 && /opt/venv-test/bin/pip install --require-hashes -r requirements.txt -r requirements-test.txt
COPY motor ./motor
COPY tests ./tests
# LA PUERTA DE LA IMAGEN: sin GPU, sin modelos y sin redis (backends dobles
# y fakeredis). Un test rojo y la imagen no existe.
RUN /opt/venv-test/bin/python -m unittest discover -s tests -t . -v

FROM registry.registry-system.svc.cluster.local:5000/python:3.12-slim@sha256:2d62568c3174136030ac1da4e534cfbfe709aec913daeb2a3d49328e27438093
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    MODELOS=/models
# CTranslate2 busca cuBLAS y cuDNN por el cargador dinámico: se le dicen
# las carpetas de los paquetes nvidia-* del venv (lo mismo que hace
# mediciones/entorno.sh con LD_LIBRARY_PATH).
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib
COPY --from=prueba /opt/venv /opt/venv
WORKDIR /app
COPY motor ./motor
# Numérico y no-root: PSS restricted. El sistema de archivos es de sólo
# lectura: los pesos se leen de /modelos y nada se escribe fuera de /tmp.
USER 65532:65532
CMD ["python", "-m", "motor"]
