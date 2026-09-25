"""conf-motor — transcribe y traduce los tramos de todas las salas.

UN proceso para todas las salas, con el modelo compartido: es lo que midió
experimental en la RTX 5070 (32 salas con large-v3-turbo fp16 + opus-mt en
un solo proceso, CTranslate2 num_workers = salas). La demo corre con
MAX_SALAS=28 para dejar margen de VRAM (plataforma, D-01).

Variables de entorno:
  REDIS_URL          redis://…
  BACKEND            el backend local que se carga al arrancar: gpu | cpu | ninguno
  MAX_SALAS          hilos del pool y num_workers de CTranslate2 (28)
  MODELOS            carpeta de pesos (/modelos): whisper-large-v3-turbo,
                     whisper-small, opus-mt-en-es-ct2, opus-mt-es-en-ct2
  ATRASO_MAX_S       atraso a partir del cual un tramo se salta (25)
  VERTEX_PROYECTO, VERTEX_REGION, GEMINI_MODELO
  GOOGLE_APPLICATION_CREDENTIALS   la cuenta de servicio montada (la lee google.auth)
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading

import redis

from .backends import Falso, Gemini, Local
from .bus import Bus
from .nucleo import Nucleo

log = logging.getLogger("conf-motor")


def url_redis(e: dict) -> str:
    """REDIS_URL si está; si no, REDIS_ADDR + REDIS_PASSWORD, que es como lo
    entrega la plataforma (servicio `bus`, Secret bus-credenciales)."""
    if e.get("REDIS_URL"):
        return e["REDIS_URL"]
    from urllib.parse import quote
    addr = e.get("REDIS_ADDR", "localhost:6379")
    pw = e.get("REDIS_PASSWORD", "")
    return f"redis://:{quote(pw, safe='')}@{addr}/0" if pw else f"redis://{addr}/0"


def ruta_modelo(modelos: str, carpeta: str, respaldo: str) -> str:
    """El volumen de la plataforma si lo tiene; si no, el nombre para que
    faster-whisper lo baje (sólo en desarrollo: en el clúster, los pesos
    vienen del volumen y la descarga sería un arranque de minutos)."""
    p = os.path.join(modelos, carpeta)
    return p if os.path.isdir(p) else respaldo


class Fabrica:
    """Crea cada backend la primera vez que se lo pide y lo reutiliza.
    Un backend que no se puede crear devuelve None (y se recuerda)."""

    def __init__(self, entorno: dict):
        self.e = entorno
        self.max_salas = int(entorno.get("MAX_SALAS", "28"))
        self.modelos = entorno.get("MODELOS", "/modelos")
        self._hechos = {}
        self._cerrojo = threading.Lock()

    def crear(self, nombre: str):
        if nombre == "gpu":
            b = Local("cuda", ruta_modelo(self.modelos, "whisper-large-v3-turbo", "large-v3-turbo"),
                      self.modelos, self.max_salas)
            b.calentar()
            return b
        if nombre == "cpu":
            # En CPU, pocos hilos por sala y pocas salas a la vez: medido,
            # small int8 con 2 hilos aguanta ~6 salas en 8 núcleos.
            b = Local("cpu", ruta_modelo(self.modelos, "whisper-small", "small"), self.modelos,
                      num_workers=min(self.max_salas, int(self.e.get("CPU_SALAS", "4"))), hilos=2)
            b.calentar()
            return b
        if nombre == "gemini":
            if not self.e.get("GOOGLE_APPLICATION_CREDENTIALS") or not self.e.get("VERTEX_PROYECTO"):
                return None
            return Gemini(self.e["VERTEX_PROYECTO"], self.e.get("VERTEX_REGION", "us-central1"),
                          self.e.get("GEMINI_MODELO", "gemini-2.5-flash-lite"))
        if nombre == "falso":
            return Falso()
        return None

    def __call__(self, nombre: str):
        with self._cerrojo:
            if nombre not in self._hechos:
                try:
                    self._hechos[nombre] = self.crear(nombre)
                    log.info("backend %s: %s", nombre, "listo" if self._hechos[nombre] else "no disponible")
                except Exception as e:  # noqa: BLE001
                    log.error("backend %s no arrancó: %s", nombre, e)
                    self._hechos[nombre] = None
            return self._hechos[nombre]


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='{"t":"%(asctime)s","nivel":"%(levelname)s","msg":"%(message)s"}')
    e = dict(os.environ)
    fabrica = Fabrica(e)
    r = redis.Redis.from_url(url_redis(e))
    respaldo = "gemini" if fabrica("gemini") else "cpu"
    nucleo = Nucleo(Bus(r), fabrica, max_salas=fabrica.max_salas,
                    atraso_max_s=float(e.get("ATRASO_MAX_S", "25")), respaldo=respaldo)

    arranque = e.get("BACKEND", "gpu")
    if arranque == "falso":
        # Pila de desarrollo sin modelos: todo tramo, sea cual sea su
        # backend, lo atiende el doble.
        doble = Falso(texto="(motor falso) texto de prueba")
        nucleo.fabrica = lambda _nombre: doble
    elif arranque in ("gpu", "cpu"):
        # Se carga ANTES de leer el primer tramo: un motor que tarda en
        # arrancar tiene que tardar al arrancar, no con la sala hablando.
        b = fabrica(arranque)
        if arranque == "gpu":
            nucleo.gpu = "ok" if b else "caida"
    log.info("motor listo: backend=%s gpu=%s respaldo=%s max_salas=%d", arranque, nucleo.gpu,
             respaldo, fabrica.max_salas)

    parar = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: parar.set())
    signal.signal(signal.SIGINT, lambda *_: parar.set())
    nucleo.correr(parar)


if __name__ == "__main__":
    main()
