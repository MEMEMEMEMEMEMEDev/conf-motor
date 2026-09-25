"""El bus: lo que el motor lee del hub y le devuelve, por redis.

Las claves y los campos son el CONTRATO entre conf-hub y conf-motor. El hub
tiene la misma lista en bus.go; cada repo tiene un test que la fija
(tests/test_motor.py::TestContrato y hub_test.go::TestContrato), así que un
campo renombrado de un lado pone rojo al otro antes de desplegar.
"""
from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass

import numpy as np

CLAVE_TRAMOS = "conf:tramos"
CLAVE_MOTOR = "conf:motor"
GRUPO = "motor"
MAX_SUBS = 5000

# Lo que el hub escribe en cada tramo (bus.go, PublicarTramo).
CAMPOS_TRAMO = ("sala", "idioma", "backend", "seq", "final", "t_ini", "t_fin", "glosario", "pcm")
# Lo que el hub lee de cada subtítulo (bus.go, subDeMensaje).
CAMPOS_SUB = ("tipo", "orig", "idioma", "es", "en", "t_ini", "t_fin", "backend", "lat_ms", "seq", "detalle")


def clave_subs(sala: str) -> str:
    return f"conf:subs:{sala}"


@dataclass
class Tramo:
    id: str
    sala: str
    idioma: str
    backend: str
    seq: int
    final: bool
    t_ini: int  # ms, reloj de pared del audio
    t_fin: int
    pcm: bytes
    glosario: str = ""  # nombres propios, separados por comas (hotwords de Whisper)

    def audio(self) -> np.ndarray:
        """PCM s16le 16 kHz → float32 en [-1, 1], lo que come faster-whisper."""
        return np.frombuffer(self.pcm, dtype="<i2").astype(np.float32) / 32768.0

    @staticmethod
    def de_mensaje(id_: str, v: dict) -> "Tramo":
        def s(k):
            x = v.get(k.encode(), v.get(k, b""))
            return x.decode() if isinstance(x, bytes) else str(x)

        pcm = v.get(b"pcm", v.get("pcm", b""))
        if isinstance(pcm, str):
            pcm = pcm.encode("latin-1")
        return Tramo(id=id_ if isinstance(id_, str) else id_.decode(), sala=s("sala"), idioma=s("idioma"),
                     backend=s("backend"), seq=int(s("seq") or 0), final=s("final") == "1",
                     t_ini=int(s("t_ini") or 0), t_fin=int(s("t_fin") or 0), pcm=pcm,
                     glosario=s("glosario"))


class Bus:
    def __init__(self, r, consumidor: str | None = None):
        self.r = r
        self.consumidor = consumidor or os.environ.get("HOSTNAME") or socket.gethostname()

    def preparar(self, desde: str = "$"):
        try:
            # "$": un motor nuevo no se come la cola vieja. Lo que quedó
            # mientras no había motor lo trae el grupo si ya existía, y la
            # regla de atraso lo salta.
            # "0" al RECUPERAR un redis reiniciado: ahí el stream es nuevo y
            # todo lo que tiene llegó después del reinicio; con "$" se
            # perdería lo que entró antes de recrear el grupo.
            self.r.xgroup_create(CLAVE_TRAMOS, GRUPO, id=desde, mkstream=True)
        except Exception as e:  # BUSYGROUP: ya existe
            if "BUSYGROUP" not in str(e):
                raise

    def leer(self, cuantos: int, bloquear_ms: int = 1000) -> list[Tramo]:
        res = self.r.xreadgroup(GRUPO, self.consumidor, {CLAVE_TRAMOS: ">"}, count=cuantos, block=bloquear_ms)
        out = []
        for _stream, mensajes in res or []:
            for id_, v in mensajes:
                out.append(Tramo.de_mensaje(id_, v))
        return out

    def confirmar(self, tramo: Tramo):
        self.r.xack(CLAVE_TRAMOS, GRUPO, tramo.id)

    def publicar(self, sala: str, **campos) -> str:
        desconocidos = set(campos) - set(CAMPOS_SUB)
        if desconocidos:
            raise ValueError(f"campos fuera del contrato: {desconocidos}")
        valores = {k: ("" if campos.get(k) is None else campos.get(k, "")) for k in CAMPOS_SUB}
        return self.r.xadd(clave_subs(sala), valores, maxlen=MAX_SUBS, approximate=True)

    def latir(self, **estado):
        self.r.hset(CLAVE_MOTOR, mapping={"latido": int(time.time() * 1000), **estado})
