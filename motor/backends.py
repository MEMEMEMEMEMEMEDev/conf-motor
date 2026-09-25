"""Los motores de transcripción y traducción, detrás de UNA interfaz.

Cada sala elige el suyo (el hub manda `backend` en cada tramo):

  gpu     faster-whisper large-v3-turbo fp16 + opus-mt, en la RTX 5070
  cpu     faster-whisper small int8 + opus-mt, en CPU (respaldo, más atraso)
  gemini  Vertex AI, gemini-2.5-flash-lite: audio → transcripción y
          traducción en UNA llamada con salida JSON

La combinación gpu es EXACTAMENTE la que midió experimental
(mediciones/bench.py:105-145): mismo modelo, mismo compute_type, mismos
parámetros de transcribe. Si cambia algo acá, el número del README deja de
describir lo que corre.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import wave
from dataclasses import dataclass

import numpy as np

DESTINO = {"en": "es", "es": "en"}
NOMBRE = {"en": "inglés", "es": "español"}


@dataclass
class Resultado:
    orig: str
    idioma: str
    es: str = ""
    en: str = ""


def completar(idioma: str, orig: str, traduccion: str) -> Resultado:
    r = Resultado(orig=orig, idioma=idioma)
    setattr(r, idioma, orig)
    setattr(r, DESTINO[idioma], traduccion)
    return r


# Un error de la placa, no del audio. Si la 5070 se cae del bus (ya pasó en
# esta máquina) o se queda sin memoria, CTranslate2 lo dice con alguna de
# estas palabras; el motor entonces pasa las salas a Gemini (fallback
# automático, plataforma D-01).
MARCAS_GPU = ("cuda", "cudnn", "cublas", "out of memory", "device-side", "nvidia", "gpu")


def humanizar(backend: str, e: BaseException) -> str:
    """Un error de un backend, dicho para quien mira el panel a las tres de
    la mañana: qué backend, qué pasó, sin volcado de Python."""
    nombre = {"gpu": "la GPU", "cpu": "la CPU", "gemini": "Gemini"}.get(backend, backend)
    t = f"{type(e).__name__} {e}".lower()
    if "timeout" in t or "timed out" in t:
        return f"{nombre} no respondió a tiempo"
    if "429" in t or "quota" in t or "resource_exhausted" in t:
        return f"{nombre}: cuota agotada"
    if "401" in t or "403" in t or "permission" in t or "credential" in t:
        return f"{nombre}: la credencial no sirve"
    if "connection" in t or "name resolution" in t or "unreachable" in t:
        return f"{nombre}: sin conexión"
    if backend == "gpu" and es_error_de_gpu(e):
        return "la GPU falló (error de CUDA)"
    if "vertex 5" in t:
        return f"{nombre}: error del servicio"
    return f"{nombre}: {type(e).__name__}"


def es_error_de_gpu(e: BaseException) -> bool:
    t = f"{type(e).__name__} {e}".lower()
    return any(m in t for m in MARCAS_GPU)


_FIN_ORACION = re.compile(r"(?<=[.!?…])\s+")


# ---- el glosario, del lado del texto ------------------------------------------
#
# Whisper recibe el glosario como hotwords, pero no alcanza: medido en la
# charla de midudev salió "Midude", "mi Dudef", y opus-mt traduce los
# nombres propios ("Miguel Ángel Durán" → "Michelangelo Durán", "Javier
# Tebas" → "Javier Thebes"). Dos pasos más, puros y con tests:
#   corregir   después de transcribir: lo que se parece MUCHO a un nombre
#              del glosario (≥ 85 %, sin espacios ni tildes ni mayúsculas)
#              se escribe como en el glosario. Sólo para nombres propios
#              (con mayúscula, punto o dígito): un término en minúscula se
#              deja como está, para no cambiar "influencers" por "influencer".
#   proteger   antes de traducir: cada término del glosario presente se
#              cambia por una marca X0X que opus-mt no toca (probado en los
#              dos sentidos) y se restaura después.

def terminos(glosario: str) -> list[str]:
    ts = [t.strip() for t in (glosario or "").split(",") if t.strip()]
    return sorted(set(ts), key=len, reverse=True)


def _norm(x: str) -> str:
    import unicodedata
    x = unicodedata.normalize("NFKD", x.lower())
    return "".join(c for c in x if c.isalnum())


def _es_nombre(t: str) -> bool:
    return any(c.isupper() or c.isdigit() or c == "." for c in t)


def corregir(texto: str, glosario: str) -> str:
    import difflib
    ts = [t for t in terminos(glosario) if _es_nombre(t) and len(_norm(t)) >= 5]
    if not ts or not texto:
        return texto
    palabras = texto.split(" ")
    i = 0
    salida = []
    while i < len(palabras):
        hecho = False
        for n in (3, 2, 1):
            if i + n > len(palabras):
                continue
            tramo = " ".join(palabras[i:i + n])
            nucleo = tramo.rstrip(".,;:!?…)\"'»")
            cola = tramo[len(nucleo):]
            cabeza = ""
            while nucleo and nucleo[0] in "(\"'«¿¡":
                cabeza, nucleo = cabeza + nucleo[0], nucleo[1:]
            k = _norm(nucleo)
            if not k:
                continue
            # El MÁS parecido; a igual parecido, el de largo más cercano a lo
            # que se escribió ("Midudev" y "Midu.dev" normalizan igual:
            # dicho "midudev", va "Midudev").
            mejor = None
            for t in ts:
                kt = _norm(t)
                # Si el tramo tiene MÁS palabras que el término, sólo vale si
                # es el mismo largo (Whisper partió una palabra: "mi Dudef"
                # por "Midudev"); si no, "de la liga" se comía el "de".
                tolera = 3 if n <= len(t.split()) else 1
                if abs(len(k) - len(kt)) > tolera:
                    continue
                r = 1.0 if k == kt else difflib.SequenceMatcher(None, k, kt).ratio()
                if r >= 0.85:
                    # A igual parecido: sin punto primero (al hablar nadie
                    # dice "Midu.dev"), después el de largo más cercano.
                    clave = (r, "." not in t, -abs(len(t) - len(nucleo)))
                    if mejor is None or clave > mejor[0]:
                        mejor = (clave, t)
            if mejor:
                salida.append(cabeza + mejor[1] + cola)
                i += n
                hecho = True
                break
        if not hecho:
            salida.append(palabras[i])
            i += 1
    return " ".join(salida)


def proteger(texto: str, glosario: str) -> tuple[str, dict[str, str]]:
    mapa = {}
    for t in terminos(glosario):
        patron = re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)", re.IGNORECASE)
        if patron.search(texto):
            marca = f"X{len(mapa)}X"
            texto = patron.sub(marca, texto)
            mapa[marca] = t
    return texto, mapa


def restaurar(texto: str, mapa: dict[str, str]) -> str:
    for marca, t in mapa.items():
        texto = texto.replace(marca, t)
    return texto


def oraciones(texto: str) -> list[str]:
    """Parte un tramo en oraciones. opus-mt se entrenó con oraciones
    sueltas: con dos o tres juntas traduce una y DESCARTA el resto sin
    avisar (visto en la primera corrida: «Do we have anyone on? Yeah,
    maybe…» salió sólo con la segunda). Se traducen todas en un lote."""
    return [o for o in (x.strip() for x in _FIN_ORACION.split(texto)) if o]


class Local:
    """Whisper + opus-mt sobre CTranslate2, en cuda o en cpu."""

    def __init__(self, device: str, modelo_asr: str, modelos_mt: str, num_workers: int, hilos: int = 2):
        import ctranslate2
        import sentencepiece as spm
        from faster_whisper import WhisperModel

        self.device = device
        comp = "float16" if device == "cuda" else "int8"
        mtcomp = "int8_float16" if device == "cuda" else "int8"
        self.asr = WhisperModel(modelo_asr, device=device, compute_type=comp, cpu_threads=hilos,
                                num_workers=num_workers)
        self.mt, self.sp = {}, {}
        for src, dst in DESTINO.items():
            d = os.path.join(modelos_mt, f"opus-mt-{src}-{dst}-ct2")
            self.mt[src] = ctranslate2.Translator(d, device=device, compute_type=mtcomp,
                                                  inter_threads=num_workers, intra_threads=hilos)
            self.sp[src] = (spm.SentencePieceProcessor(model_file=f"{d}/source.spm"),
                            spm.SentencePieceProcessor(model_file=f"{d}/target.spm"))

    def calentar(self):
        """El primer tramo en CUDA paga la inicialización de la placa:
        medido en la 5070, 4,7 s contra 0,17 s de los siguientes. Se paga
        al arrancar (mediciones/bench.py hace lo mismo) y no con la primera
        frase de la charla. Un tono de 2 s: el VAD va apagado para que el
        modelo corra de verdad sobre algo."""
        t = np.arange(32000, dtype=np.float32) / 16000
        tono = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        for idioma in DESTINO:
            list(self.asr.transcribe(tono, language=idioma, beam_size=1, vad_filter=False,
                                     without_timestamps=True, temperature=[0.0])[0])
            self.traducir("Hola.", idioma)

    def transcribir(self, audio: np.ndarray, idioma: str, glosario: str = "") -> str:
        segs, _ = self.asr.transcribe(
            audio, language=idioma, beam_size=1, vad_filter=True,
            condition_on_previous_text=False, without_timestamps=True,
            # Una sola temperatura: la escalera de reintentos de faster-
            # whisper multiplica el costo justo con el audio repetitivo
            # ("probando, probando"). Medido en el motor CPU de aegis:
            # 10,5 s contra 3,0 s para el mismo audio (engine-cpu/motor.py).
            temperature=[0.0],
            hotwords=glosario or None,
        )
        return " ".join(s.text.strip() for s in segs).strip()

    def traducir(self, texto: str, de: str, glosario: str = "") -> str:
        if not texto:
            return ""
        s_src, s_tgt = self.sp[de]
        protegido, mapa = proteger(texto, glosario)
        lote = [s_src.encode(o, out_type=str) + ["</s>"] for o in oraciones(protegido)]
        out = self.mt[de].translate_batch(lote, beam_size=2)
        return restaurar(" ".join(s_tgt.decode(r.hypotheses[0]) for r in out), mapa)

    def procesar(self, tramo, glosario: str = "") -> Resultado | None:
        idioma, final = tramo.idioma, tramo.final
        orig = corregir(self.transcribir(tramo.audio(), idioma, glosario), glosario)
        if not orig:
            return None
        # Los parciales van sin traducción: se reescriben en segundos, y la
        # traducción de media oración confunde más de lo que ayuda.
        return completar(idioma, orig, self.traducir(orig, idioma, glosario) if final else "")


def wav(pcm: bytes) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return b.getvalue()


ESQUEMA = {"type": "OBJECT", "properties": {"transcripcion": {"type": "STRING"}, "traduccion": {"type": "STRING"}},
           "required": ["transcripcion", "traduccion"]}


def instruccion(idioma: str) -> str:
    # El MISMO texto que midió experimental (gemini_bench.py): 63 tokens de
    # prompt → 0,057 USD por sala-hora con Flash-Lite. Alargarlo cambia ese
    # número; un glosario de ~500 tokens lo lleva a ~0,10.
    return (f"Audio en vivo de una charla técnica en {NOMBRE[idioma]}. Devuelve JSON con 'transcripcion' "
            f"(literal, en {NOMBRE[idioma]}) y 'traduccion' (al {NOMBRE[DESTINO[idioma]]}, para subtítulos). "
            "Si no hay voz, ambos vacíos.")


def interpretar_gemini(respuesta: dict, idioma: str) -> Resultado | None:
    texto = respuesta["candidates"][0]["content"]["parts"][0]["text"]
    d = json.loads(texto)
    orig = (d.get("transcripcion") or "").strip()
    if not orig:
        return None
    return completar(idioma, orig, (d.get("traduccion") or "").strip())


class Gemini:
    """Vertex AI por REST, autenticado con la cuenta de servicio montada
    (GOOGLE_APPLICATION_CREDENTIALS). La credencial no se lee, no se copia
    y no se loguea: google.auth la usa en su lugar."""

    def __init__(self, proyecto: str, region: str, modelo: str, sesion=None):
        import google.auth
        import google.auth.transport.requests
        import requests

        self.url = (f"https://{region}-aiplatform.googleapis.com/v1/projects/{proyecto}/locations/{region}/"
                    f"publishers/google/models/{modelo}:generateContent")
        self.modelo = modelo
        self.cred, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self._pedido = google.auth.transport.requests.Request()
        self._cerrojo = threading.Lock()
        self.http = sesion or requests.Session()

    def _token(self) -> str:
        with self._cerrojo:
            if not self.cred.valid:
                self.cred.refresh(self._pedido)
            return self.cred.token

    def procesar(self, tramo, glosario: str = "") -> Resultado | None:
        # El glosario NO va al prompt de Gemini: cada término son tokens en
        # cada llamada (experimental: ~500 tokens de glosario llevan
        # Flash-Lite de 0,057 a ~0,10 USD por sala-hora). Sí va a Whisper.
        pcm, idioma, final = tramo.pcm, tramo.idioma, tramo.final
        cuerpo = {
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav(pcm)).decode()}},
                {"text": instruccion(idioma) + (f" Nombres propios y términos, escribilos así: {glosario}." if glosario else "")},
            ]}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json",
                                 "responseSchema": ESQUEMA, "thinkingConfig": {"thinkingBudget": 0}},
        }
        # 8 s y no 20: un tramo que tarda más ya llega tarde; mejor que lo
        # atienda el siguiente backend de la cascada.
        r = self.http.post(self.url, json=cuerpo, timeout=8,
                           headers={"Authorization": f"Bearer {self._token()}"})
        if r.status_code != 200:
            # El cuerpo de error de Vertex no trae la credencial; igual se
            # recorta: un log no es lugar para respuestas enteras.
            raise RuntimeError(f"Vertex {r.status_code}: {r.text[:200]}")
        res = interpretar_gemini(r.json(), idioma)
        if res and glosario:
            res.orig = corregir(res.orig, glosario)
            setattr(res, idioma, res.orig)
        if res and not final:
            setattr(res, DESTINO[idioma], "")
        return res


class Falso:
    """Para los tests y para correr la pila sin modelos: devuelve el texto
    que se le pida, o lanza lo que se le pida."""

    def __init__(self, texto="hola mundo", error: BaseException | None = None):
        self.texto, self.error, self.llamadas = texto, error, 0

    def procesar(self, tramo, glosario=""):
        self.llamadas += 1
        if self.error:
            raise self.error
        i = tramo.idioma
        return completar(i, self.texto, f"[{DESTINO[i]}] {self.texto}" if tramo.final else "")
