#!/usr/bin/env python3
"""calidad.py — WER de la transcripción, recall del glosario y chrF de la traducción.

Contra la referencia de referencias/ (HOY: borrador de Gemini 2.5 Flash, NO
humano; si alguien la corrige a mano, el mismo guion da el número contra
humano). Dos modos de transcribir, porque en vivo no se ve el archivo entero:
  entero   el fragmento completo de una vez (techo de calidad del modelo);
  vivo     tramos de --tramo s, cada uno por separado (lo que hace bench.py).
Con y sin glosario (faster-whisper `hotwords`). El glosario de cada charla
está abajo, escrito a mano con los términos técnicos y nombres propios que
aparecen en la referencia: sube el recall EN ESTE audio por construcción;
lo que mide es cuánto ayuda un glosario preparado antes, como el de CART.

La traducción (opus-mt sobre CTranslate2) se aplica a la transcripción del
modelo (cascada ASR → traducción) y se compara con la traducción de
referencia: chrF (sacrebleu).
Salida: una línea JSON por combinación a stdout.
"""
import argparse, json, re, subprocess, sys, time, unicodedata

import numpy as np

AUDIOS = {
    "en": "fuentes/subtitula/samples/thor-schaeff-multilingual-agents-en.mp3",
    "es": "fuentes/subtitula/samples/midudev-programming-is-dead-es.mp3",
}
DST = {"en": "es", "es": "en"}
# Términos técnicos y nombres propios QUE APARECEN en la referencia (leída el
# 25-09): así el recall mide términos dichos de verdad. Un glosario de evento
# real se prepara antes con los títulos y los nombres del programa.
GLOSARIO = {
    "en": ["ElevenLabs", "live stream", "HDMI", "stream", "audio", "support agent", "omni model", "cutlasses", "mutiny", "research"],
    "es": ["Midudev", "Midu.dev", "Miguel Ángel Durán", "programación", "Instagram", "Javier Tebas", "la Liga", "piratería", "influencer", "part-time"],
}
SR = 16000


def cargar(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def normalizar(t):
    t = unicodedata.normalize("NFKC", t.lower())
    t = re.sub(r"[^\w\s']", " ", t)
    # muletillas: la referencia de Gemini las escribe (se le pidió literal) y
    # Whisper las omite; en subtítulos se omiten a propósito. Fuera de los dos.
    t = re.sub(r"\b(um+|uh+|eh+|ah+|mm+|hmm+)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def recall(glos, ref, hyp):
    r, h = normalizar(ref), normalizar(hyp)
    tot = hit = 0
    for g in glos:
        g = normalizar(g)
        n = len(re.findall(rf"\b{re.escape(g)}\b", r))
        if n:
            tot += n
            hit += min(n, len(re.findall(rf"\b{re.escape(g)}\b", h)))
    return (hit / tot) if tot else None, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hilos", type=int, default=8)
    ap.add_argument("--modelos", default="small:int8,large-v3-turbo:int8")
    ap.add_argument("--tramo", type=float, default=4.0)
    a = ap.parse_args()
    import ctranslate2, jiwer, sacrebleu, sentencepiece as spm
    from faster_whisper import WhisperModel
    audio = {k: cargar(v) for k, v in AUDIOS.items()}
    ref = {k: open(f"referencias/{k}.txt").read() for k in AUDIOS}
    ref_tr = {k: open(f"referencias/{k}.{DST[k]}.txt").read() for k in AUDIOS}
    tr, sp = {}, {}
    for s, d in DST.items():
        p = f"modelos/opus-mt-{s}-{d}-ct2"
        tr[s] = ctranslate2.Translator(p, device=a.device, compute_type="int8", intra_threads=a.hilos)
        sp[s] = (spm.SentencePieceProcessor(model_file=f"{p}/source.spm"), spm.SentencePieceProcessor(model_file=f"{p}/target.spm"))

    for spec in a.modelos.split(","):
        nombre, comp = spec.split(":")
        m = WhisperModel(nombre, device=a.device, compute_type=comp, cpu_threads=a.hilos)
        for lang in AUDIOS:
            for modo in ("entero", "vivo"):
                for glos in (False, True):
                    kw = dict(language=lang, beam_size=5 if modo == "entero" else 1, condition_on_previous_text=(modo == "entero"),
                              vad_filter=True)
                    if glos:
                        kw["hotwords"] = ", ".join(GLOSARIO[lang])
                    t0 = time.time()
                    if modo == "entero":
                        segs = [s.text.strip() for s in m.transcribe(audio[lang], **kw)[0]]
                    else:
                        n = int(a.tramo * SR); segs = []
                        for i in range(0, len(audio[lang]), n):
                            segs += [s.text.strip() for s in m.transcribe(audio[lang][i:i + n], without_timestamps=True, **kw)[0]]
                    dt = time.time() - t0
                    hyp = " ".join(x for x in segs if x)
                    wer = jiwer.wer(normalizar(ref[lang]), normalizar(hyp))
                    rc, nterm = recall(GLOSARIO[lang], ref[lang], hyp)
                    s_src, s_tgt = sp[lang]
                    frases = [x for x in segs if x]
                    out = tr[lang].translate_batch([s_src.encode(x, out_type=str) + ["</s>"] for x in frases], beam_size=2) if frases else []
                    trad = " ".join(s_tgt.decode(o.hypotheses[0]) for o in out)
                    chrf = sacrebleu.corpus_chrf([trad], [[ref_tr[lang]]]).score
                    fila = {"asr": nombre, "compute": comp, "device": a.device, "lang": lang, "modo": modo, "tramo_s": a.tramo if modo == "vivo" else None, "glosario": glos,
                            "wer": round(wer, 4), "precision_palabras": round(1 - wer, 4), "recall_glosario": None if rc is None else round(rc, 3),
                            "terminos_en_ref": nterm, "chrf_traduccion": round(chrf, 2), "segundos": round(dt, 1),
                            "referencia": "borrador gemini-2.5-flash (NO humano)"}
                    print(json.dumps(fila, ensure_ascii=False), flush=True)
                    print(f"{nombre} {comp} {lang} {modo} glos={glos}: WER {wer:.3f} recall {rc} chrF {chrf:.1f} ({dt:.0f}s)", file=sys.stderr, flush=True)
        del m


if __name__ == "__main__":
    main()
