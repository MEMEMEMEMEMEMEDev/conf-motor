#!/usr/bin/env python3
"""bench.py — capacidad de subtítulos EN VIVO: N salas en paralelo, a ritmo real.

Cada sala es un hilo que recibe su audio en tramos fijos de --tramo segundos,
cada uno disponible en el instante en que habría terminado de hablarse (ritmo
real). Apenas está disponible, la sala lo transcribe (faster-whisper) y
traduce el texto (opus-mt sobre CTranslate2). Latencia de un tramo = desde
que está disponible hasta que su subtítulo traducido está listo. Si la
máquina no da abasto, la cola crece y la latencia también: eso es lo que
distingue AGUANTA de NO AGUANTA.

Las salas alternan dos fragmentos reales de Nerdearla 2025 (repo Subtitula,
Apache 2.0): inglés → español y español → inglés, en bucle.

Un modelo de ASR y uno de traducción por dirección, compartidos, con
--salas trabajadores concurrentes cada uno (num_workers / inter_threads de
CTranslate2: una réplica por trabajador). En CPU cada trabajador usa
--hilos hilos.

Veredicto por nivel: AGUANTA si p95 de latencia <= --p95-max segundos Y el
atraso no crece (la latencia media del último cuarto no supera en más de
--deriva-max s a la del primer cuarto). RTF por sala = tiempo de cómputo /
audio procesado.

Salida: una línea JSON por nivel a stdout; progreso a stderr.
"""
import argparse, json, os, statistics, subprocess, sys, threading, time

import numpy as np
import psutil

AUDIOS = {
    "en": "fuentes/subtitula/samples/thor-schaeff-multilingual-agents-en.mp3",
    "es": "fuentes/subtitula/samples/midudev-programming-is-dead-es.mp3",
}
DESTINO = {"en": "es", "es": "en"}
SR = 16000


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def cargar(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


class Muestreo(threading.Thread):
    """CPU del proceso y del sistema, y VRAM/uso de GPU, cada 0,5 s."""
    def __init__(self, gpu):
        super().__init__(daemon=True)
        self.fin = threading.Event(); self.cpu = []; self.cpu_sis = []; self.vram = []; self.gpu_util = []
        self.proc = psutil.Process(); self.nv = None
        if gpu:
            import pynvml
            pynvml.nvmlInit(); self.nv = pynvml; self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def run(self):
        self.proc.cpu_percent(None); psutil.cpu_percent(None)
        while not self.fin.wait(0.5):
            self.cpu.append(self.proc.cpu_percent(None)); self.cpu_sis.append(psutil.cpu_percent(None))
            if self.nv:
                self.vram.append(self.nv.nvmlDeviceGetMemoryInfo(self.h).used / 2**20)
                self.gpu_util.append(self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "cpu"], required=True)
    ap.add_argument("--asr", default="large-v3-turbo")
    ap.add_argument("--compute", default=None, help="compute_type del ASR (def: float16 en cuda, int8 en cpu)")
    ap.add_argument("--mt-compute", default=None, help="compute_type de la traducción (def: int8_float16 en cuda, int8 en cpu)")
    ap.add_argument("--niveles", default="1,2,4", help="salas por nivel, separadas por coma")
    ap.add_argument("--hilos", type=int, default=2, help="hilos por trabajador (CPU)")
    ap.add_argument("--tramo", type=float, default=4.0)
    ap.add_argument("--duracion", type=float, default=90.0, help="segundos de audio en vivo por sala y nivel")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--p95-max", type=float, default=2.0)
    ap.add_argument("--deriva-max", type=float, default=1.0)
    ap.add_argument("--parar", action="store_true", help="parar en el primer NO AGUANTA")
    ap.add_argument("--cero", type=float, default=None, help="instante de arranque común (epoch), para salas en procesos separados")
    ap.add_argument("--indice", type=int, default=0, help="índice de la primera sala de este proceso")
    ap.add_argument("--crudo", action="store_true", help="incluir las latencias de cada tramo en la salida")
    a = ap.parse_args()

    import ctranslate2, sentencepiece as spm
    from faster_whisper import WhisperModel
    comp = a.compute or ("float16" if a.device == "cuda" else "int8")
    mtcomp = a.mt_compute or ("int8_float16" if a.device == "cuda" else "int8")
    audio = {k: cargar(v) for k, v in AUDIOS.items()}
    muestras_tramo = int(a.tramo * SR)
    n_tramos = int(a.duracion // a.tramo)
    base = os.path.dirname(os.path.abspath(__file__))

    for n in [int(x) for x in a.niveles.split(",")]:
        t_carga = time.time()
        asr = WhisperModel(a.asr, device=a.device, compute_type=comp, cpu_threads=a.hilos, num_workers=n)
        mt, sp = {}, {}
        for src, dst in DESTINO.items():
            d = os.path.join(base, f"modelos/opus-mt-{src}-{dst}-ct2")
            mt[src] = ctranslate2.Translator(d, device=a.device, compute_type=mtcomp, inter_threads=n, intra_threads=a.hilos)
            sp[src] = (spm.SentencePieceProcessor(model_file=f"{d}/source.spm"), spm.SentencePieceProcessor(model_file=f"{d}/target.spm"))
        # calentamiento: un tramo por idioma
        for lang in audio:
            list(asr.transcribe(audio[lang][:muestras_tramo], language=lang, beam_size=a.beam, condition_on_previous_text=False)[0])
        t_carga = time.time() - t_carga
        mu = Muestreo(a.device == "cuda"); mu.start()
        res = [[] for _ in range(n)]
        cero = a.cero if a.cero else time.time() + 1.0

        def sala(j):
            i = j + a.indice
            lang = "en" if i % 2 == 0 else "es"
            pista = audio[lang]
            desfase = (i * 0.37) % a.tramo            # que las salas no vayan en fila
            inicio_audio = (i * 17 * SR) % len(pista)  # y que no digan lo mismo a la vez
            s_src, s_tgt = sp[lang]
            for k in range(n_tramos):
                disponible = cero + desfase + (k + 1) * a.tramo
                espera = disponible - time.time()
                if espera > 0:
                    time.sleep(espera)
                ini = (inicio_audio + k * muestras_tramo) % len(pista)
                trozo = pista[ini:ini + muestras_tramo]
                if len(trozo) < muestras_tramo:
                    trozo = np.concatenate([trozo, pista[:muestras_tramo - len(trozo)]])
                t0 = time.time()
                segs, _ = asr.transcribe(trozo, language=lang, beam_size=a.beam, condition_on_previous_text=False,
                                         vad_filter=True, without_timestamps=True)
                texto = " ".join(s.text.strip() for s in segs).strip()
                t1 = time.time()
                if texto:
                    out = mt[lang].translate_batch([s_src.encode(texto, out_type=str) + ["</s>"]], beam_size=2)
                    s_tgt.decode(out[0].hypotheses[0])
                t2 = time.time()
                res[j].append({"lat": t2 - disponible, "asr": t1 - t0, "mt": t2 - t1, "vacio": not texto})

        hs = [threading.Thread(target=sala, args=(i,)) for i in range(n)]
        t_ini = time.time()
        for h in hs:
            h.start()
        for h in hs:
            h.join()
        muro = time.time() - t_ini
        mu.fin.set(); mu.join()
        todo = [r for rs in res for r in rs]
        lats = [r["lat"] for r in todo]
        comp_s = sum(r["asr"] + r["mt"] for r in todo)
        audio_s = len(todo) * a.tramo
        q = max(1, n_tramos // 4)
        primero = [r["lat"] for rs in res for r in rs[:q]]
        ultimo = [r["lat"] for rs in res for r in rs[-q:]]
        deriva = statistics.mean(ultimo) - statistics.mean(primero)
        p95 = pct(lats, 95)
        aguanta = p95 <= a.p95_max and deriva <= a.deriva_max
        fila = {
            "device": a.device, "asr": a.asr, "compute": comp, "mt": "opus-mt", "mt_compute": mtcomp,
            "salas": n, "hilos_por_trabajador": a.hilos if a.device == "cpu" else None,
            "tramo_s": a.tramo, "beam": a.beam, "tramos": len(todo),
            "lat_p50_s": round(pct(lats, 50), 3), "lat_p95_s": round(p95, 3), "lat_max_s": round(max(lats), 3),
            "deriva_s": round(deriva, 3),
            "asr_p50_s": round(pct([r["asr"] for r in todo], 50), 3), "mt_p50_s": round(pct([r["mt"] for r in todo], 50), 3),
            "rtf_por_sala": round(comp_s / audio_s, 4),
            "cpu_proceso_pct_media": round(statistics.mean(mu.cpu), 1) if mu.cpu else None,
            "cpu_sistema_pct_media": round(statistics.mean(mu.cpu_sis), 1) if mu.cpu_sis else None,
            "vram_pico_mib": round(max(mu.vram)) if mu.vram else None,
            "gpu_util_media_pct": round(statistics.mean(mu.gpu_util), 1) if mu.gpu_util else None,
            "carga_s": round(t_carga, 1), "muro_s": round(muro, 1),
            "veredicto": "AGUANTA" if aguanta else "NO AGUANTA",
        }
        if a.crudo:
            fila["lats"] = [[round(r["lat"], 3) for r in rs] for rs in res]
            fila["comp_s"] = round(comp_s, 3); fila["audio_s"] = audio_s
        print(json.dumps(fila, ensure_ascii=False), flush=True)
        log(f"[{a.device} {a.asr} {comp}] salas={n}: p50 {fila['lat_p50_s']}s p95 {p95:.2f}s max {fila['lat_max_s']}s "
            f"deriva {deriva:+.2f}s RTF/sala {fila['rtf_por_sala']} VRAM {fila['vram_pico_mib']} CPU {fila['cpu_proceso_pct_media']}% -> {fila['veredicto']}")
        del asr, mt
        if a.parar and not aguanta:
            break


if __name__ == "__main__":
    main()
