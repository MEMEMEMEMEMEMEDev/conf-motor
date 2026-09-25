#!/usr/bin/env python3
"""gemini_bench.py — Gemini en Vertex AI: referencia borrador, costo real por tokens y latencia.

Modos:
  referencia  cada fragmento ENTERO a gemini-2.5-flash: transcripción literal
              + traducción (en→es, es→en). Deja referencias/<lang>.txt y
              referencias/<lang>.<dst>.txt: un BORRADOR de modelo, no humano.
  tramos      cada fragmento en tramos de --tramo s, una llamada por tramo
              (el esquema de Subtitula): transcripción + traducción en JSON.
              Mide los tokens que devuelve la API (usageMetadata) y la latencia.
              El costo por sala-hora sale de esos tokens × el precio de hoy.

Credencial: `gcloud auth print-access-token` leído DENTRO del proceso; nunca
en argv, logs ni ficheros. Proyecto por --proyecto (def. aegis-505709).
Salida: JSON a stdout; progreso a stderr.
"""
import argparse, base64, json, subprocess, sys, time, urllib.request, urllib.error, os

AUDIOS = {
    "en": "fuentes/subtitula/samples/thor-schaeff-multilingual-agents-en.mp3",
    "es": "fuentes/subtitula/samples/midudev-programming-is-dead-es.mp3",
}
DST = {"en": "es", "es": "en"}
NOMBRE = {"en": "inglés", "es": "español"}
# precios USD por 1M tokens, ai.google.dev/gemini-api/docs/pricing (actualizada 2026-09-24),
# archivada en crudos/gemini-pricing-*.html; Vertex publica los mismos para estos modelos
PRECIO = {
    "gemini-2.5-flash": {"audio_in": 1.00, "texto_in": 0.30, "salida": 2.50},
    "gemini-2.5-flash-lite": {"audio_in": 0.30, "texto_in": 0.10, "salida": 0.40},
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def token():
    return subprocess.run([os.path.expanduser("~/google-cloud-sdk/bin/gcloud"), "auth", "print-access-token"], check=True, capture_output=True, text=True).stdout.strip()


def llamar(tok, proyecto, modelo, partes, esquema=None, temperatura=0.0):
    url = (f"https://us-central1-aiplatform.googleapis.com/v1/projects/{proyecto}/locations/us-central1/"
           f"publishers/google/models/{modelo}:generateContent")
    cuerpo = {"contents": [{"role": "user", "parts": partes}],
              "generationConfig": {"temperature": temperatura}}
    if esquema:
        cuerpo["generationConfig"]["responseMimeType"] = "application/json"
        cuerpo["generationConfig"]["responseSchema"] = esquema
    if modelo.startswith("gemini-2.5-flash"):
        cuerpo["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
    req = urllib.request.Request(url, data=json.dumps(cuerpo).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} de Vertex: {e.read().decode()[:400]}")
    dt = time.time() - t0
    texto = out["candidates"][0]["content"]["parts"][0]["text"]
    return texto, out.get("usageMetadata", {}), dt


def audio_parte(datos, mime="audio/mp3"):
    return {"inlineData": {"mimeType": mime, "data": base64.b64encode(datos).decode()}}


def recortar(path, ini, dur):
    return subprocess.run(["ffmpeg", "-v", "error", "-ss", str(ini), "-t", str(dur), "-i", path,
                           "-ac", "1", "-ar", "16000", "-f", "mp3", "-"], check=True, capture_output=True).stdout


def costo(modelo, uso):
    p = PRECIO[modelo]
    audio_t = sum(d.get("tokenCount", 0) for d in uso.get("promptTokensDetails", []) if d.get("modality") == "AUDIO")
    prompt_t = uso.get("promptTokenCount", 0)
    salida_t = uso.get("candidatesTokenCount", 0) + uso.get("thoughtsTokenCount", 0)
    return (audio_t * p["audio_in"] + (prompt_t - audio_t) * p["texto_in"] + salida_t * p["salida"]) / 1e6, audio_t, prompt_t - audio_t, salida_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["referencia", "tramos"])
    ap.add_argument("--proyecto", default="aegis-505709")
    ap.add_argument("--modelo", default="gemini-2.5-flash-lite")
    ap.add_argument("--tramo", type=float, default=4.0)
    ap.add_argument("--max-tramos", type=int, default=32, help="por fragmento")
    a = ap.parse_args()
    tok = token()

    if a.modo == "referencia":
        os.makedirs("referencias", exist_ok=True)
        for lang, path in AUDIOS.items():
            datos = open(path, "rb").read()
            txt, uso, dt = llamar(tok, a.proyecto, "gemini-2.5-flash", [
                audio_parte(datos),
                {"text": f"Transcribe este audio en {NOMBRE[lang]} de forma LITERAL, palabra por palabra, "
                         "incluidas repeticiones y muletillas. Sin marcas de tiempo, sin nombres de hablante, "
                         "sin comentarios: sólo el texto dicho."}])
            open(f"referencias/{lang}.txt", "w").write(txt.strip() + "\n")
            tr, uso2, dt2 = llamar(tok, a.proyecto, "gemini-2.5-flash", [
                {"text": f"Traduce al {NOMBRE[DST[lang]]} este texto, fiel y natural, como subtítulos. "
                         f"Devuelve sólo la traducción.\n\n{txt}"}])
            open(f"referencias/{lang}.{DST[lang]}.txt", "w").write(tr.strip() + "\n")
            c1 = costo("gemini-2.5-flash", uso)[0]; c2 = costo("gemini-2.5-flash", uso2)[0]
            print(json.dumps({"modo": "referencia", "lang": lang, "palabras": len(txt.split()),
                              "latencia_transcripcion_s": round(dt, 2), "costo_usd": round(c1 + c2, 5), "uso": uso}))
            log(f"referencia {lang}: {len(txt.split())} palabras, {dt:.1f}s, US${c1 + c2:.5f}")
        return

    esquema = {"type": "OBJECT", "properties": {"transcripcion": {"type": "STRING"}, "traduccion": {"type": "STRING"}},
               "required": ["transcripcion", "traduccion"]}
    filas = []
    for lang, path in AUDIOS.items():
        instr = (f"Audio en vivo de una charla técnica en {NOMBRE[lang]}. Devuelve JSON con 'transcripcion' "
                 f"(literal, en {NOMBRE[lang]}) y 'traduccion' (al {NOMBRE[DST[lang]]}, para subtítulos). "
                 "Si no hay voz, ambos vacíos.")
        for k in range(a.max_tramos):
            ini = k * a.tramo
            if ini >= 130:
                break
            trozo = recortar(path, ini, a.tramo)
            _, uso, dt = llamar(tok, a.proyecto, a.modelo, [audio_parte(trozo), {"text": instr}], esquema=esquema)
            c, at, tt, st = costo(a.modelo, uso)
            filas.append({"lang": lang, "k": k, "lat_s": dt, "usd": c, "audio_tok": at, "texto_tok": tt, "salida_tok": st})
        log(f"{a.modelo} {lang}: {len([f for f in filas if f['lang'] == lang])} tramos")
    n = len(filas)
    lat = sorted(f["lat_s"] for f in filas)
    usd = sum(f["usd"] for f in filas)
    llamadas_hora = 3600 / a.tramo
    res = {"modo": "tramos", "modelo": a.modelo, "tramo_s": a.tramo, "llamadas": n,
           "lat_p50_s": round(lat[len(lat) // 2], 3), "lat_p95_s": round(lat[int(0.95 * (n - 1))], 3), "lat_max_s": round(lat[-1], 3),
           "tokens_por_llamada": {k: round(sum(f[k] for f in filas) / n, 1) for k in ("audio_tok", "texto_tok", "salida_tok")},
           "usd_por_llamada": round(usd / n, 7), "usd_medido_total": round(usd, 5),
           "usd_por_sala_hora": round(usd / n * llamadas_hora, 4),
           "llamadas_por_sala_hora": llamadas_hora}
    print(json.dumps(res))
    log(json.dumps(res))


if __name__ == "__main__":
    main()
