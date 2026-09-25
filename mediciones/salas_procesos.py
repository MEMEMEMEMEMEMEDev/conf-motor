#!/usr/bin/env python3
"""salas_procesos.py — N salas, CADA UNA EN SU PROCESO (como un pod por sala).

Lanza N procesos de bench.py (1 sala cada uno, --hilos hilos), fijados a los
CPUs de --cpus, con un instante de arranque común, y junta las latencias de
todos: p50/p95/máx, deriva, RTF medio, y el uso de CPU del conjunto medido con
psutil. Sin GIL compartido: mide la CPU, no el intérprete.
Uso: salas_procesos.py --niveles 2,4,6 --hilos 2 --tramo 10 --cpus 8-15,24-31
"""
import argparse, json, statistics, subprocess, sys, time, psutil

ap = argparse.ArgumentParser()
ap.add_argument("--niveles", default="2,4")
ap.add_argument("--hilos", type=int, default=2)
ap.add_argument("--tramo", type=float, default=10.0)
ap.add_argument("--duracion", type=float, default=80.0)
ap.add_argument("--asr", default="small")
ap.add_argument("--cpus", default="8-15,24-31")
ap.add_argument("--p95-max", type=float, default=3.0)
ap.add_argument("--deriva-max", type=float, default=1.0)
ap.add_argument("--parar", action="store_true")
a = ap.parse_args()

def pct(xs, p):
    xs = sorted(xs); return xs[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))]

for n in [int(x) for x in a.niveles.split(",")]:
    cero = time.time() + 25 + 2 * n       # tiempo para cargar el modelo en cada proceso
    ps = [subprocess.Popen(["taskset", "-c", a.cpus, sys.executable, "bench.py", "--device", "cpu", "--asr", a.asr,
                            "--hilos", str(a.hilos), "--niveles", "1", "--tramo", str(a.tramo), "--duracion", str(a.duracion),
                            "--cero", str(cero), "--indice", str(i), "--crudo"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) for i in range(n)]
    while time.time() < cero:
        time.sleep(0.2)
    psutil.cpu_percent(percpu=True)
    muestras = []
    while any(p.poll() is None for p in ps):
        time.sleep(1.0)
        muestras.append(psutil.cpu_percent(percpu=True))
    filas = [json.loads(p.stdout.read().strip().splitlines()[-1]) for p in ps]
    lats = [x for f in filas for rs in f["lats"] for x in rs]
    q = max(1, len(filas[0]["lats"][0]) // 4)
    deriva = statistics.mean(x for f in filas for rs in f["lats"] for x in rs[-q:]) - statistics.mean(x for f in filas for rs in f["lats"] for x in rs[:q])
    comp = sum(f["comp_s"] for f in filas); aud = sum(f["audio_s"] for f in filas)
    cpus = []
    for part in a.cpus.split(","):
        lo, _, hi = part.partition("-"); cpus += list(range(int(lo), int(hi or lo) + 1))
    uso = [sum(m[c] for c in cpus) / 100 for m in muestras[2:-2]] or [0]
    p95 = pct(lats, 95)
    ok = p95 <= a.p95_max and deriva <= a.deriva_max
    fila = {"modo": "procesos", "asr": a.asr, "compute": "int8", "salas": n, "hilos_por_sala": a.hilos, "tramo_s": a.tramo,
            "cpus": a.cpus, "lat_p50_s": round(pct(lats, 50), 3), "lat_p95_s": round(p95, 3), "lat_max_s": round(max(lats), 3),
            "deriva_s": round(deriva, 3), "rtf_por_sala": round(comp / aud, 4),
            "hilos_ocupados_media": round(statistics.mean(uso), 2), "hilos_ocupados_por_sala": round(statistics.mean(uso) / n, 2),
            "veredicto": "AGUANTA" if ok else "NO AGUANTA"}
    print(json.dumps(fila), flush=True)
    print(f"[procesos {a.asr} h={a.hilos} tramo={a.tramo}] salas={n}: p50 {fila['lat_p50_s']} p95 {p95:.2f} deriva {deriva:+.2f} "
          f"RTF {fila['rtf_por_sala']} hilos ocupados {fila['hilos_ocupados_media']} ({fila['hilos_ocupados_por_sala']}/sala) -> {fila['veredicto']}", file=sys.stderr, flush=True)
    if a.parar and not ok:
        break
