"""El núcleo del motor: decide qué hacer con cada tramo.

Sin modelos ni redis de verdad adentro: recibe el bus y una fábrica de
backends, así los tests lo ejercitan entero con dobles.

Las tres reglas que importan, y por qué:

1. LLEGAR TARDE ES PEOR QUE PERDER UNA FRASE. Un tramo que ya tiene más de
   ATRASO_MAX_S de atraso se salta (y se cuenta, y el panel lo muestra). Si
   no, un tropiezo de la GPU deja la sala subtitulando lo de hace un minuto
   para siempre. Es la regla de Subtitula, y la razón por la que después de
   un Recreate del motor la cola vieja no se procesa.
2. LOS FINALES ANTES QUE LOS PARCIALES. Un parcial es una mejora de
   latencia; un final es el subtítulo. Con el motor ocupado, los parciales
   se descartan en silencio.
3. SI LA GPU SE CAE, LAS SALAS SIGUEN. Un error de CUDA marca la GPU como
   caída y todo lo que pedía `gpu` pasa a `gemini` (o a `cpu` si no hay
   credencial). No vuelve sola: la GPU que se cayó del bus no se recupera
   sin reiniciar el proceso, y alternar entre las dos confunde más que
   ayuda. Reiniciar el motor es la decisión del operador.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .backends import es_error_de_gpu, humanizar
from .bus import Bus, Tramo

log = logging.getLogger("conf-motor")


class Nucleo:
    def __init__(self, bus: Bus, fabrica, max_salas: int, atraso_max_s: float = 25.0,
                 respaldo: str = "gemini", reloj=time.time):
        self.bus = bus
        self.fabrica = fabrica  # nombre → backend (o None si no está disponible)
        self.max_salas = max_salas
        self.atraso_max_ms = int(atraso_max_s * 1000)
        self.respaldo = respaldo
        self.reloj = reloj

        self.gpu = "sin"  # sin | ok | caida
        self.en_vuelo = 0
        self.saltados = 0
        self.errores = 0
        self.procesados = 0
        self.lat = deque(maxlen=200)
        self._cerrojo = threading.Lock()
        self._ultimo_final = {}  # sala → seq del último final visto
        self._pool = ThreadPoolExecutor(max_workers=max_salas, thread_name_prefix="sala")

    # ---- elección de backend -------------------------------------------------

    def backend_para(self, pedido: str):
        """Devuelve (nombre, backend) que va a atender un tramo que pidió
        `pedido`. Con la GPU caída, lo que pedía gpu va al respaldo."""
        nombre = pedido if pedido in ("gpu", "cpu", "gemini") else "gpu"
        # "sin" = este motor no arrancó con GPU (BACKEND=cpu): no se la
        # crea al vuelo con el primer tramo que la pida, porque eso sería
        # bajar large-v3-turbo y tomar la placa sin que nadie lo decida.
        # (Pasó en la primera corrida local: la sala demo pedía gpu.)
        if nombre == "gpu" and self.gpu != "ok":
            nombre = self.respaldo
        b = self.fabrica(nombre)
        if b is None and nombre != "cpu":
            # Sin credencial de Gemini o sin GPU en esta máquina: el último
            # recurso siempre existe, porque corre en el mismo proceso.
            nombre, b = "cpu", self.fabrica("cpu")
        return nombre, b

    # ---- un tramo ---------------------------------------------------------------

    def _viejo(self, t: Tramo) -> bool:
        return int(self.reloj() * 1000) - t.t_fin > self.atraso_max_ms

    def procesar(self, t: Tramo):
        try:
            self._procesar(t)
        finally:
            with self._cerrojo:
                self.en_vuelo -= 1
            try:
                self.bus.confirmar(t)
            except Exception as e:  # noqa: BLE001 — el ack perdido no tumba al motor
                log.warning("ack perdido %s: %s", t.id, e)

    def _publicar(self, t: Tramo, tipo: str, **campos):
        self.bus.publicar(t.sala, tipo=tipo, seq=t.seq, t_ini=t.t_ini, t_fin=t.t_fin, **campos)

    def _procesar(self, t: Tramo):
        if self._viejo(t):
            if t.final:
                with self._cerrojo:
                    self.saltados += 1
                self._publicar(t, "saltado", detalle=f"{(self.reloj() * 1000 - t.t_fin) / 1000:.1f} s de atraso")
            return
        if not t.final and self._ultimo_final.get(t.sala, 0) >= t.seq:
            return  # el final de ese tramo ya salió: el parcial no aporta

        # EN CASCADA: el que pidió la sala, y si falla, los demás en orden
        # (gpu → gemini → cpu, sin repetir). Un respaldo que atiende el tramo
        # NO es un error: el subtítulo sale igual, con el backend que lo
        # atendió, y el panel lo muestra como "pide X → atiende Y". Sólo si
        # fallan TODOS se publica un error, dicho en castellano.
        intentos = []
        res = None
        nombre = None
        for pedido in self.cascada(t.backend):
            nombre, b = self.backend_para(pedido)
            if b is None or nombre in [n for n, _ in intentos]:
                continue
            try:
                res = b.procesar(t, t.glosario)
                break
            except Exception as e:  # noqa: BLE001
                if nombre == "gpu" and es_error_de_gpu(e):
                    self._caer_gpu(e)
                log.warning("tramo %s de %s: %s falló (%s); pruebo el siguiente", t.seq, t.sala, nombre, e)
                intentos.append((nombre, e))
        else:
            if intentos:
                return self._error(t, intentos)
            return self._error(t, [("motor", RuntimeError("no hay ningún backend disponible"))])

        if t.final:
            with self._cerrojo:
                self._ultimo_final[t.sala] = max(self._ultimo_final.get(t.sala, 0), t.seq)
        if res is None:
            return  # sin voz reconocible: no se publica una línea vacía
        lat = int(self.reloj() * 1000) - t.t_fin
        if t.final:
            with self._cerrojo:
                self.procesados += 1
                self.lat.append(lat)
        self._publicar(t, "final" if t.final else "parcial", orig=res.orig, idioma=res.idioma,
                       es=res.es, en=res.en, backend=nombre, lat_ms=lat)

    def _error(self, t: Tramo, intentos: list):
        with self._cerrojo:
            self.errores += 1
        dicho = "; ".join(humanizar(n, e) for n, e in intentos)
        log.warning("tramo %s de %s sin subtítulo: %s", t.seq, t.sala, dicho)
        if t.final:
            self._publicar(t, "error", backend=intentos[-1][0], detalle=f"No salió el subtítulo: {dicho}."[:240])

    def cascada(self, pedido: str) -> list[str]:
        """El orden en que se prueba: lo pedido primero, después el resto."""
        orden = ["gpu", "gemini", "cpu"]
        primero = pedido if pedido in orden else "gpu"
        return [primero] + [x for x in orden if x != primero]

    def _caer_gpu(self, e: BaseException):
        with self._cerrojo:
            if self.gpu == "caida":
                return
            self.gpu = "caida"
        log.error("GPU CAÍDA, las salas pasan a %s: %s", self.respaldo, e)

    # ---- el bucle ---------------------------------------------------------------

    def libres(self) -> int:
        with self._cerrojo:
            return self.max_salas - self.en_vuelo

    def despachar(self, tramos: list[Tramo]):
        """Ordena lo leído (finales primero; de los parciales, sólo el más
        nuevo de cada sala) y lo manda al pool."""
        finales = [t for t in tramos if t.final]
        ultimos_parciales = {}
        for t in tramos:
            if not t.final:
                ultimos_parciales[t.sala] = t
        descartados = [t for t in tramos if not t.final and ultimos_parciales.get(t.sala) is not t]
        for t in descartados:
            self.bus.confirmar(t)
        for t in finales + list(ultimos_parciales.values()):
            if not t.final and self.libres() <= self.max_salas // 4:
                self.bus.confirmar(t)  # motor cargado: los parciales ceden
                continue
            with self._cerrojo:
                self.en_vuelo += 1
            self._pool.submit(self.procesar, t)

    def latir(self):
        with self._cerrojo:
            lat = sorted(self.lat)
            p50 = lat[len(lat) // 2] if lat else 0
            p95 = lat[int(0.95 * (len(lat) - 1))] if lat else 0
            estado = dict(gpu=self.gpu, backend="gpu" if self.gpu == "ok" else self.respaldo,
                          cola=self.en_vuelo, saltados=self.saltados, errores=self.errores,
                          procesados=self.procesados, lat_p50_ms=p50, lat_p95_ms=p95,
                          max_salas=self.max_salas)
        self.bus.latir(**estado)

    def correr(self, parar: threading.Event):
        self.bus.preparar()
        ultimo_latido = 0.0
        while not parar.is_set():
            if self.reloj() - ultimo_latido >= 5:
                try:
                    self.latir()
                except Exception as e:  # noqa: BLE001
                    log.warning("latido: %s", e)
                ultimo_latido = self.reloj()
            libres = self.libres()
            if libres <= 0:
                time.sleep(0.05)
                continue
            try:
                tramos = self.bus.leer(max(libres, 1), bloquear_ms=1000)
            except Exception as e:  # noqa: BLE001
                # El redis de la plataforma no tiene disco: si se reinicia,
                # el stream y el grupo de consumo desaparecen y XREADGROUP
                # contesta NOGROUP para siempre. Se recrea y se sigue.
                if "NOGROUP" in str(e) or "requires the key to exist" in str(e):
                    log.warning("el grupo de consumo no existe (¿redis reiniciado?): lo recreo")
                    try:
                        self.bus.preparar(desde="0")
                    except Exception as e2:  # noqa: BLE001
                        log.warning("no pude recrear el grupo: %s", e2)
                else:
                    log.warning("redis: %s", e)
                time.sleep(1)
                continue
            if tramos:
                self.despachar(tramos)
        self._pool.shutdown(wait=True, cancel_futures=True)
