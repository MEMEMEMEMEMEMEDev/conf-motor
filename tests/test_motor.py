"""La suite del motor. Corre en el build (Containerfile) sin GPU, sin
modelos y sin redis: los backends son dobles y redis es fakeredis, que
implementa los streams y los grupos de consumo de verdad."""
import io
import json
import threading
import unittest
import wave

import fakeredis
import numpy as np

from motor.backends import Falso, completar, es_error_de_gpu, interpretar_gemini, oraciones, wav
from motor.bus import CAMPOS_SUB, CAMPOS_TRAMO, CLAVE_TRAMOS, Bus, Tramo, clave_subs
from motor.nucleo import Nucleo

AHORA = 1_800_000_000.0


def tramo(seq=1, final=True, sala="s1", idioma="en", backend="gpu", atraso_s=1.0, id_="1-0"):
    fin = int((AHORA - atraso_s) * 1000)
    return Tramo(id=id_, sala=sala, idioma=idioma, backend=backend, seq=seq, final=final,
                 t_ini=fin - 9000, t_fin=fin, pcm=b"\x00\x00" * 16000)


class BusDeMentira:
    """Registra lo publicado; mismo contrato que Bus (valida los campos)."""

    def __init__(self):
        self.subs, self.acks = [], []

    def publicar(self, sala, **campos):
        assert set(campos) <= set(CAMPOS_SUB), set(campos) - set(CAMPOS_SUB)
        self.subs.append((sala, campos))

    def confirmar(self, t):
        self.acks.append(t.id)


def nucleo(backends, respaldo="gemini"):
    bus = BusDeMentira()
    n = Nucleo(bus, lambda nombre: backends.get(nombre), max_salas=4, atraso_max_s=25,
               respaldo=respaldo, reloj=lambda: AHORA)
    n.gpu = "ok"
    return n, bus


class TestUnTramo(unittest.TestCase):
    def test_final_publica_original_y_traduccion_en_su_lugar(self):
        n, bus = nucleo({"gpu": Falso("hello world")})
        n.procesar(tramo(idioma="en"))
        (sala, s), = bus.subs
        self.assertEqual((sala, s["tipo"], s["idioma"], s["backend"]), ("s1", "final", "en", "gpu"))
        self.assertEqual(s["en"], "hello world")
        self.assertEqual(s["es"], "[es] hello world")
        self.assertEqual(s["lat_ms"], 1000)
        self.assertEqual(bus.acks, ["1-0"])

        n2, bus2 = nucleo({"gpu": Falso("hola")})
        n2.procesar(tramo(idioma="es"))
        self.assertEqual((bus2.subs[0][1]["es"], bus2.subs[0][1]["en"]), ("hola", "[en] hola"))

    def test_tramo_viejo_se_salta_sin_tocar_el_modelo(self):
        b = Falso()
        n, bus = nucleo({"gpu": b})
        n.procesar(tramo(atraso_s=40))
        self.assertEqual(b.llamadas, 0, "un tramo de hace 40 s no debe pagar GPU")
        self.assertEqual(bus.subs[0][1]["tipo"], "saltado")
        self.assertEqual(n.saltados, 1)
        self.assertEqual(bus.acks, ["1-0"], "saltado también se confirma, o queda pendiente para siempre")

    def test_sin_voz_no_publica_una_linea_vacia(self):
        class Mudo:
            def procesar(self, t, glosario=""):
                return None
        n, bus = nucleo({"gpu": Mudo()})
        n.procesar(tramo())
        self.assertEqual(bus.subs, [])


class TestRespaldoDeLaGPU(unittest.TestCase):
    def test_error_de_cuda_pasa_la_sala_a_gemini_y_se_queda_ahi(self):
        gpu = Falso(error=RuntimeError("CUDA failed with error unspecified launch failure"))
        gem = Falso("from gemini")
        n, bus = nucleo({"gpu": gpu, "gemini": gem})
        n.procesar(tramo(seq=1, id_="1-0"))
        self.assertEqual(n.gpu, "caida")
        self.assertEqual(bus.subs[0][1]["backend"], "gemini")
        self.assertEqual(bus.subs[0][1]["en"], "from gemini", "el tramo que encontró la falla no se pierde")
        n.procesar(tramo(seq=2, id_="2-0"))
        self.assertEqual(gpu.llamadas, 1, "con la GPU caída no se la vuelve a intentar")
        self.assertEqual(gem.llamadas, 2)

    def test_sin_credencial_de_gemini_el_respaldo_es_cpu(self):
        gpu = Falso(error=RuntimeError("cuDNN error: CUDNN_STATUS_EXECUTION_FAILED"))
        cpu = Falso("en cpu")
        n, bus = nucleo({"gpu": gpu, "cpu": cpu, "gemini": None})
        n.procesar(tramo())
        self.assertEqual(bus.subs[0][1]["backend"], "cpu")

    def test_un_error_que_no_es_de_la_placa_no_tumba_la_gpu_y_el_tramo_sale_igual(self):
        n, bus = nucleo({"gpu": Falso(error=ValueError("audio corrupto")), "gemini": Falso("por gemini")})
        n.procesar(tramo())
        self.assertEqual(n.gpu, "ok")
        self.assertEqual((bus.subs[0][1]["tipo"], bus.subs[0][1]["backend"]), ("final", "gemini"))

    def test_un_motor_sin_gpu_no_la_crea_al_vuelo(self):
        pedidas = []

        def fabrica(nombre):
            pedidas.append(nombre)
            return Falso(nombre)
        n = Nucleo(BusDeMentira(), fabrica, max_salas=2, respaldo="cpu", reloj=lambda: AHORA)
        self.assertEqual(n.gpu, "sin")
        n.procesar(tramo(backend="gpu"))
        self.assertNotIn("gpu", pedidas, "una sala que pide gpu no puede cargar la GPU en un motor de CPU")

    def test_reconoce_errores_de_gpu(self):
        for m in ["CUDA error: out of memory", "cuBLAS failed", "RuntimeError: CUDA driver version is insufficient"]:
            self.assertTrue(es_error_de_gpu(RuntimeError(m)), m)
        self.assertFalse(es_error_de_gpu(ValueError("invalid literal for int()")))


class TestGlosario(unittest.TestCase):
    def test_el_glosario_del_tramo_llega_al_backend(self):
        visto = []

        class Espia:
            def procesar(self, t, glosario=""):
                visto.append(glosario)
                return completar(t.idioma, "x", "y")
        n, _bus = nucleo({"gpu": Espia()})
        t = tramo()
        t.glosario = "ElevenLabs, Nerdearla"
        n.procesar(t)
        self.assertEqual(visto, ["ElevenLabs, Nerdearla"])


class TestCascada(unittest.TestCase):
    def test_gemini_que_no_responde_cae_a_la_gpu(self):
        import requests
        gem = Falso(error=requests.exceptions.ReadTimeout("Read timed out. (read timeout=20)"))
        n, bus = nucleo({"gemini": gem, "gpu": Falso("por gpu")})
        n.procesar(tramo(backend="gemini"))
        s = bus.subs[0][1]
        self.assertEqual((s["tipo"], s["backend"], s["en"]), ("final", "gpu", "por gpu"),
                         "un Gemini lento no puede dejar a la sala sin subtítulo")

    def test_sin_gpu_ni_gemini_cae_a_cpu(self):
        n, bus = nucleo({"gemini": Falso(error=RuntimeError("Vertex 503: unavailable")),
                         "gpu": Falso(error=RuntimeError("CUDA error: device lost")), "cpu": Falso("por cpu")})
        n.procesar(tramo(backend="gemini"))
        self.assertEqual(bus.subs[0][1]["backend"], "cpu")
        self.assertEqual(n.gpu, "caida")

    def test_si_fallan_todos_el_error_se_entiende(self):
        import requests
        n, bus = nucleo({"gemini": Falso(error=requests.exceptions.ReadTimeout("Read timed out. (read timeout=20)")),
                         "gpu": Falso(error=RuntimeError("CUDA error: device lost")),
                         "cpu": Falso(error=MemoryError())})
        n.procesar(tramo(backend="gemini"))
        s = bus.subs[0][1]
        self.assertEqual(s["tipo"], "error")
        self.assertEqual(s["detalle"], "No salió el subtítulo: Gemini no respondió a tiempo; "
                                       "la GPU falló (error de CUDA); la CPU: MemoryError.")
        self.assertNotIn("HTTPSConnectionPool", s["detalle"])

    def test_humanizar(self):
        from motor.backends import humanizar
        self.assertEqual(humanizar("gemini", RuntimeError("Vertex 429: RESOURCE_EXHAUSTED")), "Gemini: cuota agotada")
        self.assertEqual(humanizar("gemini", RuntimeError("Vertex 403: permission denied")), "Gemini: la credencial no sirve")


class TestParciales(unittest.TestCase):
    def test_parcial_de_un_tramo_cuyo_final_ya_salio_no_se_publica(self):
        n, bus = nucleo({"gpu": Falso()})
        n.procesar(tramo(seq=3, final=True, id_="1-0"))
        n.procesar(tramo(seq=3, final=False, id_="2-0"))
        self.assertEqual([s["tipo"] for _, s in bus.subs], ["final"])

    def test_parcial_va_sin_traduccion(self):
        n, bus = nucleo({"gpu": Falso("partial text")})
        n.procesar(tramo(final=False))
        s = bus.subs[0][1]
        self.assertEqual((s["tipo"], s["en"], s["es"]), ("parcial", "partial text", ""))

    def test_despachar_queda_con_el_parcial_mas_nuevo_de_cada_sala(self):
        b = Falso()
        n, bus = nucleo({"gpu": b})
        lote = [tramo(seq=1, final=False, id_="1-0"), tramo(seq=1, final=False, id_="2-0"),
                tramo(seq=1, final=False, sala="s2", id_="3-0")]
        n.despachar(lote)
        n._pool.shutdown(wait=True)
        self.assertEqual(b.llamadas, 2, "uno por sala: el parcial viejo de s1 ya no sirve")
        self.assertEqual(sorted(bus.acks), ["1-0", "2-0", "3-0"], "todos se confirman, procesados o no")


class TestOraciones(unittest.TestCase):
    def test_parte_por_signo_final_y_conserva_todo(self):
        t = "Do we have anyone on? Yeah, maybe we need to translate. Come on, Matt… please!"
        self.assertEqual(oraciones(t), ["Do we have anyone on?", "Yeah, maybe we need to translate.",
                                        "Come on, Matt…", "please!"])
        self.assertEqual(" ".join(oraciones(t)), t)

    def test_sin_signos_es_una_sola(self):
        self.assertEqual(oraciones("um you know laughs wickedly"), ["um you know laughs wickedly"])


class TestGlosarioEnElTexto(unittest.TestCase):
    G = "Midudev, Midu.dev, Miguel Ángel Durán, Javier Tebas, la Liga, Instagram, piratería, influencer"

    def test_corrige_lo_que_whisper_escribio_raro(self):
        from motor.backends import corregir
        self.assertEqual(corregir("como Midude, que tengo una comunidad", self.G), "como Midudev, que tengo una comunidad")
        self.assertEqual(corregir("pues soy mi Dudef en las redes", self.G), "pues soy Midudev en las redes")
        self.assertEqual(corregir("Soy Miguel Angel Duran, más de 15 años", self.G), "Soy Miguel Ángel Durán, más de 15 años")
        self.assertEqual(corregir("el presidente de la liga, se llama", self.G), "el presidente de la Liga, se llama",
                         "no puede comerse el «de»")

    def test_no_toca_lo_que_no_se_parece_ni_los_terminos_en_minuscula(self):
        from motor.backends import corregir
        t = "los influencers de la piratería hablan de mi casa"
        self.assertEqual(corregir(t, self.G), t)
        self.assertEqual(corregir("si buscáis mi ludez lo vais a encontrar", self.G), "si buscáis mi ludez lo vais a encontrar")

    def test_proteger_y_restaurar(self):
        from motor.backends import proteger, restaurar
        p, m = proteger("Soy Miguel Ángel Durán, como Midudev.", self.G)
        self.assertNotIn("Durán", p)
        self.assertEqual(restaurar("I'm X0X, like X1X.", m), "I'm Miguel Ángel Durán, like Midudev.")


class TestGemini(unittest.TestCase):
    def test_interpreta_la_respuesta_con_esquema(self):
        resp = {"candidates": [{"content": {"parts": [{"text": json.dumps(
            {"transcripcion": " Hola a todos ", "traduccion": "Hello everyone"})}]}}]}
        r = interpretar_gemini(resp, "es")
        self.assertEqual((r.orig, r.es, r.en, r.idioma), ("Hola a todos", "Hola a todos", "Hello everyone", "es"))

    def test_sin_voz_es_none(self):
        resp = {"candidates": [{"content": {"parts": [{"text": '{"transcripcion": "", "traduccion": ""}'}]}}]}
        self.assertIsNone(interpretar_gemini(resp, "en"))

    def test_wav_16k_mono(self):
        w = wave.open(io.BytesIO(wav(b"\x01\x00" * 16000)))
        self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()), (16000, 1, 2, 16000))


class TestBusDeVerdad(unittest.TestCase):
    """Contra fakeredis: grupo de consumo, lectura, ack y publicación, con el
    formato EXACTO que escribe el hub (bus.go, PublicarTramo)."""

    def setUp(self):
        self.r = fakeredis.FakeRedis()
        self.bus = Bus(self.r, consumidor="prueba")
        self.bus.preparar()
        self.bus.preparar()  # idempotente: el segundo arranque no falla

    def test_lee_un_tramo_escrito_como_lo_escribe_el_hub(self):
        pcm = (np.arange(16000, dtype="<i2") % 100).tobytes()
        self.r.xadd(CLAVE_TRAMOS, {"sala": "s1", "idioma": "es", "backend": "gpu", "seq": 7, "final": "1",
                                   "t_ini": 1000, "t_fin": 11000, "glosario": "Javier Tebas, midudev", "pcm": pcm})
        (t,) = self.bus.leer(10, bloquear_ms=10)
        self.assertEqual((t.sala, t.idioma, t.backend, t.seq, t.final, t.t_ini, t.t_fin),
                         ("s1", "es", "gpu", 7, True, 1000, 11000))
        self.assertEqual(t.glosario, "Javier Tebas, midudev")
        a = t.audio()
        self.assertEqual((a.dtype, a.shape), (np.float32, (16000,)))
        self.assertAlmostEqual(float(a[99]), 99 / 32768, places=6)
        self.bus.confirmar(t)
        self.assertEqual(self.r.xpending(CLAVE_TRAMOS, "motor")["pending"], 0)

    def test_publica_todos_los_campos_del_contrato_y_nada_mas(self):
        self.bus.publicar("s1", tipo="final", orig="hola", idioma="es", es="hola", en="hello",
                          t_ini=1, t_fin=2, backend="gpu", lat_ms=300, seq=1)
        (_id, v), = self.r.xrange(clave_subs("s1"))
        self.assertEqual(set(k.decode() for k in v), set(CAMPOS_SUB))
        with self.assertRaises(ValueError):
            self.bus.publicar("s1", tipo="final", texto="campo inventado")

    def test_el_contrato_es_el_que_lee_el_hub(self):
        # Copia literal de bus.go. Si alguien cambia un lado sin el otro,
        # este test (o su gemelo en hub_test.go) se pone rojo.
        self.assertEqual(CAMPOS_TRAMO, ("sala", "idioma", "backend", "seq", "final", "t_ini", "t_fin",
                                        "glosario", "pcm"))
        self.assertEqual(CAMPOS_SUB, ("tipo", "orig", "idioma", "es", "en", "t_ini", "t_fin", "backend",
                                      "lat_ms", "seq", "detalle"))


class TestBucle(unittest.TestCase):
    def test_redis_reiniciado_sin_grupo_se_recupera_solo(self):
        import time as _t
        r = fakeredis.FakeRedis()
        n = Nucleo(Bus(r, consumidor="prueba"), lambda nombre: Falso("de nuevo"), max_salas=2)
        n.gpu = "ok"
        parar = threading.Event()
        h = threading.Thread(target=n.correr, args=(parar,))
        h.start()
        try:
            _t.sleep(0.2)
            r.flushall()  # redis sin disco que se reinicia: sin stream ni grupo
            _t.sleep(1.5)  # una vuelta de NOGROUP y la recreación
            ahora = int(_t.time() * 1000)
            r.xadd(CLAVE_TRAMOS, {"sala": "s7", "idioma": "en", "backend": "gpu", "seq": 1, "final": "1",
                                  "t_ini": ahora - 9000, "t_fin": ahora, "pcm": b"\x00\x00" * 1600})
            for _ in range(60):
                if r.exists(clave_subs("s7")):
                    break
                _t.sleep(0.05)
            self.assertTrue(r.exists(clave_subs("s7")), "el motor no volvió a leer después de perder el grupo")
        finally:
            parar.set()
            h.join(5)

    def test_de_punta_a_punta_sobre_fakeredis(self):
        r = fakeredis.FakeRedis()
        bus = Bus(r, consumidor="prueba")
        import time
        n = Nucleo(bus, lambda nombre: Falso("end to end"), max_salas=2, atraso_max_s=25)
        n.gpu = "ok"
        parar = threading.Event()
        h = threading.Thread(target=n.correr, args=(parar,))
        h.start()
        try:
            ahora = int(time.time() * 1000)
            import time as _t
            _t.sleep(0.2)  # que el grupo exista antes del primer tramo ("$")
            r.xadd(CLAVE_TRAMOS, {"sala": "s9", "idioma": "en", "backend": "gpu", "seq": 1, "final": "1",
                                  "t_ini": ahora - 9000, "t_fin": ahora, "pcm": b"\x00\x00" * 1600})
            for _ in range(100):
                if r.xlen(clave_subs("s9")):
                    break
                _t.sleep(0.05)
            (_id, v), = r.xrange(clave_subs("s9"))
            self.assertEqual(v[b"orig"], b"end to end")
            self.assertEqual(v[b"es"], b"[es] end to end")
            self.assertEqual(r.hget("conf:motor", "gpu"), b"ok")
        finally:
            parar.set()
            h.join(5)


if __name__ == "__main__":
    unittest.main()


class TestConfig(unittest.TestCase):
    def test_redis_como_lo_entrega_la_plataforma(self):
        from motor.__main__ import url_redis
        self.assertEqual(url_redis({"REDIS_ADDR": "bus:6379", "REDIS_PASSWORD": "a/b@c"}), "redis://:a%2Fb%40c@bus:6379/0")
        self.assertEqual(url_redis({"REDIS_URL": "redis://x:1/0", "REDIS_PASSWORD": "no"}), "redis://x:1/0")
        self.assertEqual(url_redis({}), "redis://localhost:6379/0")
