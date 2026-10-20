"""
tts.py - voz que lê as palavras reconhecidas (offline, usa as vozes do sistema).

Roda em uma thread própria, então falar nunca trava o vídeo. Se a voz não puder
ser iniciada (biblioteca ou voz ausente), o programa continua funcionando só com
a legenda.

Vozes em português do Brasil:
  Windows: Configurações > Hora e idioma > Fala > Adicionar vozes (ex.: "Maria").
  macOS:   já vem com a voz "Luciana".
  Linux:   sudo apt install espeak-ng   (voz mais robótica)
"""
import queue
import re
import threading

_PT_BR = ("pt-br", "pt_br", "brazil", "brasil")
_PT = ("portug",)
_PT_TOKEN = re.compile(r"(^|[^a-z])pt([^a-z]|$)")


def _voice_text(voice):
    langs = getattr(voice, "languages", None) or []
    langs = " ".join(l.decode(errors="ignore") if isinstance(l, bytes) else str(l) for l in langs)
    return f"{voice.id} {voice.name} {langs}".lower()


def pick_portuguese_voice(voices):
    """Devolve (voice, exata) - preferindo pt-BR, depois qualquer português."""
    for v in voices:
        if any(k in _voice_text(v) for k in _PT_BR):
            return v, True
    for v in voices:
        t = _voice_text(v)
        if any(k in t for k in _PT) or _PT_TOKEN.search(t):
            return v, False
    return None, False


class Speaker:
    def __init__(self, rate=190, volume=1.0, enabled=True):
        self.enabled = enabled
        self.available = False
        self.voice_name = None
        self.message = ""                       # explica o estado da voz para o usuário
        self._rate, self._volume = rate, volume
        self._q = queue.Queue()
        self._ready = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()
        self._ready.wait(timeout=10)

    # -- thread da voz --
    # O pyttsx3 costuma falar só a PRIMEIRA vez quando o mesmo motor é reaproveitado
    # (problema conhecido, comum no Windows). Por isso cada fala usa um motor novo.
    def _new_engine(self, pyttsx3):
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)
        if self._voice_id:
            engine.setProperty("voice", self._voice_id)
        return engine

    def _worker(self):
        self._voice_id = None
        try:
            import pyttsx3
            probe = pyttsx3.init()
            voice, exact = pick_portuguese_voice(probe.getProperty("voices"))
            if voice is not None:
                self._voice_id = voice.id
                self.voice_name = voice.name
                self.message = f"Voz: {voice.name}" + ("" if exact else " (português, não pt-BR)")
            else:
                self.message = ("Nenhuma voz em português encontrada: usando a voz padrão do "
                                "sistema (a pronúncia pode ficar estranha).")
            try:
                probe.stop()
            except Exception:
                pass
            del probe
            self.available = True
        except Exception as exc:                 # sem pyttsx3, sem driver de áudio etc.
            self.message = f"Voz indisponível ({type(exc).__name__}: {exc}). Seguindo só com legenda."
        finally:
            self._ready.set()
        if not self.available:
            return
        while True:
            text = self._q.get()
            if text is None:
                break
            # palavras que chegaram enquanto a anterior era falada saem numa fala só:
            # sem pausa nem recriação do motor entre elas
            parts, stop = [text], False
            try:
                while True:
                    nxt = self._q.get_nowait()
                    if nxt is None:
                        stop = True
                        break
                    parts.append(nxt)
            except queue.Empty:
                pass
            text = " ".join(parts)
            engine = None
            try:
                print(f"[voz] falando: {text}")
                engine = self._new_engine(pyttsx3)
                engine.say(text)
                engine.runAndWait()
                try:
                    engine.stop()
                except Exception:
                    pass
            except Exception as exc:
                print(f"[voz] erro ao falar '{text}': {type(exc).__name__}: {exc}")
            finally:
                engine = None                    # solta o motor para o próximo ser novo
            if stop:
                break

    # -- API --
    def say(self, text):
        if not (self.enabled and self.available and text):
            return
        if self._q.qsize() >= 8:                 # fila muito atrasada: descarta a mais antiga
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        self._q.put(text)

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def close(self):
        self._q.put(None)