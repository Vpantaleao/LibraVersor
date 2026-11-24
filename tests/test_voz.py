"""Voz (tts.py): escolha da voz em português e comportamento com pyttsx3 simulado."""
import sys
import types
import weakref

import pytest

import tts
from helpers import wait_for


class Voz:
    def __init__(self, id_, nome, idiomas=None):
        self.id, self.name, self.languages = id_, nome, idiomas or []


# ---------------------------------------------------------------- escolha da voz
def test_prefere_pt_br():
    vozes = [Voz("en-1", "Microsoft Anna - English (United States)", [b"en-US"]),
             Voz("TTS_MS_PT-BR_MARIA_11.0", "Microsoft Maria Desktop - Portuguese(Brazil)")]
    voz, exata = tts.pick_portuguese_voice(vozes)
    assert voz.name.startswith("Microsoft Maria") and exata


def test_aceita_voz_do_mac():
    voz, exata = tts.pick_portuguese_voice([Voz("com.apple.voice.compact.pt-BR.Luciana", "Luciana")])
    assert voz.name == "Luciana" and exata


def test_aceita_portugues_de_portugal_como_segunda_opcao():
    voz, exata = tts.pick_portuguese_voice(
        [Voz("x", "Portuguese (Portugal)", ["pt-pt"]), Voz("y", "English")])
    assert voz.id == "x" and not exata


def test_sem_voz_em_portugues():
    assert tts.pick_portuguese_voice([Voz("a", "English"), Voz("b", "Script reader")]) == (None, False)


# ---------------------------------------------------------------- Speaker
@pytest.fixture
def pyttsx3_falso(monkeypatch):
    """pyttsx3 falso que reproduz o defeito real: um motor reaproveitado só fala na 1ª vez."""
    faladas, cache = [], weakref.WeakValueDictionary()

    class Motor:
        def __init__(self):
            self.chamadas, self.texto = 0, None

        def getProperty(self, nome):
            return [Voz("en", "English"), Voz("pt", "Microsoft Maria - Portuguese(Brazil)")]

        def setProperty(self, nome, valor):
            pass

        def say(self, texto):
            self.texto = texto

        def runAndWait(self):
            self.chamadas += 1
            if self.chamadas == 1:
                faladas.append(self.texto)

        def stop(self):
            pass

    def init():
        motor = cache.get("motor")
        if motor is None:
            motor = Motor()
            cache["motor"] = motor
        return motor

    monkeypatch.setitem(sys.modules, "pyttsx3", types.SimpleNamespace(init=init))
    return faladas


def test_fala_todas_as_palavras_em_ordem(pyttsx3_falso):
    voz = tts.Speaker()
    assert voz.available and "Maria" in voz.message
    for palavra in ("oi", "dor", "febre"):
        voz.say(palavra)
        assert wait_for(lambda p=palavra: p in " ".join(pyttsx3_falso).split())
    voz.close()
    assert " ".join(pyttsx3_falso).split() == ["oi", "dor", "febre"]


def test_palavras_acumuladas_saem_juntas_e_em_ordem(pyttsx3_falso):
    voz = tts.Speaker()
    for palavra in ("sim", "não", "oi"):
        voz.say(palavra)
    assert wait_for(lambda: len(" ".join(pyttsx3_falso).split()) == 3)
    voz.close()
    assert " ".join(pyttsx3_falso).split() == ["sim", "não", "oi"]


def test_desligada_nao_fala_e_toggle_alterna(pyttsx3_falso):
    voz = tts.Speaker(enabled=False)
    voz.say("oi")
    assert not wait_for(lambda: pyttsx3_falso, timeout=0.3)
    assert voz.toggle() is True
    voz.say("dor")
    assert wait_for(lambda: pyttsx3_falso)
    voz.close()


def test_sem_pyttsx3_o_programa_segue_so_com_legenda(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyttsx3", None)      # força ImportError
    voz = tts.Speaker()
    assert not voz.available
    assert "indisponível" in voz.message
    voz.say("oi")                                           # não deve dar erro
    voz.close()
