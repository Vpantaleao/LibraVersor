"""Regras que decidem O QUE aparece na legenda (StreamRecognizer) e a legenda (CaptionBuffer)."""
import libra_core as lc
from helpers import FixedModel, NO_HAND, feed, hand_frame

LABELS = {0: "dor", 1: "febre", 2: "repouso", 3: "oi"}
DOR, FEBRE, REPOUSO = 0, 1, 2
HAND = hand_frame()


def novo(**kw):
    model = FixedModel(4)
    return model, lc.StreamRecognizer(model, LABELS, **kw)


# ---------------------------------------------------------------- confirmação
def test_aquece_antes_de_decidir():
    model, rec = novo()
    assert feed(rec, HAND, lc.SEQUENCE_LENGTH - 1) == []
    assert rec.state == "aquecendo" and model.calls == 0


def test_confirma_a_palavra_uma_unica_vez_mesmo_com_o_gesto_mantido():
    model, rec = novo()
    model.pick = (DOR, 0.9)
    assert feed(rec, HAND, 120) == ["dor"]


def test_repouso_nunca_vira_palavra():
    model, rec = novo()
    model.pick = (REPOUSO, 0.95)
    assert feed(rec, HAND, 100) == []
    assert rec.state == "repouso"


def test_confianca_baixa_nao_vira_palavra():
    model, rec = novo()
    model.pick = (DOR, 0.45)                     # abaixo dos 60% padrão
    assert feed(rec, HAND, 100) == []
    assert rec.state == "analisando"


def test_margem_pequena_sobre_a_segunda_colocada_nao_vira_palavra():
    model, rec = novo(conf_min=0.4)
    model.pick = [0.45, 0.40, 0.10, 0.05]        # confiança ok, mas só 5 pontos de margem
    assert feed(rec, HAND, 100) == []
    model2, rec2 = novo(conf_min=0.4, margin_min=0.0)
    model2.pick = [0.45, 0.40, 0.10, 0.05]
    assert feed(rec2, HAND, 100) == ["dor"]      # sem exigir margem, a palavra passa


def test_exige_estabilidade_antes_de_confirmar():
    model, rec = novo(stable_frames=20)
    model.pick = (DOR, 0.9)
    assert feed(rec, HAND, 40) == []             # ainda não ficou estável por 20 frames
    assert feed(rec, HAND, 20) == ["dor"]


def test_usa_os_ids_de_classe_do_modelo_e_nao_a_posicao():
    labels = {10: "dor", 20: "febre", 30: "repouso"}
    model = FixedModel(3, class_ids=[10, 20, 30])
    rec = lc.StreamRecognizer(model, labels)
    model.pick = (1, 0.9)                        # coluna 1 = classe de id 20
    assert feed(rec, HAND, 80) == ["febre"]


# ---------------------------------------------------------------- repetição
def test_cauda_do_gesto_nao_gera_palavra_repetida():
    """Regressão: a mão sai da câmera, mas a janela ainda contém o gesto anterior."""
    model, rec = novo()
    model.pick = (DOR, 0.9)
    assert feed(rec, HAND, 60) == ["dor"]
    assert feed(rec, NO_HAND, 25) == []
    assert feed(rec, NO_HAND, 60) == []


def test_mesma_palavra_repete_depois_que_a_mao_sai_e_volta():
    model, rec = novo()
    model.pick = (DOR, 0.9)
    assert feed(rec, HAND, 60) == ["dor"]
    assert feed(rec, NO_HAND, 60) == []          # janela esvazia e o sistema rearma
    assert feed(rec, HAND, 60) == ["dor"]


def test_mesma_palavra_repete_depois_de_repouso_previsto():
    model, rec = novo()
    model.pick = (DOR, 0.9)
    assert feed(rec, HAND, 60) == ["dor"]
    model.pick = (REPOUSO, 0.95)
    assert feed(rec, HAND, 40) == []
    model.pick = (DOR, 0.9)
    assert feed(rec, HAND, 40) == ["dor"]


def test_palavra_diferente_espera_o_intervalo_refratario():
    model, rec = novo(refractory_frames=200)
    model.pick = (DOR, 0.9)
    assert feed(rec, HAND, 60) == ["dor"]
    model.pick = (FEBRE, 0.9)
    assert feed(rec, HAND, 60) == []             # ainda dentro do intervalo


def test_palavra_diferente_passa_depois_do_intervalo():
    model, rec = novo()
    model.pick = (DOR, 0.9)
    assert feed(rec, HAND, 60) == ["dor"]
    model.pick = (FEBRE, 0.9)
    assert feed(rec, HAND, 60) == ["febre"]


# ---------------------------------------------------------------- mão ausente / janela
def test_sem_mao_nao_consulta_o_modelo():
    model, rec = novo()
    feed(rec, NO_HAND, 100)
    assert model.calls == 0 and rec.state == "sem_mao"


def test_poucos_frames_com_mao_contam_como_sem_mao():
    model, rec = novo()
    feed(rec, NO_HAND, 40)
    feed(rec, HAND, 6)                           # só 6 frames com mão na janela (< 12)
    assert rec.state == "sem_mao" and model.calls == 0


def test_so_preve_a_cada_stride_frames():
    model, rec = novo(stride=2)
    feed(rec, HAND, lc.SEQUENCE_LENGTH + 10)     # previsões nos frames 30, 32, ..., 40
    assert model.calls == 6
    model5, rec5 = novo(stride=5)
    feed(rec5, HAND, lc.SEQUENCE_LENGTH + 10)    # frames 30, 35, 40
    assert model5.calls == 3


def test_reset_limpa_o_estado():
    model, rec = novo()
    model.pick = (DOR, 0.9)
    assert feed(rec, HAND, 60) == ["dor"]
    rec.reset()
    assert len(rec.window) == 0 and rec.frame_i == 0
    assert rec.state == "aquecendo" and rec.last_word is None and rec.armed
    assert feed(rec, HAND, lc.SEQUENCE_LENGTH - 1) == []


# ---------------------------------------------------------------- legenda
def test_legenda_guarda_palavras_em_ordem():
    cap = lc.CaptionBuffer()
    cap.add("dor", 0.0)
    cap.add("febre", 1.0)
    assert cap.words() == ["dor", "febre"]
    assert cap.last_age(3.0) == 2.0


def test_legenda_vazia_nao_tem_idade():
    assert lc.CaptionBuffer().last_age(5.0) is None


def test_legenda_limita_o_numero_de_palavras():
    cap = lc.CaptionBuffer(max_words=3)
    for i, w in enumerate(["a", "b", "c", "d", "e"]):
        cap.add(w, float(i))
    assert cap.words() == ["c", "d", "e"]


def test_legenda_some_so_depois_do_tempo_limite():
    cap = lc.CaptionBuffer(timeout=7.0)
    cap.add("dor", 100.0)
    cap.prune(107.0)                             # exatamente no limite: ainda fica
    assert cap.words() == ["dor"]
    cap.prune(107.1)
    assert cap.words() == []


def test_legenda_limpar():
    cap = lc.CaptionBuffer()
    cap.add("dor", 0.0)
    cap.clear()
    assert cap.words() == []
