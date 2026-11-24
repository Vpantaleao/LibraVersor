"""main.py: carregamento do modelo, fontes, legenda na tela, thread de reconhecimento e laço principal."""
import pickle
import time
import types

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("mediapipe")
Image = pytest.importorskip("PIL.Image")

import main  # noqa: E402
import libra_core as lc  # noqa: E402
from helpers import FixedModel, hand_frame, make_sequence, wait_for  # noqa: E402


# ---------------------------------------------------------------- carregamento do modelo
def _salva(caminho, obj):
    with open(caminho, "wb") as f:
        pickle.dump(obj, f)


def test_modelo_ausente_manda_treinar(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MODEL_PATH", str(tmp_path / "nao_existe.p"))
    with pytest.raises(SystemExit) as erro:
        main.load_model()
    assert "treinar_modelo.py" in str(erro.value)


@pytest.mark.parametrize("payload", [
    {"model": object(), "labels_dict": {}},                                # formato antigo, sem versão
    {"model": object(), "labels_dict": {}, "feature_version": 1},          # versão antiga
])
def test_modelo_antigo_manda_treinar(tmp_path, monkeypatch, payload):
    _salva(tmp_path / "m.p", payload)
    monkeypatch.setattr(main, "MODEL_PATH", str(tmp_path / "m.p"))
    with pytest.raises(SystemExit) as erro:
        main.load_model()
    assert "treinar_modelo.py" in str(erro.value)


def test_arquivo_de_modelo_corrompido_manda_treinar(tmp_path, monkeypatch):
    (tmp_path / "m.p").write_bytes(b"isto nao e um pickle")
    monkeypatch.setattr(main, "MODEL_PATH", str(tmp_path / "m.p"))
    with pytest.raises(SystemExit) as erro:
        main.load_model()
    assert "treinar_modelo.py" in str(erro.value)


def test_modelo_valido_carrega_pronto_para_uso_ao_vivo(tmp_path, monkeypatch):
    seqs = [make_sequence(s, noise_seed=n) for s in (11, 22) for n in range(8)]
    y = np.array([0] * 8 + [1] * 8)
    modelo = lc.GestureEnsemble(n_trees=10).fit(lc.features_matrix(seqs), y)
    _salva(tmp_path / "m.p", {"model": modelo, "labels_dict": {0: "a", 1: "b"},
                              "feature_version": lc.FEATURE_VERSION})
    monkeypatch.setattr(main, "MODEL_PATH", str(tmp_path / "m.p"))
    carregado, nomes = main.load_model()
    assert nomes == {0: "a", 1: "b"} and carregado.rf.n_jobs == 1


def test_argumentos_padrao_e_personalizados():
    padrao = main.parse_args([])
    assert (padrao.camera, padrao.sem_voz, padrao.debug, padrao.confianca) == (0, False, False, 0.60)
    custom = main.parse_args(["--camera", "2", "--sem-voz", "--debug", "--confianca", "0.7"])
    assert (custom.camera, custom.sem_voz, custom.debug, custom.confianca) == (2, True, True, 0.7)


# ---------------------------------------------------------------- fontes
def _largura(texto, tamanho):
    from PIL import ImageDraw
    return ImageDraw.Draw(Image.new("RGB", (10, 10))).textlength(texto, font=main.font(tamanho))


def test_letras_crescem_com_o_tamanho():
    main._fonts.clear()
    assert _largura("Bom dia dor", 84) > 1.3 * _largura("Bom dia dor", 58)


def test_letras_crescem_mesmo_sem_a_fonte_embutida(monkeypatch):
    """Regressão: sem a pasta assets/fonts só a caixa crescia e as letras ficavam minúsculas."""
    monkeypatch.setattr(main, "FONT_BOLD", "/nao/existe/Bold.ttf")
    monkeypatch.setattr(main, "FONT_REGULAR", "/nao/existe/Regular.ttf")
    main._fonts.clear()
    assert _largura("Bom dia dor", 84) > 1.3 * _largura("Bom dia dor", 58)


def test_letras_crescem_mesmo_sem_nenhuma_fonte_em_disco(monkeypatch):
    real = main.ImageFont.truetype

    def so_memoria(fonte, *a, **k):
        if isinstance(fonte, str):
            raise OSError("sem fonte no disco")
        return real(fonte, *a, **k)

    monkeypatch.setattr(main.ImageFont, "truetype", so_memoria)
    main._fonts.clear()
    assert _largura("Bom dia dor", 84) > 1.3 * _largura("Bom dia dor", 58)
    main._fonts.clear()


# ---------------------------------------------------------------- legenda na tela
def _ui(**mudancas):
    ui = dict(words=[], age=0.0, hand=False, state="aquecendo", candidate=None, confidence=0.0,
              voice_on=True, voice_available=True, debug=False, fps=30.0)
    ui.update(mudancas)
    return ui


def _desenha(ui, tamanho=(1280, 720)):
    img = Image.new("RGB", tamanho, (120, 120, 120))
    main.draw_hud(img, ui)
    return np.asarray(img).astype(int)


def test_legenda_aparece_na_parte_de_baixo():
    sem, com = _desenha(_ui()), _desenha(_ui(words=["dor"], age=0.2))
    assert np.abs(sem[500:] - com[500:]).sum() > 0


def test_palavra_recente_fica_destacada_so_por_um_tempo():
    cor = np.array(main.ACCENT)

    def tem_destaque(img):
        return (np.abs(img[430:] - cor).sum(axis=2) < 12).any()

    assert tem_destaque(_desenha(_ui(words=["dor"], age=0.2)))
    assert not tem_destaque(_desenha(_ui(words=["dor"], age=main.HIGHLIGHT_SECONDS + 1)))


def test_constante_do_tamanho_da_legenda_controla_o_tamanho_das_letras(monkeypatch):
    def brancos(img):
        return (img[430:].min(axis=2) > 245).sum()

    monkeypatch.setattr(main, "CAPTION_FONT_SIZE", 40)
    pequeno = brancos(_desenha(_ui(words=["dor"], age=5)))
    monkeypatch.setattr(main, "CAPTION_FONT_SIZE", 96)
    grande = brancos(_desenha(_ui(words=["dor"], age=5)))
    assert grande > 2 * pequeno


@pytest.mark.parametrize("tamanho", [(1280, 720), (640, 360), (1920, 1080)])
@pytest.mark.parametrize("ui", [
    _ui(),
    _ui(hand=True, state="repouso", words=["bom dia"], age=1.0),
    _ui(hand=True, state="reconhecendo", candidate="dor", confidence=0.9, words=["dor", "febre"], debug=True),
    _ui(voice_available=False, words=["oi"], age=3.0),
    _ui(voice_on=False, hand=True, state="analisando"),
    _ui(words=["bom dia", "sentindo", "dor", "de cabeça", "febre", "muito", "remédio", "vômito",
               "emergência", "obrigado", "pouco", "gripe"], age=2.0, hand=True),
])
def test_interface_desenha_em_qualquer_estado_e_tamanho_sem_erro(ui, tamanho):
    img = _desenha(ui, tamanho)
    assert img.shape == (tamanho[1], tamanho[0], 3)


def test_legenda_longa_nao_ultrapassa_a_largura_da_tela():
    ui = _ui(words=["bom dia", "sentindo", "dor", "de cabeça", "febre", "muito", "remédio", "vômito",
                    "emergência", "obrigado", "pouco", "gripe"], age=2.0)
    img = _desenha(ui)
    fundo = (np.abs(img[430:690] - 120).sum(axis=2) > 30)       # tudo que não é o fundo da cena
    colunas = np.where(fundo.any(axis=0))[0]
    assert colunas.min() > 20 and colunas.max() < 1280 - 20


# ---------------------------------------------------------------- thread de reconhecimento
class Reconhecedor:
    state, candidate, confidence = "pronto", None, 0.0

    def __init__(self, falha_na_primeira=False, lento=0.0):
        self.resets, self.vistos, self.falha, self.lento = 0, 0, falha_na_primeira, lento

    def update(self, feats):
        self.vistos += 1
        if self.falha and self.vistos == 1:
            raise RuntimeError("erro simulado")
        if self.lento:
            time.sleep(self.lento)
        return "dor" if feats[0] == 1 else None

    def reset(self):
        self.resets += 1


def _frame(valor):
    v = np.zeros(63)
    v[0] = valor
    return v


def test_thread_entrega_a_palavra_confirmada():
    ass = main.AsyncRecognizer(Reconhecedor())
    ass.push(_frame(0))
    ass.push(_frame(1))
    palavras = []
    assert wait_for(lambda: palavras.extend(ass.pop_words()) or palavras)
    assert palavras == ["dor"]
    ass.stop()


def test_erro_em_um_frame_nao_derruba_a_thread():
    ass = main.AsyncRecognizer(Reconhecedor(falha_na_primeira=True))
    ass.push(_frame(0))                    # provoca o erro
    ass.push(_frame(1))
    palavras = []
    assert wait_for(lambda: palavras.extend(ass.pop_words()) or palavras)
    assert palavras == ["dor"]
    ass.stop()


def test_reset_chega_ao_reconhecedor():
    rec = Reconhecedor()
    ass = main.AsyncRecognizer(rec)
    ass.reset()
    assert wait_for(lambda: rec.resets == 1)
    ass.stop()


def test_fila_cheia_nao_trava_o_video():
    ass = main.AsyncRecognizer(Reconhecedor(lento=0.01))
    inicio = time.time()
    for _ in range(600):                   # bem mais que o tamanho da fila
        ass.push(_frame(0))
    assert time.time() - inicio < 1.0
    ass.stop()


# ---------------------------------------------------------------- laço principal (câmera, mão e voz simuladas)
class LM:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def _mao(vec):
    return types.SimpleNamespace(landmark=[LM(0.5 + x, 0.5 + y, z) for x, y, z in vec.reshape(21, 3)])


@pytest.fixture
def ambiente(monkeypatch):
    """Troca câmera, janela, MediaPipe, modelo e voz por dublês e devolve o que foi observado."""
    obs = types.SimpleNamespace(falas=[], toggles=0, fechou_voz=False, camera_liberada=False,
                                teclas=[], chamadas_waitkey=0)
    mao = _mao(hand_frame(1))

    class Camera:
        def __init__(self, *a):
            self.n = 0

        def set(self, *a):
            pass

        def isOpened(self):
            return True

        def read(self):
            time.sleep(0.003)
            self.n += 1
            return True, np.full((180, 320, 3), 90, dtype=np.uint8)

        def release(self):
            obs.camera_liberada = True

    class Maos:
        def __init__(self, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def process(self, rgb):
            return types.SimpleNamespace(multi_hand_landmarks=[mao])

    class Voz:
        available, enabled, message = True, True, "voz simulada"

        def __init__(self, enabled=True):
            pass

        def say(self, texto):
            obs.falas.append(texto)

        def toggle(self):
            obs.toggles += 1

        def close(self):
            obs.fechou_voz = True

    def waitkey(_):
        obs.chamadas_waitkey += 1
        n = obs.chamadas_waitkey
        if obs.teclas and n in obs.teclas:
            return ord(obs.teclas[n])
        if (obs.falas and n > 8) or n > 800:
            return ord("q")
        return -1

    modelo = FixedModel(4)
    modelo.pick = (0, 0.9)
    nomes = {0: "dor", 1: "febre", 2: "repouso", 3: "oi"}
    desenho = types.SimpleNamespace(DrawingSpec=lambda **k: k, draw_landmarks=lambda *a, **k: None)
    monkeypatch.setattr(main.cv2, "VideoCapture", Camera)
    monkeypatch.setattr(main.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(main.cv2, "waitKey", waitkey)
    monkeypatch.setattr(main.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(main.cv2, "resizeWindow", lambda *a, **k: None)
    monkeypatch.setattr(main.cv2, "setWindowProperty", lambda *a, **k: None)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(main, "mp", types.SimpleNamespace(solutions=types.SimpleNamespace(
        hands=types.SimpleNamespace(Hands=Maos, HAND_CONNECTIONS=[]), drawing_utils=desenho)))
    monkeypatch.setattr(main, "Speaker", Voz)
    monkeypatch.setattr(main, "load_model", lambda: (modelo, nomes))
    return obs


def test_laco_principal_reconhece_fala_e_encerra_com_q(ambiente):
    main.run(main.parse_args([]))
    assert ambiente.falas and set(ambiente.falas) == {"dor"}
    assert ambiente.fechou_voz and ambiente.camera_liberada


def test_teclas_de_voz_e_de_limpar_nao_quebram_o_laco(ambiente):
    ambiente.teclas = {3: "v", 5: "c", 6: "d", 7: "v"}
    main.run(main.parse_args([]))
    assert ambiente.toggles == 2
    assert ambiente.falas and ambiente.camera_liberada
