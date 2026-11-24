"""Features, variações simuladas (aumento de dados) e o modelo, em libra_core."""
from types import SimpleNamespace

import numpy as np
import pytest

import libra_core as lc
from helpers import hand_frame, make_sequence


# ---------------------------------------------------------------- landmarks
def _mao_falsa():
    pontos = [SimpleNamespace(x=0.5 + i * 0.01, y=0.4 + i * 0.02, z=0.01 * i) for i in range(21)]
    return SimpleNamespace(landmark=pontos)


def test_sem_mao_vira_vetor_de_zeros():
    assert np.array_equal(lc.extract_landmarks(None), np.zeros(63))


def test_landmarks_sao_relativos_ao_punho():
    v = lc.extract_landmarks(_mao_falsa())
    assert v.shape == (63,)
    assert np.allclose(v[:3], 0)                           # o punho vira a origem
    assert np.allclose(v[3:6], [0.01, 0.02, 0.01])         # ponto 1 - punho


def test_nomes_de_exibicao():
    assert lc.display_name("emergencia") == "emergência"
    assert lc.display_name("vômito") == "vômito"
    assert lc.display_name("gesto-novo") == "gesto-novo"   # sem cadastro: usa o próprio nome


# ---------------------------------------------------------------- features
def test_features_tem_o_tamanho_esperado_e_sao_finitas():
    f = lc.sequence_features(make_sequence(1))
    assert f.shape == (lc.FEATURE_DIM,)
    assert np.isfinite(f).all()


def test_features_matrix():
    m = lc.features_matrix([make_sequence(1), make_sequence(2), make_sequence(3)])
    assert m.shape == (3, lc.FEATURE_DIM)


def test_menos_de_3_frames_com_mao_vira_zeros():
    seq = np.zeros((lc.SEQUENCE_LENGTH, 63))
    seq[:2] = make_sequence(1)[:2]
    assert not lc.sequence_features(seq).any()
    assert not lc.sequence_features(np.zeros((lc.SEQUENCE_LENGTH, 63))).any()


def test_fracao_de_frames_com_mao_e_a_ultima_feature():
    seq = make_sequence(1)
    seq[15:] = 0
    assert lc.sequence_features(seq)[-1] == pytest.approx(0.5)


def test_features_nao_dependem_da_distancia_da_mao_a_camera():
    seq = make_sequence(1)
    assert np.allclose(lc.sequence_features(seq), lc.sequence_features(seq * 2), atol=1e-3)
    assert np.allclose(lc.sequence_features(seq), lc.sequence_features(seq * 0.5), atol=1e-3)


def test_gestos_diferentes_geram_features_bem_diferentes():
    a1 = lc.sequence_features(make_sequence(11, noise_seed=1))
    a2 = lc.sequence_features(make_sequence(11, noise_seed=2))
    b1 = lc.sequence_features(make_sequence(22, noise_seed=1))
    mesmo = np.linalg.norm(a1 - a2)
    outro = np.linalg.norm(a1 - b1)
    assert outro > 3 * mesmo


# ---------------------------------------------------------------- variações simuladas
def test_transform_sem_parametros_nao_altera_nada():
    seq = make_sequence(1)
    assert np.allclose(lc.transform(seq), seq)


def test_transform_escala():
    seq = make_sequence(1)
    assert np.allclose(lc.transform(seq, scale=2.0), seq * 2)


def test_transform_espelho_inverte_so_o_eixo_x():
    seq = make_sequence(1)
    out = lc.transform(seq, mirror=True)
    assert np.allclose(out[:, 0::3], -seq[:, 0::3])
    assert np.allclose(out[:, 1::3], seq[:, 1::3])
    assert np.allclose(out[:, 2::3], seq[:, 2::3])


def test_transform_rotacao_preserva_as_distancias_ao_punho():
    def normas(s):
        p = s.reshape(len(s), 21, 3).copy()
        p[..., 1] *= lc.ASPECT
        return np.linalg.norm(p, axis=2)

    seq = make_sequence(1)
    assert np.allclose(normas(lc.transform(seq, rot=37)), normas(seq))


def test_transform_deslocamento_no_tempo():
    seq = np.arange(lc.SEQUENCE_LENGTH * 63, dtype=float).reshape(lc.SEQUENCE_LENGTH, 63) + 1
    tarde = lc.transform(seq, shift=3)
    assert np.allclose(tarde[3:], seq[:-3]) and np.allclose(tarde[:3], seq[0])
    cedo = lc.transform(seq, shift=-3)
    assert np.allclose(cedo[:-3], seq[3:]) and np.allclose(cedo[-3:], seq[-1])


def test_transform_perda_total_de_frames():
    assert not lc.transform(make_sequence(1), drop=1.0).any()


def test_transform_velocidade_mantem_o_formato():
    seq = make_sequence(1)
    for warp in (0.8, 1.25):
        out = lc.transform(seq, warp=warp)
        assert out.shape == seq.shape and np.isfinite(out).all()


def test_frames_sem_mao_continuam_sem_mao_depois_de_qualquer_variacao():
    seq = make_sequence(1)
    seq[5:10] = 0
    out = lc.transform(seq, scale=1.3, rot=15, mirror=True, noise=0.01)
    assert not out[5:10].any()


def test_augment_gera_copias_com_os_mesmos_rotulos():
    seqs = [make_sequence(1), make_sequence(2)]
    x, y = lc.augment(seqs, [7, 9], copies=4, rng=np.random.default_rng(0))
    assert x.shape == (8, lc.SEQUENCE_LENGTH, 63)
    assert list(y) == [7] * 4 + [9] * 4


def test_augment_e_reprodutivel_com_a_mesma_semente():
    seqs = [make_sequence(1)]
    a, _ = lc.augment(seqs, [0], 3, np.random.default_rng(5))
    b, _ = lc.augment(seqs, [0], 3, np.random.default_rng(5))
    assert np.allclose(a, b)


# ---------------------------------------------------------------- modelo
def _dados_sinteticos(noise_offset=0):
    seqs, y = [], []
    for classe, shape_seed in enumerate((11, 22, 33)):
        for n in range(10):
            seqs.append(make_sequence(shape_seed, noise_seed=noise_offset + n))
            y.append(classe)
    return lc.features_matrix(seqs), np.array(y)


def test_ensemble_aprende_e_gera_probabilidades_validas():
    x, y = _dados_sinteticos()
    modelo = lc.GestureEnsemble(n_trees=30).fit(x, y)
    assert list(modelo.classes_) == [0, 1, 2]
    xt, yt = _dados_sinteticos(noise_offset=100)       # variações que não viu no treino
    proba = modelo.predict_proba(xt)
    assert proba.shape == (30, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (modelo.predict(xt) == yt).all()


def test_for_inference_evita_threads_por_frame():
    x, y = _dados_sinteticos()
    modelo = lc.GestureEnsemble(n_trees=10).fit(x, y)
    assert modelo.for_inference() is modelo
    assert modelo.rf.n_jobs == 1
