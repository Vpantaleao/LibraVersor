"""Qualidade dos dados gravados (dados_gestos/) e do modelo que está no repositório."""
import pickle

import numpy as np
import pytest

import libra_core as lc

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def dados(request):
    raiz = request.config.rootpath
    pasta = raiz / "dados_gestos"
    if not pasta.is_dir():
        pytest.skip("pasta dados_gestos/ não encontrada")
    saida = {}
    for classe in sorted(p.name for p in pasta.iterdir() if p.is_dir()):
        arquivos = sorted((pasta / classe).glob("*.npy"), key=lambda p: int(p.stem) if p.stem.isdigit() else 10**9)
        saida[classe] = [(a.name, np.load(a)) for a in arquivos]
    return saida


def test_todas_as_gravacoes_tem_o_formato_e_valores_validos(dados):
    for classe, gravacoes in dados.items():
        for nome, arr in gravacoes:
            assert arr.shape == (lc.SEQUENCE_LENGTH, 63), f"{classe}/{nome}: formato {arr.shape}"
            assert np.isfinite(arr).all(), f"{classe}/{nome}: tem NaN/infinito"


def test_ha_a_classe_de_repouso_e_gravacoes_suficientes(dados):
    assert any(c in lc.REST_LABELS for c in dados), "grave a classe 'repouso'"
    poucas = {c: len(g) for c, g in dados.items() if len(g) < 20}
    assert not poucas, f"classes com menos de 20 gravações: {poucas}"


def test_toda_classe_tem_nome_de_exibicao(dados):
    """Gesto novo sem nome cadastrado apareceria na legenda sem acento e seria falado sem acento."""
    sem_nome = [c for c in dados if c not in lc.DISPLAY_NAMES and c not in lc.REST_LABELS]
    assert not sem_nome, f"cadastre em DISPLAY_NAMES (libra_core.py): {sem_nome}"


def test_gravacoes_de_gestos_tem_mao_suficiente(dados):
    """Aviso (não falha): gravação de gesto com poucos frames de mão vira 'sem mão' ao vivo."""
    ruins = [f"{c}/{n} ({int((np.abs(a).sum(1) > 0).sum())} frames)"
             for c, g in dados.items() if c not in lc.REST_LABELS
             for n, a in g if (np.abs(a).sum(1) > 0).sum() < 12]
    if ruins:
        pytest.xfail(f"{len(ruins)} gravações de gestos com menos de 12 frames com mão "
                     f"(regrave ou apague): {', '.join(ruins)}")


def test_modelo_do_repositorio_carrega_e_e_compativel(root, dados):
    caminho = root / "modelo_gestos.p"
    if not caminho.exists():
        pytest.skip("modelo_gestos.p não encontrado")
    with open(caminho, "rb") as f:
        payload = pickle.load(f)
    assert payload["feature_version"] == lc.FEATURE_VERSION, "rode python treinar_modelo.py"
    assert payload["sequence_length"] == lc.SEQUENCE_LENGTH
    modelo, nomes = payload["model"], payload["labels_dict"]
    assert sorted(nomes.values()) == sorted(dados), "o modelo foi treinado com outro conjunto de gestos"
    assert sorted(int(c) for c in modelo.classes_) == sorted(nomes)
    qualquer = next(iter(dados.values()))[0][1]
    proba = modelo.predict_proba(lc.sequence_features(qualquer).reshape(1, -1))
    assert proba.shape == (1, len(nomes)) and np.isclose(proba.sum(), 1.0)
