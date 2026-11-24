"""Teste lento (use --runslow): treina com os dados reais e confere precisão e fluxo ao vivo.

Treina com as 2/3 primeiras gravações de cada gesto e testa nas últimas, que o modelo não viu.
Serve para perceber se uma mudança no código ou nos dados piorou o reconhecimento.
"""
import numpy as np
import pytest

import libra_core as lc
import treinar_modelo as tm

pytestmark = [pytest.mark.slow, pytest.mark.filterwarnings("ignore::UserWarning")]


@pytest.fixture(scope="module")
def treinado(request):
    raiz = request.config.rootpath
    if not (raiz / "dados_gestos").is_dir():
        pytest.skip("pasta dados_gestos/ não encontrada")
    import os
    antigo = os.getcwd()
    os.chdir(raiz)
    try:
        seqs, labels, order, labels_dict = tm.load_dataset()
    finally:
        os.chdir(antigo)
    val = order > 2 / 3
    modelo = tm.fit_model(seqs[~val], labels[~val]).for_inference()
    return modelo, seqs, labels, val, labels_dict


def test_acuracia_nas_gravacoes_nao_vistas(treinado):
    modelo, seqs, labels, val, _ = treinado
    pred = modelo.predict(lc.features_matrix(seqs[val]))
    acuracia = (pred == labels[val]).mean()
    assert acuracia >= 0.90, f"acurácia caiu para {acuracia:.1%}"


def test_fluxo_ao_vivo_simulado_reconhece_sem_palavras_erradas(treinado):
    """Cola gravações não vistas com pausas sem mão e confere as palavras que sairiam na legenda."""
    modelo, seqs, labels, val, labels_dict = treinado
    rng = np.random.default_rng(7)
    repouso = [i for i, n in labels_dict.items() if n in lc.REST_LABELS]
    clipes = {c: [s for s, l, v in zip(seqs, labels, val)
                  if v and l == c and (np.abs(s).sum(1) > 0).sum() >= 12]   # ignora gravações sem mão
              for c in labels_dict}
    gestos = [c for c in labels_dict if c not in repouso and clipes[c]]

    quadros, eventos = [np.zeros((20, 63))], []
    posicao = 20
    for _ in range(40):
        classe = int(rng.choice(gestos))
        clipe = clipes[classe][rng.integers(len(clipes[classe]))]
        eventos.append((posicao, posicao + len(clipe), labels_dict[classe]))
        quadros.append(clipe)
        posicao += len(clipe)
        pausa = np.zeros((int(rng.integers(20, 41)), 63))
        quadros.append(pausa)
        posicao += len(pausa)
    fluxo = np.concatenate(quadros)

    rec = lc.StreamRecognizer(modelo, labels_dict)
    emitidas = [(f, w) for f, quadro in enumerate(fluxo) if (w := rec.update(quadro))]

    usadas, certas = set(), 0
    for inicio, fim, nome in eventos:
        for k, (f, palavra) in enumerate(emitidas):
            if k not in usadas and inicio <= f <= fim + 25 and palavra == nome:
                usadas.add(k)
                certas += 1
                break
    erradas = len(emitidas) - certas
    assert certas / len(eventos) >= 0.85, f"só {certas}/{len(eventos)} palavras reconhecidas"
    assert erradas <= 3, f"{erradas} palavras erradas ou repetidas na legenda"
