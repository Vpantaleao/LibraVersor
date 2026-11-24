"""treinar_modelo.py: leitura dos dados, treino e arquivo salvo (com dados pequenos e sintéticos)."""
import pickle
import sys

import numpy as np
import pytest

import libra_core as lc
import treinar_modelo as tm
from helpers import make_sequence


def _grava(pasta, classes, n=9):
    for classe, shape_seed in classes.items():
        (pasta / classe).mkdir(parents=True)
        for i in range(n):
            np.save(pasta / classe / f"{i}.npy", make_sequence(shape_seed, noise_seed=i))


@pytest.fixture
def dados(tmp_path, monkeypatch):
    pasta = tmp_path / "dados"
    _grava(pasta, {"a": 11, "b": 22, "repouso": 33})
    monkeypatch.setattr(tm, "DATA_PATH", str(pasta))
    monkeypatch.setattr(tm, "MODEL_OUTPUT_PATH", str(tmp_path / "modelo.p"))
    monkeypatch.setattr(tm, "COPIES", 1)
    return pasta


def test_load_dataset_le_classes_e_ordena_numericamente(tmp_path, monkeypatch):
    pasta = tmp_path / "d"
    (pasta / "x").mkdir(parents=True)
    for n in (0, 1, 2, 10):                                  # "10" deve vir depois de "2"
        np.save(pasta / "x" / f"{n}.npy", np.full((lc.SEQUENCE_LENGTH, 63), float(n)))
    monkeypatch.setattr(tm, "DATA_PATH", str(pasta))
    seqs, labels, order, labels_dict = tm.load_dataset()
    assert [int(s[0, 0]) for s in seqs] == [0, 1, 2, 10]
    assert labels_dict == {0: "x"} and list(order) == sorted(order)
    assert order[0] == 0 and order[-1] == 1


def test_load_dataset_ignora_gravacao_com_formato_errado(tmp_path, monkeypatch, capsys):
    pasta = tmp_path / "d"
    (pasta / "x").mkdir(parents=True)
    np.save(pasta / "x" / "0.npy", make_sequence(1))
    np.save(pasta / "x" / "1.npy", np.zeros((10, 63)))        # quantidade de frames errada
    monkeypatch.setattr(tm, "DATA_PATH", str(pasta))
    seqs, *_ = tm.load_dataset()
    assert len(seqs) == 1
    assert "ignorando" in capsys.readouterr().out


def test_fit_model_aprende_os_gestos_sinteticos(dados):
    seqs, labels, _, _ = tm.load_dataset()
    modelo = tm.fit_model(seqs, labels)
    assert len(modelo.classes_) == 3
    assert (modelo.predict(lc.features_matrix(seqs)) == labels).mean() > 0.95


def test_main_rapido_salva_o_modelo_com_o_formato_esperado(dados, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["treinar_modelo.py", "--rapido"])
    tm.main()
    with open(tmp_path / "modelo.p", "rb") as f:
        payload = pickle.load(f)
    assert set(payload) == {"model", "labels_dict", "feature_version", "sequence_length"}
    assert payload["feature_version"] == lc.FEATURE_VERSION
    assert payload["sequence_length"] == lc.SEQUENCE_LENGTH
    assert payload["labels_dict"] == {0: "a", 1: "b", 2: "repouso"}
    assert payload["model"].predict_proba(lc.features_matrix([make_sequence(11)])).shape == (1, 3)


def test_main_com_validacao_mostra_a_acuracia(dados, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["treinar_modelo.py"])
    tm.main()
    assert "Acurácia na validação" in capsys.readouterr().out


def test_avisa_quando_nao_ha_classe_repouso(tmp_path, monkeypatch, capsys):
    pasta = tmp_path / "d"
    _grava(pasta, {"a": 11, "b": 22})
    monkeypatch.setattr(tm, "DATA_PATH", str(pasta))
    monkeypatch.setattr(tm, "MODEL_OUTPUT_PATH", str(tmp_path / "m.p"))
    monkeypatch.setattr(tm, "COPIES", 1)
    monkeypatch.setattr(sys, "argv", ["treinar_modelo.py", "--rapido"])
    tm.main()
    saida = capsys.readouterr().out
    assert "não há classe 'repouso'" in saida
    assert "poucas gravações" in saida                      # só 9 por gesto (< 20)
