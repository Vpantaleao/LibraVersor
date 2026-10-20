"""
Treina o modelo de gestos a partir de dados_gestos/<gesto>/<n>.npy

Uso:
    python treinar_modelo.py            # valida (usa o último terço das repetições) e treina
    python treinar_modelo.py --rapido   # pula a validação

A validação separa as ÚLTIMAS repetições de cada gesto (que o modelo não vê no
treino). É mais honesta do que sortear amostras, porque repetições gravadas em
sequência são muito parecidas entre si.
"""
import argparse
import os
import pickle
import time
from collections import Counter

import numpy as np
from sklearn.metrics import classification_report

import libra_core as lc

DATA_PATH = "dados_gestos"
MODEL_OUTPUT_PATH = "modelo_gestos.p"
COPIES = 6              # versões aumentadas de cada gravação (distância, inclinação, velocidade...)


def load_dataset():
    actions = sorted(d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d)))
    labels_dict, seqs, labels, order = {}, [], [], []
    for idx, action in enumerate(actions):
        labels_dict[idx] = action
        files = [f for f in os.listdir(os.path.join(DATA_PATH, action)) if f.endswith(".npy")]
        files.sort(key=lambda f: (0, int(f[:-4])) if f[:-4].isdigit() else (1, f))
        for pos, f in enumerate(files):
            arr = np.load(os.path.join(DATA_PATH, action, f))
            if arr.shape != (lc.SEQUENCE_LENGTH, 63):
                print(f"[aviso] ignorando {action}/{f}: formato {arr.shape}")
                continue
            seqs.append(arr)
            labels.append(idx)
            order.append(pos / max(len(files) - 1, 1))      # 0 = primeira gravação, 1 = última
    return np.array(seqs), np.array(labels), np.array(order), labels_dict


def fit_model(seqs, labels, seed=1):
    rng = np.random.default_rng(seed)
    xa, ya = lc.augment(seqs, labels, COPIES, rng)
    x = np.concatenate([seqs, xa])
    y = np.concatenate([labels, ya])
    return lc.GestureEnsemble(seed=seed).fit(lc.features_matrix(x), y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="pula a validação")
    args = ap.parse_args()

    t0 = time.time()
    seqs, labels, order, labels_dict = load_dataset()
    counts = Counter(labels)
    print(f"{len(seqs)} gravações | {len(labels_dict)} classes: {list(labels_dict.values())}")
    few = [labels_dict[i] for i, n in counts.items() if n < 20]
    if few:
        print(f"[aviso] poucas gravações (<20) em: {few} - grave mais repetições para melhorar")
    if not any(n in lc.REST_LABELS for n in labels_dict.values()):
        print("[aviso] não há classe 'repouso': grave uma para a mão parada não virar palavra")

    if not args.rapido:
        val = order > 2 / 3
        if val.sum() and (~val).sum():
            print(f"\nValidação: treino com {int((~val).sum())} gravações, "
                  f"teste com as {int(val.sum())} últimas (nunca vistas)...")
            m = fit_model(seqs[~val], labels[~val])
            pred = m.predict(lc.features_matrix(seqs[val]))
            acc = (pred == labels[val]).mean()
            print(f"Acurácia na validação: {acc * 100:.1f}%\n")
            names = [labels_dict[i] for i in sorted(labels_dict)]
            print(classification_report(labels[val], pred, labels=sorted(labels_dict),
                                        target_names=names, zero_division=0))

    print("Treinando o modelo final com todas as gravações...")
    model = fit_model(seqs, labels)
    payload = {
        "model": model,
        "labels_dict": labels_dict,
        "feature_version": lc.FEATURE_VERSION,
        "sequence_length": lc.SEQUENCE_LENGTH,
    }
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"Modelo salvo em '{MODEL_OUTPUT_PATH}' ({time.time() - t0:.0f}s).")


if __name__ == "__main__":
    main()
