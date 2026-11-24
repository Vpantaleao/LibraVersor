"""Dados sintéticos e dublês usados pelos testes (sem câmera, sem áudio, sem modelo treinado)."""
import time

import numpy as np


def hand_shape(shape_seed):
    """21 pontos (x, y, z) relativos ao punho, que fica na origem."""
    pts = np.random.default_rng(shape_seed).normal(size=(21, 3)) * 0.08
    pts[0] = 0.0
    return pts


def make_sequence(shape_seed, noise_seed=0, frames=30, motion=0.004):
    """Sequência (frames x 63): o mesmo formato de mão com pequena oscilação a cada frame."""
    rng = np.random.default_rng(10_000 + noise_seed)
    base = hand_shape(shape_seed)
    seq = []
    for _ in range(frames):
        p = base + rng.normal(size=(21, 3)) * motion
        p[0] = 0.0
        seq.append(p.reshape(63))
    return np.array(seq)


def hand_frame(shape_seed=1):
    return hand_shape(shape_seed).reshape(63)


NO_HAND = np.zeros(63)


class FixedModel:
    """Modelo falso: devolve sempre a distribuição de probabilidade configurada em `pick`.

    pick = (índice_da_classe, confiança)  ou  um vetor de probabilidades completo.
    """

    def __init__(self, n_classes=4, class_ids=None):
        self.n = n_classes
        self.classes_ = np.array(class_ids if class_ids is not None else range(n_classes))
        self.calls = 0
        self.pick = (0, 0.9)

    def predict_proba(self, X):
        self.calls += 1
        if isinstance(self.pick, tuple):
            idx, conf = self.pick
            p = np.full(self.n, (1 - conf) / (self.n - 1))
            p[idx] = conf
        else:
            p = np.asarray(self.pick, dtype=float)
        return p.reshape(1, -1)


def feed(rec, frame, n):
    """Alimenta o reconhecedor com `n` frames iguais e devolve as palavras confirmadas."""
    words = []
    for _ in range(n):
        w = rec.update(frame)
        if w:
            words.append(w)
    return words


def wait_for(cond, timeout=3.0, step=0.01):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(step)
    return bool(cond())
