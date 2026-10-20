"""
libra_core.py - núcleo compartilhado do LibraVersor.

Concentra tudo que o treino (treinar_modelo.py) e a execução (main.py) precisam
usar DE FORMA IDÊNTICA: extração de landmarks, features da sequência, aumento de
dados, o modelo (ensemble) e a lógica que transforma previsões em palavras.
Assim, o que foi treinado é exatamente o que roda ao vivo.
"""
import collections
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEQUENCE_LENGTH = 30        # frames por janela (~1 s)
FEATURE_VERSION = 2         # mude se alterar as features: força retreinar o modelo
ASPECT = 9 / 16             # y é normalizado pela altura e x pela largura (câmera 16:9)
REST_LABELS = {"repouso", "neutro"}   # classes de "mão parada": nunca viram palavra

# Como cada classe aparece na legenda e é falada pela voz (acentos corretos).
DISPLAY_NAMES = {
    "bom dia": "bom dia", "de cabeça": "de cabeça", "dor": "dor",
    "emergencia": "emergência", "febre": "febre", "gripe": "gripe",
    "muito": "muito", "não": "não", "obrigado": "obrigado", "oi": "oi",
    "pouco": "pouco", "remédio": "remédio", "sentindo": "sentindo",
    "sim": "sim", "vômito": "vômito",
}


def display_name(label):
    return DISPLAY_NAMES.get(label, label)


# --------------------------------------------------------------------------
# Landmarks
# --------------------------------------------------------------------------
def extract_landmarks(hand_landmarks):
    """21 pontos (x, y, z) relativos ao punho (ponto 0). Sem mão -> zeros."""
    if hand_landmarks is None:
        return np.zeros(63)
    lm = hand_landmarks.landmark
    bx, by, bz = lm[0].x, lm[0].y, lm[0].z
    coords = []
    for p in lm:
        coords.extend([p.x - bx, p.y - by, p.z - bz])
    return np.array(coords)


# --------------------------------------------------------------------------
# Features da sequência (30 x 63) -> vetor
# --------------------------------------------------------------------------
_IU = np.triu_indices(21, 1)


def sequence_features(seq):
    """
    Em vez de usar os 1890 números crus, resume a janela de forma que:
      - não dependa da distância da mão até a câmera (normaliza pelo tamanho da mão);
      - não dependa de em que momento da janela o gesto começou;
      - use distâncias entre pontos (não mudam se a mão inclinar ou for a outra mão).
    """
    seq = np.asarray(seq, dtype=float).reshape(-1, 21, 3).copy()
    valid = np.abs(seq).sum((1, 2)) > 0
    nv = int(valid.sum())
    if nv < 3:
        return np.zeros(FEATURE_DIM)
    seq[..., 1] *= ASPECT
    palm = np.linalg.norm(seq[valid][:, [5, 9, 13, 17]], axis=2).mean(1)
    pn = seq / (np.median(palm) + 1e-6)
    pn[~valid] = 0
    coords = pn.reshape(len(pn), 63)
    dist = np.linalg.norm(pn[:, :, None, :] - pn[:, None, :, :], axis=3)[:, _IU[0], _IU[1]]
    per_frame = np.concatenate([coords, dist], axis=1)               # (30, 273)
    fv = per_frame[valid]
    thirds = [per_frame[t].mean(0) if len(t) else np.zeros(per_frame.shape[1])
              for t in np.array_split(np.where(valid)[0], 3)]
    cv = coords[valid]
    vel = np.abs(np.diff(cv, axis=0)).mean(0) if nv > 1 else np.zeros(63)
    tips = pn[valid][:, [4, 8, 12, 16, 20]]
    path = np.linalg.norm(np.diff(tips, axis=0), axis=2).sum(0) if nv > 1 else np.zeros(5)
    return np.concatenate([fv.mean(0), fv.std(0), cv.min(0), cv.max(0), *thirds,
                           vel, path, [nv / len(seq)]])


FEATURE_DIM = len(sequence_features(np.random.default_rng(0).normal(size=(SEQUENCE_LENGTH, 63))))


def features_matrix(sequences):
    return np.array([sequence_features(s) for s in sequences])


# --------------------------------------------------------------------------
# Aumento de dados (só no treino): simula situações reais de uso
# --------------------------------------------------------------------------
def transform(seq, scale=1.0, rot=0.0, mirror=False, noise=0.0, shift=0,
              warp=1.0, drop=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    p = np.asarray(seq, dtype=float).reshape(len(seq), 21, 3).copy()
    valid = np.abs(p).sum((1, 2)) > 0
    p[..., 1] *= ASPECT
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    x, y = p[..., 0].copy(), p[..., 1].copy()
    p[..., 0] = c * x - s * y
    p[..., 1] = s * x + c * y
    if mirror:
        p[..., 0] *= -1
    p *= scale
    if noise > 0:
        p[valid] += rng.normal(0, noise, p[valid].shape)
    p[..., 1] /= ASPECT
    p[~valid] = 0
    out = p.reshape(len(seq), 63)
    if warp != 1.0:                                   # gesto mais rápido / mais lento
        n = len(out)
        t = np.clip(np.arange(n) * warp, 0, n - 1)
        out = np.stack([np.interp(t, np.arange(n), out[:, k]) for k in range(63)], 1)
    if shift:                                          # gesto começa antes/depois na janela
        n = len(out)
        o2 = np.empty_like(out)
        if shift > 0:
            o2[shift:], o2[:shift] = out[:n - shift], out[0]
        else:
            o2[:n + shift], o2[n + shift:] = out[-shift:], out[-1]
        out = o2
    if drop > 0:                                       # MediaPipe perdeu a mão em alguns frames
        out[rng.random(len(out)) < drop] = 0
    return out


def augment(sequences, labels, copies, rng, mirror_p=0.3):
    xa, ya = [], []
    for seq, lab in zip(sequences, labels):
        for _ in range(copies):
            xa.append(transform(
                seq, scale=rng.uniform(0.7, 1.4), rot=rng.uniform(-20, 20),
                mirror=rng.random() < mirror_p, noise=rng.uniform(0, 0.004),
                shift=int(rng.integers(-8, 9)), warp=rng.uniform(0.8, 1.25),
                drop=rng.uniform(0, 0.08), rng=rng))
            ya.append(lab)
    return np.array(xa), np.array(ya)


# --------------------------------------------------------------------------
# Modelo: Random Forest + SVM (média das probabilidades)
# --------------------------------------------------------------------------
class GestureEnsemble:
    def __init__(self, n_trees=300, seed=1):
        self.rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1)
        self.svm = make_pipeline(StandardScaler(),
                                 SVC(C=10, gamma="scale", probability=True, random_state=seed))
        self.classes_ = None

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.svm.fit(X, y)
        self.classes_ = self.rf.classes_
        return self

    def predict_proba(self, X):
        return (self.rf.predict_proba(X) + self.svm.predict_proba(X)) / 2

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def for_inference(self):
        self.rf.n_jobs = 1          # evita criar threads a cada frame
        return self


# --------------------------------------------------------------------------
# Previsões -> palavras (a parte que decide O QUE aparece na legenda)
# --------------------------------------------------------------------------
class StreamRecognizer:
    """
    Recebe um vetor de landmarks por frame e devolve, quando houver, uma palavra
    NOVA para a legenda. Regras principais:
      * "repouso" e mão fora da câmera nunca viram palavra (só rearmam o sistema);
      * uma palavra só é aceita se ficar estável por `stable_frames` frames,
        com confiança mínima e boa margem sobre a segunda colocada;
      * depois de aceitar uma palavra, há um intervalo (`refractory_frames`) em que a
        janela ainda contém o gesto anterior e nada novo é aceito;
      * a mesma palavra só se repete depois que a JANELA inteira volta a ser repouso
        (ou sem mão) - assim a cauda do gesto anterior não gera palavra repetida.
    """

    def __init__(self, model, labels_dict, conf_min=0.60, margin_min=0.15,
                 stable_frames=6, refractory_frames=15, rearm_frames=8,
                 min_hand_frames=12, stride=2):
        self.model, self.labels = model, labels_dict
        self.conf_min, self.margin_min = conf_min, margin_min
        self.stable_frames, self.refractory_frames = stable_frames, refractory_frames
        self.rearm_frames, self.min_hand_frames, self.stride = rearm_frames, min_hand_frames, stride
        self.window = collections.deque(maxlen=SEQUENCE_LENGTH)
        self.reset()

    def reset(self):
        self.window.clear()
        self.frame_i = 0
        self.run_label, self.run_len, self.misses = None, 0, 0
        self.last_word, self.armed = None, True
        self.rest_run = 0
        self.refractory_until = 0
        self.state, self.candidate, self.confidence = "aquecendo", None, 0.0

    def update(self, feats):
        """Retorna a palavra recém-confirmada (str) ou None."""
        self.window.append(feats)
        self.frame_i += 1
        if len(self.window) < SEQUENCE_LENGTH or self.frame_i % self.stride:
            return None

        w = np.array(self.window)
        if (np.abs(w).sum(1) > 0).sum() < self.min_hand_frames:
            self.state, self.candidate, self.confidence = "sem_mao", None, 0.0
            self._rest_step()
            self._drop_run()
            return None

        proba = self.model.predict_proba(sequence_features(w).reshape(1, -1))[0]
        order = np.argsort(proba)[::-1]
        top, second = proba[order[0]], proba[order[1]]
        label = self.labels[int(self.model.classes_[order[0]])] \
            if self.model.classes_ is not None else self.labels[int(order[0])]
        self.confidence = float(top)

        if label in REST_LABELS:
            self.state, self.candidate = "repouso", None
            self._rest_step()
            self._drop_run()
            return None
        self.rest_run = 0

        if top >= self.conf_min and (top - second) >= self.margin_min:
            self.state, self.candidate = "reconhecendo", label
            self.misses = 0
            if label == self.run_label:
                self.run_len += self.stride
            else:
                self.run_label, self.run_len = label, self.stride
        else:
            self.state, self.candidate = "analisando", None
            self.misses += 1
            if self.misses > 1:
                self._drop_run()
            return None

        if (self.run_len >= self.stable_frames and self.frame_i >= self.refractory_until
                and (label != self.last_word or self.armed)):
            self.last_word, self.armed = label, False
            self.refractory_until = self.frame_i + self.refractory_frames
            self._drop_run()
            return label
        return None

    def _rest_step(self):
        """Janela inteira em repouso/sem mão: depois de `rearm_frames`, libera repetir palavra."""
        self.rest_run += self.stride
        if self.rest_run >= self.rearm_frames:
            self.armed = True

    def _drop_run(self):
        self.run_label, self.run_len, self.misses = None, 0, 0


class CaptionBuffer:
    """Guarda as palavras da legenda e apaga tudo depois de `timeout` s sem novidade."""

    def __init__(self, timeout=7.0, max_words=12):
        self.timeout, self.max_words = timeout, max_words
        self.items = []                 # [(texto, instante)]

    def add(self, text, now):
        self.items.append((text, now))
        self.items = self.items[-self.max_words:]

    def prune(self, now):
        if self.items and now - self.items[-1][1] > self.timeout:
            self.items = []

    def clear(self):
        self.items = []

    def words(self):
        return [t for t, _ in self.items]

    def last_age(self, now):
        return now - self.items[-1][1] if self.items else None
