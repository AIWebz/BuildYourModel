"""The Brain: a NumPy neural network that understands questions.

Architecture (written from scratch, no ML frameworks):
    hashed text features (32k sparse dims: words, bigrams, char trigrams)
      -> embedding-bag layer (sparse x dense, 256 units) + ReLU + dropout
      -> hidden layer (128 units) + ReLU + dropout
      -> softmax over classes (defined intents + knowledge facts + 'unknown')
Trained with Adam, label smoothing, and heavy data augmentation.

A TF-IDF retriever runs alongside the network; the two scores are blended so
the brain stays robust on phrasings the network has not seen.
"""
import difflib
import math
import random

import numpy as np

from .text import N_FEATURES, augment, features, negative_examples, tokenize

UNKNOWN = "__unknown__"


def _pack(feat_dicts):
    """Turn a list of {idx: weight} dicts into padded (idx, weight) arrays."""
    L = max(1, max(len(f) for f in feat_dicts))
    idx = np.zeros((len(feat_dicts), L), dtype=np.int64)
    wts = np.zeros((len(feat_dicts), L), dtype=np.float32)
    for r, f in enumerate(feat_dicts):
        if f:
            k = list(f.keys())
            idx[r, :len(k)] = k
            wts[r, :len(k)] = list(f.values())
    return idx, wts


class NeuralNet:
    def __init__(self, n_classes, h1=256, h2=128, seed=0, n_features=N_FEATURES):
        rng = np.random.default_rng(seed)
        self.p = {
            "W1": (rng.standard_normal((n_features, h1)) * 0.1).astype(np.float32),
            "b1": np.zeros(h1, np.float32),
            "W2": (rng.standard_normal((h1, h2)) * math.sqrt(2 / h1)).astype(np.float32),
            "b2": np.zeros(h2, np.float32),
            "W3": (rng.standard_normal((h2, n_classes)) * math.sqrt(1 / h2)).astype(np.float32),
            "b3": np.zeros(n_classes, np.float32),
        }

    def forward(self, idx, wts, train=False, drop=0.2, rng=None):
        p = self.p
        e = (p["W1"][idx] * wts[..., None]).sum(1) + p["b1"]
        a1 = np.maximum(e, 0)
        m1 = (rng.random(a1.shape) > drop) / (1 - drop) if train else 1.0
        d1 = a1 * m1
        z2 = d1 @ p["W2"] + p["b2"]
        a2 = np.maximum(z2, 0)
        m2 = (rng.random(a2.shape) > drop) / (1 - drop) if train else 1.0
        d2 = a2 * m2
        logits = d2 @ p["W3"] + p["b3"]
        logits -= logits.max(1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(1, keepdims=True)
        cache = (idx, wts, e, m1, d1, z2, m2, d2)
        return probs, cache

    def backward(self, probs, target, cache):
        idx, wts, e, m1, d1, z2, m2, d2 = cache
        p, n = self.p, probs.shape[0]
        g = {}
        dlog = (probs - target) / n
        g["W3"] = d2.T @ dlog
        g["b3"] = dlog.sum(0)
        dz2 = (dlog @ p["W3"].T) * m2 * (z2 > 0)
        g["W2"] = d1.T @ dz2
        g["b2"] = dz2.sum(0)
        de = (dz2 @ p["W2"].T) * m1 * (e > 0)
        g["b1"] = de.sum(0)
        # sparse gradient for the embedding-bag layer
        rows = idx.reshape(-1)
        vals = (wts[..., None] * de[:, None, :]).reshape(-1, de.shape[1])
        order = np.argsort(rows, kind="stable")
        rows_s = rows[order]
        starts = np.flatnonzero(np.r_[True, rows_s[1:] != rows_s[:-1]])
        gW1 = np.add.reduceat(vals[order], starts, axis=0)
        g["W1"] = (rows_s[starts], gW1)
        return g


class Adam:
    def __init__(self, params, lr=0.005, b1=0.9, b2=0.999, eps=1e-8, wd=1e-5):
        self.lr, self.b1, self.b2, self.eps, self.wd, self.t = lr, b1, b2, eps, wd, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, g in grads.items():
            if k == "W1":  # lazy sparse Adam on touched rows only
                rows, g = g
                m, v, P = self.m[k], self.v[k], params[k]
                m[rows] = self.b1 * m[rows] + (1 - self.b1) * g
                v[rows] = self.b2 * v[rows] + (1 - self.b2) * g * g
                P[rows] -= self.lr * (m[rows] / c1) / (np.sqrt(v[rows] / c2) + self.eps)
            else:
                g = g + self.wd * params[k]
                self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
                self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
                params[k] -= self.lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)


class Retriever:
    """TF-IDF cosine retriever over the same hashed features."""

    def __init__(self, n=N_FEATURES):
        self.n = n
        self.docs, self.labels, self.idf = [], [], {}

    def fit(self, texts, labels):
        self.labels = list(labels)
        raw = [features(t, self.n) for t in texts]
        df = {}
        for f in raw:
            for k in f:
                df[k] = df.get(k, 0) + 1
        n = len(raw)
        self.idf = {k: math.log((1 + n) / (1 + c)) + 1 for k, c in df.items()}
        self.docs = [self._weight(f) for f in raw]

    def _weight(self, f):
        w = {k: v * self.idf.get(k, 1.0) for k, v in f.items()}
        norm = math.sqrt(sum(x * x for x in w.values())) or 1.0
        return {k: x / norm for k, x in w.items()}

    def add(self, text, label):
        self.docs.append(self._weight(features(text, self.n)))
        self.labels.append(label)

    def scores(self, text):
        q = self._weight(features(text, self.n))
        best = {}
        for d, lab in zip(self.docs, self.labels):
            s = sum(v * d.get(k, 0.0) for k, v in q.items())
            if s > best.get(lab, 0.0):
                best[lab] = s
        return best


class Brain:
    def __init__(self, n_features=N_FEATURES, h1=256, h2=128):
        self.n_features, self.h1, self.h2 = n_features, h1, h2
        self.classes, self.net, self.retriever = [], None, Retriever(n_features)

    @classmethod
    def sized_for(cls, examples):
        """A small brain sized to its data: hash table ~2x the distinct features."""
        distinct = set()
        for t, _ in examples:
            distinct.update(features(t))
        n = 1 << max(10, min(13, (2 * len(distinct) - 1).bit_length()))
        return cls(n_features=n, h1=64, h2=32)

    def n_params(self):
        return int(sum(v.size for v in self.net.p.values()))

    # ------------------------------------------------------------------ train
    def train(self, examples, epochs=12, batch=64, aug=12, seed=0, verbose=True, unknown=True):
        """examples: list of (text, label). Returns final training accuracy."""
        rng = random.Random(seed)
        nrng = np.random.default_rng(seed)
        self.classes = sorted({lab for _, lab in examples} | ({UNKNOWN} if unknown else set()))
        cid = {c: i for i, c in enumerate(self.classes)}
        data = []
        for text, lab in examples:
            data += [(t, lab) for t in augment(text, aug, rng)]
        if unknown:
            data += [(t, UNKNOWN) for t in negative_examples(len(examples) // 2, rng)]
        idx, wts = _pack([features(t, self.n_features) for t, _ in data])
        y = np.array([cid[lab] for _, lab in data])
        C = len(self.classes)
        smooth = 0.05
        target_all = np.full((len(y), C), smooth / C, np.float32)
        target_all[np.arange(len(y)), y] += 1 - smooth

        self.net = NeuralNet(C, self.h1, self.h2, seed=seed, n_features=self.n_features)
        opt = Adam(self.net.p)
        acc = 0.0
        for ep in range(epochs):
            order = nrng.permutation(len(y))
            loss, correct = 0.0, 0
            for s in range(0, len(order), batch):
                b = order[s:s + batch]
                probs, cache = self.net.forward(idx[b], wts[b], train=True, rng=nrng)
                tgt = target_all[b]
                loss += -(tgt * np.log(probs + 1e-9)).sum()
                correct += (probs.argmax(1) == y[b]).sum()
                opt.step(self.net.p, self.net.backward(probs, tgt, cache))
            opt.lr *= 0.93
            acc = correct / len(y)
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"  brain epoch {ep + 1:2d}/{epochs}  loss {loss / len(y):.4f}  acc {acc:.3f}")

        self._fit_side(examples)
        return acc

    def _fit_side(self, examples):
        self.retriever.fit([t for t, _ in examples], [lab for _, lab in examples])
        self.vocab = sorted({w for t, _ in examples for w in tokenize(t)})
        self._known = set(self.vocab)

    def learn(self, text, label):
        """Add a new fact at runtime (retrieval + spelling vocabulary)."""
        self.retriever.add(text, label)
        new = [w for w in tokenize(text) if w not in self._known]
        self._known.update(new)
        self.vocab = sorted(self._known)

    def correct(self, text):
        """Fix misspelled words by snapping them to the closest known word."""
        out = []
        for w in tokenize(text):
            if w not in self._known and len(w) >= 4 and not w.isdigit():
                m = difflib.get_close_matches(w, self.vocab, n=1, cutoff=0.81)
                w = m[0] if m else w
            out.append(w)
        return " ".join(out)

    # ---------------------------------------------------------------- predict
    def predict(self, text, top=3):
        """Blend network probability with retrieval similarity.
        Returns list of (label, score, net_prob, retrieval_sim)."""
        text = self.correct(text)
        idx, wts = _pack([features(text, self.n_features)])
        probs = self.net.forward(idx, wts)[0][0]
        sims = self.retriever.scores(text)
        out = []
        labels = set(self.classes) | set(sims)
        for lab in labels:
            p = float(probs[self.classes.index(lab)]) if lab in self.classes else 0.0
            s = sims.get(lab, 0.0)
            if lab == UNKNOWN:
                score = p
            elif lab in self.classes:
                score = 0.6 * p + 0.4 * s
            else:  # facts learned at runtime: retrieval only
                score = s
            out.append((lab, score, p, s))
        out.sort(key=lambda r: -r[1])
        return out[:top]

    # ------------------------------------------------------------ persistence
    def save(self, path):
        np.savez(path, classes=np.array(self.classes), dims=np.array([self.n_features, self.h1, self.h2]), **self.net.p)

    def save_compact(self, path, init_seed=0):
        """Smaller file for the browser: drop rows never touched in training, store float16."""
        init = (np.random.default_rng(init_seed).standard_normal((self.n_features, self.h1)) * 0.1).astype(np.float32)
        W1 = self.net.p["W1"]
        rows = np.flatnonzero(np.any(W1 != init, axis=1)).astype(np.int32)
        rest = {k: v.astype(np.float16) for k, v in self.net.p.items() if k != "W1"}
        np.savez_compressed(path, classes=np.array(self.classes), dims=np.array([self.n_features, self.h1, self.h2]),
                            W1_rows=rows, W1_vals=W1[rows].astype(np.float16), **rest)

    def load(self, path, examples):
        d = np.load(path, allow_pickle=False)
        self.classes = list(d["classes"])
        if "dims" in d.files:
            self.n_features, self.h1, self.h2 = (int(x) for x in d["dims"])
            self.retriever = Retriever(self.n_features)
        self.net = NeuralNet(len(self.classes), self.h1, self.h2, n_features=self.n_features)
        for k in self.net.p:
            if k == "W1" and "W1_rows" in d.files:  # compact format: only trained rows stored
                W1 = np.zeros((self.n_features, self.h1), np.float32)
                W1[d["W1_rows"]] = d["W1_vals"].astype(np.float32)
                self.net.p[k] = W1
            else:
                self.net.p[k] = d[k].astype(np.float32)
        self._fit_side(examples)
        return self
