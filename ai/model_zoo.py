"""More small model types Basis can build, all from scratch in NumPy.

    ClusterModel    k-means with automatic k (silhouette)        -> groups rows
    AnomalyModel    distance-to-centroid with learned threshold  -> flags unusual rows
    Forecaster      autoregressive ridge model on lagged values  -> predicts next values
    Recommender     item-item cosine collaborative filtering     -> suggests items
    CharNN          neural character-level language model (MLP)  -> writes new text
    ImageClassifier small neural net on 16x16 grayscale images   -> recognizes pictures
    MazeAgent       tabular Q-learning (reinforcement learning)  -> learns to solve a maze
"""
import math
import os
import pickle
from collections import deque

import numpy as np

from .brain import Adam


def _num(v):
    try:
        x = float(str(v).replace(",", ""))
        return x if math.isfinite(x) else None
    except ValueError:
        return None


class Saveable:
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        m = cls.__new__(cls)
        with open(path, "rb") as f:
            m.__dict__.update(pickle.load(f))
        return m

    def n_params(self):
        return int(sum(v.size for v in self.__dict__.values() if isinstance(v, np.ndarray)))


# ------------------------------------------------------------- table encoding
class Encoder:
    """Numeric columns -> standardized; category columns -> one-hot."""

    def __init__(self, cols, rows):
        self.cols, self.spec = list(cols), []
        for j in range(len(cols)):
            col = [r[j] for r in rows]
            nums = [_num(v) for v in col]
            if all(x is not None for x in nums):
                self.spec.append({"type": "num", "mean": float(np.mean(nums)), "std": float(np.std(nums)) or 1.0})
            else:
                self.spec.append({"type": "cat", "values": sorted({str(v).strip().lower() for v in col})})

    def encode(self, rows):
        out = []
        for r in rows:
            v = []
            for val, sp in zip(r, self.spec):
                if sp["type"] == "num":
                    x = _num(val)
                    v.append(((x if x is not None else sp["mean"]) - sp["mean"]) / sp["std"])
                else:
                    v.extend(1.0 if str(val).strip().lower() == c else 0.0 for c in sp["values"])
            out.append(v)
        return np.array(out, np.float64)

    def describe(self, vec):
        parts, i = [], 0
        for name, sp in zip(self.cols, self.spec):
            if sp["type"] == "num":
                x = vec[i] * sp["std"] + sp["mean"]
                parts.append(f"{name} ~{x:,.0f}" if abs(x) >= 100 else f"{name} ~{x:.3g}")
                i += 1
            else:
                k = len(sp["values"])
                parts.append(f"{name} mostly {sp['values'][int(np.argmax(vec[i:i + k]))]}")
                i += k
        return ", ".join(parts)


def kmeans(X, k, seed=0, iters=60, restarts=5):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        C = [X[rng.integers(len(X))]]
        for _ in range(1, k):  # k-means++ init
            d = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
            C.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
        C = np.array(C)
        for _ in range(iters):
            lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, C, lab)
    return best[1], best[2]


def silhouette(X, lab):
    if len(set(lab)) < 2:
        return -1.0
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    s = []
    for i in range(len(X)):
        same = lab == lab[i]
        a = D[i, same].sum() / max(same.sum() - 1, 1)
        b = min(D[i, lab == j].mean() for j in set(lab) if j != lab[i])
        s.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(s))


class ClusterModel(Saveable):
    kind = "cluster"

    def fit(self, header, rows, k=None):
        self.enc = Encoder(header, rows)
        X = self.enc.encode(rows)
        sample = X[:400]
        if k is None:
            scores = {}
            for kk in range(2, min(7, len(X))):
                C, lab = kmeans(sample, kk)
                scores[kk] = silhouette(sample, lab)
            k = max(scores, key=scores.get)
        self.centers, lab = kmeans(X, k)
        self.sizes = np.bincount(lab, minlength=k)
        self.score = silhouette(sample, lab[:len(sample)])
        return self

    def summary(self):
        return " | ".join(f"Group {j + 1} ({self.sizes[j]} rows): {self.enc.describe(c)}"
                          for j, c in enumerate(self.centers))

    def predict(self, row):
        x = self.enc.encode([row])[0]
        j = int(((self.centers - x) ** 2).sum(1).argmin())
        return j, self.enc.describe(self.centers[j])


class AnomalyModel(Saveable):
    kind = "anomaly"

    def fit(self, header, rows, quantile=97.5):
        self.cm = ClusterModel().fit(header, rows)
        X = self.cm.enc.encode(rows)
        d = np.sqrt(((X[:, None] - self.cm.centers[None]) ** 2).sum(-1)).min(1)
        self.threshold = float(np.percentile(d, quantile))
        self.flagged = int((d > self.threshold).sum())
        return self

    def n_params(self):
        return self.cm.n_params() + 1

    def predict(self, row):
        x = self.cm.enc.encode([row])[0]
        d = float(np.sqrt(((self.cm.centers - x) ** 2).sum(1)).min())
        return d > self.threshold, d


# ----------------------------------------------------------------- forecaster
class Forecaster(Saveable):
    kind = "forecast"

    def _design(self, z, w):
        X = np.array([z[i - w:i] for i in range(w, len(z))])
        return np.hstack([X, np.ones((len(X), 1))]), z[w:]

    def _fit_w(self, z, w, lam=0.1):
        X, y = self._design(z, w)
        A = X.T @ X + lam * np.eye(X.shape[1])
        return np.linalg.solve(A, X.T @ y)

    def fit(self, values, labels=None):
        v = np.array(values, np.float64)
        self.mean, self.std = float(v.mean()), float(v.std()) or 1.0
        z = (v - self.mean) / self.std
        cut = int(len(z) * 0.8)
        best = None
        for w in [w for w in (2, 3, 4, 6, 7, 12, 24) if w < cut // 2]:
            coef = self._fit_w(z[:cut], w)
            preds = [z[i - w:i] @ coef[:-1] + coef[-1] for i in range(cut, len(z))]
            err = float(np.mean(np.abs(np.array(preds) - z[cut:]))) * self.std
            if best is None or err < best[0]:
                best = (err, w)
        self.mae, self.w = best
        self.naive_mae = float(np.mean(np.abs(np.diff(v[cut - 1:]))))
        self.coef = self._fit_w(z, self.w)
        self.history = z[-self.w:].copy()
        self.last_label = labels[-1] if labels else None
        return self

    def predict(self, steps):
        hist, out = list(self.history), []
        for _ in range(steps):
            nxt = float(np.array(hist[-self.w:]) @ self.coef[:-1] + self.coef[-1])
            hist.append(nxt)
            out.append(nxt * self.std + self.mean)
        return out


# ---------------------------------------------------------------- recommender
class Recommender(Saveable):
    kind = "recommend"

    def fit(self, rows):
        self.users = sorted({r[0] for r in rows})
        self.items = sorted({r[1] for r in rows})
        ui, ii = {u: i for i, u in enumerate(self.users)}, {t: i for i, t in enumerate(self.items)}
        R = np.zeros((len(self.users), len(self.items)))
        for r in rows:
            R[ui[r[0]], ii[r[1]]] = _num(r[2]) if len(r) > 2 and _num(r[2]) is not None else 1.0
        self.R = R
        norm = np.linalg.norm(R, axis=0) + 1e-9
        self.S = (R.T @ R) / np.outer(norm, norm)
        np.fill_diagonal(self.S, 0)
        # leave-one-out hit rate @5 vs popularity
        rng = np.random.default_rng(0)
        hits = pop_hits = n = 0
        pop = (R > 0).sum(0)
        for u in range(len(self.users)):
            liked = np.flatnonzero(R[u])
            if len(liked) < 2:
                continue
            hide = rng.choice(liked)
            r = R[u].copy()
            r[hide] = 0
            scores = self.S @ r
            scores[r > 0] = -1
            hits += hide in np.argsort(-scores)[:5]
            p = pop.astype(float).copy()
            p[r > 0] = -1
            pop_hits += hide in np.argsort(-p)[:5]
            n += 1
        self.hit, self.pop_hit, self.n_eval = hits, pop_hits, n
        return self

    def similar(self, item, k=5):
        low = {t.lower(): i for i, t in enumerate(self.items)}
        if item.lower() not in low:
            return None
        j = low[item.lower()]
        return [self.items[i] for i in np.argsort(-self.S[j])[:k] if self.S[j, i] > 0]

    def for_user(self, user, k=5):
        if user not in self.users:
            return None
        r = self.R[self.users.index(user)]
        scores = self.S @ r
        scores[r > 0] = -1
        return [self.items[i] for i in np.argsort(-scores)[:k] if scores[i] > 0]


# ----------------------------------------------------- neural character model
class CharNN(Saveable):
    """Bengio-style neural LM over characters: embed last K chars -> tanh MLP -> softmax."""
    kind = "charnn"

    def fit(self, text, K=6, emb=12, hidden=64, steps=5000, seed=0):
        self.K = K
        self.chars = sorted(set(text))
        self.ix = {c: i for i, c in enumerate(self.chars)}
        data = np.array([self.ix[c] for c in text])
        cut = int(len(data) * 0.9)
        rng = np.random.default_rng(seed)
        V = len(self.chars)
        self.E = rng.standard_normal((V, emb)) * 0.1
        self.U1 = rng.standard_normal((K * emb, hidden)) / math.sqrt(K * emb)
        self.d1 = np.zeros(hidden)
        self.U2 = rng.standard_normal((hidden, V)) * 0.01
        self.d2 = np.zeros(V)
        names = ("E", "U1", "d1", "U2", "d2")
        params = {n: getattr(self, n) for n in names}
        opt = Adam(params, lr=0.003, wd=1e-4)
        train, val = data[:cut], data[cut - K:]
        best = (float("inf"), None)
        for step in range(steps):
            if step % 250 == 0:  # early stopping on held-out text
                v = self._nll(val)
                if v < best[0]:
                    best = (v, {n: params[n].copy() for n in names})
                elif step > 1500 and v > best[0] * 1.02:
                    break
            pos = rng.integers(K, len(train), 128)
            ctx = np.stack([train[p - K:p] for p in pos])
            y = train[pos]
            e = self.E[ctx].reshape(len(pos), -1)
            h = np.tanh(e @ self.U1 + self.d1)
            o = h @ self.U2 + self.d2
            p = np.exp(o - o.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            do = p.copy()
            do[np.arange(len(y)), y] -= 1
            do /= len(y)
            g = {"U2": h.T @ do, "d2": do.sum(0)}
            dh = (do @ self.U2.T) * (1 - h ** 2)
            g["U1"], g["d1"] = e.T @ dh, dh.sum(0)
            de = (dh @ self.U1.T).reshape(len(pos), K, -1)
            gE = np.zeros_like(self.E)
            np.add.at(gE, ctx, de)
            g["E"] = gE
            opt.step(params, g)
            if step == int(steps * 0.7):
                opt.lr *= 0.3
        if self._nll(val) > best[0]:
            for n in names:
                params[n][...] = best[1][n]
        test = val
        self.ppl = float(math.exp(self._nll(val)))
        self.unigram_ppl = float(math.exp(-np.mean(np.log(np.bincount(train, minlength=V)[test[K:]] / len(train) + 1e-12))))
        return self

    def _nll(self, seq):
        return float(-np.mean([np.log(self._probs(seq[i - self.K:i])[seq[i]] + 1e-12)
                               for i in range(self.K, len(seq))]))

    def _probs(self, ctx):
        e = self.E[ctx].reshape(-1)
        o = np.tanh(e @ self.U1 + self.d1) @ self.U2 + self.d2
        p = np.exp(o - o.max())
        return p / p.sum()

    def generate(self, prompt="", n=160, temperature=0.7, seed=None):
        rng = np.random.default_rng(seed)
        pad = self.chars.index(" ") if " " in self.chars else 0
        ids = [pad] * self.K + [self.ix[c] for c in prompt if c in self.ix]
        out = prompt
        for _ in range(n):
            p = self._probs(np.array(ids[-self.K:])) ** (1 / temperature)
            j = int(rng.choice(len(p), p=p / p.sum()))
            ids.append(j)
            out += self.chars[j]
        return out


# ----------------------------------------------------------- image classifier
SIZE = 16


def load_image(path):
    from PIL import Image  # optional dependency, only for image models
    img = Image.open(path).convert("L").resize((SIZE, SIZE))
    return np.asarray(img, np.float64).reshape(-1) / 255.0


class ImageClassifier(Saveable):
    kind = "image"

    @staticmethod
    def read_folder(folder):
        X, y = [], []
        for cls in sorted(os.listdir(folder)):
            d = os.path.join(folder, cls)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
                    X.append(load_image(os.path.join(d, f)))
                    y.append(cls)
        return X, y

    def fit(self, X, y, epochs=400, seed=0):
        self.classes = sorted(set(y))
        X = np.array(X)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xs = (X - self.mu) / self.sd
        Y = np.zeros((len(y), len(self.classes)))
        Y[np.arange(len(y)), [self.classes.index(c) for c in y]] = 1
        rng = np.random.default_rng(seed)
        h = 64
        self.A1 = rng.standard_normal((X.shape[1], h)) * math.sqrt(2 / X.shape[1])
        self.c1 = np.zeros(h)
        self.A2 = rng.standard_normal((h, len(self.classes))) * math.sqrt(1 / h)
        self.c2 = np.zeros(len(self.classes))
        params = {n: getattr(self, n) for n in ("A1", "c1", "A2", "c2")}
        opt = Adam(params, lr=0.003, wd=1e-3)
        for _ in range(epochs):
            Xa = Xs + rng.normal(0, 0.1, Xs.shape)  # noise augmentation
            z = Xa @ self.A1 + self.c1
            a = np.maximum(z, 0)
            o = a @ self.A2 + self.c2
            p = np.exp(o - o.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            do = (p - Y) / len(Y)
            g = {"A2": a.T @ do, "c2": do.sum(0)}
            dz = (do @ self.A2.T) * (z > 0)
            g["A1"], g["c1"] = Xa.T @ dz, dz.sum(0)
            opt.step(params, g)
        return self

    def n_params(self):
        return int(sum(getattr(self, n).size for n in ("A1", "c1", "A2", "c2")))

    def predict_vec(self, x):
        o = np.maximum(((x - self.mu) / self.sd) @ self.A1 + self.c1, 0) @ self.A2 + self.c2
        p = np.exp(o - o.max())
        p /= p.sum()
        return self.classes[int(p.argmax())], float(p.max())


# ------------------------------------------------------- reinforcement learning
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def is_maze(text):
    chars = set(text) - set("\n\r")
    return bool(chars) and chars <= set("#.SGX ") and "S" in text and "G" in text


class MazeAgent(Saveable):
    """Learns to reach G from S with tabular Q-learning. # = wall, X = trap."""
    kind = "rl"

    def fit(self, text, episodes=3000, seed=0):
        self.grid = [list(line) for line in text.strip("\n").splitlines()]
        H, W = len(self.grid), max(len(r) for r in self.grid)
        self.grid = [r + ["#"] * (W - len(r)) for r in self.grid]
        self.start = next((i, j) for i in range(H) for j in range(W) if self.grid[i][j] == "S")
        self.Q = np.zeros((H, W, 4))
        rng = np.random.default_rng(seed)
        alpha, gamma = 0.5, 0.95
        for ep in range(episodes):
            eps = max(0.05, 1 - ep / (episodes * 0.6))
            s = self.start
            for _ in range(H * W * 2):
                a = int(rng.integers(4)) if rng.random() < eps else int(self.Q[s].argmax())
                s2, r, done = self._step(s, a)
                target = r + (0 if done else gamma * self.Q[s2].max())
                self.Q[s][a] += alpha * (target - self.Q[s][a])
                s = s2
                if done:
                    break
        self.path = self.solve(self.start)
        self.optimal = self._bfs()
        return self

    def _step(self, s, a):
        i, j = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
        if not (0 <= i < len(self.grid) and 0 <= j < len(self.grid[0])) or self.grid[i][j] == "#":
            return s, -2.0, False
        c = self.grid[i][j]
        if c == "G":
            return (i, j), 50.0, True
        if c == "X":
            return (i, j), -30.0, True
        return (i, j), -1.0, False

    def solve(self, start):
        s, path = start, [start]
        for _ in range(len(self.grid) * len(self.grid[0])):
            s, _, done = self._step(s, int(self.Q[s].argmax()))
            path.append(s)
            if done:
                return path if self.grid[s[0]][s[1]] == "G" else None
        return None

    def _bfs(self):
        q, seen = deque([(self.start, 0)]), {self.start}
        while q:
            (i, j), d = q.popleft()
            if self.grid[i][j] == "G":
                return d
            for di, dj in MOVES:
                n = (i + di, j + dj)
                if (0 <= n[0] < len(self.grid) and 0 <= n[1] < len(self.grid[0]) and n not in seen
                        and self.grid[n[0]][n[1]] not in "#X"):
                    seen.add(n)
                    q.append((n, d + 1))
        return None

    def draw(self, path):
        g = [row[:] for row in self.grid]
        for (i, j) in path[1:-1]:
            g[i][j] = "*"
        return "\n".join("".join(r) for r in g)
