"""Even more model types for Solen 1.0, from scratch in NumPy (+ stdlib).

    ChatbotModel     question -> answer bot from Q&A pairs (neural intent net + retriever)
    DocQA            answers questions from a document (TF-IDF passage retrieval)
    Summarizer       extractive summaries (TextRank over TF-IDF sentence graph)
    Translator       statistical translation (IBM Model 1 + target trigram LM, beam search)
    MiniGPT          a real decoder-only transformer, char-level, trained with hand-written backprop
    ImageGenerator   generates new images (PCA + Gaussian latent model, per class)
    AudioClassifier  classifies .wav sounds (FFT band features + neural net)
"""
import math
import os
import re
import time
import wave

import numpy as np

from .brain import UNKNOWN, Adam, Brain, Retriever
from .language_model import LanguageModel
from .model_zoo import ImageClassifier, Saveable, load_image
from .text import tokenize, tokenize_unicode

SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text, min_words=3):
    return [s.strip() for s in SENT_RE.split(text) if len(s.split()) >= min_words]


# ------------------------------------------------------------------ chatbot
class ChatbotModel(Saveable):
    kind = "chatbot"

    def fit(self, pairs):
        self.answers = sorted({a for _, a in pairs})
        idx = {a: i for i, a in enumerate(self.answers)}
        self.examples = [(q, f"a{idx[a]}") for q, a in pairs]
        self.brain = Brain.sized_for(self.examples)
        self.brain.train(self.examples, epochs=15, verbose=False, unknown=True)
        return self

    def n_params(self):
        return self.brain.n_params()

    def save(self, path):
        self.brain.save(path + ".npz")
        import json
        with open(path, "w") as f:
            json.dump({"answers": self.answers, "examples": self.examples}, f)

    @classmethod
    def load(cls, path):
        import json
        m = cls.__new__(cls)
        with open(path) as f:
            d = json.load(f)
        m.answers, m.examples = d["answers"], [tuple(e) for e in d["examples"]]
        m.brain = Brain().load(path + ".npz", m.examples)
        return m

    def reply(self, text):
        label, score, p, s = self.brain.predict(text, top=1)[0]
        if label == UNKNOWN or score < 0.3:
            return "I don't know the answer to that yet."
        return self.answers[int(label[1:])]


# ------------------------------------------------------------ document Q&A
class DocQA(Saveable):
    """Answers questions from a document: BM25 ranking over sentences and sentence pairs."""
    kind = "docqa"
    STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "and", "or", "what", "who",
            "how", "why", "when", "where", "which", "do", "does", "did", "it", "its", "that", "this", "for", "by",
            "with", "as", "be", "can", "you", "me", "tell", "about", "i", "my", "much", "many", "there"}

    @staticmethod
    def _stem(w):
        for suf in ("est", "ing", "ed", "er", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[:-len(suf)]
                if len(w) > 3 and w[-1] == w[-2]:
                    w = w[:-1]  # hottest -> hott -> hot
                break
        return w

    def _terms(self, text):
        from .text import canonical
        return [self._stem(w) for w in canonical(tokenize(text)) if w not in self.STOP]

    def fit(self, text):
        sents = sentences(text)
        self.passages = sents + [f"{a} {b}" for a, b in zip(sents, sents[1:])]
        self.docs = [self._terms(p) for p in self.passages]
        df = {}
        for d in self.docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.avg = sum(len(d) for d in self.docs) / max(n, 1)
        self.n_sents = len(sents)
        return self

    def n_params(self):
        return len(self.idf)

    def answer(self, question, k1=1.5, b=0.75):
        q = set(self._terms(question))
        if not q:
            return None, 0.0
        best, best_s = None, 0.0
        for p, d in zip(self.passages, self.docs):
            tf = {}
            for w in d:
                tf[w] = tf.get(w, 0) + 1
            s = sum(self.idf[w] * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / self.avg))
                    for w in q if w in tf)
            m = re.match(r"\s*(?:what|who)\s+(?:is|are|was|were)\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$", question.lower())
            if m and re.match(rf"(?:a |an |the )?{re.escape(m.group(1))}\b", p.lower()):
                s *= 1.6  # "what is X" -> prefer sentences that define X
            if s > best_s:
                best, best_s = p, s
        if best is None:
            return None, 0.0
        coverage = sum(1 for w in q if w in self._terms(best)) / len(q)
        return best, best_s * coverage


# --------------------------------------------------------------- summarizer
class Summarizer(Saveable):
    kind = "summarize"

    def fit(self, text):
        self.text = text
        self.sents = sentences(text)
        df = {}
        for s in self.sents:
            for w in set(tokenize(s)):
                df[w] = df.get(w, 0) + 1
        n = max(len(self.sents), 1)
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
        return self

    def n_params(self):
        return len(self.idf)

    def summarize(self, text=None, k=3):
        sents = sentences(text) if text else self.sents
        if len(sents) <= k:
            return " ".join(sents)
        vecs = []
        for s in sents:
            tf = {}
            for w in tokenize(s):
                tf[w] = tf.get(w, 0) + self.idf.get(w, 2.0)
            norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
            vecs.append({w: v / norm for w, v in tf.items()})
        n = len(sents)
        S = np.array([[sum(v * vecs[j].get(w, 0) for w, v in vecs[i].items()) if i != j else 0
                       for j in range(n)] for i in range(n)])
        S = S / np.maximum(S.sum(1, keepdims=True), 1e-9)
        r = np.ones(n) / n
        for _ in range(40):  # PageRank (TextRank)
            r = 0.15 / n + 0.85 * S.T @ r
        keep = sorted(np.argsort(-r)[:k])
        return " ".join(sents[i] for i in keep)


# --------------------------------------------------------------- translator
class Translator(Saveable):
    """IBM Model 1 word translation probabilities (EM) + target trigram LM + sentence-wide
    co-occurrence (for agreement such as el/la ... rojo/roja), decoded with beam search."""
    kind = "translate"

    def fit(self, pairs, iters=15):
        tok = tokenize_unicode
        self.memory = {" ".join(tok(s)): t for s, t in pairs}
        src = [tok(s) + ["<null>"] for s, _ in pairs]
        tgt = [tok(t) for _, t in pairs]
        tv = sorted({w for t in tgt for w in t})
        t_prob, uniform = {}, 1.0 / len(tv)
        for _ in range(iters):
            count, total = {}, {}
            for s, t in zip(src, tgt):
                for f in t:
                    z = sum(t_prob.get((f, e), uniform) for e in s)
                    for e in s:
                        c = t_prob.get((f, e), uniform) / z
                        count[(f, e)] = count.get((f, e), 0) + c
                        total[e] = total.get(e, 0) + c
            t_prob = {(f, e): c / total[e] for (f, e), c in count.items()}
        self.table = {}
        for (f, e), p in t_prob.items():
            self.table.setdefault(e, []).append((p, f))
        for e in self.table:
            self.table[e] = sorted(self.table[e], reverse=True)[:4]
        # co-occurrence of target words anywhere in the same sentence
        self.co, self.uni = {}, {}
        for t in tgt:
            for i, f in enumerate(t):
                self.uni[f] = self.uni.get(f, 0) + 1
                for w in t[:i]:
                    self.co[(w, f)] = self.co.get((w, f), 0) + 1
        self.lm = LanguageModel(unicode=True).train([t for _, t in pairs])
        self.n_pairs = len(pairs)
        return self

    def n_params(self):
        return sum(len(v) for v in self.table.values()) + len(self.lm.c3) + len(self.co)

    def _agree(self, words, f):
        if not words:
            return 0.0
        return sum(math.log((self.co.get((w, f), 0) + 0.1) / (self.uni.get(w, 0) + 0.1 * len(self.uni)))
                   for w in words) / len(words)

    def translate(self, text, beam=12):
        src = tokenize_unicode(text)
        if " ".join(src) in self.memory:
            return self.memory[" ".join(src)]

        def step(score, words, f, p):
            hist = (["<s>", "<s>"] + words)[-2:]
            return (score + math.log(p + 1e-9) + 0.7 * math.log(self.lm.prob(hist[0], hist[1], f))
                    + 0.5 * self._agree(words, f), words + [f])

        beams = [(0.0, [])]
        for w in src:
            options = self.table.get(w) or [(0.5, w)]  # unknown word: copy it through
            new = []
            for score, words in beams:
                for p, f in options:
                    new.append(step(score, words, f, p))
                # one source word may produce two target words (e.g. here -> está aquí)
                strong = [(p, f) for p, f in options if p > 0.25]
                for (p1, f1) in strong:
                    for (p2, f2) in strong:
                        if f1 != f2:
                            s1, w1 = step(score, words, f1, p1)
                            new.append(step(s1, w1, f2, p2))
                if "<null>" in self.table:  # or nothing (words like "do")
                    new.append((score + math.log(1e-3), words))
            beams = sorted(new, key=lambda b: -b[0])[:beam]
        end = lambda b: b[0] + 0.7 * math.log(self.lm.prob(*(["<s>", "<s>"] + b[1])[-2:], "</s>"))
        return " ".join(max(beams, key=end)[1])


# ----------------------------------------------------------------- MiniGPT
def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = (x - mu) * inv
    return xh * g + b, (xh, inv)


def _ln_back(dy, g, cache):
    xh, inv = cache
    dxh = dy * g
    N = xh.shape[-1]
    dx = inv / N * (N * dxh - dxh.sum(-1, keepdims=True) - xh * (dxh * xh).sum(-1, keepdims=True))
    return dx, (dy * xh).reshape(-1, N).sum(0), dy.reshape(-1, N).sum(0)


class MiniGPT(Saveable):
    """Decoder-only transformer (token + position embeddings, causal self-attention,
    MLP, residuals, LayerNorm) over characters, trained with hand-written backprop."""
    kind = "gpt"

    def _init(self, V, T, d, L, seed=0):
        rng = np.random.default_rng(seed)
        s = 0.02
        p = {"wte": rng.normal(0, s, (V, d)), "wpe": rng.normal(0, s, (T, d)),
             "gf": np.ones(d), "bf": np.zeros(d), "Wout": rng.normal(0, s, (d, V)), "bout": np.zeros(V)}
        for l in range(L):
            p.update({f"g1_{l}": np.ones(d), f"b1_{l}": np.zeros(d), f"g2_{l}": np.ones(d), f"b2_{l}": np.zeros(d),
                      f"Wq_{l}": rng.normal(0, s, (d, d)), f"Wk_{l}": rng.normal(0, s, (d, d)),
                      f"Wv_{l}": rng.normal(0, s, (d, d)), f"Wo_{l}": rng.normal(0, s / math.sqrt(2 * L), (d, d)),
                      f"M1_{l}": rng.normal(0, s, (d, 4 * d)), f"c1_{l}": np.zeros(4 * d),
                      f"M2_{l}": rng.normal(0, s / math.sqrt(2 * L), (4 * d, d)), f"c2_{l}": np.zeros(d)})
        self.p = {k: v.astype(np.float32) for k, v in p.items()}
        self.T, self.d, self.L = T, d, L

    def forward(self, X, Y=None):
        p, B, T = self.p, X.shape[0], X.shape[1]
        x = p["wte"][X] + p["wpe"][:T]
        mask = np.triu(np.full((T, T), -1e9, np.float32), 1)
        caches = []
        for l in range(self.L):
            h, c1 = _ln(x, p[f"g1_{l}"], p[f"b1_{l}"])
            q, k, v = h @ p[f"Wq_{l}"], h @ p[f"Wk_{l}"], h @ p[f"Wv_{l}"]
            att = q @ k.transpose(0, 2, 1) / math.sqrt(self.d) + mask
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            a = att @ v
            x = x + a @ p[f"Wo_{l}"]
            h2, c2 = _ln(x, p[f"g2_{l}"], p[f"b2_{l}"])
            z = h2 @ p[f"M1_{l}"] + p[f"c1_{l}"]
            m = np.maximum(z, 0)
            x = x + m @ p[f"M2_{l}"] + p[f"c2_{l}"]
            caches.append((h, c1, q, k, v, att, a, h2, c2, z, m))
        xf, cf = _ln(x, p["gf"], p["bf"])
        logits = xf @ p["Wout"] + p["bout"]
        if Y is None:
            return logits
        lg = logits - logits.max(-1, keepdims=True)
        prob = np.exp(lg)
        prob /= prob.sum(-1, keepdims=True)
        loss = -np.log(prob.reshape(-1, prob.shape[-1])[np.arange(B * T), Y.reshape(-1)] + 1e-9).mean()
        return loss, (X, caches, xf, cf, prob, Y)

    def backward(self, cache):
        X, caches, xf, cf, prob, Y = cache
        p, g = self.p, {}
        B, T, V = prob.shape
        dl = prob.copy()
        dl.reshape(-1, V)[np.arange(B * T), Y.reshape(-1)] -= 1
        dl /= B * T
        g["Wout"] = xf.reshape(-1, self.d).T @ dl.reshape(-1, V)
        g["bout"] = dl.reshape(-1, V).sum(0)
        dx, g["gf"], g["bf"] = _ln_back(dl @ p["Wout"].T, p["gf"], cf)
        for l in reversed(range(self.L)):
            h, c1, q, k, v, att, a, h2, c2, z, m = caches[l]
            flat = lambda t: t.reshape(-1, t.shape[-1])
            # MLP block
            g[f"M2_{l}"] = flat(m).T @ flat(dx)
            g[f"c2_{l}"] = flat(dx).sum(0)
            dz = (dx @ p[f"M2_{l}"].T) * (z > 0)
            g[f"M1_{l}"] = flat(h2).T @ flat(dz)
            g[f"c1_{l}"] = flat(dz).sum(0)
            dh2, g[f"g2_{l}"], g[f"b2_{l}"] = _ln_back(dz @ p[f"M1_{l}"].T, p[f"g2_{l}"], c2)
            dx = dx + dh2
            # attention block
            g[f"Wo_{l}"] = flat(a).T @ flat(dx)
            da = dx @ p[f"Wo_{l}"].T
            datt = da @ v.transpose(0, 2, 1)
            dv = att.transpose(0, 2, 1) @ da
            ds = att * (datt - (datt * att).sum(-1, keepdims=True)) / math.sqrt(self.d)
            dq = ds @ k
            dk = ds.transpose(0, 2, 1) @ q
            g[f"Wq_{l}"] = flat(h).T @ flat(dq)
            g[f"Wk_{l}"] = flat(h).T @ flat(dk)
            g[f"Wv_{l}"] = flat(h).T @ flat(dv)
            dh = dq @ p[f"Wq_{l}"].T + dk @ p[f"Wk_{l}"].T + dv @ p[f"Wv_{l}"].T
            dh1, g[f"g1_{l}"], g[f"b1_{l}"] = _ln_back(dh, p[f"g1_{l}"], c1)
            dx = dx + dh1
        g["wpe"] = np.zeros_like(p["wpe"])
        g["wpe"][:T] = dx.sum(0)
        gw = np.zeros_like(p["wte"])
        np.add.at(gw, X.reshape(-1), dx.reshape(-1, self.d))
        g["wte"] = gw
        return g

    def fit(self, text, T=32, d=48, L=2, steps=3000, batch=16, time_budget=60, seed=0):
        self.chars = sorted(set(text))
        self.ix = {c: i for i, c in enumerate(self.chars)}
        data = np.array([self.ix[c] for c in text])
        T = min(T, max(8, len(data) // 20))
        self._init(len(self.chars), T, d, L, seed)
        cut = int(len(data) * 0.9)
        train, val = data[:cut], data[cut:]
        rng = np.random.default_rng(seed)
        opt = Adam(self.p, lr=3e-3, wd=0)
        start, best, self.steps = time.time(), (float("inf"), None), 0
        for step in range(steps):
            pos = rng.integers(0, len(train) - T - 1, batch)
            Xb = np.stack([train[i:i + T] for i in pos])
            Yb = np.stack([train[i + 1:i + T + 1] for i in pos])
            loss, cache = self.forward(Xb, Yb)
            opt.step(self.p, self.backward(cache))
            self.steps = step + 1
            if step % 100 == 0 or step == steps - 1:
                v = self.val_loss(val)
                if v < best[0]:
                    best = (v, {k: a.copy() for k, a in self.p.items()})
                if time.time() - start > time_budget:
                    break
            if step == int(steps * 0.7):
                opt.lr *= 0.3
        for k in self.p:
            self.p[k][...] = best[1][k]
        self.ppl = float(math.exp(best[0]))
        self.train_seconds = time.time() - start
        return self

    def val_loss(self, val):
        T = self.T
        if len(val) <= T + 1:
            return float("inf")
        n = min(16, (len(val) - 1) // T)
        X = np.stack([val[i * T:(i + 1) * T] for i in range(n)])
        Y = np.stack([val[i * T + 1:(i + 1) * T + 1] for i in range(n)])
        return float(self.forward(X, Y)[0])

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def generate(self, prompt="", n=200, temperature=0.8, seed=None):
        rng = np.random.default_rng(seed)
        ids = [self.ix[c] for c in prompt if c in self.ix] or [self.ix.get(" ", 0)]
        out = prompt
        for _ in range(n):
            ctx = np.array([ids[-self.T:]])
            logits = self.forward(ctx)[0, -1] / temperature
            pr = np.exp(logits - logits.max())
            j = int(rng.choice(len(pr), p=pr / pr.sum()))
            ids.append(j)
            out += self.chars[j]
        return out


# ----------------------------------------------------------- image generator
class ImageGenerator(Saveable):
    """Learns the distribution of a set of images and samples new ones:
    PCA to a small latent space + a Gaussian over the latent codes (one per class)."""
    kind = "imagegen"
    SIZE = 16

    def fit(self, folder):
        groups = {}
        subdirs = [d for d in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, d))]
        for cls in (subdirs or [""]):
            d = os.path.join(folder, cls)
            imgs = [load_image(os.path.join(d, f)) for f in sorted(os.listdir(d))
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))]
            if len(imgs) >= 3:
                groups[cls or "all"] = np.array(imgs)
        if not groups:
            raise ValueError("I need at least 3 images (or sub-folders with 3+ images each).")
        self.models, self.explained = {}, {}
        for cls, X in groups.items():
            mean = X.mean(0)
            U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
            var = S ** 2 / max(len(X) - 1, 1)
            k = int(min(len(S), max(2, np.searchsorted(np.cumsum(var) / var.sum(), 0.9) + 1), 24))
            self.models[cls] = (mean, Vt[:k], np.sqrt(var[:k]))
            self.explained[cls] = float(var[:k].sum() / var.sum())
        self.classes = sorted(self.models)
        return self

    def n_params(self):
        return int(sum(m.size + c.size + s.size for m, c, s in self.models.values()))

    def sample(self, cls=None, seed=None):
        rng = np.random.default_rng(seed)
        cls = cls if cls in self.models else rng.choice(self.classes)
        mean, comps, std = self.models[cls]
        z = rng.normal(0, 1, len(std)) * std
        img = np.clip(mean + z @ comps, 0, 1).reshape(self.SIZE, self.SIZE)
        return cls, img

    @staticmethod
    def ascii(img):
        ramp = " .:-=+*#%@"
        return "\n".join("".join(ramp[int(v * 9.99)] * 2 for v in row) for row in img)

    @staticmethod
    def save_png(img, path, scale=8):
        from PIL import Image
        Image.fromarray((img * 255).astype(np.uint8)).resize(
            (img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST).save(path)


# --------------------------------------------------------- audio classifier
def read_wav(path, rate=8000, seconds=2.0):
    with wave.open(path, "rb") as w:
        n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(min(n, int(sr * seconds)))
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype).astype(np.float64)
    if sw == 1:
        x = x - 128
    x = x.reshape(-1, ch).mean(1) / (np.abs(x).max() or 1)
    if sr != rate:  # simple resample
        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * rate / sr)), np.arange(len(x)), x)
    return x, rate


def audio_features(x, rate, frame=256, hop=128, bands=16):
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    frames = np.stack([x[i:i + frame] * np.hanning(frame) for i in range(0, len(x) - frame + 1, hop)])
    mag = np.abs(np.fft.rfft(frames, axis=1))
    edges = np.unique(np.geomspace(1, mag.shape[1] - 1, bands + 1).astype(int))
    E = np.log1p(np.stack([mag[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])], 1))
    zcr = (np.diff(np.sign(frames), axis=1) != 0).mean(1)
    rms = np.sqrt((frames ** 2).mean(1))
    return np.concatenate([E.mean(0), E.std(0), [zcr.mean(), zcr.std(), rms.mean(), rms.std()]])


class AudioClassifier(Saveable):
    kind = "audio"

    @staticmethod
    def read_folder(folder):
        X, y = [], []
        for cls in sorted(os.listdir(folder)):
            d = os.path.join(folder, cls)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith(".wav"):
                        X.append(audio_features(*read_wav(os.path.join(d, f))))
                        y.append(cls)
        return X, y

    def fit(self, X, y):
        self.net = ImageClassifier().fit(X, y, epochs=300)
        self.classes = self.net.classes
        return self

    def n_params(self):
        return self.net.n_params()

    def predict_file(self, path):
        return self.net.predict_vec(audio_features(*read_wav(path)))
