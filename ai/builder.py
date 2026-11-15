"""Model builder: Solen 1.0 builds and trains small, separate AI models from your files.

    .csv with 2 columns (text,label)        -> text classifier  (small neural net)
    .csv with 2+ columns, last column = target
        numeric target                      -> number predictor (regression MLP)
        category target                     -> table classifier (softmax MLP)
    .txt (any plain text)                   -> tiny language model (Kneser-Ney trigram)

Every model is sized to its data, scored on held-out examples it never trained
on, and saved to models/custom/ with its parameter count and file size.
"""
import csv
import json
import math
import os
import re

import numpy as np

from .brain import UNKNOWN, Adam, Brain
from .language_model import LanguageModel
from .model_zoo import (AnomalyModel, CharNN, ClusterModel, Forecaster, ImageClassifier, MazeAgent,
                        Recommender, is_maze, load_image)
from .model_zoo2 import (AudioClassifier, ChatbotModel, DocQA, ImageGenerator, MiniGPT, Summarizer, Translator,
                         sentences)

KINDS = {"text": "text classifier", "number": "number predictor", "table": "table classifier",
         "lm": "tiny language model", "cluster": "clustering model", "anomaly": "anomaly detector",
         "forecast": "forecaster", "recommend": "recommender", "charnn": "neural character model",
         "image": "image classifier", "rl": "reinforcement learning agent", "chatbot": "chatbot",
         "docqa": "document Q&A model", "summarize": "summarizer", "translate": "translator",
         "gpt": "transformer language model (mini GPT)", "imagegen": "image generator", "audio": "audio classifier"}
ZOO = {"cluster": ClusterModel, "anomaly": AnomalyModel, "forecast": Forecaster, "recommend": Recommender,
       "charnn": CharNN, "image": ImageClassifier, "rl": MazeAgent, "chatbot": ChatbotModel, "docqa": DocQA,
       "summarize": Summarizer, "translate": Translator, "gpt": MiniGPT, "imagegen": ImageGenerator,
       "audio": AudioClassifier}
TEXT_KINDS = ("lm", "charnn", "rl", "docqa", "summarize", "gpt")


def _num(v):
    try:
        x = float(v.replace(",", "")) if isinstance(v, str) else float(v)
        return x if math.isfinite(x) else None
    except ValueError:
        return None


def _size(n):
    return f"{n / 1024:.1f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.2f} MB"


def _fmt(x):
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


# --------------------------------------------------------------- table model
class TableModel:
    """Small MLP for tabular rows: numeric columns are standardized,
    category columns are one-hot encoded. Regression or classification."""

    def __init__(self):
        self.cols, self.spec, self.target, self.task = [], [], "", ""
        self.classes, self.y_mean, self.y_std, self.p = [], 0.0, 1.0, {}

    def _encode(self, rows):
        out = []
        for r in rows:
            v = []
            for val, sp in zip(r, self.spec):
                if sp["type"] == "num":
                    x = _num(val)
                    v.append(((x if x is not None else sp["mean"]) - sp["mean"]) / sp["std"])
                else:
                    v.extend(1.0 if val.strip().lower() == c else 0.0 for c in sp["values"])
            out.append(v)
        return np.array(out, np.float32)

    def fit(self, header, rows, epochs=600, seed=0, arch=None):
        self.cols, self.target = header[:-1], header[-1]
        feats = [r[:-1] for r in rows]
        ys = [r[-1] for r in rows]
        self.spec = []
        for j in range(len(self.cols)):
            col = [f[j] for f in feats]
            nums = [_num(v) for v in col]
            if all(x is not None for x in nums):
                m, sd = float(np.mean(nums)), float(np.std(nums)) or 1.0
                self.spec.append({"type": "num", "mean": m, "std": sd})
            else:
                self.spec.append({"type": "cat", "values": sorted({v.strip().lower() for v in col})})
        X = self._encode(feats)
        ynum = [_num(v) for v in ys]
        if all(y is not None for y in ynum):
            self.task = "number"
            y = np.array(ynum, np.float32)
            self.y_mean, self.y_std = float(y.mean()), float(y.std()) or 1.0
            Y = ((y - self.y_mean) / self.y_std)[:, None]
            out = 1
        else:
            self.task = "table"
            self.classes = sorted({v.strip() for v in ys})
            Y = np.zeros((len(ys), len(self.classes)), np.float32)
            Y[np.arange(len(ys)), [self.classes.index(v.strip()) for v in ys]] = 1
            out = len(self.classes)
        rng = np.random.default_rng(seed)
        arch = arch or {}
        self.act = arch.get("act", "relu")
        sizes = [X.shape[1]] + list(arch.get("layers") or [32]) + [out]
        self.p = {}
        for i in range(len(sizes) - 1):
            scale = math.sqrt((2 if self.act == "relu" else 1) / max(sizes[i], 1))
            self.p[f"A{i + 1}"] = (rng.standard_normal((sizes[i], sizes[i + 1])) * scale).astype(np.float32)
            self.p[f"c{i + 1}"] = np.zeros(sizes[i + 1], np.float32)
        opt = Adam(self.p, lr=float(arch.get("lr", 0.01)), wd=1e-4)
        for _ in range(int(arch.get("epochs", epochs))):
            o, cache = self._forward(X)
            if self.task == "number":
                dout = 2 * (o - Y) / len(X)
            else:
                e = np.exp(o - o.max(1, keepdims=True))
                dout = (e / e.sum(1, keepdims=True) - Y) / len(X)
            opt.step(self.p, self._backward(dout, cache))
        return self

    def _n_layers(self):
        return sum(1 for k in self.p if k.startswith("A"))

    def _act(self, z):
        act = getattr(self, "act", "relu")
        if act == "tanh":
            return np.tanh(z)
        if act == "sigmoid":
            return 1 / (1 + np.exp(-z))
        return np.maximum(z, 0)

    def _dact(self, z, a):
        act = getattr(self, "act", "relu")
        if act == "tanh":
            return 1 - a ** 2
        if act == "sigmoid":
            return a * (1 - a)
        return (z > 0).astype(z.dtype)

    def _forward(self, X):
        cache, a = [], X
        n = self._n_layers()
        for i in range(1, n + 1):
            z = a @ self.p[f"A{i}"] + self.p[f"c{i}"]
            cache.append((a, z))
            a = z if i == n else self._act(z)
        return a, cache

    def _backward(self, dout, cache):
        g, d = {}, dout
        n = len(cache)
        for i in range(n, 0, -1):
            a_in, z = cache[i - 1]
            g[f"A{i}"], g[f"c{i}"] = a_in.T @ d, d.sum(0)
            if i > 1:
                a_prev, z_prev = cache[i - 2][1], cache[i - 2][1]
                d = (d @ self.p[f"A{i}"].T) * self._dact(z_prev, self._act(z_prev))
        return g

    def predict(self, feature_rows):
        o, _ = self._forward(self._encode(feature_rows))
        if self.task == "number":
            return [(float(v) * self.y_std + self.y_mean, None) for v in o[:, 0]]
        e = np.exp(o - o.max(1, keepdims=True))
        pr = e / e.sum(1, keepdims=True)
        return [(self.classes[int(i)], float(pr[k, i])) for k, i in enumerate(pr.argmax(1))]

    def architecture(self):
        sizes = [self.p[f"A{i}"].shape[1] for i in range(1, self._n_layers())]
        return f"{len(sizes)} hidden layer{'s' if len(sizes) != 1 else ''} ({', '.join(map(str, sizes))} units, {getattr(self, 'act', 'relu')})"

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def save(self, path):
        np.savez(path, **self.p)
        with open(path + ".json", "w") as f:
            json.dump({k: getattr(self, k) for k in
                       ("cols", "spec", "target", "task", "classes", "y_mean", "y_std")} | {"act": getattr(self, "act", "relu")}, f)

    @classmethod
    def load(cls, path):
        m = cls()
        with open(path + ".json") as f:
            m.__dict__.update(json.load(f))
        d = np.load(path)
        m.p = {k: d[k] for k in d.files}
        return m


# ------------------------------------------------------------------ builder
class ModelBuilder:
    KIND_NAMES = ["text", "number", "table", "lm", "charnn", "cluster", "anomaly", "forecast", "recommend",
                  "image", "rl", "chatbot", "docqa", "summarize", "translate", "gpt", "imagegen", "audio"]
    def __init__(self, model_dir):
        self.dir = os.path.join(model_dir, "custom")
        self.cache = {}

    def _paths(self, name):
        os.makedirs(self.dir, exist_ok=True)
        base = os.path.join(self.dir, name)
        return base, base + ".meta.json"

    # ---------------------------------------------------------------- build
    def build(self, name, path, kind=None, target=None, k=None, arch=None):
        if not re.fullmatch(r"[a-z0-9_\-]{1,40}", name):
            return "Model names can use lowercase letters, numbers, - and _."
        if os.path.isdir(path):
            has_wav = any(f.lower().endswith(".wav") for _, _, fs in os.walk(path) for f in fs)
            if kind not in ("imagegen", "audio"):
                kind = "audio" if has_wav else "image"
        elif not os.path.isfile(path):
            return f"I could not find the file {path}."
        elif kind in ("image", "imagegen", "audio"):
            return "That model type needs a folder (or a ZIP) with one sub-folder of files per class."
        try:
            if kind == "image":
                return self._build_image(name, path)
            if kind == "imagegen":
                return self._build_imagegen(name, path)
            if kind == "audio":
                return self._build_audio(name, path)
            if path.endswith(".txt") or kind in TEXT_KINDS:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
                if kind == "rl" or (kind is None and is_maze(text)):
                    return self._build_rl(name, text) if is_maze(text) else \
                        "A maze file needs # for walls, . for open cells, S for start and G for goal."
                if kind == "docqa":
                    return self._build_docqa(name, text)
                if kind == "summarize":
                    return self._build_summarizer(name, text)
                if kind == "gpt":
                    return self._build_gpt(name, text)
                return self._build_charnn(name, text) if kind == "charnn" else self._build_lm(name, path)
            header, rows = self._read_table(path)
            if target:
                header, rows = self._move_target(header, rows, target)
        except (OSError, ValueError, KeyError, csv.Error, UnicodeDecodeError, ImportError) as e:
            return f"I could not build that: {e}"
        if len(rows) < 6:
            return "I need at least 6 rows of data."
        low = [h.lower() for h in header]
        if kind is None and any("user" in h for h in low[:2]) and \
                any(w in h for h in low[:2] for w in ("item", "product", "movie", "song", "book")):
            kind = "recommend"
        if kind is None and len(header) >= 2 and "question" in low[0] and "answer" in low[1]:
            kind = "chatbot"
        langs = ("english", "spanish", "french", "german", "italian", "portuguese", "source", "target", "src", "tgt")
        if kind is None and len(header) == 2 and all(any(w in h for w in langs) for h in low):
            kind = "translate"
        if kind == "chatbot":
            return self._build_chatbot(name, rows)
        if kind == "translate":
            return self._build_translator(name, header, rows)
        if kind == "forecast":
            return self._build_forecast(name, header, rows)
        if kind == "recommend":
            return self._build_rec(name, rows)
        if kind in ("cluster", "anomaly"):
            return self._build_cluster(name, header, rows, kind, k)
        text_cls = len(header) == 2 and _num(rows[0][0]) is None and any(_num(r[1]) is None for r in rows)
        if kind == "text" or (kind in (None, "classify") and text_cls):
            return self._build_text(name, rows)
        return self._build_table(name, header, rows, arch)

    @staticmethod
    def _move_target(header, rows, target):
        """Move the column the user wants predicted to the end. target may be a phrase."""
        low = [h.lower() for h in header]
        words = [w for w in re.findall(r"[a-z_]+", target.lower()) if len(w) > 2]
        j = None
        for w in words + [w.rstrip("s") for w in words]:
            j = next((i for i, h in enumerate(low) if h == w or w in h or (len(h) > 2 and h in w)), None)
            if j is not None:
                break
        if j is None:
            return header, rows
        order = [i for i in range(len(header)) if i != j] + [j]
        return [header[i] for i in order], [[r[i] for i in order] for r in rows]

    # ------------------------------------------------------- new model types
    def _save_zoo(self, name, model, kind, usage, extra_meta=None):
        base, _ = self._paths(name)
        model.save(base + ".zoo.pkl")
        meta = {"kind": kind, "params": model.n_params(), "usage": usage, **(extra_meta or {})}
        files = [f for f in (base + ".zoo.pkl", base + ".zoo.pkl.npz") if os.path.exists(f)]
        meta = self._save_meta(name, meta, files)
        self.cache[name] = model
        return meta

    def _build_cluster(self, name, header, rows, kind, k):
        if kind == "anomaly":
            m = AnomalyModel().fit(header, rows)
            meta = self._save_zoo(name, m, kind, ", ".join(f"<{c}>" for c in header))
            return self._done(name, meta, f"Learned what normal rows look like ({len(m.cm.centers)} typical "
                                          f"patterns). It flags rows beyond the 97.5th percentile of distance; "
                                          f"{m.flagged} of {len(rows)} training rows were flagged.")
        m = ClusterModel().fit(header, rows, k=k)
        meta = self._save_zoo(name, m, kind, ", ".join(f"<{c}>" for c in header))
        return self._done(name, meta, f"Found {len(m.centers)} groups (silhouette score {m.score:.2f}, "
                                      f"1.0 is perfect separation). {m.summary()}.")

    def _build_forecast(self, name, header, rows):
        vals = [_num(r[-1]) for r in rows]
        if any(v is None for v in vals) or len(vals) < 12:
            return "A forecaster needs at least 12 rows with a number in the last column."
        labels = [r[0] for r in rows] if len(header) > 1 else None
        m = Forecaster().fit(vals, labels)
        meta = self._save_zoo(name, m, "forecast", "<how many steps ahead>")
        return self._done(name, meta, f"Uses the last {m.w} values of {header[-1]} to predict the next one. "
                                      f"Held-out average error {_fmt(m.mae)} (repeating the last value would "
                                      f"be off by {_fmt(m.naive_mae)}).")

    def _build_rec(self, name, rows):
        m = Recommender().fit(rows)
        meta = self._save_zoo(name, m, "recommend", "<item>   or   /ask " + name + " user <user id>")
        return self._done(name, meta, f"Learned from {len(m.users)} users and {len(m.items)} items. "
                                      f"Held-out hit rate in top 5: {m.hit}/{m.n_eval} "
                                      f"(recommending popular items: {m.pop_hit}/{m.n_eval}).")

    def _build_charnn(self, name, text):
        if len(text) < 500:
            return "A neural character model needs at least 500 characters of text."
        m = CharNN().fit(text)
        meta = self._save_zoo(name, m, "charnn", "<starting text>")
        return self._done(name, meta, f"A neural network that writes one character at a time ({len(m.chars)} "
                                      f"characters known). Held-out perplexity {m.ppl:.1f} per character "
                                      f"(guessing by letter frequency: {m.unigram_ppl:.1f}).")

    def _build_image(self, name, folder):
        X, y = ImageClassifier.read_folder(folder)
        classes = sorted(set(y))
        if len(classes) < 2 or len(X) < 6:
            return "An image model needs a folder with one sub-folder of pictures per class (2+ classes)."
        idx = list(range(len(X)))
        train, test = self._split(idx, 5)
        probe = ImageClassifier().fit([X[i] for i in train], [y[i] for i in train])
        hits = sum(probe.predict_vec(X[i])[0] == y[i] for i in test)
        m = ImageClassifier().fit(X, y)
        meta = self._save_zoo(name, m, "image", "<path to an image>")
        return self._done(name, meta, f"Classes: {', '.join(classes)} ({len(X)} images). "
                                      f"Held-out accuracy {hits}/{len(test)}.")

    def _build_rl(self, name, text):
        m = MazeAgent().fit(text)
        meta = self._save_zoo(name, m, "rl", "(no input)   or   /ask " + name + " <row>,<col> to start elsewhere")
        if m.optimal is None:
            return self._done(name, meta, "There is no safe route from S to G in this maze (walls or traps block it).")
        if m.path is None:
            return self._done(name, meta, "It trained but has not found a path to G yet.")
        return self._done(name, meta, f"Learned by trial and error over 3,000 games. Its route takes "
                                      f"{len(m.path) - 1} steps (shortest possible: {m.optimal}).\n{m.draw(m.path)}\n")

    @staticmethod
    def _read_table(path):
        if path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            return ["text", "label"], [[str(r["text"]), str(r["label"])] for r in data]
        with open(path, newline="") as f:
            rows = [[c.strip() for c in r] for r in csv.reader(f) if r and any(c.strip() for c in r)]
        if not rows:
            raise ValueError("the file is empty")
        width = len(rows[0])
        def looks_like_names(row):  # short identifiers that never appear again in their column
            return all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_ ]{0,24}", c) and len(c.split()) <= 3 and
                       all(r[j] != c for r in rows[1:] if len(r) > j) for j, c in enumerate(row))
        has_header = all(_num(c) is None for c in rows[0]) and (
            width > 2 or [c.lower() for c in rows[0]] == ["text", "label"]
            or (len(rows) > 1 and any(_num(c) is not None for c in rows[1])) or looks_like_names(rows[0]))
        header = rows[0] if has_header else [f"col{i + 1}" for i in range(width - 1)] + ["target"]
        body = [r for r in (rows[1:] if has_header else rows) if len(r) == width]
        return header, body

    @staticmethod
    def _split(items, every):
        train = [x for i, x in enumerate(items) if i % every]
        test = [x for i, x in enumerate(items) if not i % every]
        return train, test

    def _save_meta(self, name, meta, files):
        base, meta_path = self._paths(name)
        meta["bytes"] = sum(os.path.getsize(f) for f in files)
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        return meta

    def _done(self, name, meta, extra):
        return (f"Built {KINDS[meta['kind']]} '{name}': {meta['params']:,} parameters, "
                f"{_size(meta['bytes'])} on disk. {extra} Use it with /ask {name} {meta['usage']}")

    def _build_text(self, name, rows):
        rows = [(r[0], r[1]) for r in rows]
        labels = sorted({lab for _, lab in rows})
        if len(labels) < 2:
            return "I need at least 2 different labels."
        report = ""
        if len(rows) >= 15:
            train, test = self._split(rows, 5)
            probe = Brain.sized_for(train)
            probe.train(train, epochs=15, verbose=False, unknown=False)
            hits = sum(probe.predict(t, top=1)[0][0] == lab for t, lab in test)
            report = f"Held-out accuracy {hits}/{len(test)}."
        model = Brain.sized_for(rows)
        model.train(rows, epochs=15, verbose=False, unknown=False)
        base, _ = self._paths(name)
        model.save(base + ".npz")
        with open(base + ".rows.json", "w") as f:
            json.dump(rows, f)
        meta = self._save_meta(name, {"kind": "text", "params": model.n_params(), "usage": "<text>",
                                      "labels": labels}, [base + ".npz", base + ".rows.json"])
        self.cache[name] = model
        return self._done(name, meta, f"Labels: {', '.join(labels)}. {report}")

    def _build_table(self, name, header, rows, arch=None):
        report = ""
        if len(rows) >= 15:
            train, test = self._split(rows, 5)
            probe = TableModel().fit(header, train, arch=arch)
            preds = probe.predict([r[:-1] for r in test])
            if probe.task == "number":
                truth = [_num(r[-1]) for r in test]
                mae = float(np.mean([abs(p - t) for (p, _), t in zip(preds, truth)]))
                base_mae = float(np.mean([abs(np.mean([_num(r[-1]) for r in train]) - t) for t in truth]))
                report = (f"Held-out average error {_fmt(mae)} "
                          f"(guessing the average would be off by {_fmt(base_mae)}).")
            else:
                hits = sum(p == r[-1].strip() for (p, _), r in zip(preds, test))
                report = f"Held-out accuracy {hits}/{len(test)}."
        model = TableModel().fit(header, rows, arch=arch)
        base, _ = self._paths(name)
        model.save(base + ".table.npz")
        usage = ", ".join(f"<{c}>" for c in model.cols)
        meta = self._save_meta(name, {"kind": model.task, "params": model.n_params(), "usage": usage},
                               [base + ".table.npz", base + ".table.npz.json"])
        self.cache[name] = model
        what = f"Predicts {model.target}" + (f" ({', '.join(model.classes)})" if model.classes else "")
        return self._done(name, meta, f"{what} from {', '.join(model.cols)}, using a neural network with "
                                      f"{model.architecture()}. {report}")

    # ---------------------------------------------------- more model types
    def _build_chatbot(self, name, rows):
        pairs = [(r[0], r[1]) for r in rows if len(r) >= 2 and r[0] and r[1]]
        m = ChatbotModel().fit(pairs)
        meta = self._save_zoo(name, m, "chatbot", "<your question>")
        return self._done(name, meta, f"Learned {len(m.answers)} answers from {len(pairs)} example questions, "
                                      f"and says when it does not know.")

    def _build_docqa(self, name, text):
        m = DocQA().fit(text)
        if m.n_sents < 3:
            return "I need a document with at least 3 sentences."
        meta = self._save_zoo(name, m, "docqa", "<question about the document>")
        return self._done(name, meta, f"Indexed {m.n_sents} sentences. It answers with the most relevant passage.")

    def _build_summarizer(self, name, text):
        m = Summarizer().fit(text)
        if len(m.sents) < 4:
            return "I need a document with at least 4 sentences."
        meta = self._save_zoo(name, m, "summarize", "<number of sentences, or text to summarize>")
        return self._done(name, meta, f"Learned word importance from {len(m.sents)} sentences. "
                                      f"Summary: {m.summarize(k=2)}")

    def _build_translator(self, name, header, rows):
        pairs = [(r[0], r[1]) for r in rows if len(r) >= 2 and r[0] and r[1]]
        report = ""
        if len(pairs) >= 20:
            train, test = self._split(pairs, 5)
            probe = Translator().fit(train)
            exact = sum(probe.translate(s).lower().split() == t.lower().split() for s, t in test)
            report = f"Held-out sentences translated exactly right: {exact}/{len(test)}."
        m = Translator().fit(pairs)
        meta = self._save_zoo(name, m, "translate", "<sentence to translate>")
        return self._done(name, meta, f"Translates {header[0]} to {header[1]}, learned from {len(pairs)} sentence "
                                      f"pairs (IBM Model 1 word alignment + a {header[1]} language model). {report}")

    def _build_gpt(self, name, text):
        if len(text) < 1000:
            return "A transformer needs at least 1,000 characters of text."
        m = MiniGPT().fit(text)
        V = len(m.chars)
        meta = self._save_zoo(name, m, "gpt", "<starting text>")
        return self._done(name, meta, f"A decoder-only transformer ({m.L} layers, {m.d}-dim attention, {m.T}-character "
                                      f"context), trained for {m.steps} steps in {m.train_seconds:.0f}s. Held-out "
                                      f"perplexity {m.ppl:.1f} per character (random guessing: {V}).")

    def _build_imagegen(self, name, folder):
        m = ImageGenerator().fit(folder)
        meta = self._save_zoo(name, m, "imagegen", "<" + " or ".join(m.classes[:4]) + ">")
        info = ", ".join(f"{c} ({m.explained[c]:.0%} of variation captured)" for c in m.classes)
        return self._done(name, meta, f"Learned to generate new 16x16 images for: {info}.")

    def _build_audio(self, name, folder):
        X, y = AudioClassifier.read_folder(folder)
        classes = sorted(set(y))
        if len(classes) < 2 or len(X) < 6:
            return "An audio model needs a folder with one sub-folder of .wav files per class (2+ classes)."
        train, test = self._split(list(range(len(X))), 5)
        probe = AudioClassifier().fit([X[i] for i in train], [y[i] for i in train])
        hits = sum(probe.net.predict_vec(X[i])[0] == y[i] for i in test)
        m = AudioClassifier().fit(X, y)
        meta = self._save_zoo(name, m, "audio", "<path to a .wav file>")
        return self._done(name, meta, f"Classes: {', '.join(classes)} ({len(X)} sounds). Held-out accuracy {hits}/{len(test)}.")

    def _build_lm(self, name, path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.split()) >= 3]
        if len(sents) < 10:
            return "I need at least 10 sentences of text to build a language model."
        train, test = self._split(sents, 10)
        probe = LanguageModel().train(train)
        ppl = math.exp(-sum(probe.logprob(s) for s in test) /
                       sum(len(s.split()) + 1 for s in test))
        model = LanguageModel().train(sents, responses={})
        base, _ = self._paths(name)
        model.save(base + ".lm.pkl")
        meta = self._save_meta(name, {"kind": "lm", "params": len(model.c3), "usage": "<starting words>"},
                               [base + ".lm.pkl"])
        self.cache[name] = model
        return self._done(name, meta, f"Learned {len(model.vocab):,} words from {len(sents)} sentences. "
                                      f"Held-out perplexity {ppl:.0f} (lower is better).")

    # ------------------------------------------------------------------ use
    def _meta(self, name):
        _, meta_path = self._paths(name)
        if not os.path.exists(meta_path):
            return None
        with open(meta_path) as f:
            return json.load(f)

    def _load(self, name, meta):
        if name in self.cache:
            return self.cache[name]
        base, _ = self._paths(name)
        if meta["kind"] == "text":
            with open(base + ".rows.json") as f:
                rows = [tuple(r) for r in json.load(f)]
            m = Brain().load(base + ".npz", rows)
        elif meta["kind"] == "lm":
            m = LanguageModel.load(base + ".lm.pkl")
        elif meta["kind"] in ZOO:
            m = ZOO[meta["kind"]].load(base + ".zoo.pkl")
        else:
            m = TableModel.load(base + ".table.npz")
        self.cache[name] = m
        return m

    def ask(self, name, text):
        meta = self._meta(name)
        if meta is None:
            return f"There is no model named '{name}'. Build one with /build {name} <file>"
        model = self._load(name, meta)
        if meta["kind"] == "text":
            label, score, _, _ = model.predict(text, top=1)[0]
            return f"[{name}] {label} (confidence {score:.2f})" if label != UNKNOWN else f"[{name}] not sure"
        if meta["kind"] == "lm":
            return f"[{name}] {model.generate(text, max_words=25)}"
        if meta["kind"] in ZOO:
            return f"[{name}] " + self._ask_zoo(meta["kind"], model, text.strip())
        vals = next(csv.reader([text]))
        if len(vals) != len(model.cols):
            return f"[{name}] give {len(model.cols)} comma-separated values: {', '.join(model.cols)}"
        value, conf = model.predict([[v.strip() for v in vals]])[0]
        if model.task == "number":
            return f"[{name}] predicted {model.target}: {_fmt(value)}"
        return f"[{name}] {model.target}: {value} (confidence {conf:.2f})"

    @staticmethod
    def _ask_zoo(kind, m, text):
        if kind == "chatbot":
            return m.reply(text)
        if kind == "docqa":
            ans, score = m.answer(text)
            return ans if ans and score >= 1.0 else "I could not find that in the document."
        if kind == "summarize":
            if not text or _num(text) is not None:
                return m.summarize(k=int(_num(text) or 3))
            return m.summarize(text, k=2) if len(sentences(text)) > 2 else "Give me a few sentences to summarize, or a number."
        if kind == "translate":
            return m.translate(text) if text else "Give me a sentence to translate."
        if kind == "gpt":
            return m.generate(text)
        if kind == "imagegen":
            import tempfile
            cls, img = m.sample(text.strip().lower() or None)
            path = os.path.join(tempfile.gettempdir(), f"solen_gen_{os.getpid()}_{np.random.randint(1e9)}.png")
            note = ""
            try:
                m.save_png(img, path)
                note = f"\n[image] {path}"
            except ImportError:
                pass
            return f"new {cls}:\n{m.ascii(img)}{note}"
        if kind == "audio":
            if not os.path.isfile(text):
                return f"I could not find the sound file {text}"
            label, conf = m.predict_file(text)
            return f"{label} (confidence {conf:.2f})"
        if kind == "forecast":
            n = int(_num(text) or 3) if text else 3
            n = max(1, min(n, 60))
            return f"next {n}: " + ", ".join(_fmt(v) for v in m.predict(n))
        if kind == "recommend":
            if text.lower().startswith("user "):
                recs = m.for_user(text[5:].strip())
                return "unknown user" if recs is None else ("recommended: " + ", ".join(recs) if recs else "nothing new")
            recs = m.similar(text)
            return f"unknown item. Known items: {', '.join(m.items[:12])}" if recs is None else \
                "people who liked that also liked: " + ", ".join(recs)
        if kind == "charnn":
            return m.generate(text)
        if kind == "image":
            if not os.path.isfile(text):
                return f"I could not find the image {text}"
            label, conf = m.predict_vec(load_image(text))
            return f"{label} (confidence {conf:.2f})"
        if kind == "rl":
            start = m.start
            if text:
                parts = [int(_num(x) or 0) for x in text.split(",")[:2]]
                if len(parts) == 2:
                    start = tuple(parts)
            path = m.solve(start)
            return "no route found from there" if path is None else f"route in {len(path) - 1} steps:\n{m.draw(path)}"
        vals = next(csv.reader([text]))
        cols = m.cm.enc.cols if kind == "anomaly" else m.enc.cols
        if len(vals) != len(cols):
            return f"give {len(cols)} comma-separated values: {', '.join(cols)}"
        vals = [v.strip() for v in vals]
        if kind == "anomaly":
            bad, d = m.predict(vals)
            return f"{'UNUSUAL' if bad else 'normal'} (distance {d:.2f}, threshold {m.threshold:.2f})"
        j, desc = m.predict(vals)
        return f"group {j + 1}: {desc}"

    def models(self):
        """Structured list of built models (for the website)."""
        os.makedirs(self.dir, exist_ok=True)
        out = []
        for f in sorted(os.listdir(self.dir)):
            if f.endswith(".meta.json"):
                n = f[:-10]
                m = self._meta(n)
                out.append({"name": n, "kind": m["kind"], "type": KINDS[m["kind"]], "params": m["params"],
                            "size": _size(m["bytes"]), "usage": m["usage"],
                            "built": os.path.getmtime(os.path.join(self.dir, f))})
        return out

    def list(self):
        os.makedirs(self.dir, exist_ok=True)
        names = sorted(f[:-10] for f in os.listdir(self.dir) if f.endswith(".meta.json"))
        if not names:
            return "I have not built any models yet. Use /build name file.csv or file.txt"
        lines = []
        for n in names:
            m = self._meta(n)
            lines.append(f"  {n}: {KINDS[m['kind']]}, {m['params']:,} parameters, {_size(m['bytes'])}")
        return "Models I have built:\n" + "\n".join(lines)
