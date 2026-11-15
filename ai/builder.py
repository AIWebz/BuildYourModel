"""Model builder: Basis builds and trains small, separate AI models from your files.

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

KINDS = {"text": "text classifier", "number": "number predictor",
         "table": "table classifier", "lm": "tiny language model"}


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

    def fit(self, header, rows, epochs=600, seed=0):
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
        d, h = X.shape[1], 32
        self.p = {"A1": (rng.standard_normal((d, h)) * math.sqrt(2 / max(d, 1))).astype(np.float32),
                  "c1": np.zeros(h, np.float32),
                  "A2": (rng.standard_normal((h, out)) * math.sqrt(1 / h)).astype(np.float32),
                  "c2": np.zeros(out, np.float32)}
        opt = Adam(self.p, lr=0.01, wd=1e-4)
        for _ in range(epochs):
            z1 = X @ self.p["A1"] + self.p["c1"]
            a1 = np.maximum(z1, 0)
            o = a1 @ self.p["A2"] + self.p["c2"]
            if self.task == "number":
                dout = 2 * (o - Y) / len(X)
            else:
                e = np.exp(o - o.max(1, keepdims=True))
                dout = (e / e.sum(1, keepdims=True) - Y) / len(X)
            g = {"A2": a1.T @ dout, "c2": dout.sum(0)}
            dz1 = (dout @ self.p["A2"].T) * (z1 > 0)
            g["A1"], g["c1"] = X.T @ dz1, dz1.sum(0)
            opt.step(self.p, g)
        return self

    def predict(self, feature_rows):
        X = self._encode(feature_rows)
        o = np.maximum(X @ self.p["A1"] + self.p["c1"], 0) @ self.p["A2"] + self.p["c2"]
        if self.task == "number":
            return [(float(v) * self.y_std + self.y_mean, None) for v in o[:, 0]]
        e = np.exp(o - o.max(1, keepdims=True))
        pr = e / e.sum(1, keepdims=True)
        return [(self.classes[int(i)], float(pr[k, i])) for k, i in enumerate(pr.argmax(1))]

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def save(self, path):
        np.savez(path, **self.p)
        with open(path + ".json", "w") as f:
            json.dump({k: getattr(self, k) for k in
                       ("cols", "spec", "target", "task", "classes", "y_mean", "y_std")}, f)

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
    def __init__(self, model_dir):
        self.dir = os.path.join(model_dir, "custom")
        self.cache = {}

    def _paths(self, name):
        os.makedirs(self.dir, exist_ok=True)
        base = os.path.join(self.dir, name)
        return base, base + ".meta.json"

    # ---------------------------------------------------------------- build
    def build(self, name, path):
        if not re.fullmatch(r"[a-z0-9_\-]{1,40}", name):
            return "Model names can use lowercase letters, numbers, - and _."
        if not os.path.isfile(path):
            return f"I could not find the file {path}."
        try:
            if path.endswith(".txt"):
                return self._build_lm(name, path)
            header, rows = self._read_table(path)
        except (OSError, ValueError, KeyError, csv.Error, UnicodeDecodeError) as e:
            return f"I could not read that file: {e}"
        if len(rows) < 6:
            return "I need at least 6 rows of data."
        text_cls = len(header) == 2 and _num(rows[0][0]) is None and any(_num(r[1]) is None for r in rows)
        return self._build_text(name, rows) if text_cls else self._build_table(name, header, rows)

    @staticmethod
    def _read_table(path):
        if path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            return ["text", "label"], [[str(r["text"]), str(r["label"])] for r in data]
        with open(path, newline="") as f:
            rows = [[c.strip() for c in r] for r in csv.reader(f) if r and any(c.strip() for c in r)]
        if not rows or len(rows[0]) < 2:
            raise ValueError("the CSV needs at least 2 columns")
        width = len(rows[0])
        has_header = all(_num(c) is None for c in rows[0]) and (
            width > 2 or [c.lower() for c in rows[0]] == ["text", "label"])
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

    def _build_table(self, name, header, rows):
        report = ""
        if len(rows) >= 15:
            train, test = self._split(rows, 5)
            probe = TableModel().fit(header, train)
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
        model = TableModel().fit(header, rows)
        base, _ = self._paths(name)
        model.save(base + ".table.npz")
        usage = ", ".join(f"<{c}>" for c in model.cols)
        meta = self._save_meta(name, {"kind": model.task, "params": model.n_params(), "usage": usage},
                               [base + ".table.npz", base + ".table.npz.json"])
        self.cache[name] = model
        what = f"Predicts {model.target}" + (f" ({', '.join(model.classes)})" if model.classes else "")
        return self._done(name, meta, f"{what} from {', '.join(model.cols)}. {report}")

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
        vals = next(csv.reader([text]))
        if len(vals) != len(model.cols):
            return f"[{name}] give {len(model.cols)} comma-separated values: {', '.join(model.cols)}"
        value, conf = model.predict([[v.strip() for v in vals]])[0]
        if model.task == "number":
            return f"[{name}] predicted {model.target}: {_fmt(value)}"
        return f"[{name}] {model.target}: {value} (confidence {conf:.2f})"

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
