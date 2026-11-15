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

KINDS = {"text": "text classifier", "number": "number predictor", "table": "table classifier",
         "lm": "tiny language model", "cluster": "clustering model", "anomaly": "anomaly detector",
         "forecast": "forecaster", "recommend": "recommender", "charnn": "neural character model",
         "image": "image classifier", "rl": "reinforcement learning agent"}
ZOO = {"cluster": ClusterModel, "anomaly": AnomalyModel, "forecast": Forecaster, "recommend": Recommender,
       "charnn": CharNN, "image": ImageClassifier, "rl": MazeAgent}


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
    KIND_NAMES = ["text", "number", "table", "lm", "charnn", "cluster", "anomaly", "forecast", "recommend",
                  "image", "rl"]
    def __init__(self, model_dir):
        self.dir = os.path.join(model_dir, "custom")
        self.cache = {}

    def _paths(self, name):
        os.makedirs(self.dir, exist_ok=True)
        base = os.path.join(self.dir, name)
        return base, base + ".meta.json"

    # ---------------------------------------------------------------- build
    def build(self, name, path, kind=None, target=None, k=None):
        if not re.fullmatch(r"[a-z0-9_\-]{1,40}", name):
            return "Model names can use lowercase letters, numbers, - and _."
        if os.path.isdir(path):
            kind = "image"
        elif not os.path.isfile(path):
            return f"I could not find the file {path}."
        try:
            if kind == "image":
                return self._build_image(name, path)
            if path.endswith(".txt") or kind in ("lm", "charnn", "rl"):
                with open(path, encoding="utf-8") as f:
                    text = f.read()
                if kind == "rl" or (kind is None and is_maze(text)):
                    return self._build_rl(name, text) if is_maze(text) else \
                        "A maze file needs # for walls, . for open cells, S for start and G for goal."
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
        if kind == "forecast":
            return self._build_forecast(name, header, rows)
        if kind == "recommend":
            return self._build_rec(name, rows)
        if kind in ("cluster", "anomaly"):
            return self._build_cluster(name, header, rows, kind, k)
        text_cls = len(header) == 2 and _num(rows[0][0]) is None and any(_num(r[1]) is None for r in rows)
        if kind == "text" or (kind in (None, "classify") and text_cls):
            return self._build_text(name, rows)
        return self._build_table(name, header, rows)

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
        meta = self._save_meta(name, meta, [base + ".zoo.pkl"])
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
        has_header = all(_num(c) is None for c in rows[0]) and (
            width > 2 or [c.lower() for c in rows[0]] == ["text", "label"]
            or (len(rows) > 1 and any(_num(c) is not None for c in rows[1])))
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
