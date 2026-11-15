"""Basis LM: a custom word-level language model with defined responses.

* Trigram model with interpolated Kneser-Ney smoothing, written from scratch.
* Trained on the knowledge base and the defined-response bank.
* Can score text (log-probability / perplexity) and generate new text.
* Holds the DEFINED RESPONSES for every intent. When the brain picks an intent,
  the LM chooses the most fluent defined response (avoiding repeats) and fills
  its slots ({name}, {answer}, ...). Responses are always defined, never random
  free text, unless /generate is used explicitly.
"""
import math
import pickle
import random
from collections import defaultdict

from .text import tokenize

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class LanguageModel:
    def __init__(self, discount=0.75):
        self.d = discount
        self.responses = {}  # intent -> list of templates
        self._last = {}      # intent -> last template used

    # ------------------------------------------------------------------ train
    def train(self, sentences, responses=None):
        if responses is not None:
            self.responses = {k: list(v) for k, v in responses.items()}
        c3 = defaultdict(int)
        for s in sentences:
            toks = [BOS, BOS] + tokenize(s) + [EOS]
            for i in range(2, len(toks)):
                c3[(toks[i - 2], toks[i - 1], toks[i])] += 1
        self._index(dict(c3))
        return self

    def _index(self, c3):
        """Build all Kneser-Ney count tables from raw trigram counts."""
        self.c3 = c3
        self.vocab = sorted(({w for k in c3 for w in k} - {BOS}) | {UNK})
        self.c12, self.n12 = defaultdict(int), defaultdict(int)      # c(u v .), N1+(u v .)
        self.cont2, self.cont2_ctx = defaultdict(int), defaultdict(int)  # N1+(. v w), N1+(. v .)
        self.n2, self.cont1 = defaultdict(int), defaultdict(int)     # N1+(v .), N1+(. w)
        self.trigram_next = defaultdict(set)
        for (u, v, w), c in c3.items():
            self.c12[(u, v)] += c
            self.n12[(u, v)] += 1
            self.cont2[(v, w)] += 1
            self.trigram_next[(u, v)].add(w)
        for (v, w), c in self.cont2.items():
            self.cont2_ctx[v] += c
            self.n2[v] += 1
            self.cont1[w] += 1
        self.cont1_total = sum(self.cont1.values()) or 1

    # ------------------------------------------------------------ probability
    def _p1(self, w):
        V = len(self.vocab)
        return 0.98 * self.cont1.get(w, 0) / self.cont1_total + 0.02 / V

    def _p2(self, v, w):
        ctx = self.cont2_ctx.get(v, 0)
        if ctx == 0:
            return self._p1(w)
        lam = self.d * self.n2[v] / ctx
        return max(self.cont2.get((v, w), 0) - self.d, 0) / ctx + lam * self._p1(w)

    def prob(self, u, v, w):
        ctx = self.c12.get((u, v), 0)
        if ctx == 0:
            return self._p2(v, w)
        lam = self.d * self.n12[(u, v)] / ctx
        return max(self.c3.get((u, v, w), 0) - self.d, 0) / ctx + lam * self._p2(v, w)

    def _norm(self, toks):
        known = set(self.vocab)
        return [t if t in known else UNK for t in toks]

    def logprob(self, text):
        toks = [BOS, BOS] + self._norm(tokenize(text)) + [EOS]
        return sum(math.log(self.prob(toks[i - 2], toks[i - 1], toks[i])) for i in range(2, len(toks)))

    def perplexity(self, text):
        n = len(tokenize(text)) + 1
        return math.exp(-self.logprob(text) / n)

    # ------------------------------------------------------------- generation
    def generate(self, prompt="", max_words=30, temperature=0.8, top_k=12, seed=None):
        rng = random.Random(seed)
        words = self._norm(tokenize(prompt))
        hist = [BOS, BOS] + words
        out = list(tokenize(prompt))
        candidates = [w for w in self.vocab if w != UNK]
        for _ in range(max_words):
            u, v = hist[-2], hist[-1]
            pool = self.trigram_next.get((u, v)) or candidates
            scored = sorted(((self.prob(u, v, w), w) for w in pool if w != UNK), reverse=True)[:top_k]
            weights = [p ** (1 / temperature) for p, _ in scored]
            w = rng.choices([w for _, w in scored], weights=weights)[0]
            if w == EOS:
                break
            out.append(w)
            hist.append(w)
        text = " ".join(out)
        return (text[:1].upper() + text[1:] + ".") if text else ""

    # ------------------------------------------------------ defined responses
    def respond(self, intent, **slots):
        """Pick the best defined response for an intent and fill its slots."""
        templates = self.responses.get(intent)
        if not templates:
            return None
        name = slots.get("name")
        slots.setdefault("name_suffix", f", {name}" if name else "")
        filled = []
        for t in templates:
            try:
                filled.append((t, t.format(**slots)))
            except (KeyError, IndexError):
                continue
        if not filled:
            return None
        last = self._last.get(intent)
        fresh = [f for f in filled if f[0] != last] or filled
        # rank by LM fluency (per-word log-prob) with a little variety
        best = max(fresh, key=lambda f: self.logprob(f[1]) / (len(tokenize(f[1])) + 1)
                   + random.uniform(0, 0.6))
        self._last[intent] = best[0]
        return best[1]

    def add_responses(self, intent, templates):
        self.responses.setdefault(intent, []).extend(templates)

    # ------------------------------------------------------------ persistence
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"d": self.d, "c3": self.c3, "responses": self.responses}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        lm = cls(state["d"])
        lm.responses = state["responses"]
        lm._index(state["c3"])
        return lm
