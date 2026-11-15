"""The connector: wires the Brain (understanding) to the Language Model (responding).

    user text
      -> tools (name memory, calculator, /teach, /generate)
      -> Brain.predict()         neural net + retriever pick an intent / fact
      -> LanguageModel.respond() picks & fills the defined response for it
      -> reply
"""
import ast
import json
import operator
import os
import re

from .brain import UNKNOWN, Brain
from .language_model import LanguageModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIDENCE = 0.35

UNKNOWN_RESPONSES = [
    "I do not know that yet. You can teach me with /teach question | answer.",
    "That is outside what I have learned so far. Teach me with /teach question | answer.",
]
MAYBE_RESPONSES = ["I am not completely sure, but this may help: {answer}"]
NO_NAME_RESPONSES = ["I do not know your name yet. Tell me by saying: my name is ..."]

_NAME_RE = re.compile(r"\b(?:my name is|call me|i am called|i am named|you can call me|i'm called)\s+([a-z][a-z'\-]*)", re.I)

# ---------------------------------------------------------------- calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos}
_WORD_OPS = [(r"\bmultiplied by\b", "*"), (r"\btimes\b", "*"), (r"\bdivided by\b", "/"),
             (r"\bover\b", "/"), (r"\bplus\b", "+"), (r"\bminus\b", "-"), (r"\bmod(?:ulo)?\b", "%"),
             (r"\bto the power of\b", "**"), (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"),
             (r"(?<=\d)\s*x\s*(?=\d)", "*"), (r"\^", "**")]


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculate(text):
    """Return (expression, result) if the text contains arithmetic, else None."""
    s = text.lower()
    for pat, rep in _WORD_OPS:
        s = re.sub(pat, rep, s)
    m = re.findall(r"[\d\.\s\+\-\*/%\(\)]+", s)
    expr = max(m, key=len).strip() if m else ""
    if not re.search(r"\d", expr) or not re.search(r"\d\s*(\*\*|[\+\-\*/%])\s*[\(\d\-]", expr):
        return None
    try:
        val = _eval(ast.parse(expr, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError):
        return None
    if isinstance(val, float):
        val = int(val) if val.is_integer() else round(val, 6)
    return expr, val


# ----------------------------------------------------------------- assistant
class Assistant:
    def __init__(self, data_dir=None, model_dir=None):
        self.data_dir = data_dir or os.path.join(ROOT, "data")
        self.model_dir = model_dir or os.path.join(ROOT, "models")
        self.learned_path = os.path.join(self.data_dir, "learned.json")
        self.memory = {}
        self.brain = Brain()
        self.lm = LanguageModel()
        self._load_data()

    # --------------------------------------------------------------- data
    def _load_data(self):
        with open(os.path.join(self.data_dir, "intents.json")) as f:
            intents = json.load(f)["intents"]
        with open(os.path.join(self.data_dir, "knowledge.json")) as f:
            facts = json.load(f)["facts"]
        self.examples, self.responses, self.corpus = [], {}, []
        for it in intents:
            self.examples += [(p, it["intent"]) for p in it["patterns"]]
            self.responses[it["intent"]] = it["responses"]
        self.topics = {}
        for i, fact in enumerate(facts):
            label = f"fact:{i}"
            self.examples += [(p, label) for p in fact["patterns"]]
            self.responses[label] = [fact["answer"]]
            self.topics[label] = fact["topic"]
            self.corpus.append(fact["answer"])
        self.responses[UNKNOWN] = UNKNOWN_RESPONSES
        self.responses["maybe"] = MAYBE_RESPONSES
        self.responses["no_name"] = NO_NAME_RESPONSES
        for templates in self.responses.values():
            self.corpus += [t.replace("{name_suffix}", "").replace("{name}", "friend")
                            .replace("{answer}", "") for t in templates]
        self.learned = []
        if os.path.exists(self.learned_path):
            with open(self.learned_path) as f:
                self.learned = json.load(f)

    # ------------------------------------------------------ train / load
    def train(self, verbose=True, epochs=12):
        if verbose:
            print(f"Training brain on {len(self.examples)} examples, "
                  f"{len(set(l for _, l in self.examples))} classes...")
        acc = self.brain.train(self.examples, epochs=epochs, verbose=verbose)
        self._train_lm()
        if verbose:
            print(f"Training language model on {len(self.corpus)} sentences, "
                  f"vocab {len(self.lm.vocab)}")
        os.makedirs(self.model_dir, exist_ok=True)
        self.brain.save(os.path.join(self.model_dir, "brain.npz"))
        self.lm.save(os.path.join(self.model_dir, "lm.pkl"))
        self._apply_learned()
        return acc

    def _train_lm(self):
        extra = [x["answer"] for x in self.learned]
        self.lm.train(self.corpus + extra, responses=self.responses)

    def load(self, verbose=False):
        bp = os.path.join(self.model_dir, "brain.npz")
        lp = os.path.join(self.model_dir, "lm.pkl")
        if not (os.path.exists(bp) and os.path.exists(lp)):
            self.train(verbose=verbose)
            return self
        self.brain.load(bp, self.examples)
        self.lm = LanguageModel.load(lp)
        self._apply_learned()
        return self

    def _apply_learned(self):
        for i, item in enumerate(self.learned):
            label = f"learned:{i}"
            self.brain.learn(item["question"], label)
            self.lm.responses[label] = [item["answer"]]
        if self.learned:
            self._train_lm()
            for i, item in enumerate(self.learned):
                self.lm.responses[f"learned:{i}"] = [item["answer"]]

    # -------------------------------------------------------------- teach
    def teach(self, question, answer):
        self.learned.append({"question": question.strip(), "answer": answer.strip()})
        with open(self.learned_path, "w") as f:
            json.dump(self.learned, f, indent=2)
        label = f"learned:{len(self.learned) - 1}"
        self.brain.learn(question, label)
        self._train_lm()
        for i, item in enumerate(self.learned):
            self.lm.responses[f"learned:{i}"] = [item["answer"]]
        return "Got it. I learned that."

    # -------------------------------------------------------------- reply
    def analyze(self, text):
        return self.brain.predict(text, top=3)

    def reply(self, text):
        text = (text or "").strip()
        if not text:
            return "Say something and I will answer."
        name = self.memory.get("name")

        # commands
        if text.startswith("/teach"):
            body = text[len("/teach"):]
            if "|" not in body:
                return "Use: /teach question | answer"
            q, a = body.split("|", 1)
            if not q.strip() or not a.strip():
                return "Use: /teach question | answer"
            return self.teach(q, a)
        if text.startswith("/generate"):
            return self.lm.generate(text[len("/generate"):].strip()) or "Give me a few words to start from."

        # tool: name memory
        m = _NAME_RE.search(text)
        if m:
            self.memory["name"] = m.group(1).capitalize()
            return self.lm.respond("set_name", name=self.memory["name"])

        # tool: calculator
        calc = calculate(text)
        if calc:
            shown = re.sub(r"\s*(\*\*|[\+\-\*/%])\s*", r" \1 ", calc[0]).replace(" ** ", "^")
            return f"{' '.join(shown.split())} = {calc[1]}"

        # brain -> language model
        top = self.brain.predict(text, top=3)
        label, score = top[0][0], top[0][1]
        if label == UNKNOWN or score < CONFIDENCE:
            alt = next((r for r in top if r[0] != UNKNOWN and r[0].startswith(("fact:", "learned:"))), None)
            if alt and alt[3] >= 0.3:
                return self.lm.respond("maybe", answer=self.lm.respond(alt[0]))
            return self.lm.respond(UNKNOWN)
        if label == "get_name" and not name:
            return self.lm.respond("no_name")
        if label == "set_name":  # recognized intent but no name extracted
            return "What is your name? Say: my name is ..."
        return self.lm.respond(label, name=name or "")
