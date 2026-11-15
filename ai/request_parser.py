"""Understands plain-English requests for a new AI model.

    "make me an AI that predicts house prices from data/examples/houses.csv"
    -> {kind: "number", target: "price", path: ".../houses.csv", name: "houses"}

It finds: is this a build request? which model type? which file? which column to
predict? how many groups? what to call it? Anything missing is reported so the
assistant can ask for it.
"""
import os
import re

WORD_NUMS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}

BUILD_VERB = re.compile(r"\b(build|make|create|train|develop|design|code|give me|i want|i need|i'd like|"
                        r"can you make|could you make|set up|teach an?)\b")
MODEL_NOUN = re.compile(r"\b(model|ai|a\.i\.|classifier|predictor|bot|network|agent|recommender|detector|"
                        r"forecaster|generator|system|clusterer|neural net|machine learning|program|tool)s?\b")
QUESTION = re.compile(r"^\s*(how|what|why|when|where|which|who|explain|tell me about|is|are|does|do)\b")
PATH_RE = re.compile(r"""(?:"([^"]+)"|'([^']+)'|([\w./\\~-]+\.(?:csv|txt|json)))""", re.I)

# keyword -> weight, per model kind
KIND_WORDS = {
    "image": ["image", "images", "photo", "photos", "picture", "pictures", "pics", "vision", "png", "jpg",
              "shapes", "recognize shapes", "see"],
    "forecast": ["forecast", "forecasting", "future", "time series", "next month", "next week", "next year",
                 "over time", "trend", "upcoming", "predict next", "next values", "predicts the next"],
    "recommend": ["recommend", "recommends", "recommendation", "recommendations", "recommender", "suggest",
                  "suggests", "people who liked", "similar items", "what to watch", "what to buy"],
    "cluster": ["group", "groups", "cluster", "clusters", "clustering", "segment", "segments", "segmentation",
                "without labels", "find patterns"],
    "anomaly": ["anomaly", "anomalies", "unusual", "outlier", "outliers", "fraud", "abnormal", "strange",
                "weird", "suspicious"],
    "rl": ["maze", "game", "games", "reinforcement", "agent", "navigate", "escape", "path finding", "play"],
    "charnn": ["neural text", "character", "characters", "invent", "new names", "names", "neural language",
               "rnn", "letter by letter"],
    "lm": ["write", "writes", "writing", "generate text", "text generator", "stories", "story", "poems",
           "language model", "autocomplete", "sentences", "in the style"],
    "number": ["price", "prices", "how much", "cost", "amount", "value", "estimate", "regression",
               "number", "score", "salary", "temperature"],
    "classify": ["classify", "classifier", "classifies", "categorize", "sort", "sorts", "detect", "spam",
                 "sentiment", "positive", "negative", "label", "labels", "which type", "identify", "tell if",
                 "tells if", "tell whether", "category", "categories"],
}

DESCRIBE = {
    "text": "a text classifier", "table": "a table classifier", "number": "a number predictor",
    "classify": "a classifier", "lm": "a tiny language model", "charnn": "a neural character model",
    "cluster": "a clustering model", "anomaly": "an anomaly detector", "forecast": "a forecaster",
    "recommend": "a recommender", "image": "an image classifier", "rl": "a reinforcement learning agent",
}

NEEDS = {
    "classify": "a CSV with text,label columns (or a table whose last column is the category)",
    "text": "a CSV with text,label columns",
    "table": "a CSV table whose last column is the category to predict",
    "number": "a CSV table whose last column is the number to predict",
    "lm": "a .txt file of example text",
    "charnn": "a .txt file of example text (500+ characters)",
    "cluster": "a CSV table of rows to group (with a header row)",
    "anomaly": "a CSV table of normal-looking rows (with a header row)",
    "forecast": "a CSV with a date column and a number column, in time order (12+ rows)",
    "recommend": "a CSV with user,item columns (optionally a rating column)",
    "image": "a folder with one sub-folder of pictures per class",
    "rl": "a .txt maze: # walls, . open, S start, G goal, X traps",
}

EXAMPLES = {
    "classify": "data/examples/sentiment.csv", "text": "data/examples/sentiment.csv",
    "table": "data/examples/fruit.csv", "number": "data/examples/houses.csv",
    "lm": "data/examples/stories.txt", "charnn": "data/examples/stories.txt",
    "cluster": "data/examples/customers.csv", "anomaly": "data/examples/transactions.csv",
    "forecast": "data/examples/sales.csv", "recommend": "data/examples/ratings.csv",
    "image": "data/examples/shapes", "rl": "data/examples/maze.txt",
}


def _score_kinds(text):
    scores = {}
    for kind, words in KIND_WORDS.items():
        s = 0.0
        for w in words:
            if re.search(rf"\b{re.escape(w)}(?:s|es|ed|ing)?\b", text):
                s += 1.0 + 0.5 * (" " in w)  # phrases count more
        if s:
            scores[kind] = s
    return scores


def find_path(text, roots):
    for m in PATH_RE.finditer(text):
        p = next(g for g in m.groups() if g)
        for root in roots:
            cand = os.path.join(root, os.path.expanduser(p))
            if os.path.exists(cand):
                return cand
        return p  # named but not found: report it
    for tok in text.split():  # a folder (for image models)
        tok = tok.strip("\"',.")
        for root in roots:
            cand = os.path.join(root, tok)
            if "/" in tok and os.path.isdir(cand):
                return cand
    return None


def wants_example(text):
    return bool(re.search(r"\b(example|sample|demo|test) (data|file|dataset|folder)\b|\buse (the |your )?(example|sample|demo)\b", text))


def is_build_request(text):
    t = text.lower()
    if QUESTION.match(t) and not PATH_RE.search(t):
        return False
    return bool(BUILD_VERB.search(t) and MODEL_NOUN.search(t))


def infer_kind(text):
    t = text.lower()
    scores = _score_kinds(t)
    if re.search(r"\bpredicts?\b", t) and "number" not in scores and "forecast" not in scores:
        scores["classify" if re.search(r"\b(which|whether|if|type|kind|class)\b", t) else "number"] = 0.5
    return max(scores, key=scores.get) if scores else None


def parse(text, roots):
    """Return a dict describing the requested model, or None if not a build request."""
    t = text.lower()
    if not is_build_request(t):
        return None
    scores = _score_kinds(t)
    path = find_path(text, roots)
    if path and os.path.isdir(path):
        scores["image"] = scores.get("image", 0) + 5
    elif path and path.endswith(".txt") and not ({"rl", "charnn", "lm"} & set(scores)):
        scores["lm"] = 1
    # "predict" + number words -> number; "predict which/whether" -> classify
    if re.search(r"\bpredicts?\b", t) and "number" not in scores and "forecast" not in scores:
        scores["classify" if re.search(r"\b(which|whether|if|type|kind|class)\b", t) else "number"] = 0.5
    kind = max(scores, key=scores.get) if scores else None
    target = None
    m = re.search(r"\b(?:predicts?|predicting|estimates?|forecasts?|guess(?:es)?)\s+(?:the\s+|a\s+|an\s+|my\s+|their\s+|next\s+)*"
                  r"((?:[a-z_]+\s*){1,3})", t)
    if m:
        words = []
        for w in m.group(1).split():
            if w in ("from", "using", "in", "with", "based", "given", "for", "of", "on", "by", "data", "file", "it", "is"):
                break
            words.append(w)
        target = " ".join(words) or None
    k = None
    m = re.search(r"\binto\s+(\d+|" + "|".join(WORD_NUMS) + r")\s+(?:groups|clusters|segments|types)", t)
    if m:
        k = int(m.group(1)) if m.group(1).isdigit() else WORD_NUMS[m.group(1)]
    m = re.search(r"\b(?:called|named)\s+[\"']?([a-z0-9_\-]+)", t)
    name = m.group(1) if m else None
    return {"kind": kind, "path": path, "target": target, "k": k, "name": name,
            "example": wants_example(t)}


def default_name(req):
    if req.get("name"):
        return req["name"]
    if req.get("path"):
        stem = os.path.splitext(os.path.basename(os.path.normpath(req["path"])))[0].lower()
        stem = re.sub(r"[^a-z0-9_\-]", "_", stem)[:30]
        if stem:
            return stem
    return (req.get("kind") or "model")
