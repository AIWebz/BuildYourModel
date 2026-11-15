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
MODEL_NOUN = re.compile(r"\b(model|ai|a\.i\.|classifier|predictor|bot|chatbot|chat bot|network|agent|recommender|detector|"
                        r"translator|summarizer|transformer|gpt|llm|language model|generator|"
                        r"forecaster|generator|system|clusterer|neural net|machine learning|program|tool)s?\b")
QUESTION = re.compile(r"^\s*(how|what|why|when|where|which|who|explain|tell me about|is|are|does|do)\b")
PATH_RE = re.compile(r"""(?:"([^"]+)"|'([^']+)'|([\w./\\~-]+\.(?:csv|txt|json)))""", re.I)

# keyword -> weight, per model kind
KIND_WORDS = {
    "chatbot": ["chatbot", "chat bot", "support bot", "faq", "customer service", "customer support",
                "answers questions", "help desk", "assistant bot"],
    "docqa": ["answer questions about", "answers questions about", "answering questions about", "question answering", "questions about my document", "about this document",
              "about the document", "search my document", "ask questions about", "from my notes"],
    "summarize": ["summarize", "summary", "summaries", "summarizer", "tldr", "tl;dr", "shorten", "key points"],
    "translate": ["translate", "translator", "translation", "into spanish", "into english", "into french",
                  "spanish", "french", "german", "italian", "portuguese"],
    "gpt": ["transformer", "gpt", "attention", "llm", "large language model", "chatgpt", "mini gpt", "nanogpt",
            "decoder"],
    "imagegen": ["generate images", "generates images", "image generator", "create images", "creates images",
                 "make images", "draw", "draws", "drawing", "new images", "generative", "gan", "diffusion", "art",
                 "generate pictures", "new pictures", "generate new"],
    "audio": ["audio", "sound", "sounds", "speech", "voice", "voices", "wav", "noise", "music", "spoken",
              "listen", "hear"],
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
    "chatbot": "a chatbot", "docqa": "a document question-answering model", "summarize": "a summarizer",
    "translate": "a translator", "gpt": "a transformer language model (mini GPT)", "imagegen": "an image generator",
    "audio": "an audio classifier",
    "text": "a text classifier", "table": "a table classifier", "number": "a number predictor",
    "classify": "a classifier", "lm": "a tiny language model", "charnn": "a neural character model",
    "cluster": "a clustering model", "anomaly": "an anomaly detector", "forecast": "a forecaster",
    "recommend": "a recommender", "image": "an image classifier", "rl": "a reinforcement learning agent",
}

NEEDS = {
    "chatbot": "a CSV with question,answer columns",
    "docqa": "a .txt document",
    "summarize": "a .txt document",
    "translate": "a CSV with two columns of matching sentences, like english,spanish",
    "gpt": "a .txt file of example text (1,000+ characters)",
    "imagegen": "a folder or ZIP of pictures (optionally one sub-folder per kind of picture)",
    "audio": "a folder or ZIP with one sub-folder of .wav files per class",
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
    "chatbot": "data/examples/faq.csv", "docqa": "data/examples/science_notes.txt",
    "summarize": "data/examples/science_notes.txt", "translate": "data/examples/translations.csv",
    "gpt": "data/examples/stories.txt", "imagegen": "data/examples/shapes", "audio": "data/examples/sounds",
    "classify": "data/examples/sentiment.csv", "text": "data/examples/sentiment.csv",
    "table": "data/examples/fruit.csv", "number": "data/examples/houses.csv",
    "lm": "data/examples/stories.txt", "charnn": "data/examples/stories.txt",
    "cluster": "data/examples/customers.csv", "anomaly": "data/examples/transactions.csv",
    "forecast": "data/examples/sales.csv", "recommend": "data/examples/ratings.csv",
    "image": "data/examples/shapes", "rl": "data/examples/maze.txt",
}


# Requests beyond what can be trained here: (pattern, closest kind or None, honest note)
OUT_OF_SCOPE = [
    (r"\b(chatgpt|gpt-?[3-5]|claude|gemini|llama|large language model|llm)\b", "gpt",
     "A model at the scale of ChatGPT needs billions of parameters, trillions of words and thousands of GPUs. "
     "I'll build the closest thing I can train here: a small transformer language model learned from your text."),
    (r"\b(photo-?realistic|stable diffusion|dall-?e|midjourney|text.to.image)\b", "imagegen",
     "Photorealistic text-to-image models need enormous datasets and GPU clusters. I'll build a small image "
     "generator that learns from your pictures instead."),
    (r"\b(videos?|movie generator|animation)\b", None,
     "Video models are far beyond what can be trained here: they need massive datasets and GPU clusters. I can build "
     "an image classifier or an image generator from frames you upload instead."),
    (r"\b(text.to.speech|speech synthesis|voice clon\w*|clone (my|a) voice|sings?|singing)\b", None,
     "Generating realistic speech or singing needs large audio models and GPUs, which I can't train here. I can build "
     "an audio classifier that recognizes sounds or short spoken words from your .wav clips."),
    (r"\b(compos\w*|generat\w*|creat\w*|writ\w*|make|makes)\s+(?:\w+\s+)?(music|songs?|melod\w*)\b", None,
     "Generating music needs large audio models and GPUs, which I can't train here. I can build an audio classifier "
     "that recognizes kinds of sounds, or a text generator that writes song lyrics from your examples."),
    (r"\b(speech recognition|speech.to.text|transcrib\w*)\b", "audio",
     "Full speech-to-text needs large models trained on thousands of hours of audio. I'll build an audio classifier "
     "that recognizes a fixed set of words or sounds from your .wav clips instead."),
    (r"\b(object detection|detect objects|bounding box\w*|yolo|face recognition|recogni[sz]e faces)\b", "image",
     "Object detection and face recognition need large labeled datasets and deep networks. I'll build a small image "
     "classifier that labels whole pictures instead."),
    (r"\b(self.driving|autonomous (car|vehicle)|robot arm|drone)\b", "rl",
     "Real-world robot control can't be trained here. I'll build a reinforcement learning agent that learns to solve "
     "a maze you describe instead."),
]
ACTIVATIONS = {"relu": "relu", "tanh": "tanh", "sigmoid": "sigmoid", "logistic": "sigmoid"}


def parse_arch(t):
    """Custom network design: '3 hidden layers of 64 units with tanh, 500 epochs, learning rate 0.005'."""
    arch = {}
    num = r"(\d+|" + "|".join(WORD_NUMS) + r"|one)"
    as_int = lambda v: int(v) if v.isdigit() else {**WORD_NUMS, "one": 1}[v]
    m = re.search(num + r"\s+(?:hidden\s+)?layers?", t)
    width = re.search(r"(\d+)\s+(?:neurons|units|nodes|hidden units)", t)
    sizes = re.search(r"layers?\s+(?:of\s+)?(?:sizes?\s+)?(\d+(?:\s*(?:,|x|and)\s*\d+)+)", t)
    if sizes:
        arch["layers"] = [int(v) for v in re.findall(r"\d+", sizes.group(1))][:6]
    elif m or width:
        n = min(as_int(m.group(1)), 6) if m else 1
        w = min(int(width.group(1)), 512) if width else 32
        arch["layers"] = [w] * n
    for word, act in ACTIVATIONS.items():
        if re.search(rf"\b{word}\b", t):
            arch["act"] = act
    m = re.search(r"(\d+)\s+epochs?", t)
    if m:
        arch["epochs"] = min(int(m.group(1)), 5000)
    m = re.search(r"learning rate\s+(?:of\s+)?([0-9]*\.?[0-9]+)", t)
    if m:
        arch["lr"] = min(float(m.group(1)), 1.0)
    return arch or None


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
    if "imagegen" in scores:  # "generate images" beats the plain word "images"
        scores.pop("image", None)
    if "docqa" in scores and scores["docqa"] >= scores.get("chatbot", 0):
        scores.pop("chatbot", None)
    path = find_path(text, roots)
    if path and path.endswith(".txt") and "chatbot" in scores:  # a chatbot over a document = document Q&A
        scores["docqa"] = scores.pop("chatbot") + scores.get("docqa", 0)
    if path and os.path.isdir(path):
        best = max(("imagegen", "audio", "image"), key=lambda k: scores.get(k, 0))
        scores[best] = scores.get(best, 0) + 5
    elif path and path.endswith(".txt") and not ({"rl", "charnn", "lm", "gpt", "docqa", "summarize"} & set(scores)):
        scores["docqa" if "chatbot" in scores else "lm"] = 1
    elif path and path.endswith(".csv") and "docqa" in scores and "chatbot" not in scores:
        scores["chatbot"] = scores.pop("docqa")
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
    note, blocked = None, False
    for pattern, closest, msg in OUT_OF_SCOPE:
        if re.search(pattern, t):
            note, blocked = msg, closest is None
            kind = closest or kind
            break
    return {"kind": kind, "path": path, "target": target, "k": k, "name": name,
            "example": wants_example(t), "arch": parse_arch(t), "note": note, "blocked": blocked}


def default_name(req):
    if req.get("name"):
        return req["name"]
    if req.get("path"):
        stem = os.path.splitext(os.path.basename(os.path.normpath(req["path"])))[0].lower()
        stem = re.sub(r"[^a-z0-9_\-]", "_", stem)[:30]
        if stem:
            return stem
    return (req.get("kind") or "model")
