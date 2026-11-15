"""Shared text utilities: tokenizing, hashed features, data augmentation."""
import random
import re
import zlib

N_FEATURES = 2 ** 15
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
CONTRACTIONS = {"what's": "what is", "whats": "what is", "who's": "who is", "where's": "where is",
                "how's": "how is", "i'm": "i am", "you're": "you are", "it's": "it is",
                "don't": "do not", "can't": "can not", "won't": "will not", "isn't": "is not"}
STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "in", "do", "does", "me", "you", "please",
             "can", "tell", "about", "what", "i", "and", "my", "your", "it", "on", "for", "be"}


# canonical forms so different wordings of the same idea share features
SYNONYMS = {
    "make": "build", "makes": "build", "making": "build", "create": "build", "creating": "build",
    "develop": "build", "construct": "build", "code": "build", "program": "build",
    "llm": "large language model", "llms": "large language model", "gpt": "large language model",
    "ml": "machine learning", "nn": "neural network", "ai": "artificial intelligence",
    "dataset": "data", "datasets": "data", "get": "find", "obtain": "find", "source": "find",
    "shrink": "smaller", "compress": "smaller", "tiny": "small", "faster": "fast", "quicker": "fast",
    "biggest": "largest", "huge": "large", "big": "large",
    "pick": "choose", "select": "choose", "set": "choose",
    "models": "model", "networks": "network", "nets": "network", "net": "network",
}


def canonical(tokens):
    out = []
    for t in tokens:
        out.extend(SYNONYMS.get(t, t).split())
    return out


def tokenize(text):
    text = text.lower()
    for k, v in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return _TOKEN_RE.findall(text)


def _h(s):
    return zlib.crc32(s.encode("utf-8")) % N_FEATURES


def features(text):
    """Return {feature_index: weight} using word 1-grams, 2-grams and char 3-grams."""
    toks = canonical(tokenize(text))
    feats = {}

    def add(key, w):
        i = _h(key)
        feats[i] = feats.get(i, 0.0) + w

    for t in toks:
        add("w:" + t, 1.0 if t not in STOPWORDS else 0.3)
        padded = f"#{t}#"
        for j in range(len(padded) - 2):
            add("c:" + padded[j:j + 3], 0.25)
    for a, b in zip(toks, toks[1:]):
        add(f"b:{a}_{b}", 0.7)
    norm = sum(v * v for v in feats.values()) ** 0.5 or 1.0
    return {k: v / norm for k, v in feats.items()}


PREFIXES = ["", "", "", "can you tell me ", "please tell me ", "i want to know ", "do you know ",
            "hey ", "tell me ", "quick question ", "i was wondering "]
SUFFIXES = ["", "", "", " please", " thanks", "?", " now"]


def _typo(word, rng):
    if len(word) < 4:
        return word
    i = rng.randrange(len(word) - 1)
    op = rng.random()
    if op < 0.4:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]  # swap
    if op < 0.7:
        return word[:i] + word[i + 1:]  # delete
    return word[:i] + word[i] + word[i:]  # double


def augment(pattern, n, rng):
    """Generate n noisy paraphrases of a pattern (fillers, word dropout, typos)."""
    out = [pattern]
    words = pattern.split()
    for _ in range(n):
        w = list(words)
        if len(w) > 3 and rng.random() < 0.35:
            del w[rng.randrange(len(w))]
        w = [_typo(x, rng) if rng.random() < 0.12 else x for x in w]
        out.append(rng.choice(PREFIXES) + " ".join(w) + rng.choice(SUFFIXES))
    return out


GIBBERISH_WORDS = ("stock price weather tomorrow play song order pizza flight ticket book hotel "
                   "car insurance lottery numbers bitcoin football score movie tonight recipe "
                   "lasagna drive route traffic email send message alarm timer purple elephant "
                   "banana keyboard dance guitar lawyer tax refund shoes sale").split()


def negative_examples(n, rng):
    """Out-of-domain sentences used to train the 'unknown' class."""
    fixed = ["what is the weather tomorrow", "play my favorite song", "order a pizza",
             "book a flight to nowhere", "what is the bitcoin price", "set an alarm for 7",
             "send an email to my boss", "who will win the football game", "asdf qwer zxcv",
             "blah blah blah", "purple monkey dishwasher"]
    out = list(fixed)
    for _ in range(n):
        out.append(" ".join(rng.choice(GIBBERISH_WORDS) for _ in range(rng.randint(2, 6))))
    return out
