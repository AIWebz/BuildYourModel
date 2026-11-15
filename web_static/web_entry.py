"""Solen 1.0 running inside the browser (Pyodide). Called from worker.js."""
import json
import os
import re
import sys
import zipfile

ROOT = "/solen"
UPLOADS = "/uploads"
BOT = None


def _restore_models():
    """Model weights ship as JSON data; turn them back into the files Solen loads."""
    import base64
    import numpy as np
    from ai.language_model import LanguageModel
    mdir = os.path.join(ROOT, "models")
    with open(os.path.join(mdir, "brain.json")) as f:
        spec = json.load(f)
    arrays = {k: np.frombuffer(base64.b64decode(v["b64"]), dtype=v["dtype"]).reshape(v["shape"])
              for k, v in spec["arrays"].items()}
    np.savez(os.path.join(mdir, "brain.npz"), classes=np.array(spec["classes"]), **arrays)
    with open(os.path.join(mdir, "lm.json")) as f:
        lm_spec = json.load(f)
    lm = LanguageModel(lm_spec["d"])
    lm.responses = lm_spec["responses"]
    lm._index({(u, v, w): c for u, v, w, c in lm_spec["c3"]})
    lm.save(os.path.join(mdir, "lm.pkl"))


def init(persist_dir):
    global BOT
    os.chdir(ROOT)
    os.makedirs(UPLOADS, exist_ok=True)
    sys.path.insert(0, ROOT)
    if os.path.exists(os.path.join(ROOT, "models", "brain.json")):
        _restore_models()
    examples = os.path.join(ROOT, "data", "examples")
    if not os.path.exists(os.path.join(examples, "maze.txt")):  # synthetic examples are generated here
        import runpy
        runpy.run_path(os.path.join(examples, "make_examples.py"))
    from ai import Assistant
    from ai.builder import ModelBuilder
    BOT = Assistant(model_dir=os.path.join(ROOT, "models")).load()
    BOT.builder = ModelBuilder(os.path.join(persist_dir, "models"))
    BOT.allowed_roots = [UPLOADS, os.path.join(ROOT, "data", "examples")]
    return "ready"


def _clean(reply):
    reply = reply.replace(UPLOADS + "/", "")
    return re.sub(r"\s*Use it with /ask .*\Z", "\nTry it in Your models.", reply, flags=re.S)


def chat(message, file_path):
    msg = message[:2000]
    if file_path:
        msg = f'{msg} "{file_path}"'.strip()
    return json.dumps({"reply": _clean(BOT.reply(msg)), "models": BOT.builder.models()})


def ask(name, value):
    if not re.fullmatch(r"[a-z0-9_\-]{1,40}", name):
        return json.dumps({"reply": "Unknown model."})
    return json.dumps({"reply": BOT.ask_model(name, value[:2000])})


def models():
    return json.dumps({"models": BOT.builder.models()})


def unzip(path):
    dest = os.path.splitext(path)[0]
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        infos = [i for i in z.infolist() if not i.is_dir() and not i.filename.startswith("__MACOSX")]
        if len(infos) > 3000 or sum(i.file_size for i in infos) > 100 * 1024 ** 2:
            raise ValueError("zip is too large")
        for i in infos:
            target = os.path.realpath(os.path.join(dest, i.filename))
            if not target.startswith(os.path.realpath(dest) + os.sep):
                raise ValueError("unsafe path in zip")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(i) as src, open(target, "wb") as out:
                out.write(src.read())
    entries = [e for e in os.listdir(dest) if not e.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(dest, entries[0])):
        return os.path.join(dest, entries[0])
    return dest
