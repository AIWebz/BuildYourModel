"""Build the browser-only Solen 1.0 site (runs the real Python model with Pyodide).

    python tools/build_web.py [out_dir]                 # artifact build (default: dist/)
    python tools/build_web.py site --standalone         # full HTML page for GitHub Pages etc.

Output: index.html, app.js, worker.js, solen.zip (code + data + compact model) and
pyodide/ (runtime + NumPy + Pillow). Serve the folder with any static file host.
"""
import io
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
VERSION = "0.26.4"
CORE = f"https://github.com/pyodide/pyodide/releases/download/{VERSION}/pyodide-core-{VERSION}.tar.bz2"
WHEELS = ["numpy-1.26.4-cp312-cp312-pyodide_2024_0_wasm32.whl", "pillow-10.2.0-cp312-cp312-pyodide_2024_0_wasm32.whl"]
CORE_FILES = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]
CACHE = os.path.join(ROOT, ".cache", f"pyodide-{VERSION}")


def fetch_pyodide():
    os.makedirs(CACHE, exist_ok=True)
    if not all(os.path.exists(os.path.join(CACHE, f)) for f in CORE_FILES):
        print("downloading Pyodide core ...")
        data = urllib.request.urlopen(CORE).read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as t:
            for m in t.getmembers():
                if os.path.basename(m.name) in CORE_FILES and m.isfile():
                    with open(os.path.join(CACHE, os.path.basename(m.name)), "wb") as f:
                        f.write(t.extractfile(m).read())
    for w in WHEELS:
        if not os.path.exists(os.path.join(CACHE, w)):
            print("downloading", w)
            urllib.request.urlretrieve(f"https://cdn.jsdelivr.net/pyodide/v{VERSION}/full/{w}", os.path.join(CACHE, w))


STDLIB_EXTRAS = ["traceback.py", "linecache.py", "tokenize.py", "token.py", "warnings.py", "textwrap.py",
                 "inspect.py", "dis.py", "opcode.py", "_opcode_metadata.py", "contextlib.py", "weakref.py",
                 "_weakrefset.py", "copy.py", "copyreg.py", "string.py", "collections/abc.py", "encodings/__init__.py",
                 "encodings/aliases.py", "encodings/utf_8.py", "encodings/utf_8_sig.py", "encodings/latin_1.py",
                 "encodings/ascii.py", "encodings/cp437.py", "runpy.py", "pkgutil.py", "zipfile/__init__.py",
                 "zipfile/_path/__init__.py", "shutil.py", "fnmatch.py", "bz2.py", "lzma.py", "struct.py", "base64.py"]


def _pyodide_files(needed):
    """Pick the files Solen needs from Pyodide's stdlib and wheels: [(archive, real_path, bytes)]."""
    picked = []
    with zipfile.ZipFile(os.path.join(CACHE, "python_stdlib.zip")) as z:
        names = set(z.namelist())
        want = set(needed["stdlib"]) | set(STDLIB_EXTRAS)
        want |= {n for n in names if n.startswith(("_pyodide/", "pyodide/")) and not n.endswith("/")}
        for n in sorted(want & names):
            picked.append(("python_stdlib.zip", n, z.read(n)))
    site = set(needed["site"])
    for whl in WHEELS:
        with zipfile.ZipFile(os.path.join(CACHE, whl)) as z:
            for n in z.namelist():
                if n.endswith("/"):
                    continue
                numpy_ok = whl.startswith("numpy") and "/tests/" not in n and "/testing/" not in n and \
                    not n.endswith((".h", ".pc", ".a", ".c", ".pxd", ".pyi", ".f2py", ".in", ".pyx"))
                if numpy_ok or n in site or n.endswith(".so") or ".dist-info/" in n:
                    picked.append((whl, n, z.read(n)))
    return picked


def build(out, standalone=False):
    """Static site where every published file is a plain web type (.txt, .wasm, .json, .js, .csv, .html)."""
    import base64
    import numpy as np
    from ai import Assistant
    fetch_pyodide()
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "pyodide"))
    os.makedirs(os.path.join(out, "f"))
    for f in ("pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "pyodide-lock.json"):
        shutil.copy(os.path.join(CACHE, f), os.path.join(out, "pyodide", f))
    for f in ("index.html", "app.js", "worker.js"):
        shutil.copy(os.path.join(ROOT, "web_static", f), os.path.join(out, f))
    if standalone:  # a complete HTML document for ordinary static hosts (GitHub Pages, Netlify, ...)
        page = open(os.path.join(out, "index.html")).read()
        with open(os.path.join(out, "index.html"), "w") as fh:
            fh.write('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                     '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                     '<link rel="icon" href="favicon.svg" type="image/svg+xml">\n'
                     '<style>[hidden]{display:none!important} body{margin:0}</style>\n</head>\n<body>\n'
                     + page + '</body>\n</html>\n')
        shutil.copy(os.path.join(ROOT, "web", "favicon.svg"), os.path.join(out, "favicon.svg"))
        open(os.path.join(out, ".nojekyll"), "w").close()

    counter = [0]

    def publish(real, data):
        ext = ".wasm" if real.endswith(".so") else os.path.splitext(real)[1] if real.endswith((".json", ".csv")) else ".txt"
        if ext == ".txt":
            data.decode("utf-8")  # text files must be valid UTF-8
        pub = f"f/{counter[0]}{ext}"
        counter[0] += 1
        with open(os.path.join(out, pub), "wb") as fh:
            fh.write(data)
        return pub

    with open(os.path.join(ROOT, "tools", "pyodide_needed.json")) as f:
        needed = json.load(f)
    manifest = {"archives": {"python_stdlib.zip": []}, "solen": []}
    for whl in WHEELS:
        manifest["archives"][whl] = []
    for archive, real, data in _pyodide_files(needed):
        manifest["archives"][archive].append([real, publish(real, data)])

    # Solen itself: code, data, and model weights as JSON
    bot = Assistant().load(verbose=False)
    tmp = os.path.join(out, "_brain.npz")
    bot.brain.save_compact(tmp)
    d = np.load(tmp)
    brain = {"classes": [str(c) for c in d["classes"]],
             "arrays": {k: {"dtype": str(d[k].dtype), "shape": list(d[k].shape), "b64": base64.b64encode(d[k].tobytes()).decode()}
                        for k in d.files if k != "classes"}}
    d.close()
    os.remove(tmp)
    lm = {"d": bot.lm.d, "responses": bot.lm.responses, "c3": [[u, v, w, c] for (u, v, w), c in bot.lm.c3.items()]}
    solen = [("models/brain.json", json.dumps(brain).encode()), ("models/lm.json", json.dumps(lm).encode()),
             ("web_entry.py", open(os.path.join(ROOT, "web_static", "web_entry.py"), "rb").read())]
    for f in sorted(os.listdir(os.path.join(ROOT, "ai"))):
        if f.endswith(".py"):
            solen.append((f"ai/{f}", open(os.path.join(ROOT, "ai", f), "rb").read()))
    for f in ("intents.json", "knowledge.json"):
        solen.append((f"data/{f}", open(os.path.join(ROOT, "data", f), "rb").read()))
    for f in ("make_examples.py", "sentiment.csv", "stories.txt", "fruit.csv", "houses.csv"):
        solen.append((f"data/examples/{f}", open(os.path.join(ROOT, "data", "examples", f), "rb").read()))
    for real, data in solen:
        manifest["solen"].append([real, publish(real, data)])
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    files = [os.path.relpath(os.path.join(dp, f), out) for dp, _, fs in os.walk(out) for f in fs]
    total = sum(os.path.getsize(os.path.join(out, f)) for f in files)
    biggest = max(files, key=lambda f: os.path.getsize(os.path.join(out, f)))
    print(f"{len(files)} files, {total / 1024 ** 2:.1f} MB total, largest {biggest} "
          f"({os.path.getsize(os.path.join(out, biggest)) / 1024 ** 2:.1f} MB)")
    with open(os.path.join(out, "_files.json"), "w") as f:
        json.dump(sorted(files), f)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    build(args[0] if args else os.path.join(ROOT, "dist"), standalone="--standalone" in sys.argv)
