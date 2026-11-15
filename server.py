"""Solen 1.0 website: chat with Solen and build your own AI models in the browser.

    python server.py            # then open http://localhost:8000
    python server.py --port 9000

Standard library only. Each browser gets its own session: its own chat memory,
its own uploads (uploads/<session>/) and its own models (models/sessions/<session>/).
"""
import argparse
import io
import json
import os
import re
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ai import Assistant
from ai.builder import ModelBuilder

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
UPLOADS = os.path.join(ROOT, "uploads")
SESSIONS_DIR = os.path.join(ROOT, "models", "sessions")
EXAMPLES = os.path.join(ROOT, "data", "examples")
MAX_UPLOAD = 20 * 1024 ** 2
MAX_UNZIPPED = 100 * 1024 ** 2
UPLOAD_EXT = (".csv", ".txt", ".json", ".zip", ".wav", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
STATIC = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
          "/style.css": ("style.css", "text/css"), "/favicon.svg": ("favicon.svg", "image/svg+xml")}
SID_RE = re.compile(r"^[a-f0-9]{32}$")
NAME_RE = re.compile(r"^[a-z0-9_\-]{1,40}$")

print("Loading Solen 1.0 ...")
BOT = Assistant().load(verbose=True)
BOT.allow_teach = False
LOCK = threading.Lock()
SESSIONS = {}


def session(sid):
    if sid not in SESSIONS:
        SESSIONS[sid] = {"memory": {}, "builder": ModelBuilder(os.path.join(SESSIONS_DIR, sid))}
    return SESSIONS[sid]


def run_as(sid, fn):
    """Run fn() with the shared model switched to this visitor's memory, models and files."""
    s = session(sid)
    with LOCK:
        BOT.memory, BOT.builder = s["memory"], s["builder"]
        BOT.allowed_roots = [os.path.join(UPLOADS, sid), EXAMPLES]
        return fn()


def attach_image(reply):
    """Turn a '[image] /path.png' marker from the image generator into a data URI."""
    import base64
    m = re.search(r"\n\[image\] (\S+\.png)\s*$", reply)
    if not m:
        return reply, None
    try:
        with open(m.group(1), "rb") as f:
            data = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        os.remove(m.group(1))
    except OSError:
        data = None
    text = reply[:m.start()]
    if data:  # the picture replaces the text preview
        text = "\n".join(l for l in text.split("\n") if not re.fullmatch(r"[ .:\-=+*#%@]{16,}", l)).strip()
    return text, data


def safe_extract(data, dest):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = [i for i in z.infolist() if not i.is_dir() and not i.filename.startswith("__MACOSX")]
        if len(infos) > 3000 or sum(i.file_size for i in infos) > MAX_UNZIPPED:
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
        return os.path.join(dest, entries[0])  # zip had one top folder
    return dest


class Handler(BaseHTTPRequestHandler):
    server_version = "Solen/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") or ctype.endswith("json") else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1024 ** 2:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _sid(self, value):
        if not isinstance(value, str) or not SID_RE.match(value):
            raise ValueError("bad session")
        return value

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path in STATIC:
                fname, ctype = STATIC[url.path]
                with open(os.path.join(WEB, fname), "rb") as f:
                    return self._send(200, f.read(), ctype)
            if url.path == "/api/models":
                sid = self._sid(q.get("session", [""])[0])
                return self._send(200, {"models": session(sid)["builder"].models()})
            if url.path == "/api/download":
                sid = self._sid(q.get("session", [""])[0])
                name = q.get("name", [""])[0]
                if not NAME_RE.match(name):
                    raise ValueError("bad model name")
                d = session(sid)["builder"].dir
                files = [f for f in os.listdir(d) if f.split(".")[0] == name]
                if not files:
                    return self._send(404, {"error": "no such model"})
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in files:
                        z.write(os.path.join(d, f), f"{name}/{f}")
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{name}-solen-model.zip"')
                self.send_header("Content-Length", str(buf.tell()))
                self.end_headers()
                return self.wfile.write(buf.getvalue())
            return self._send(404, {"error": "not found"})
        except (ValueError, OSError) as e:
            return self._send(400, {"error": str(e)})

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/chat":
                body = self._json()
                sid = self._sid(body.get("session"))
                msg = str(body.get("message", ""))[:2000]
                if body.get("file"):
                    msg = f'{msg} "{body["file"]}"'.strip()
                reply, image = attach_image(run_as(sid, lambda: BOT.reply(msg)))
                reply = reply.replace(f"uploads/{sid}/", "")  # show the user's file name, not the server path
                reply = re.sub(r"\s*Use it with /ask .*\Z", "\nTry it in Your models.", reply, flags=re.S)
                return self._send(200, {"reply": reply, "image": image, "models": session(sid)["builder"].models()})
            if url.path == "/api/ask":
                body = self._json()
                sid = self._sid(body.get("session"))
                name = str(body.get("name", ""))
                if not NAME_RE.match(name):
                    raise ValueError("bad model name")
                inp = str(body.get("input", ""))[:2000]
                reply, image = attach_image(run_as(sid, lambda: BOT.ask_model(name, inp)))
                return self._send(200, {"reply": reply, "image": image})
            if url.path == "/api/upload":
                q = parse_qs(url.query)
                sid = self._sid(q.get("session", [""])[0])
                fname = os.path.basename(q.get("filename", [""])[0])
                fname = re.sub(r"[^A-Za-z0-9._\-]", "_", fname)[:80]
                if not fname.lower().endswith(UPLOAD_EXT):
                    raise ValueError("allowed files: " + ", ".join(UPLOAD_EXT))
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > MAX_UPLOAD:
                    raise ValueError("files must be under 20 MB")
                data = self.rfile.read(n)
                dest_dir = os.path.join(UPLOADS, sid)
                os.makedirs(dest_dir, exist_ok=True)
                if fname.lower().endswith(".zip"):
                    folder = os.path.join(dest_dir, os.path.splitext(fname)[0])
                    os.makedirs(folder, exist_ok=True)
                    path = safe_extract(data, folder)
                else:
                    path = os.path.join(dest_dir, fname)
                    with open(path, "wb") as f:
                        f.write(data)
                return self._send(200, {"path": os.path.relpath(path, ROOT), "name": fname})
            return self._send(404, {"error": "not found"})
        except (ValueError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as e:
            return self._send(400, {"error": str(e)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"Solen 1.0 is live at http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
