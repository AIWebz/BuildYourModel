// Runs Solen 1.0 (Python + NumPy via Pyodide) off the main thread.
// Pyodide's standard library and the NumPy/Pillow packages are published as individual
// files (see manifest.json); this worker reassembles them in memory when Pyodide asks.
const realFetch = self.fetch.bind(self);
let manifestPromise;
const manifest = () => (manifestPromise ??= realFetch("manifest.json").then((r) => {
  if (!r.ok) throw new Error("could not load manifest.json");
  return r.json();
}));
const status = (text) => postMessage({ type: "status", text });

// ---- minimal zip writer (stored entries + CRC32), so Pyodide receives the archives it expects
const CRC = new Uint32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
function crc32(b) { let c = 0xffffffff; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
function makeZip(entries) {
  const enc = new TextEncoder(), parts = [], central = [];
  let offset = 0;
  for (const [name, data] of entries) {
    const n = enc.encode(name), crc = crc32(data);
    const h = new DataView(new ArrayBuffer(30));
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true);
    h.setUint32(14, crc, true); h.setUint32(18, data.length, true); h.setUint32(22, data.length, true);
    h.setUint16(26, n.length, true);
    parts.push(new Uint8Array(h.buffer), n, data);
    const c = new DataView(new ArrayBuffer(46));
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true);
    c.setUint32(16, crc, true); c.setUint32(20, data.length, true); c.setUint32(24, data.length, true);
    c.setUint16(28, n.length, true); c.setUint32(42, offset, true);
    central.push(new Uint8Array(c.buffer), n);
    offset += 30 + n.length + data.length;
  }
  const cdSize = central.reduce((s, p) => s + p.length, 0);
  const e = new DataView(new ArrayBuffer(22));
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, entries.length, true); e.setUint16(10, entries.length, true);
  e.setUint32(12, cdSize, true); e.setUint32(16, offset, true);
  return new Blob([...parts, ...central, new Uint8Array(e.buffer)]);
}

async function fetchAll(pairs) {  // [[realPath, publishedPath], ...] -> [[realPath, bytes], ...]
  const out = new Array(pairs.length);
  let i = 0;
  async function lane() {
    while (i < pairs.length) {
      const k = i++;
      const r = await realFetch(pairs[k][1]);
      if (!r.ok) throw new Error(`missing file ${pairs[k][1]}`);
      out[k] = [pairs[k][0], new Uint8Array(await r.arrayBuffer())];
    }
  }
  await Promise.all(Array.from({ length: 12 }, lane));
  return out;
}

self.fetch = async (input, init) => {
  const url = input instanceof Request ? input.url : String(input);
  const name = decodeURIComponent(url.split("?")[0].split("/").pop());
  const m = await manifest();
  if (m.archives[name]) {
    return new Response(makeZip(await fetchAll(m.archives[name])), { headers: { "Content-Type": "application/zip" } });
  }
  return realFetch(input, init);
};

importScripts("pyodide/pyodide.js");
let py, entry, persist = false;

function syncfs(populate) {
  if (!persist) return Promise.resolve();
  return new Promise((res) => py.FS.syncfs(populate, () => res()));
}

const ready = (async () => {
  status("Starting the Python runtime…");
  py = await loadPyodide({ indexURL: new URL("pyodide/", self.location.href).href });
  status("Loading NumPy and Pillow…");
  await py.loadPackage(["numpy", "pillow"]);
  status("Loading the Solen 1.0 model…");
  const m = await manifest();
  for (const [real, data] of await fetchAll(m.solen)) {
    py.FS.mkdirTree("/solen/" + real.split("/").slice(0, -1).join("/"));
    py.FS.writeFile("/solen/" + real, data);
  }
  let dir = "/tmp/solen";
  try {  // keep built models in this browser between visits
    py.FS.mkdirTree("/persist");
    py.FS.mount(py.FS.filesystems.IDBFS, {}, "/persist");
    persist = true;
    await syncfs(true);
    dir = "/persist";
  } catch (e) { persist = false; }
  py.runPython("import sys; sys.path.insert(0, '/solen')");
  entry = py.pyimport("web_entry");
  status("Waking up Solen 1.0…");
  entry.init(dir);
  postMessage({ type: "ready" });
})().catch((e) => postMessage({ type: "fatal", text: String(e && e.message || e).split("\n").filter(Boolean).pop() }));

onmessage = async ({ data }) => {
  const { id, op, args } = data;
  try {
    await ready;
    if (!entry) throw new Error("Solen did not start");
    let result;
    if (op === "upload") {
      const safe = args.name.replace(/[^A-Za-z0-9._-]/g, "_").slice(0, 80);
      let path = `/uploads/${safe}`;
      py.FS.writeFile(path, new Uint8Array(args.data));
      if (safe.toLowerCase().endsWith(".zip")) path = entry.unzip(path);
      result = { path, name: safe };
    } else if (op === "chat") {
      result = JSON.parse(entry.chat(args.message, args.file || ""));
      await syncfs(false);
    } else if (op === "ask") {
      result = JSON.parse(entry.ask(args.name, args.input));
    } else if (op === "models") {
      result = JSON.parse(entry.models());
    }
    postMessage({ id, result });
  } catch (e) {
    postMessage({ id, error: String(e && e.message || e).split("\n").filter(Boolean).pop() });
  }
};
