// Solen 1.0 website: chat to build models, then test and download them.
(() => {
  const $ = (id) => document.getElementById(id);
  const messages = $("messages"), input = $("input"), send = $("send"), fileInput = $("fileInput");
  let attached = null, busy = false, known = new Set();


  // ---- theme toggle
  $("themeBtn").onclick = () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("solen-theme", root.dataset.theme); } catch {}
  };
  try { const t = localStorage.getItem("solen-theme"); if (t) document.documentElement.dataset.theme = t; } catch {}

  // ---- example prompts
  const EXAMPLES = [
    ["Predict house prices", "Make me an AI that predicts house prices from data/examples/houses.csv"],
    ["Review sentiment", "Make a model that tells if a review is positive or negative from data/examples/sentiment.csv"],
    ["Group customers", "I want an AI that groups customers using data/examples/customers.csv"],
    ["Spot fraud", "Build an AI that spots unusual transactions from data/examples/transactions.csv"],
    ["Forecast sales", "Build a model that forecasts sales from data/examples/sales.csv"],
    ["Recommend movies", "Create a recommender using data/examples/ratings.csv"],
    ["Recognize shapes", "Train an AI to recognize shapes in data/examples/shapes"],
    ["Solve a maze", "Make an AI agent that learns to escape the maze in data/examples/maze.txt"],
    ["Write stories", "Build a neural character model called writer from data/examples/stories.txt"],
    ["FAQ chatbot", "Make a customer support chatbot from data/examples/faq.csv"],
    ["Mini GPT", "Build a mini GPT transformer called gpt from data/examples/stories.txt"],
    ["Translate", "Make a translator from English to Spanish using data/examples/translations.csv"],
    ["Generate images", "Train an AI that generates images of shapes from data/examples/shapes called artist"],
    ["Ask a document", "Build an AI that answers questions about data/examples/science_notes.txt"],
    ["Summarize", "Create a summarizer called digest for data/examples/science_notes.txt"],
    ["Recognize sounds", "Make an AI that recognizes sounds in data/examples/sounds"],
    ["Custom network", "Build a model that predicts house prices from data/examples/houses.csv with 3 hidden layers of 64 neurons, tanh, 400 epochs, called deephouse"],
  ];
  for (const [label, prompt] of EXAMPLES) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.textContent = label;
    b.onclick = () => submit(prompt);
    $("chips").appendChild(b);
  }

  // ---- render text: monospace blocks for grids/code, plain text otherwise
  function renderText(el, text) {
    const lines = text.split("\n");
    let buf = [], grid = [];
    const flushText = () => { if (buf.length) { el.appendChild(document.createTextNode(buf.join("\n"))); buf = []; } };
    const flushGrid = () => { if (grid.length) { const pre = document.createElement("pre"); pre.textContent = grid.join("\n"); el.appendChild(pre); grid = []; } };
    const isGrid = (l) => (/^[#.*SGX ]{3,}$/.test(l) && l.includes("#")) || (/^[ .:\-=+*#%@]{16,}$/.test(l));
    const isCode = (l) => /^(import |from |model|opt|loss|for |    |clf|print\(|X_train)/.test(l);
    for (const l of lines) {
      if (isGrid(l) || isCode(l)) { flushText(); grid.push(l); }
      else { flushGrid(); buf.push(l); }
    }
    flushText(); flushGrid();
  }

  function addMsg(who, text, extra) {
    $("welcome")?.remove();
    const div = document.createElement("div");
    div.className = `msg ${who}`;
    if (who === "bot") {
      const w = document.createElement("span"); w.className = "who"; w.textContent = "Solen 1.0"; div.appendChild(w);
      if (/^Understood: building|^Built /m.test(text)) div.classList.add("built");
    }
    const body = document.createElement("div");
    renderText(body, text);
    div.appendChild(body);
    if (extra) { const f = document.createElement("span"); f.className = "file"; f.textContent = extra; div.appendChild(f); }
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
  }

  function showImage(el, src) {
    if (!src || !/^data:image\/png;base64,/.test(src)) return;
    const img = document.createElement("img");
    img.src = src; img.alt = "Generated image"; img.className = "gen";
    el.appendChild(img);
  }

  function typing() {
    const div = addMsg("bot", "");
    const t = document.createElement("span"); t.className = "typing"; t.innerHTML = "<i></i><i></i><i></i>";
    div.lastChild.appendChild(t);
    return div;
  }

  // ---- Solen runs in a background worker (Python + NumPy in the browser)
  const worker = new Worker("worker.js");
  const pending = new Map();
  let nextId = 1;
  worker.onmessage = ({ data }) => {
    if (data.type === "status") { $("loadingText") && ($("loadingText").textContent = data.text); return; }
    if (data.type === "ready") {
      $("loading")?.remove();
      input.disabled = false; send.disabled = false; input.focus();
      rpc("models").then((d) => renderModels(d.models)).catch(() => {});
      return;
    }
    if (data.type === "fatal") {
      const l = $("loading");
      if (l) { l.classList.add("error"); $("loadingText").textContent = `Solen could not start in this browser: ${data.text}`; }
      return;
    }
    const p = pending.get(data.id);
    if (!p) return;
    pending.delete(data.id);
    data.error ? p.reject(new Error(data.error)) : p.resolve(data.result);
  };
  worker.onerror = (e) => {
    const l = $("loading");
    if (l) { l.classList.add("error"); $("loadingText").textContent = `Solen could not start in this browser: ${e.message || "worker error"}`; }
  };
  function rpc(op, args = {}, transfer = []) {
    return new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, op, args }, transfer);
    });
  }

  // ---- upload
  async function upload(file) {
    if (file.size > 20 * 1024 * 1024) throw new Error("Files must be under 20 MB.");
    const data = await file.arrayBuffer();
    return rpc("upload", { name: file.name, data }, [data]);
  }

  async function attach(file) {
    if (!file) return;
    $("attachName").textContent = `Uploading ${file.name}…`;
    $("attachment").hidden = false;
    try {
      attached = await upload(file);
      $("attachName").textContent = `📎 ${attached.name}`;
    } catch (e) {
      attached = null; $("attachment").hidden = true;
      addMsg("bot", `Upload failed: ${e.message}`);
    }
  }
  fileInput.onchange = () => { attach(fileInput.files[0]); fileInput.value = ""; };
  $("attachRemove").onclick = () => { attached = null; $("attachment").hidden = true; };

  const chat = document.querySelector(".chat");
  chat.addEventListener("dragover", (e) => { e.preventDefault(); chat.classList.add("drag"); });
  chat.addEventListener("dragleave", () => chat.classList.remove("drag"));
  chat.addEventListener("drop", (e) => { e.preventDefault(); chat.classList.remove("drag"); attach(e.dataTransfer.files[0]); });

  // ---- chat
  async function submit(text) {
    text = (text ?? input.value).trim();
    if (busy || input.disabled || (!text && !attached)) return;
    const file = attached;
    attached = null; $("attachment").hidden = true;
    input.value = ""; autosize();
    addMsg("me", text || "Use this file", file ? `📎 ${file.name}` : "");
    busy = true; send.disabled = true;
    const t = typing();
    try {
      const data = await rpc("chat", { message: text || "use this file", file: file?.path });
      t.remove();
      showImage(addMsg("bot", data.reply).lastChild, data.image);
      renderModels(data.models);
    } catch (e) {
      t.remove();
      addMsg("bot", `Something went wrong: ${e.message}`);
    } finally {
      busy = false; send.disabled = false; input.focus();
    }
  }
  $("composer").onsubmit = (e) => { e.preventDefault(); submit(); };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } });
  const autosize = () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; };
  input.addEventListener("input", autosize);

  // ---- model cards
  function placeholderFor(m) {
    if (m.kind === "forecast") return "steps ahead, e.g. 6";
    if (m.kind === "recommend") return "an item, or: user u3";
    if (m.kind === "rl") return "optional start row,col";
    if (m.kind === "lm" || m.kind === "charnn") return "starting words";
    if (m.kind === "text") return "type some text";
    if (m.kind === "chatbot") return "ask a question";
    if (m.kind === "docqa") return "ask about the document";
    if (m.kind === "summarize") return "number of sentences, or paste text";
    if (m.kind === "translate") return "a sentence to translate";
    if (m.kind === "gpt") return "starting text";
    if (m.kind === "imagegen") return m.usage.replace(/[<>]/g, "") + " (or leave blank)";
    return m.usage.replace(/[<>]/g, "");
  }

  const cards = new Map();  // name -> card element (kept, so test results survive re-renders)
  function renderModels(list) {
    const box = $("models");
    $("count").textContent = list.length;
    if (!list.length) return;
    box.querySelector(".empty")?.remove();
    const fresh = cards.size > 0 || known.size > 0;
    for (const m of [...list].sort((a, b) => b.built - a.built)) {
      if (!cards.has(m.name)) {
        const card = buildCard(m);
        if (fresh && !known.has(m.name)) card.classList.add("new");
        cards.set(m.name, card);
      }
      box.appendChild(cards.get(m.name));  // re-append = reorder newest first
    }
    known = new Set(list.map((m) => m.name));
  }

  function buildCard(m) {
      const card = document.createElement("div");
      card.className = "card";
      const h = document.createElement("h3"); h.textContent = m.name;
      const badge = document.createElement("span"); badge.className = "badge"; badge.textContent = m.type;
      const stats = document.createElement("div"); stats.className = "stats";
      stats.textContent = `${m.params.toLocaleString()} parameters · ${m.size}`;
      const form = document.createElement("form"); form.className = "try";
      const out = document.createElement("div"); out.className = "out";
      let field;
      if (m.kind === "image" || m.kind === "audio") {
        field = document.createElement("input"); field.type = "file";
        field.accept = m.kind === "audio" ? ".wav,audio/wav" : "image/*";
      } else {
        field = document.createElement("input"); field.placeholder = placeholderFor(m);
      }
      field.setAttribute("aria-label", `Input for ${m.name}`);
      const run = document.createElement("button"); run.type = "submit"; run.textContent = "Run";
      form.append(field, run);
      form.onsubmit = async (e) => {
        e.preventDefault();
        out.textContent = "Running…";
        try {
          let value = field.value;
          if (m.kind === "image" || m.kind === "audio") {
            if (!field.files[0]) { out.textContent = `Choose ${m.kind === "audio" ? "a .wav file" : "an image"} first.`; return; }
            value = (await upload(field.files[0])).path;
          }
          const data = await rpc("ask", { name: m.name, input: value });
          out.textContent = "";
          renderText(out, data.reply.replace(/^\[[^\]]+\]\s*/, ""));
          showImage(out, data.image);
        } catch (err) { out.textContent = err.message; }
      };
      card.append(h, badge, stats, form, out);
      return card;
  }

})();
