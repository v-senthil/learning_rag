// ============================================================
// RAG Studio — frontend logic
// Communicates with the FastAPI backend via fetch + NDJSON streaming
// ============================================================

const API = ""; // same origin

// ---------- DOM refs ----------
const $ = (id) => document.getElementById(id);
const els = {
  health: $("health-indicator"),
  themeToggle: $("theme-toggle"),
  dropzone: $("dropzone"),
  fileInput: $("file-input"),
  uploadQueue: $("upload-queue"),
  statChunks: $("stat-chunks"),
  statFiles: $("stat-files"),
  fileList: $("file-list"),
  saveName: $("save-name"),
  saveBtn: $("save-btn"),
  clearStore: $("clear-store"),
  storesList: $("stores-list"),
  refreshStores: $("refresh-stores"),
  llmSelect: $("llm-select"),
  embedSelect: $("embed-select"),
  embedLock: $("embed-lock"),
  refreshModels: $("refresh-models"),
  topkSlider: $("topk-slider"),
  topkValue: $("topk-value"),
  alphaSlider: $("alpha-slider"),
  alphaValue: $("alpha-value"),
  chatContainer: $("chat-container"),
  emptyState: $("empty-state"),
  messages: $("messages"),
  input: $("composer-input"),
  sendBtn: $("send-btn"),
  composerContext: $("composer-context"),
  toasts: $("toasts"),
};

// ---------- State ----------
const state = {
  storeLoaded: false,
  chunkCount: 0,
  fileCount: 0,
  llmModel: "",
  embedModel: "",
  topK: 5,
  alpha: 0.5,
  streaming: false,
};

// ============================================================
// API helpers
// ============================================================
async function apiGet(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(API + path, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
async function apiDelete(path) {
  const r = await fetch(API + path, { method: "DELETE" });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

// ============================================================
// Toasts
// ============================================================
function toast(message, type = "info", duration = 3200) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  els.toasts.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transform = "translateX(12px)";
    setTimeout(() => el.remove(), 240);
  }, duration);
}

// ============================================================
// Theme
// ============================================================
function initTheme() {
  const saved = localStorage.getItem("theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = saved || (prefersDark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", theme);
}
els.themeToggle.addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
});

// ============================================================
// Health + initial state
// ============================================================
async function refreshHealth() {
  try {
    const h = await apiGet("/health");
    els.health.classList.remove("err");
    els.health.classList.add("ok");
    state.chunkCount = h.chunks;
    state.fileCount = h.files;
    state.storeLoaded = h.store_loaded;
    updateStoreUI();
  } catch (e) {
    els.health.classList.remove("ok");
    els.health.classList.add("err");
    toast("Backend unreachable. Is the server running?", "error");
  }
}

async function refreshStoreDetail() {
  try {
    const d = await apiGet("/store/detail");
    els.statChunks.textContent = d.total_chunks;
    els.statFiles.textContent = d.files.length;
    els.fileList.innerHTML = d.files.length
      ? d.files.map(f => `<li><span class="name" title="${esc(f.name)}">${esc(f.name)}</span><span class="count">${f.chunks}</span></li>`).join("")
      : `<li class="empty-line">No files yet.</li>`;
    state.chunkCount = d.total_chunks;
    state.fileCount = d.files.length;
    state.storeLoaded = d.total_chunks > 0;
    updateStoreUI();
  } catch (e) {}
}

function updateStoreUI() {
  // composer context line
  if (state.storeLoaded) {
    els.composerContext.textContent = `${state.chunkCount} chunks · ${state.fileCount} files · top-k=${state.topK} · α=${state.alpha.toFixed(2)}`;
    els.emptyState.classList.add("hidden");
  } else {
    els.composerContext.textContent = "No documents loaded";
  }
  // embedding lock
  els.embedSelect.disabled = state.storeLoaded;
  els.embedLock.classList.toggle("hidden", !state.storeLoaded);
  // send button availability
  els.sendBtn.disabled = !state.storeLoaded || state.streaming;
}

// ============================================================
// Models
// ============================================================
async function refreshModels() {
  try {
    const m = await apiGet("/models");
    state.llmModel = m.current_llm;
    state.embedModel = m.current_embed;
    renderSelect(els.llmSelect, m.llm, m.current_llm);
    renderSelect(els.embedSelect, m.embed, m.current_embed);
  } catch (e) {
    toast("Failed to load models", "error");
  }
}

function renderSelect(selectEl, options, current) {
  selectEl.innerHTML = options.map(o =>
    `<option value="${esc(o)}" ${o === current ? "selected" : ""}>${esc(o)}</option>`
  ).join("");
}

els.llmSelect.addEventListener("change", async () => {
  const model = els.llmSelect.value;
  try {
    await apiPost("/models/llm", { model });
    state.llmModel = model;
    toast(`LLM switched to ${model}`, "success");
  } catch (e) {
    toast("Failed to switch LLM", "error");
  }
});

els.embedSelect.addEventListener("change", async () => {
  const model = els.embedSelect.value;
  try {
    await apiPost("/models/embed", { model });
    state.embedModel = model;
    toast(`Embedding model → ${model}`, "success");
  } catch (e) {
    toast("Clear the store first to change embeddings", "error");
    els.embedSelect.value = state.embedModel;
  }
});

els.refreshModels.addEventListener("click", refreshModels);

// ============================================================
// Settings
// ============================================================
els.topkSlider.addEventListener("input", () => {
  state.topK = parseInt(els.topkSlider.value, 10);
  els.topkValue.textContent = state.topK;
  updateStoreUI();
});
els.alphaSlider.addEventListener("input", () => {
  state.alpha = parseFloat(els.alphaSlider.value);
  els.alphaValue.textContent = state.alpha.toFixed(2);
  updateStoreUI();
});

// ============================================================
// Upload / Ingest
// ============================================================
function setupDropzone() {
  els.dropzone.addEventListener("click", () => els.fileInput.click());
  els.dropzone.addEventListener("dragover", (e) => { e.preventDefault(); els.dropzone.classList.add("drag"); });
  els.dropzone.addEventListener("dragleave", () => els.dropzone.classList.remove("drag"));
  els.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("drag");
    if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
  });
  els.fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleFiles(e.target.files);
  });
}

async function handleFiles(fileList) {
  const files = Array.from(fileList);
  els.uploadQueue.classList.remove("hidden");
  els.uploadQueue.innerHTML = files.map(f =>
    `<div class="upload-item" data-name="${esc(f.name)}">
       <div class="spinner"></div>
       <span class="name">${esc(f.name)}</span>
       <span class="status">uploading…</span>
     </div>`
  ).join("");

  const fd = new FormData();
  files.forEach(f => fd.append("files", f));

  try {
    const r = await fetch("/ingest", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const result = await r.json();

    // mark each item
    els.uploadQueue.querySelectorAll(".upload-item").forEach(item => {
      const name = item.dataset.name;
      const spinner = item.querySelector(".spinner");
      const status = item.querySelector(".status");
      spinner.remove();
      if (result.files_ingested.includes(name)) {
        status.textContent = "✓ ingested";
        status.classList.add("done");
      } else if (result.files_skipped.includes(name)) {
        status.textContent = "skipped";
        status.classList.add("err");
      } else {
        status.textContent = "unknown";
      }
    });

    toast(`Added ${result.new_chunks} chunks from ${result.files_ingested.length} file(s)`, "success");
    await refreshStoreDetail();

    setTimeout(() => els.uploadQueue.classList.add("hidden"), 4000);
  } catch (e) {
    toast(`Upload failed: ${e.message}`, "error");
  }
}

// ============================================================
// Stores
// ============================================================
els.saveBtn.addEventListener("click", async () => {
  const name = els.saveName.value.trim();
  if (!name) { toast("Enter a name first", "error"); return; }
  try {
    await apiPost(`/stores/${encodeURIComponent(name)}/save`);
    toast(`Saved "${name}"`, "success");
    els.saveName.value = "";
    refreshStores();
  } catch (e) {
    toast("Save failed", "error");
  }
});

els.clearStore.addEventListener("click", async () => {
  if (!confirm("Clear all chunks and chat? This cannot be undone.")) return;
  try {
    await apiDelete("/store");
    els.messages.innerHTML = "";
    els.emptyState.classList.remove("hidden");
    await refreshStoreDetail();
    toast("Store cleared", "info");
  } catch (e) { toast("Clear failed", "error"); }
});

async function refreshStores() {
  try {
    const list = await apiGet("/stores");
    if (!list.length) {
      els.storesList.innerHTML = `<div class="empty-line">No saved knowledge bases yet.</div>`;
      return;
    }
    els.storesList.innerHTML = list.map(name =>
      `<div class="store-row" data-name="${esc(name)}">
         <span class="store-name">${esc(name)}</span>
         <span class="load-link">Load →</span>
       </div>`
    ).join("");
    els.storesList.querySelectorAll(".store-row").forEach(row => {
      row.addEventListener("click", () => loadStore(row.dataset.name));
    });
  } catch (e) {}
}

async function loadStore(name) {
  try {
    const r = await apiPost(`/stores/${encodeURIComponent(name)}/load`);
    els.messages.innerHTML = "";
    els.emptyState.classList.remove("hidden");
    await refreshStoreDetail();
    toast(`Loaded "${name}" (${r.chunks} chunks)`, "success");
  } catch (e) { toast("Load failed", "error"); }
}

els.refreshStores.addEventListener("click", refreshStores);

// ============================================================
// Chat — streaming
// ============================================================
function createMessage(role, content = "") {
  const wrap = document.createElement("div");
  wrap.className = "message";
  wrap.innerHTML = `
    <div class="avatar ${role}">${role === "user" ? "U" : "AI"}</div>
    <div class="bubble">
      <div class="bubble-role">${role === "user" ? "You" : "Assistant"}</div>
      <div class="bubble-body"></div>
      <div class="details-group"></div>
    </div>
  `;
  const body = wrap.querySelector(".bubble-body");
  body.textContent = content;
  els.messages.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function scrollToBottom() {
  els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
}

function renderMarkdown(raw) {
  // highlight [filename.ext] as a pill
  const withPills = raw.replace(/\[([^\]]+\.(?:md|pdf|txt|json))\]/gi,
    (_, name) => `<span class="citation-pill">${esc(name)}</span>`);
  try {
    return marked.parse(withPills, { breaks: true, gfm: true });
  } catch {
    return withPills;
  }
}

function renderSources(hits) {
  if (!hits || !hits.length) return "";
  const cards = hits.map((h, i) => `
    <div class="source-card">
      <div class="source-head">
        <span class="source-title"><span class="rank">#${i + 1}</span>${esc(h.source)} · chunk ${h.chunk_index}</span>
        <span class="source-scores">
          <span class="score-pill hybrid">hybrid ${h.score.toFixed(3)}</span>
          <span class="score-pill">vec ${h.vec_score.toFixed(3)}</span>
          <span class="score-pill">bm25 ${h.bm25_score.toFixed(2)}</span>
        </span>
      </div>
      <div class="source-text">${esc(h.text)}</div>
    </div>
  `).join("");
  return `
    <details class="disclosure">
      <summary class="disclosure-summary">
        <span>📄 Sources <span class="badge">${hits.length}</span></span>
        <svg class="disclosure-arrow" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
      </summary>
      <div class="disclosure-content">${cards}</div>
    </details>
  `;
}

function renderPrompt(prompt) {
  if (!prompt) return "";
  return `
    <details class="disclosure">
      <summary class="disclosure-summary">
        <span>🧠 AI Input (prompt) <span class="badge">${prompt.length.toLocaleString()} chars</span></span>
        <svg class="disclosure-arrow" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
      </summary>
      <div class="disclosure-content">
        <div class="prompt-block">${esc(prompt)}</div>
      </div>
    </details>
  `;
}

// ============================================================
// UMAP scatter (custom SVG)
// ============================================================
const UMAP_PALETTE = [
  "#6366f1", "#14b8a6", "#f59e0b", "#ec4899", "#8b5cf6",
  "#10b981", "#f97316", "#06b6d4", "#64748b", "#84cc16",
];

function colorForSource(source, sourceColors) {
  if (sourceColors[source]) return sourceColors[source];
  const idx = Object.keys(sourceColors).length % UMAP_PALETTE.length;
  sourceColors[source] = UMAP_PALETTE[idx];
  return sourceColors[source];
}

function renderUMAPShell(query, sources) {
  // placeholder — data loads lazily when disclosure is opened
  const payload = encodeURIComponent(JSON.stringify({ query, sources }));
  return `
    <details class="disclosure umap-disclosure" data-payload="${payload}">
      <summary class="disclosure-summary">
        <span>🗺️ Embedding Map (UMAP) <span class="badge">lazy</span></span>
        <svg class="disclosure-arrow" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
      </summary>
      <div class="disclosure-content umap-slot"></div>
    </details>
  `;
}

async function loadUMAPIntoSlot(disclosureEl) {
  const slot = disclosureEl.querySelector(".umap-slot");
  if (slot.dataset.loaded === "true") return;
  slot.dataset.loaded = "true";

  const payload = JSON.parse(decodeURIComponent(disclosureEl.dataset.payload));
  slot.innerHTML = `<div class="umap-loading">Computing projection…</div>`;

  try {
    const data = await apiPost("/umap", { query: payload.query });
    if (!data.available) {
      slot.innerHTML = `<div class="umap-empty">${esc(data.reason || "UMAP unavailable.")}</div>`;
      return;
    }
    slot.innerHTML = renderUMAP(data, payload.sources);
    attachUMAPInteractions(slot);
  } catch (e) {
    slot.innerHTML = `<div class="umap-empty" style="color:var(--danger)">Failed: ${esc(e.message)}</div>`;
  }
}

function renderUMAP(data, sources) {
  const W = 600, H = 360;
  const PAD = 24;

  // bounds
  const allX = data.chunks.map(c => c.x);
  const allY = data.chunks.map(c => c.y);
  if (data.query_point) {
    allX.push(data.query_point[0]);
    allY.push(data.query_point[1]);
  }
  const xMin = Math.min(...allX), xMax = Math.max(...allX);
  const yMin = Math.min(...allY), yMax = Math.max(...allY);
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;

  const sx = (x) => PAD + ((x - xMin) / xSpan) * (W - 2 * PAD);
  const sy = (y) => H - PAD - ((y - yMin) / ySpan) * (H - 2 * PAD);

  // color map per source
  const sourceColors = {};
  data.chunks.forEach(c => colorForSource(c.source, sourceColors));

  // retrieved set
  const retrievedKeys = new Set(
    (sources || []).map(s => `${s.source}#${s.chunk_index}`)
  );

  // dots
  const dots = data.chunks.map((c, i) => {
    const key = `${c.source}#${c.chunk_index}`;
    const retrieved = retrievedKeys.has(key);
    return `<circle
      class="dot ${retrieved ? "retrieved" : ""}"
      cx="${sx(c.x)}" cy="${sy(c.y)}" r="5"
      fill="${colorForSource(c.source, sourceColors)}"
      data-source="${esc(c.source)}"
      data-chunk="${c.chunk_index}"
      data-preview="${esc(c.preview)}"
      data-retrieved="${retrieved ? "yes" : "no"}"
    />`;
  }).join("");

  // query point
  let queryMarker = "";
  if (data.query_point) {
    const [qx, qy] = [sx(data.query_point[0]), sy(data.query_point[1])];
    queryMarker = `
      <g>
        <circle class="query-star" cx="${qx}" cy="${qy}" r="9" />
        <text class="query-label" x="${qx}" y="${qy - 14}">QUERY</text>
      </g>
    `;
  }

  // legend
  const legendItems = Object.entries(sourceColors).map(([src, color]) =>
    `<span class="umap-legend-item">
       <span class="umap-legend-dot" style="background:${color}"></span>${esc(src)}
     </span>`
  ).join("");

  return `
    <div class="umap-wrap">
      <div class="umap-caption">
        Each dot is a chunk in 2-D UMAP space.
        <strong style="color:var(--text)">Outlined</strong> dots are the top-k retrieved.
        <span style="color:#ef4444;font-weight:600">Red star</span> = your query.
        UMAP is a <em>lossy</em> projection — trust scores, not visual distance.
      </div>
      <svg class="umap-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
        ${dots}
        ${queryMarker}
      </svg>
      <div class="umap-legend">
        ${legendItems}
        <span class="umap-legend-item umap-legend-special">⬤ outlined = retrieved</span>
        <span class="umap-legend-item umap-legend-special" style="color:#ef4444">★ query</span>
      </div>
      <div class="umap-tooltip"></div>
    </div>
  `;
}

function attachUMAPInteractions(slot) {
  const tooltip = slot.querySelector(".umap-tooltip");
  const wrap = slot.querySelector(".umap-wrap");
  if (!tooltip || !wrap) return;

  slot.querySelectorAll(".dot").forEach(dot => {
    dot.addEventListener("mouseenter", (e) => {
      const { source, chunk, preview, retrieved } = e.target.dataset;
      tooltip.innerHTML = `
        <strong>${esc(source)} · chunk ${chunk}${retrieved === "yes" ? " · <span style='color:#fbbf24'>retrieved</span>" : ""}</strong>
        ${esc(preview)}${preview.length >= 140 ? "…" : ""}
      `;
      tooltip.classList.add("show");
    });
    dot.addEventListener("mousemove", (e) => {
      const rect = wrap.getBoundingClientRect();
      tooltip.style.left = (e.clientX - rect.left + 12) + "px";
      tooltip.style.top = (e.clientY - rect.top + 12) + "px";
    });
    dot.addEventListener("mouseleave", () => {
      tooltip.classList.remove("show");
    });
  });
}

async function sendQuery(query) {
  if (!query.trim() || state.streaming) return;
  if (!state.storeLoaded) {
    toast("Upload or load a knowledge base first", "error");
    return;
  }

  els.emptyState.classList.add("hidden");
  state.streaming = true;
  updateStoreUI();

  createMessage("user", query);
  const aiWrap = createMessage("assistant", "");
  const aiBody = aiWrap.querySelector(".bubble-body");
  const aiDetails = aiWrap.querySelector(".details-group");

  // cursor during streaming
  aiBody.innerHTML = `<span class="typing-cursor"></span>`;

  let buffer = "";
  let rawAnswer = "";
  let meta = null;

  try {
    const res = await fetch("/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: state.topK, alpha: state.alpha }),
    });
    if (!res.ok) throw new Error(await res.text());

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let newlineIdx;
      while ((newlineIdx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newlineIdx).trim();
        buffer = buffer.slice(newlineIdx + 1);
        if (!line) continue;

        let event;
        try { event = JSON.parse(line); } catch { continue; }

        if (event.type === "meta") {
          meta = event;
        } else if (event.type === "token") {
          rawAnswer += event.text;
          aiBody.innerHTML = renderMarkdown(rawAnswer) + `<span class="typing-cursor"></span>`;
          scrollToBottom();
        } else if (event.type === "done") {
          aiBody.innerHTML = renderMarkdown(rawAnswer);
        }
      }
    }

    if (meta) {
      aiDetails.innerHTML =
        renderSources(meta.sources) +
        renderPrompt(meta.prompt) +
        renderUMAPShell(query, meta.sources);

      // lazy-load the UMAP when its disclosure opens (save work if never opened)
      const umapDisc = aiDetails.querySelector(".umap-disclosure");
      if (umapDisc) {
        umapDisc.addEventListener("toggle", () => {
          if (umapDisc.open) loadUMAPIntoSlot(umapDisc);
        });
      }
    }
  } catch (e) {
    aiBody.innerHTML = `<em style="color:var(--danger)">Error: ${esc(e.message)}</em>`;
    toast("Query failed", "error");
  } finally {
    state.streaming = false;
    updateStoreUI();
    els.input.focus();
  }
}

// ============================================================
// Composer behavior
// ============================================================
els.input.addEventListener("input", () => {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
});
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    fireSend();
  }
});
els.sendBtn.addEventListener("click", fireSend);

function fireSend() {
  const q = els.input.value.trim();
  if (!q) return;
  els.input.value = "";
  els.input.style.height = "auto";
  sendQuery(q);
}

// starter prompt chips
document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    els.input.value = chip.textContent;
    els.input.focus();
    els.input.dispatchEvent(new Event("input"));
  });
});

// ============================================================
// Helpers
// ============================================================
function esc(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// ============================================================
// Boot
// ============================================================
(async function boot() {
  initTheme();
  setupDropzone();
  await refreshHealth();
  await refreshStoreDetail();
  await refreshModels();
  await refreshStores();
  els.input.focus();
  // periodic health ping
  setInterval(refreshHealth, 20_000);
})();
