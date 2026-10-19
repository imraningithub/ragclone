// RAG chat UI — vanilla JS, talks to the FastAPI backend described in CONTRACTS.md.
const $ = (id) => document.getElementById(id);
const ACCEPTED = [".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".json", ".html"];

const state = {
  threads: [],
  currentId: null,
  files: [],
  uploading: [], // {key, name}
  settings: null,
  controller: null, // AbortController while streaming
};

const el = {
  threadList: $("threadList"), threadEmpty: $("threadEmpty"), status: $("status"),
  messages: $("messages"), emptyState: $("emptyState"), title: $("threadTitle"),
  attachments: $("attachments"), input: $("input"), composer: $("composer"),
  sendBtn: $("sendBtn"), stopBtn: $("stopBtn"), attachBtn: $("attachBtn"), fileInput: $("fileInput"),
  sidebar: $("sidebar"), scrim: $("scrim"), toggle: $("toggleSidebar"),
  main: $("main"), dropOverlay: $("dropOverlay"), toasts: $("toasts"),
  settingsDialog: $("settingsDialog"), settingsForm: $("settingsForm"), settingsError: $("settingsError"),
  parseDialog: $("parseDialog"), parseFile: $("parseFile"),
};

// ---------- helpers ----------
class ApiError extends Error {
  constructor(status, detail) { super(detail); this.status = status; this.detail = detail; }
}

function formatDetail(detail, fallback) {
  if (!detail) return fallback;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  return JSON.stringify(detail);
}

async function errorFrom(res) {
  let detail = null;
  try { detail = (await res.json()).detail; } catch { /* non-JSON body */ }
  return new ApiError(res.status, formatDetail(detail, `${res.status} ${res.statusText || "Request failed"}`));
}

async function api(path, { method = "GET", json, body, signal } = {}) {
  const opts = { method, signal, headers: {} };
  if (json !== undefined) { opts.body = JSON.stringify(json); opts.headers["Content-Type"] = "application/json"; }
  else if (body) opts.body = body;
  let res;
  try { res = await fetch(`/api${path}`, opts); }
  catch (e) { if (e.name === "AbortError") throw e; throw new ApiError(0, "Cannot reach the server"); }
  if (!res.ok) throw await errorFrom(res);
  return res.status === 204 ? null : res.json();
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

function svg(path) {
  const span = document.createElement("span");
  span.className = "svg-wrap";
  span.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${path}</svg>`; // static, trusted markup
  return span;
}
const ICON_X = '<path d="M18 6L6 18M6 6l12 12"/>';
const ICON_TRASH = '<path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/>';
const ICON_FILE = '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>';

function toast(msg, kind = "error") {
  const t = h("div", { class: `toast ${kind}` }, msg);
  el.toasts.append(t);
  setTimeout(() => t.classList.add("hide"), 4500);
  setTimeout(() => t.remove(), 5000);
}

// ---------- minimal safe markdown ----------
// Everything is escaped first; only our own tags are introduced afterwards.
function inlineMd(text) {
  const codes = [];
  let s = escapeHtml(text).replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/\[(\d{1,3})\]/g, '<button type="button" class="cite-ref" data-n="$1" aria-label="Show source $1">[$1]</button>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[+i]}</code>`);
}

function renderMarkdown(src) {
  const out = [];
  const parts = String(src || "").replace(/\r\n/g, "\n").split(/^```/m);
  parts.forEach((part, i) => {
    if (i % 2 === 1) { // fenced code block (may be unterminated while streaming)
      const nl = part.indexOf("\n");
      const lang = nl >= 0 ? part.slice(0, nl).trim() : "";
      const code = nl >= 0 ? part.slice(nl + 1) : "";
      out.push(`<pre><code${lang ? ` data-lang="${escapeHtml(lang)}"` : ""}>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
      return;
    }
    for (const block of part.split(/\n{2,}/)) {
      const lines = block.split("\n").filter((l) => l.trim() !== "");
      if (!lines.length) continue;
      const ul = lines.every((l) => /^\s*[-*+]\s+/.test(l));
      const ol = lines.every((l) => /^\s*\d+[.)]\s+/.test(l));
      const head = lines.length === 1 && /^(#{1,6})\s+(.*)$/.exec(lines[0]);
      if (ul || ol) {
        const tag = ul ? "ul" : "ol";
        const items = lines.map((l) => `<li>${inlineMd(l.replace(/^\s*(?:[-*+]|\d+[.)])\s+/, ""))}</li>`).join("");
        out.push(`<${tag}>${items}</${tag}>`);
      } else if (head) {
        out.push(`<p class="md-h"><strong>${inlineMd(head[2])}</strong></p>`);
      } else {
        out.push(`<p>${lines.map(inlineMd).join("<br>")}</p>`);
      }
    }
  });
  return out.join("");
}

// ---------- sidebar / threads ----------
async function loadStatus() {
  try {
    const s = await api("/status");
    el.status.textContent = `${s.provider} · ${s.chat_model} · ${s.embed_model} · ANN ${s.ann_available ? "on" : "off"}`;
    el.status.title = el.status.textContent;
  } catch (e) { el.status.textContent = `Status unavailable: ${e.message}`; }
}

async function loadSettings() {
  try { state.settings = await api("/settings"); } catch { state.settings = null; }
  updateAttachEnabled();
}

function updateAttachEnabled() {
  const on = !state.settings || state.settings.enabled !== false;
  el.attachBtn.disabled = !on;
  el.attachBtn.title = on ? "Attach files" : "Attachments are disabled in settings";
}

async function loadThreads() {
  try { state.threads = (await api("/threads")) || []; }
  catch (e) { state.threads = []; toast(`Could not load chats: ${e.message}`); }
  renderThreads();
}

function renderThreads() {
  el.threadList.replaceChildren(...state.threads.map(threadItem));
  el.threadEmpty.hidden = state.threads.length > 0;
}

function threadItem(t) {
  const active = t.id === state.currentId;
  const label = h("span", { class: "thread-name" }, t.title || "New chat");
  const open = h("button", {
    type: "button", class: "thread-open", "aria-current": active ? "page" : null,
    title: "Open (double-click or F2 to rename)",
    onclick: () => openThread(t.id),
    ondblclick: (e) => { e.preventDefault(); startRename(t, li); },
    onkeydown: (e) => { if (e.key === "F2") { e.preventDefault(); startRename(t, li); } },
  }, label);
  const del = h("button", {
    type: "button", class: "icon-btn thread-del", "aria-label": `Delete chat ${t.title || ""}`, title: "Delete",
    onclick: () => deleteThread(t),
  }, svg(ICON_TRASH));
  const li = h("li", { class: `thread${active ? " active" : ""}`, "data-id": t.id }, open, del);
  return li;
}

function startRename(t, li) {
  const input = h("input", { class: "rename-input", value: t.title || "", "aria-label": "Chat title", maxlength: "200" });
  let done = false;
  const finish = async (save) => {
    if (done) return; done = true;
    const title = input.value.trim();
    if (save && title && title !== t.title) {
      try { Object.assign(t, await api(`/threads/${t.id}`, { method: "PATCH", json: { title } })); }
      catch (e) { toast(`Rename failed: ${e.message}`); }
    }
    renderThreads();
    if (t.id === state.currentId) el.title.textContent = t.title || "New chat";
    el.threadList.querySelector(`[data-id="${CSS.escape(t.id)}"] .thread-open`)?.focus();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    else if (e.key === "Escape") { e.preventDefault(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
  li.replaceChildren(input);
  input.focus(); input.select();
}

async function deleteThread(t) {
  if (!confirm(`Delete "${t.title || "New chat"}" and its attachments?`)) return;
  try { await api(`/threads/${t.id}`, { method: "DELETE" }); }
  catch (e) { if (e.status !== 404) { toast(`Delete failed: ${e.message}`); return; } }
  state.threads = state.threads.filter((x) => x.id !== t.id);
  if (t.id === state.currentId) newChat();
  renderThreads();
}

function newChat() {
  stopStreaming();
  state.currentId = null; state.files = []; state.uploading = [];
  el.title.textContent = "New chat";
  el.messages.replaceChildren(el.emptyState);
  el.emptyState.hidden = false;
  renderAttachments(); renderThreads(); closeSidebar();
  el.input.focus();
}

async function openThread(id) {
  if (id === state.currentId) { closeSidebar(); return; }
  stopStreaming();
  state.currentId = id; state.files = []; state.uploading = [];
  renderThreads(); closeSidebar();
  try {
    const t = await api(`/threads/${id}`);
    if (state.currentId !== id) return;
    el.title.textContent = t.title || "New chat";
    renderMessages(t.messages || []);
    await loadFiles();
  } catch (e) {
    if (e.status === 404) {
      toast("That chat no longer exists");
      state.threads = state.threads.filter((x) => x.id !== id);
      newChat();
    } else toast(`Could not open chat: ${e.message}`);
  }
}

async function ensureThread() {
  if (state.currentId) return state.currentId;
  const t = await api("/threads", { method: "POST", json: {} });
  state.threads.unshift(t);
  state.currentId = t.id;
  el.title.textContent = t.title || "New chat";
  renderThreads();
  return t.id;
}

// ---------- messages ----------
function renderMessages(messages) {
  el.messages.replaceChildren(el.emptyState);
  el.emptyState.hidden = messages.length > 0;
  for (const m of messages) {
    const node = messageNode(m.role);
    fillMessage(node, m.content, m.citations || []);
    el.messages.append(node);
  }
  scrollToBottom(true);
}

function messageNode(role) {
  const body = h("div", { class: "bubble" });
  const cites = h("div", { class: "citations" });
  const node = h("article", { class: `msg ${role === "user" ? "user" : "assistant"}`, "aria-label": role === "user" ? "You" : "Assistant" }, body, cites);
  node._body = body; node._cites = cites;
  return node;
}

function fillMessage(node, content, citations) {
  if (node.classList.contains("user")) node._body.textContent = content;
  else node._body.innerHTML = renderMarkdown(content); // renderMarkdown escapes all input
  if (citations) renderCitations(node, citations);
}

function renderCitations(node, citations) {
  node._citations = citations;
  node._cites.replaceChildren(...citations.map((c) => {
    const score = typeof c.score === "number" ? c.score.toFixed(2) : c.score;
    const excerpt = h("div", { class: "excerpt", hidden: true }, c.text || "");
    const chip = h("button", {
      type: "button", class: "chip cite-chip", "aria-expanded": "false", "data-n": c.n,
      title: c.file_name,
      onclick: () => {
        const open = excerpt.hidden;
        excerpt.hidden = !open;
        chip.setAttribute("aria-expanded", String(open));
      },
    }, `[${c.n}] ${c.file_name} · chunk ${c.chunk_index} · ${score}`);
    return h("div", { class: "cite" }, chip, excerpt);
  }));
}

function scrollToBottom(force = false) {
  const m = el.messages;
  if (force || m.scrollHeight - m.scrollTop - m.clientHeight < 120) m.scrollTop = m.scrollHeight;
}

// Clicking an inline [n] marker expands the matching citation.
el.messages.addEventListener("click", (e) => {
  const ref = e.target.closest(".cite-ref");
  if (!ref) return;
  const chip = ref.closest(".msg")?.querySelector(`.cite-chip[data-n="${ref.dataset.n}"]`);
  if (chip) { if (chip.getAttribute("aria-expanded") !== "true") chip.click(); chip.focus(); chip.scrollIntoView({ block: "nearest" }); }
});

// ---------- chat streaming ----------
function setStreaming(on) {
  el.sendBtn.hidden = on; el.stopBtn.hidden = !on;
  el.messages.setAttribute("aria-busy", String(on));
}

function stopStreaming() {
  if (state.controller) { state.controller.abort(); state.controller = null; }
  setStreaming(false);
}

async function* sseEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const parse = (block) => {
    let event = "message"; const data = [];
    for (const line of block.split("\n")) {
      if (!line || line.startsWith(":")) continue;
      const i = line.indexOf(":");
      const field = i < 0 ? line : line.slice(0, i);
      let value = i < 0 ? "" : line.slice(i + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") event = value;
      else if (field === "data") data.push(value);
    }
    if (!data.length) return null;
    const raw = data.join("\n");
    try { return { event, data: JSON.parse(raw) }; } catch { return { event, data: raw }; }
  };
  for (;;) {
    const { value, done } = await reader.read();
    buf += decoder.decode(value || new Uint8Array(), { stream: !done });
    buf = buf.replace(/\r\n?/g, "\n");
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const ev = parse(buf.slice(0, idx));
      buf = buf.slice(idx + 2);
      if (ev) yield ev;
    }
    if (done) { const ev = buf.trim() && parse(buf); if (ev) yield ev; return; }
  }
}

async function sendMessage() {
  const text = el.input.value.trim();
  if (!text || state.controller) return;
  let threadId;
  try { threadId = await ensureThread(); } catch (e) { toast(`Could not create chat: ${e.message}`); return; }

  el.input.value = ""; autosize();
  el.emptyState.hidden = true;
  const userNode = messageNode("user");
  fillMessage(userNode, text, null);
  const botNode = messageNode("assistant");
  botNode.classList.add("streaming");
  botNode._body.innerHTML = '<span class="typing" aria-label="Assistant is thinking"><i></i><i></i><i></i></span>';
  el.messages.append(userNode, botNode);
  scrollToBottom(true);

  const controller = new AbortController();
  state.controller = controller;
  setStreaming(true);
  let answer = "", frame = 0;
  const paint = () => { frame = 0; botNode._body.innerHTML = renderMarkdown(answer); scrollToBottom(); };

  try {
    const res = await fetch(`/api/threads/${threadId}/chat`, {
      method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ message: text }), signal: controller.signal,
    });
    if (!res.ok) throw await errorFrom(res);
    for await (const { event, data } of sseEvents(res)) {
      if (event === "citations") renderCitations(botNode, Array.isArray(data) ? data : []);
      else if (event === "token") { answer += (data && data.text) || ""; if (!frame) frame = requestAnimationFrame(paint); }
      else if (event === "done") {
        const t = state.threads.find((x) => x.id === threadId);
        if (t && data && data.title) { t.title = data.title; if (state.currentId === threadId) el.title.textContent = data.title; }
        if (t) { t.updated_at = Date.now() / 1000; state.threads = [t, ...state.threads.filter((x) => x !== t)]; }
        renderThreads();
      } else if (event === "error") throw new ApiError(500, formatDetail(data && data.detail, "Generation failed"));
    }
  } catch (e) {
    if (e.name === "AbortError") answer += answer ? "\n\n*[stopped]*" : "*[stopped]*";
    else { toast(e.message); botNode.classList.add("error"); if (!answer) answer = `*Error: ${e.message}*`; }
  } finally {
    if (frame) cancelAnimationFrame(frame);
    paint();
    botNode.classList.remove("streaming");
    if (state.controller === controller) { state.controller = null; setStreaming(false); }
  }
}

// ---------- attachments ----------
async function loadFiles() {
  const id = state.currentId;
  if (!id) { state.files = []; renderAttachments(); return; }
  try {
    const files = await api(`/threads/${id}/files`);
    if (state.currentId === id) state.files = files || [];
  } catch (e) { if (e.status !== 404) toast(`Could not load attachments: ${e.message}`); state.files = []; }
  renderAttachments();
}

function renderAttachments() {
  const chips = state.files.map((f) => {
    const badge = f.mode === "embeddings" ? `embedded · ${f.chunk_count ?? 0} chunk${f.chunk_count === 1 ? "" : "s"}` : "inline";
    return h("div", { class: "file-chip", title: f.name },
      svg(ICON_FILE),
      h("span", { class: "file-name" }, f.name),
      h("span", { class: `badge ${f.mode === "embeddings" ? "emb" : "inl"}` }, badge),
      h("button", { type: "button", class: "chip-x", "aria-label": `Remove ${f.name}`, onclick: () => removeFile(f) }, svg(ICON_X)));
  });
  const pending = state.uploading.map((u) => h("div", { class: "file-chip uploading", "aria-busy": "true" },
    h("span", { class: "spinner", "aria-hidden": "true" }), h("span", { class: "file-name" }, u.name),
    h("span", { class: "badge" }, "uploading…")));
  el.attachments.replaceChildren(...chips, ...pending);
  el.attachments.hidden = chips.length + pending.length === 0;
}

async function removeFile(f) {
  const id = state.currentId;
  try { await api(`/threads/${id}/files/${f.id}`, { method: "DELETE" }); }
  catch (e) { if (e.status !== 404) { toast(`Remove failed: ${e.message}`); return; } }
  state.files = state.files.filter((x) => x.id !== f.id);
  renderAttachments();
}

function askParseMode(name) {
  return new Promise((resolve) => {
    el.parseFile.textContent = name;
    el.parseDialog.returnValue = "";
    el.parseDialog.addEventListener("close", () => {
      const v = el.parseDialog.returnValue;
      resolve(v === "inline" || v === "embeddings" ? v : null);
    }, { once: true });
    el.parseDialog.showModal();
  });
}

async function uploadFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  if (state.settings && state.settings.enabled === false) { toast("Attachments are disabled in settings"); return; }
  if (!state.settings) await loadSettings();
  for (const file of files) {
    const ext = file.name.includes(".") ? file.name.slice(file.name.lastIndexOf(".")).toLowerCase() : "";
    if (!ACCEPTED.includes(ext)) { toast(`${file.name}: unsupported file type`); continue; }
    let parseMode = null;
    if (state.settings && state.settings.parse_mode === "prompt") {
      parseMode = await askParseMode(file.name);
      if (!parseMode) continue;
    }
    let threadId;
    try { threadId = await ensureThread(); } catch (e) { toast(`Could not create chat: ${e.message}`); return; }
    const form = new FormData();
    form.append("file", file, file.name);
    if (parseMode) form.append("parse_mode", parseMode);
    const entry = { key: Math.random().toString(36).slice(2), name: file.name };
    state.uploading.push(entry); renderAttachments();
    try {
      const info = await api(`/threads/${threadId}/files`, { method: "POST", body: form });
      if (state.currentId === threadId && info && !state.files.some((x) => x.id === info.id)) state.files.push(info);
    } catch (e) {
      const known = { 409: "Already attached", 413: "File too large", 415: "Unsupported file type", 503: "Service unavailable" };
      toast(`${file.name}: ${e.detail || known[e.status] || e.message}`);
    } finally {
      state.uploading = state.uploading.filter((u) => u !== entry);
      if (state.currentId === threadId) renderAttachments();
    }
  }
}

// ---------- settings ----------
const NUMERIC = { auto_inline_context_ratio: parseFloat, retrieval_threshold: parseFloat, max_file_size_mb: (v) => parseInt(v, 10), retrieval_limit: (v) => parseInt(v, 10), chunk_size_chars: (v) => parseInt(v, 10), overlap_chars: (v) => parseInt(v, 10) };

async function openSettings() {
  el.settingsError.hidden = true;
  try { state.settings = await api("/settings"); }
  catch (e) { toast(`Could not load settings: ${e.message}`); return; }
  const f = el.settingsForm.elements;
  for (const [k, v] of Object.entries(state.settings)) {
    const input = f.namedItem(k);
    if (!input) continue;
    if (input.type === "checkbox") input.checked = !!v; else input.value = v;
  }
  el.settingsDialog.showModal();
}

async function saveSettings(e) {
  e.preventDefault();
  const f = el.settingsForm.elements;
  const payload = { enabled: f.namedItem("enabled").checked, parse_mode: f.namedItem("parse_mode").value, search_mode: f.namedItem("search_mode").value };
  for (const [k, conv] of Object.entries(NUMERIC)) {
    const n = conv(f.namedItem(k).value);
    if (Number.isNaN(n)) { showSettingsError(`${k} must be a number`); return; }
    payload[k] = n;
  }
  try {
    state.settings = await api("/settings", { method: "PUT", json: payload });
    updateAttachEnabled();
    el.settingsDialog.close();
    toast("Settings saved", "ok");
  } catch (err) { showSettingsError(err.message); }
}

function showSettingsError(msg) { el.settingsError.textContent = msg; el.settingsError.hidden = false; }

// ---------- layout / events ----------
function openSidebar() { el.sidebar.classList.add("open"); el.scrim.hidden = false; el.toggle.setAttribute("aria-expanded", "true"); }
function closeSidebar() { el.sidebar.classList.remove("open"); el.scrim.hidden = true; el.toggle.setAttribute("aria-expanded", "false"); }

function autosize() {
  el.input.style.height = "auto";
  el.input.style.height = `${Math.min(el.input.scrollHeight, 200)}px`;
}

function bindEvents() {
  $("newChat").addEventListener("click", newChat);
  $("openSettings").addEventListener("click", openSettings);
  $("settingsCancel").addEventListener("click", () => el.settingsDialog.close());
  el.settingsForm.addEventListener("submit", saveSettings);
  el.toggle.addEventListener("click", () => (el.sidebar.classList.contains("open") ? closeSidebar() : openSidebar()));
  el.scrim.addEventListener("click", closeSidebar);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && el.sidebar.classList.contains("open")) closeSidebar(); });

  el.composer.addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });
  el.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendMessage(); }
  });
  el.input.addEventListener("input", autosize);
  el.stopBtn.addEventListener("click", stopStreaming);
  el.attachBtn.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", () => { const fl = [...el.fileInput.files]; el.fileInput.value = ""; uploadFiles(fl); });

  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  el.main.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; el.dropOverlay.hidden = false; });
  el.main.addEventListener("dragover", (e) => { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } });
  el.main.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) el.dropOverlay.hidden = true; });
  el.main.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); depth = 0; el.dropOverlay.hidden = true;
    uploadFiles(e.dataTransfer.files);
  });
}

async function init() {
  bindEvents();
  renderAttachments();
  await Promise.all([loadStatus(), loadSettings(), loadThreads()]);
  el.input.focus();
}

init();
