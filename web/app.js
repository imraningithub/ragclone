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
  mode: "chat", // "chat" | "kb", from GET /api/status
  documents: [], // kb mode: [{name, chunk_count}]
};

// ---------- kb mode (DTM Studienassistent) ----------
const DE = String(navigator.language || "").toLowerCase().startsWith("de");
const IS_ADMIN = (() => { try { return new URLSearchParams(location.search).get("admin") === "1"; } catch { return false; } })();
const KB_TITLE = "DTM Studienassistent";
const T = DE ? {
  newChat: "Neuer Chat",
  greeting: "Hallo! Ich beantworte Fragen zu deinem DTM-Studium – ausschließlich auf Basis der offiziellen Dokumente.",
  examplesLabel: "Beispielfragen",
  placeholder: "Frag etwas zu deinem DTM-Studium…",
  disclaimer: "Antworten basieren nur auf den offiziellen DTM-Dokumenten. Verbindlich sind die Originale.",
  page: "S.", openPdf: (p) => `PDF öffnen (S. ${p})`, openDoc: "Dokument öffnen",
  docs: (n) => `${n} Dokument${n === 1 ? "" : "e"}`, noDocs: "Keine Dokumente geladen.",
  tooMany: "Zu viele Anfragen – bitte kurz warten.", notice: "Hinweis",
  tokenPrompt: "Admin-Token eingeben:", tokenMissing: "Admin-Token erforderlich.", tokenBad: "Admin-Token ungültig.",
  message: "Nachricht",
} : {
  newChat: "New chat",
  greeting: "Hi! I answer questions about your DTM studies – based only on the official documents.",
  examplesLabel: "Example questions",
  placeholder: "Ask something about your DTM studies…",
  disclaimer: "Answers are based only on the official DTM documents. The originals are binding.",
  page: "p.", openPdf: (p) => `Open PDF (p. ${p})`, openDoc: "Open document",
  docs: (n) => `${n} document${n === 1 ? "" : "s"}`, noDocs: "No documents loaded.",
  tooMany: "Too many requests – please wait a moment.", notice: "Notice",
  tokenPrompt: "Enter admin token:", tokenMissing: "Admin token required.", tokenBad: "Invalid admin token.",
  message: "Message",
};
const EXAMPLES = DE
  ? ["Wie viele ECTS-Punkte hat das erste Semester?", "Wie melde ich mich zu Prüfungen an?", "Welche Module habe ich im ersten Semester?", "Wie oft darf ich eine Prüfung wiederholen?"]
  : ["How many ECTS credits is the first semester?", "How do I register for exams?", "Which modules do I take in the first semester?", "How often can I retake an exam?"];
const isKb = () => state.mode === "kb";
const defaultTitle = () => (isKb() ? KB_TITLE : "New chat");
// Backend names fresh threads "New chat"; in kb mode show the localized/default label instead.
const threadLabel = (title) => (isKb() && (!title || title === "New chat") ? T.newChat : title || "New chat");
const headerTitle = (title) => (isKb() && (!title || title === "New chat") ? KB_TITLE : title || "New chat");
const pageOf = (c) => {
  const p = c && (c.page ?? (c.meta && c.meta.page));
  return p === null || p === undefined || p === "" ? null : p;
};
const ADMIN_KEY = "dtm_admin_token";
function getAdminToken() {
  let tok = null;
  try { tok = sessionStorage.getItem(ADMIN_KEY); } catch { /* storage blocked */ }
  if (!tok) {
    tok = (prompt(T.tokenPrompt) || "").trim() || null;
    if (tok) { try { sessionStorage.setItem(ADMIN_KEY, tok); } catch { /* storage blocked */ } }
  }
  return tok;
}
function clearAdminToken() { try { sessionStorage.removeItem(ADMIN_KEY); } catch { /* storage blocked */ } }

const el = {
  threadList: $("threadList"), threadEmpty: $("threadEmpty"), status: $("status"),
  messages: $("messages"), emptyState: $("emptyState"), title: $("threadTitle"),
  attachments: $("attachments"), input: $("input"), composer: $("composer"),
  sendBtn: $("sendBtn"), stopBtn: $("stopBtn"), attachBtn: $("attachBtn"), fileInput: $("fileInput"),
  sidebar: $("sidebar"), scrim: $("scrim"), toggle: $("toggleSidebar"),
  main: $("main"), dropOverlay: $("dropOverlay"), toasts: $("toasts"),
  settingsDialog: $("settingsDialog"), settingsForm: $("settingsForm"), settingsError: $("settingsError"),
  parseDialog: $("parseDialog"), parseFile: $("parseFile"),
  brand: $("brand"), newChatLabel: $("newChatLabel"), sources: $("sources"), sourceList: $("sourceList"),
  sourceCount: $("sourceCount"), emptyText: $("emptyText"), examples: $("examples"), disclaimer: $("disclaimer"),
  openSettings: $("openSettings"),
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

async function api(path, { method = "GET", json, body, signal, headers } = {}) {
  const opts = { method, signal, headers: { ...(headers || {}) } };
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
const ICON_INFO = '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>';
const ICON_EXT = '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3"/>';

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
    const s = (await api("/status")) || {};
    const tech = `${s.provider} · ${s.chat_model} · ${s.embed_model} · ANN ${s.ann_available ? "on" : "off"}`;
    if (s.mode === "kb") {
      state.documents = Array.isArray(s.documents) ? s.documents.filter((d) => d && d.name) : [];
      applyKbMode();
      el.status.textContent = T.docs(state.documents.length);
      el.status.title = tech;
    } else {
      el.status.textContent = tech;
      el.status.title = el.status.textContent;
    }
  } catch (e) { el.status.textContent = `Status unavailable: ${e.message}`; }
}

// Switches the static chat-mode markup into the DTM Studienassistent layout. Called once, when status says "kb".
function applyKbMode() {
  state.mode = "kb";
  document.body.classList.add("kb-mode");
  document.documentElement.lang = DE ? "de" : "en";
  document.title = KB_TITLE;
  el.brand.hidden = false;
  el.newChatLabel.textContent = T.newChat;
  el.attachBtn.hidden = true; el.fileInput.disabled = true;
  el.dropOverlay.hidden = true;
  state.files = []; state.uploading = []; renderAttachments();
  el.openSettings.hidden = !IS_ADMIN;
  el.input.placeholder = T.placeholder;
  const lbl = document.querySelector('label[for="input"]');
  if (lbl) lbl.textContent = T.message;
  el.disclaimer.textContent = T.disclaimer; el.disclaimer.hidden = false;
  el.emptyText.textContent = T.greeting;
  el.examples.setAttribute("role", "group");
  el.examples.setAttribute("aria-label", T.examplesLabel);
  el.examples.replaceChildren(...EXAMPLES.map((q) => h("button", {
    type: "button", class: "example-chip", onclick: () => { if (state.controller) return; el.input.value = q; sendMessage(); },
  }, q)));
  el.examples.hidden = false;
  if (!state.currentId) el.title.textContent = KB_TITLE;
  renderSources();
  renderThreads();
}

function renderSources() {
  const docs = state.documents;
  el.sourceCount.textContent = String(docs.length);
  el.sourceList.replaceChildren(...(docs.length ? docs.map((d) => h("li", {},
    h("a", { href: `/kb/${encodeURIComponent(d.name)}`, target: "_blank", rel: "noopener", title: `${d.name} – ${T.openDoc}` },
      svg(ICON_FILE), h("span", { class: "file-name" }, d.name)),
    typeof d.chunk_count === "number" ? h("span", { class: "badge" }, String(d.chunk_count)) : null,
  )) : [h("li", { class: "muted" }, T.noDocs)]));
  el.sources.hidden = false;
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
  const label = h("span", { class: "thread-name" }, threadLabel(t.title));
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
    if (t.id === state.currentId) el.title.textContent = headerTitle(t.title);
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
  el.title.textContent = defaultTitle();
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
    el.title.textContent = headerTitle(t.title);
    renderMessages(t.messages || []);
    if (!isKb()) await loadFiles();
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
  el.title.textContent = headerTitle(t.title);
  renderThreads();
  return t.id;
}

// ---------- messages ----------
function renderMessages(messages) {
  el.messages.replaceChildren(el.emptyState);
  el.emptyState.hidden = messages.length > 0;
  for (const m of messages) {
    const node = messageNode(m.role);
    const cites = Array.isArray(m.citations) ? m.citations : [];
    // Stored kb messages carry no refusal flag; accepted kb answers always have citations,
    // so an assistant message without any is treated as a refusal (unless m.refused says otherwise).
    const refused = isKb() && m.role !== "user" && (m.refused !== undefined ? !!m.refused : cites.length === 0);
    fillMessage(node, m.content, refused ? [] : cites);
    if (refused) markRefused(node);
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

// Refused kb answer: muted info bubble, no citations.
function markRefused(node) {
  if (node.classList.contains("refused")) return;
  node.classList.add("refused");
  node.setAttribute("aria-label", T.notice);
  renderCitations(node, []);
  node._cites.hidden = true;
  const icon = svg(ICON_INFO);
  icon.classList.add("info-icon");
  node.insertBefore(icon, node._body);
}

let citeSeq = 0;
function renderCitations(node, citations) {
  node._citations = citations;
  node._cites.replaceChildren(...citations.filter(Boolean).map((c) => {
    const score = typeof c.score === "number" ? c.score.toFixed(2) : c.score;
    const page = pageOf(c);
    const file = c.file_name || "?";
    const excerptId = `excerpt-${++citeSeq}`;
    const excerpt = h("div", { class: "excerpt", id: excerptId, hidden: true }, h("div", { class: "excerpt-text" }, c.text || ""));
    if (isKb() && c.file_name) {
      const href = `/kb/${encodeURIComponent(c.file_name)}${page != null ? `#page=${encodeURIComponent(page)}` : ""}`;
      excerpt.append(h("a", { class: "excerpt-link", href, target: "_blank", rel: "noopener" },
        page != null ? T.openPdf(page) : T.openDoc, svg(ICON_EXT)));
    }
    const label = page != null
      ? `[${c.n}] ${file} · ${DE ? "S." : "p."} ${page}`
      : `[${c.n}] ${file} · chunk ${c.chunk_index} · ${score}`;
    const chip = h("button", {
      type: "button", class: "chip cite-chip", "aria-expanded": "false", "aria-controls": excerptId, "data-n": c.n,
      title: page != null && typeof c.score === "number" ? `${file} · ${score}` : file,
      onclick: () => {
        const open = excerpt.hidden;
        excerpt.hidden = !open;
        chip.setAttribute("aria-expanded", String(open));
      },
    }, label);
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
  let answer = "", frame = 0, started = false;
  const paint = () => { frame = 0; botNode._body.innerHTML = renderMarkdown(answer); scrollToBottom(); };

  try {
    const res = await fetch(`/api/threads/${threadId}/chat`, {
      method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ message: text }), signal: controller.signal,
    });
    if (!res.ok) {
      const err = await errorFrom(res);
      if (err.status === 429) err.message = err.detail = T.tooMany;
      throw err;
    }
    started = true;
    for await (const { event, data } of sseEvents(res)) {
      if (event === "citations") renderCitations(botNode, Array.isArray(data) ? data : []);
      else if (event === "token") { answer += (data && data.text) || ""; if (!frame) frame = requestAnimationFrame(paint); }
      // kb mode: the final check changed what was streamed (e.g. a late refusal) -> show the final text instead
      else if (event === "reset") { answer = (data && data.text) || ""; if (!frame) frame = requestAnimationFrame(paint); }
      else if (event === "done") {
        if (data && data.refused) markRefused(botNode);
        const t = state.threads.find((x) => x.id === threadId);
        if (t && data && data.title) { t.title = data.title; if (state.currentId === threadId) el.title.textContent = headerTitle(data.title); }
        if (t) { t.updated_at = Date.now() / 1000; state.threads = [t, ...state.threads.filter((x) => x !== t)]; }
        renderThreads();
      } else if (event === "error") throw new ApiError(500, formatDetail(data && data.detail, "Generation failed"));
    }
  } catch (e) {
    if (e.name === "AbortError") answer += answer ? "\n\n*[stopped]*" : "*[stopped]*";
    else if (isKb() && !started && (e.status === 429 || e.status === 400)) {
      // Rejected before streaming (rate limit / invalid question): nothing was answered, so drop the
      // placeholder bubbles and give the question back to the user.
      toast(e.message);
      if (frame) { cancelAnimationFrame(frame); frame = 0; }
      userNode.remove(); botNode.remove();
      if (!el.input.value) { el.input.value = text; autosize(); }
      el.emptyState.hidden = !!el.messages.querySelector(".msg");
      if (state.controller === controller) { state.controller = null; setStreaming(false); }
      return;
    }
    else { toast(e.message); botNode.classList.add("error"); if (!answer) answer = `*Error: ${e.message}*`; }
  } finally {
    if (frame) cancelAnimationFrame(frame);
    if (botNode.isConnected) paint();
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
  el.attachments.hidden = isKb() || chips.length + pending.length === 0;
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
  if (!files.length || isKb()) return;
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
  let headers;
  if (isKb()) {
    const tok = getAdminToken();
    if (!tok) { showSettingsError(T.tokenMissing); return; }
    headers = { "X-Admin-Token": tok };
  }
  try {
    state.settings = await api("/settings", { method: "PUT", json: payload, headers });
    updateAttachEnabled();
    el.settingsDialog.close();
    toast("Settings saved", "ok");
  } catch (err) {
    if (isKb() && (err.status === 401 || err.status === 403)) { clearAdminToken(); showSettingsError(`${T.tokenBad} (${err.message})`); }
    else showSettingsError(err.message);
  }
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
  el.attachBtn.addEventListener("click", () => { if (!isKb()) el.fileInput.click(); });
  el.fileInput.addEventListener("change", () => { const fl = [...el.fileInput.files]; el.fileInput.value = ""; uploadFiles(fl); });

  let depth = 0;
  // In kb mode drag-and-drop is ignored entirely (no overlay, no preventDefault, no upload).
  const hasFiles = (e) => !isKb() && [...(e.dataTransfer?.types || [])].includes("Files");
  el.main.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; el.dropOverlay.hidden = false; });
  el.main.addEventListener("dragover", (e) => { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } });
  el.main.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) el.dropOverlay.hidden = true; });
  el.main.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); depth = 0; el.dropOverlay.hidden = true;
    uploadFiles(e.dataTransfer.files);
  });
  // kb mode: swallow file drops so the browser doesn't navigate away to the dropped file.
  const isFileDrag = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  for (const type of ["dragover", "drop"]) {
    window.addEventListener(type, (e) => {
      if (!isKb() || !isFileDrag(e)) return;
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = "none";
    });
  }
}

async function init() {
  bindEvents();
  renderAttachments();
  await Promise.all([loadStatus(), loadSettings(), loadThreads()]);
  el.input.focus();
}

init();
