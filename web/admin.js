// Admin cost dashboard. The page holds no data: everything comes from /api/admin/costs with the admin token,
// which is asked once and kept in sessionStorage under the same key as the settings dialog in app.js.
const ADMIN_KEY = "dtm_admin_token";
const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";

function getAdminToken(ask) {
  let tok = null;
  try { tok = sessionStorage.getItem(ADMIN_KEY); } catch { /* storage blocked */ }
  if (!tok && ask) {
    tok = (prompt("Enter admin token:") || "").trim() || null;
    if (tok) { try { sessionStorage.setItem(ADMIN_KEY, tok); } catch { /* storage blocked */ } }
  }
  return tok;
}
function clearAdminToken() { try { sessionStorage.removeItem(ADMIN_KEY); } catch { /* storage blocked */ } }

// Sub-cent amounts are the norm here, so show enough digits to tell questions apart.
function usd(v) {
  if (v == null) return "–";
  if (v === 0) return "$0";
  if (v > 0 && v < 0.000005) return "<$0.00001";  // e.g. one embeddings call: not free, just tiny
  return v < 0.01 ? `$${v.toFixed(5)}` : v < 100 ? `$${v.toFixed(2)}` : `$${Math.round(v).toLocaleString()}`;
}

let range = "today";
try { range = new URLSearchParams(location.search).get("range") || range; } catch { /* ignore */ }

async function load(retry = true) {
  $("error").hidden = true;
  document.querySelectorAll("[data-range]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.range === range)));
  const tok = getAdminToken(true);
  if (!tok) { showError("Admin token required."); return; }
  let res;
  try {
    res = await fetch(`/api/admin/costs?range=${encodeURIComponent(range)}`, { headers: { "X-Admin-Token": tok } });
  } catch (e) { showError(`Could not load costs: ${e.message}`); return; }
  if (res.status === 403) {
    clearAdminToken();
    if (retry) return load(false);
    showError("Invalid admin token."); return;
  }
  if (!res.ok) {
    // A 400 carries the server's reason (e.g. no semester configured) in a JSON "detail" field.
    let detail = null;
    if (res.status === 400) { try { detail = (await res.json()).detail; } catch { /* body is not JSON */ } }
    showError(typeof detail === "string" && detail ? detail : `Could not load costs (${res.status}).`);
    return;
  }
  render(await res.json());
}

function render(d) {
  const pq = d.per_question || {};
  $("mean").textContent = usd(pq.mean);
  $("median").textContent = usd(pq.median);
  $("p95").textContent = usd(pq.p95);
  $("spend").textContent = usd(d.spend_usd);
  $("questions").textContent = String(d.questions ?? "–");
  const cache = d.cache || null;
  $("cache").textContent = !cache || cache.share == null ? "–" : `${(cache.share * 100).toFixed(1)}%`;
  $("cache-sub").textContent = cache
    ? `${(cache.cached || 0).toLocaleString()} of ${(cache.input || 0).toLocaleString()} tokens` : "";
  renderDaily(d.daily || []);
  renderReasons(d.by_reason || [], d.spend_usd || 0);
  renderUsage("steps", d.by_step || [], (r) => STEP_NAMES[r.step] || r.step);
  renderUsage("models", d.by_model || [], (r) => r.model);
  renderTopQuestions(d.top_questions || [], d.spend_usd || 0, d.retention_days);
  renderVisitors(d.top_visitors || [], d.spend_usd || 0);
  const retention = d.retention_days;
  $("retention").hidden = retention == null;
  $("retention").textContent = retention == null
    ? "" : `Question text is deleted after ${retention} days; tokens and costs are kept.`;
  const est = d.estimated_questions || 0;
  $("estimated").hidden = !est;
  $("estimated").textContent = est
    ? `${est} question(s) include estimated tokens (the answer stream was cut before OpenAI reported usage).` : "";
  const unknown = d.unknown_cost_questions || 0;
  $("unknown").hidden = !unknown;
  $("unknown").textContent = unknown
    ? `${unknown} question(s) used a model missing from the price table; their cost is incomplete.` : "";
  const p = d.pricing || {};
  $("pricing-error").hidden = !p.override_error;
  $("pricing-error").textContent = p.override_error
    ? `The admin pricing.json file is invalid: ${p.override_error}` : "";
  $("pricing").textContent = `Estimated from token counts, ${p.currency || "USD"}. Prices as of ${p.as_of || "?"} (${p.source || "?"}).`;
}

const REASONS = {
  answered: ["Answered", "grounded answer with sources"],
  no_context: ["Refused: nothing relevant found", "planner + search only, no answer call"],
  no_answer: ["Refused by the model", "off-topic, injection or not in the documents"],
  uncited: ["Refused: answer without sources", "the model answered but cited nothing"],
  aborted: ["Aborted", "the answer stream was cut mid-way"],
  error: ["Failed", "a model call failed"],
};

const STEP_NAMES = { answer: "Answer", planner: "Planner", embeddings: "Search embeddings" };

function renderReasons(rows, total) {
  const body = $("reasons").tBodies[0];
  body.replaceChildren(...rows.map((r) => {
    const [label, hint] = REASONS[r.reason] || [r.reason, ""];
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    name.append(label);
    if (hint) { const h = document.createElement("div"); h.className = "hint"; h.textContent = hint; name.append(h); }
    const cells = [String(r.questions), usd(r.spend_usd), total ? `${Math.round((r.spend_usd / total) * 100)}%` : "–"]
      .map((text) => { const td = document.createElement("td"); td.className = "num"; td.textContent = text; return td; });
    tr.append(name, ...cells);
    return tr;
  }));
  if (!rows.length) {
    const tr = document.createElement("tr"), td = document.createElement("td");
    td.colSpan = 4; td.className = "hint"; td.textContent = "No questions in this range.";
    tr.append(td); body.append(tr);
  }
}

// Rows for the "By step" and "By model" tables: one label column, then calls, tokens and spend.
function renderUsage(id, rows, labelOf) {
  const body = $(id).tBodies[0];
  body.replaceChildren(...rows.map((r) => {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = labelOf(r);
    const cells = [
      (r.calls || 0).toLocaleString(),
      (r.input || 0).toLocaleString(),
      (r.cached || 0).toLocaleString(),
      (r.output || 0).toLocaleString(),
      usd(r.spend_usd),
    ].map((text) => { const td = document.createElement("td"); td.className = "num"; td.textContent = text; return td; });
    tr.append(name, ...cells);
    return tr;
  }));
  if (!rows.length) {
    const tr = document.createElement("tr"), td = document.createElement("td");
    td.colSpan = 6; td.className = "hint"; td.textContent = "No calls in this range.";
    tr.append(td); body.append(tr);
  }
}

function makeCell(text, cls = "") {
  const td = document.createElement("td");
  if (cls) td.className = cls;
  td.textContent = text;
  return td;
}

function emptyRow(body, colSpan) {
  const tr = document.createElement("tr"), td = document.createElement("td");
  td.colSpan = colSpan; td.className = "hint"; td.textContent = "No questions in this range.";
  tr.append(td); body.append(tr);
}

// Most expensive questions. The question text is deleted after retention_days, but tokens and
// costs are kept, so a row can outlive its text.
function renderTopQuestions(rows, total, retentionDays) {
  const body = $("top-questions").tBodies[0];
  body.replaceChildren(...rows.map((q) => {
    const tr = document.createElement("tr");
    const textTd = document.createElement("td");
    const text = document.createElement("div");
    text.className = "q-text";
    if (q.question) {
      text.textContent = q.question;
    } else {
      const h = document.createElement("div");
      h.className = "hint";
      h.textContent = retentionDays != null ? `deleted after ${retentionDays} days` : "deleted";
      text.append(h);
    }
    textTd.append(text);
    const steps = q.steps || {};
    const tokens = `${(q.input || 0).toLocaleString()} / ${(q.cached || 0).toLocaleString()} / ${(q.output || 0).toLocaleString()}`;
    tr.append(
      textTd,
      makeCell(q.asked_at ? new Date(q.asked_at).toLocaleString() : "–"),
      makeCell((REASONS[q.reason] || [q.reason || "–"])[0]),
      makeCell(tokens, "num"),
      makeCell(usd(steps.answer), "num"),
      makeCell(usd(steps.planner), "num"),
      makeCell(usd(steps.embeddings), "num"),
      makeCell(usd(q.spend_usd), "num"),
    );
    return tr;
  }));
  if (!rows.length) emptyRow(body, 8);
}

// Top visitors by spend. Visitor ids are anonymous browser ids.
function renderVisitors(rows, total) {
  const body = $("visitors").tBodies[0];
  body.replaceChildren(...rows.map((v) => {
    const tr = document.createElement("tr");
    tr.append(
      makeCell(v.visitor || "–", "mono"),
      makeCell((v.questions || 0).toLocaleString(), "num"),
      makeCell(usd(v.spend_usd), "num"),
      makeCell(total ? `${Math.round(((v.spend_usd || 0) / total) * 100)}%` : "–", "num"),
    );
    return tr;
  }));
  if (!rows.length) emptyRow(body, 4);
}

function svgEl(name, attrs = {}, cls = null) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (cls) node.setAttribute("class", cls);
  return node;
}

// Stacked daily bars: answer at the bottom, then planner, then embeddings. Drawn in a fixed
// 480x240 viewBox so the SVG scales to the container width without horizontal scrolling.
function renderDaily(days) {
  const box = $("daily");
  const max = days.reduce((m, day) => Math.max(m, day.total || 0), 0);
  if (!days.length || max <= 0) {
    const p = document.createElement("p");
    p.className = "note";
    p.textContent = "No spend in this range.";
    box.replaceChildren(p);
    return;
  }

  const W = 480, H = 240, L = 64, R = 8, T = 12, B = 30;
  const plotW = W - L - R, plotH = H - T - B, base = T + plotH;
  const slot = plotW / days.length;
  const barW = Math.max(1, slot * 0.75);
  const segments = [["answer", "seg-answer"], ["planner", "seg-planner"], ["embeddings", "seg-embeddings"]];

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Daily spend by step" }, "daily-svg");
  svg.append(svgEl("line", { x1: L, x2: W - R, y1: base, y2: base }, "axis-line"));
  days.forEach((day, i) => {
    const g = svgEl("g");
    const title = svgEl("title");
    title.textContent = `${day.date}: ${usd(day.total)} (answer ${usd(day.answer)}, planner ${usd(day.planner)}, embeddings ${usd(day.embeddings)})`;
    g.append(title);
    // Full-slot transparent target so zero-spend days still show their tooltip.
    g.append(svgEl("rect", { x: (L + i * slot).toFixed(2), y: T, width: slot.toFixed(2), height: plotH }, "hit"));
    const x = L + i * slot + (slot - barW) / 2;
    let stack = 0;
    for (const [key, cls] of segments) {
      const v = day[key] || 0;
      if (v <= 0) continue;
      const h = (v / max) * plotH;
      g.append(svgEl("rect", {
        x: x.toFixed(2), y: (base - stack - h).toFixed(2), width: barW.toFixed(2), height: h.toFixed(2),
      }, cls));
      stack += h;
    }
    svg.append(g);
  });

  const yLabel = svgEl("text", { x: L - 6, y: T + 10, "text-anchor": "end" }, "axis");
  yLabel.textContent = usd(max);
  svg.append(yLabel);
  if (days.length > 1) {
    const first = svgEl("text", { x: L, y: H - 8, "text-anchor": "start" }, "axis");
    first.textContent = days[0].date;
    const last = svgEl("text", { x: W - R, y: H - 8, "text-anchor": "end" }, "axis");
    last.textContent = days[days.length - 1].date;
    svg.append(first, last);
  }

  const legend = document.createElement("div");
  legend.className = "legend";
  for (const [label, cls] of [["Answer", "sw-answer"], ["Planner", "sw-planner"], ["Search embeddings", "sw-embeddings"]]) {
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.className = `sw ${cls}`;
    item.append(swatch, label);
    legend.append(item);
  }
  box.replaceChildren(legend, svg);
}

function showError(msg) { $("error").textContent = msg; $("error").hidden = false; }

document.querySelectorAll("[data-range]").forEach((b) => b.addEventListener("click", () => {
  range = b.dataset.range;
  try { history.replaceState(null, "", `?range=${range}`); } catch { /* ignore */ }
  load();
}));
load();
