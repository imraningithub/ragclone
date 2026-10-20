// Admin cost dashboard. The page holds no data: everything comes from /api/admin/costs with the admin token,
// which is asked once and kept in sessionStorage under the same key as the settings dialog in app.js.
const ADMIN_KEY = "dtm_admin_token";
const $ = (id) => document.getElementById(id);

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
  if (!res.ok) { showError(`Could not load costs (${res.status}).`); return; }
  render(await res.json());
}

function render(d) {
  const pq = d.per_question || {};
  $("mean").textContent = usd(pq.mean);
  $("median").textContent = usd(pq.median);
  $("p95").textContent = usd(pq.p95);
  $("spend").textContent = usd(d.spend_usd);
  $("questions").textContent = String(d.questions ?? "–");
  const unknown = d.unknown_cost_questions || 0;
  $("unknown").hidden = !unknown;
  $("unknown").textContent = unknown
    ? `${unknown} question(s) used a model missing from the price table; their cost is incomplete.` : "";
  const p = d.pricing || {};
  $("pricing").textContent = `Estimated from token counts, ${p.currency || "USD"}. Prices as of ${p.as_of || "?"} (${p.source || "?"}).`;
}

function showError(msg) { $("error").textContent = msg; $("error").hidden = false; }

document.querySelectorAll("[data-range]").forEach((b) => b.addEventListener("click", () => {
  range = b.dataset.range;
  try { history.replaceState(null, "", `?range=${range}`); } catch { /* ignore */ }
  load();
}));
load();
