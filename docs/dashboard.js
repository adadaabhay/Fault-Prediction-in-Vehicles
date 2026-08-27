/* Dashboard: live telemetry feed + in-browser LSTM inference.
 * - Module cards are clickable and open a detail view (live trends, RUL, events)
 * - Failure-mode detections are logged with mission timestamps
 * - Per-module RUL from the LSTM regression head, fault mode from classifier */

"use strict";

const SPARK_N = 220;          // samples kept per parameter trend
const EVENT_CAP = 400;
const ENTER_N = 8;            // frames a breach must persist before logging
const EXIT_N = 25;            // frames back in band before re-arming a param

const DASH = {
  config: null,
  parts: {},        // pid -> {gaugeParam, hist:{key:[]}, prevStatus:{}, failLatched}
  windowBuf: null,
  events: [],
  clsEpisode: { name: null, armed: true },
  modalPid: null,
};

function $(id) { return document.getElementById(id); }

/* Health lookup with bounds and presence checks. Returns null when the stream
 * does not carry health for that part/index, so callers can distinguish
 * "no data" from "healthy". */
function healthAt(pid, idx, fallback) {
  const h = LiveFeed.stream && LiveFeed.stream.health;
  if (!h || !h[pid] || idx < 0 || idx >= h[pid].length) {
    return (fallback === undefined) ? null : fallback;
  }
  const v = h[pid][idx];
  return (typeof v === "number" && isFinite(v)) ? v : ((fallback === undefined) ? null : fallback);
}

/* Channels a stream declares as schema placeholders rather than measurements. */
function isSyntheticChannel(key) {
  const m = LiveFeed.stream && LiveFeed.stream.meta;
  return !!(m && Array.isArray(m.channels_synthetic) && m.channels_synthetic.indexOf(key) !== -1);
}

/* ---------------- parameter value helpers ---------------- */

/* Dynamic chip-flip hook for the REMEDIATION ROUND 1 bar.
 *
 * The retrain chip starts as "pending" in the HTML so a slow network
 * cannot show a falsely-green chip.  Once ``init()`` has loaded both
 * ``config.json`` and ``model.json`` (or failed to), this function
 * re-fetches them with ``cache: no-store`` and flips the chip to one
 * of three states:
 *
 *   - ``ok``      : both responses are 2xx AND both bodies parse as
 *                   JSON. A 200 with a malformed body is reported as
 *                   ``error`` -- a green chip on a broken page is the
 *                   exact lie this gate is meant to prevent.
 *   - ``pending`` : exactly one probe fails its HTTP check (4xx, 5xx,
 *                   network, or CORS error). The page is mid-deploy;
 *                   refresh once the trainer has finished. A parse
 *                   failure on a 2xx is NOT pending -- see ``error``.
 *   - ``error``   : both probes fail their HTTP check, OR either body
 *                   fails to parse as JSON, OR both fail. A parse
 *                   failure on a 2xx is a build defect (the artifact
 *                   was published truncated or empty), not a mid-
 *                   deploy state; the operator needs to rebuild, not
 *                   refresh.
 *
 * The chip is identified by id ``chip_retrain`` so a future round can
 * add more dynamic chips without re-engineering this hook.
 */
async function verifyArtifacts() {
  const chip = $("chip_retrain");
  if (!chip) return;
  /* Returns one of:
   *   "ok"     -- 2xx AND parseable JSON
   *   "http"   -- 4xx / 5xx / network / CORS (probe never saw a body)
   *   "parse"  -- 2xx but the body is not parseable JSON (build defect)
   */
  async function probe(url) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) return "http";
      try { await r.json(); return "ok"; }
      catch (_) { return "parse"; }
    } catch (_) { return "http"; }
  }
  const [cfgR, mdlR] = await Promise.all([probe("config.json"), probe("model.json")]);
  chip.classList.remove("ok", "pending", "error");
  // Parse failures are always error, never pending -- a 200 with a
  // broken body is a build defect, not a mid-deploy state.
  const anyParse = (cfgR === "parse" || mdlR === "parse");
  if (cfgR === "ok" && mdlR === "ok") {
    chip.classList.add("ok");
    chip.textContent = "MODEL RETRAIN COMPLETE";
    chip.title = "config.json and model.json both 2xx with parseable bodies; LSTM weights present.";
  } else if ((cfgR === "http" && mdlR === "http") || anyParse) {
    chip.classList.add("error");
    chip.textContent = "ARTIFACTS UNREACHABLE";
    const why = anyParse
      ? "one or both artifact bodies failed to parse as JSON (truncated or empty publish). Re-run \`python -m ml.train\` and refresh."
      : "config.json and model.json both failed the HTTP probe. Run \`python -m ml.train\` and refresh.";
    chip.title = why;
  } else {
    chip.classList.add("pending");
    chip.textContent = "MODEL RETRAIN PENDING";
    const missing = cfgR !== "ok" ? "config.json" : "model.json";
    const why = cfgR === "parse" || mdlR === "parse"
      ? `${missing} returned a non-parseable body (build defect -- re-run training, do not just refresh)`
      : `${missing} failed the HTTP probe (4xx, 5xx, network, or CORS). The page is mid-deploy; refresh.`;
    chip.title = why;
  }
}

function scaledValue(param, raw) {
  return (param.scale !== undefined) ? raw * param.scale : raw;
}
function paramStatus(param, raw) {
  const v = scaledValue(param, raw);
  if (param.crit_lo !== undefined && v < param.crit_lo) return "crit";
  if (param.crit_hi !== undefined && v > param.crit_hi) return "crit";
  if (param.warn_lo !== undefined && v < param.warn_lo) return "warn";
  if (param.warn_hi !== undefined && v > param.warn_hi) return "warn";
  return "ok";
}

/* ---------------- build the parts grid ---------------- */
function buildGrid() {
  const grid = $("parts-grid");
  for (const pid of DASH.config.part_order) {
    if (pid === "overall") continue;
    const part = DASH.config.parts[pid];
    const card = document.createElement("section");
    card.className = "panel part-card clickable";
    card.id = "card_" + pid;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", "Open " + part.label + " details");

    const gaugeParam = part.params.find(p => p.key === part.gauge);

    let rows = "";
    for (const p of part.params) {
      // health_exclude parameters are display-only: the LSTM never sees
      // them and the simulator does not produce them.  Render with a
      // faded modifier so the operator can tell at a glance which rows
      // are real measurements and which are placeholders.  See
      // docs/REMEDIATION.md round 1, finding #4.
      const ex = p.health_exclude ? " display-only" : "";
      const labelSuffix = p.health_exclude
        ? ` <span class="display-only-tag" title="not a measurement; placeholder until a real sensor feeds the gateway">n/a</span>`
        : "";
      rows += `
        <div class="param-row${ex}">
          <span class="p-label">${p.label}${labelSuffix}</span>
          <span class="p-val" id="val_${pid}_${p.key}">--</span>
          <span class="p-unit">${p.unit}</span>
        </div>
        <div class="param-track${ex}" id="track_${pid}_${p.key}"></div>`;
    }

    card.innerHTML = `
      <span class="open-hint" title="Open details">&#9906;</span>
      <div class="part-meta">
        <span class="p-name">${part.label}</span>
        <span class="p-health">HEALTH <b id="health_${pid}">--</b>/100</span>
      </div>
      <div class="part-body">
        <div class="gauge-wrap">
          <canvas id="gauge_${pid}" width="170" height="150"></canvas>
        </div>
        <div class="params">
          <div class="metric part-rul">
            <span class="m-label">LSTM RUL — ${part.label}</span>
            <div class="rul-bar"><i id="rulbar_${pid}" class="green" style="width:0%"></i></div>
            <span class="rul-text" id="rultext_${pid}">--</span>
          </div>
          ${rows}
        </div>
      </div>`;
    grid.appendChild(card);

    card.addEventListener("click", () => openModule(pid));
    card.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openModule(pid); }
    });

    DASH.parts[pid] = { gaugeParam, hist: {}, prevStatus: {}, statusCount: {},
                        lastStatus: {}, failLatched: false };
    for (const p of part.params) DASH.parts[pid].hist[p.key] = [];
    Gauge.draw($("gauge_" + pid), gaugeParam.min, gaugeParam, { label: gaugeParam.label });
  }
}

/* ---------------- track (threshold bar) rendering ---------------- */
function renderTrack(pid, p, raw) {
  const track = $("track_" + pid + "_" + p.key);
  if (!track) return;
  track.innerHTML = "";
  const pct = v => Math.max(0, Math.min(100, (v - p.min) / (p.max - p.min) * 100));
  const zones = [
    [p.crit_lo, p.warn_lo, "#f03a3a"],
    [p.warn_lo, p.warn_hi, "#f5b83d"],
    [p.warn_hi, p.crit_hi, "#f5b83d"],
  ];
  for (const [lo, hi, color] of zones) {
    if (lo === undefined || hi === undefined) continue;
    const div = document.createElement("div");
    div.className = "zone";
    div.style.left = pct(Math.min(lo, hi)) + "%";
    div.style.width = Math.max(0, pct(Math.max(lo, hi)) - pct(Math.min(lo, hi))) + "%";
    div.style.background = color;
    track.appendChild(div);
  }
  const cur = document.createElement("div");
  cur.className = "cur";
  cur.style.left = pct(scaledValue(p, raw)) + "%";
  track.appendChild(cur);
}

/* ---------------- event log ---------------- */
function missionTimeString(rec, cycle) {
  const recTime = (rec && typeof rec.time === "number") ? rec.time : (LiveFeed.idx * (DASH.config ? DASH.config.dt : 0.1));
  const totalS = recTime + (cycle || 0) * (LiveFeed.durationS || 60.0);
  const h = Math.floor(totalS / 3600), m = Math.floor(totalS / 60) % 60,
        s = Math.floor(totalS % 60);
  const pad = x => String(x).padStart(2, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

function addEvent(pid, sev, msg, tstr) {
  const ev = { t: tstr, pid, sev, msg };
  DASH.events.unshift(ev);
  if (DASH.events.length > EVENT_CAP) DASH.events.pop();
  prependEventRow($("eventLog"), ev, true);
  const n = $("eventCount");
  n.textContent = DASH.events.length + " events";
  if (DASH.modalPid) renderModalEvents();
}

function prependEventRow(container, ev, animate) {
  const empty = container.querySelector(".ev-empty");
  if (empty) empty.remove();
  const row = document.createElement("div");
  row.className = "ev-row " + ev.sev + (animate ? " ev-new" : "");
  const mod = ev.pid ? DASH.config.parts[ev.pid].label : "AI MODEL";
  row.innerHTML = `
    <span class="ev-t">${ev.t}</span>
    <span class="ev-mod">${mod}</span>
    <span class="ev-msg">${ev.msg}</span>`;
  container.prepend(row);
  while (container.children.length > EVENT_CAP) container.lastChild.remove();
}

const RANK = { ok: 0, warn: 1, crit: 2 };

function detectParamEvents(pid, rec, health, tstr) {
  const st = DASH.parts[pid];
  const part = DASH.config.parts[pid];
  for (const p of part.params) {
    const raw = rec[p.key];
    if (raw === undefined || isSyntheticChannel(p.key)) continue;
    const s = paramStatus(p, raw);
    const prev = st.prevStatus[p.key] || "ok";
    if (s === st.lastStatus[p.key]) st.statusCount[p.key] += 1;
    else st.statusCount[p.key] = 1;
    st.lastStatus[p.key] = s;
    const cnt = st.statusCount[p.key];

    if (cnt === ENTER_N && RANK[s] > RANK[prev]) {
      if (s === "crit") {
        addEvent(pid, "crit",
          `${p.label} CRITICAL — ${fmtVal(scaledValue(p, raw), p)} ${p.unit}`, tstr);
      } else {
        addEvent(pid, "warn",
          `${p.label} entered warning band — ${fmtVal(scaledValue(p, raw), p)} ${p.unit}`, tstr);
      }
      st.prevStatus[p.key] = s;
    } else if (RANK[s] < RANK[prev] && cnt >= EXIT_N) {
      st.prevStatus[p.key] = s;   // recovered long enough; re-arm
    }
  }
  if (!st.failLatched && health < DASH.config.fail_health) {
    st.failLatched = true;
    addEvent(pid, "fail", `health ${health.toFixed(0)} — failure threshold crossed`, tstr);
  } else if (st.failLatched && health > DASH.config.fail_health + 10) {
    st.failLatched = false;
  }
}

function detectAiEvent(cls, tstr) {
  const names = DASH.config.class_names;
  let top = 0;
  for (let i = 1; i < cls.length; i++) if (cls[i] > cls[top]) top = i;
  const name = names[top], prob = cls[top];
  const ep = DASH.clsEpisode;
  if (top !== 0 && ep.armed && prob >= 0.5) {
    addEvent(null, "ai",
      `fault mode "${name.replace(/_/g, " ")}" detected (${Math.round(prob * 100)}% confidence)`, tstr);
    ep.armed = false;
    ep.name = name;
  } else if (top === 0 || prob < 0.3) {
    ep.armed = true;
    if (top === 0) ep.name = null;
  }
}

/* ---------------- stream provenance banner ---------------- */
function renderProvenance(stream) {
  const el = $("streamProvenance");
  if (!el) return;
  const m = (stream && stream.meta) || {};
  if (!m.health_provenance && !m.channels_measured) { el.textContent = ""; return; }
  const measured = (m.channels_measured || []).length + (m.channels_measured_extra || []).length;
  const synth = (m.channels_synthetic || []).length;
  const bits = [];
  if (m.source) bits.push(m.source);
  bits.push(measured + " measured / " + synth + " placeholder channels");
  if (m.health_provenance) bits.push("health: " + m.health_provenance.replace(/_/g, " "));
  el.textContent = bits.join(" · ");
}

/* ---------------- per-part update ---------------- */
function updatePart(pid, rec, healthVal, rulFrac) {
  const part = DASH.config.parts[pid];
  const st = DASH.parts[pid];
  const card = $("card_" + pid);

  const hEl = $("health_" + pid);
  hEl.textContent = healthVal.toFixed(0);
  hEl.className = healthVal < DASH.config.fail_health ? "crit" : (healthVal < 50 ? "warn" : "");

  const rulPct = Math.max(0, Math.min(100, rulFrac * 100));
  const rulBar = $("rulbar_" + pid);
  rulBar.style.width = rulPct + "%";
  rulBar.className = rulPct < 15 ? "red" : (rulPct < 40 ? "amber" : "green");
  $("rultext_" + pid).textContent = rulText(rulFrac);

  let worst = "ok";
  for (const p of part.params) {
    const raw = rec[p.key];
    if (raw === undefined) continue;
    const val = scaledValue(p, raw);
    const stat = isSyntheticChannel(p.key) ? "ok" : paramStatus(p, raw);
    if (stat === "crit") worst = "crit"; else if (stat === "warn" && worst !== "crit") worst = "warn";

    const vEl = $("val_" + pid + "_" + p.key);
    const synthetic = isSyntheticChannel(p.key);
    vEl.textContent = synthetic ? "n/a" : fmtVal(val, p) + (p.unit ? " " + p.unit : "");
    vEl.className = "p-val " + (synthetic ? "synthetic" : stat);
    vEl.title = synthetic
      ? "This stream does not carry this channel; value is a schema placeholder, not a measurement."
      : "";
    renderTrack(pid, p, raw);

    const hist = st.hist[p.key];
    hist.push(val);
    if (hist.length > SPARK_N) hist.shift();
  }

  const gp = st.gaugeParam;
  const gval = rec[gp.key] !== undefined ? scaledValue(gp, rec[gp.key]) : gp.min;
  Gauge.draw($("gauge_" + pid), gval, gp, { label: gp.label });

  card.classList.remove("sos", "warn");
  if (worst === "crit" || healthVal < DASH.config.fail_health || rulFrac < 0.05) {
    card.classList.add("sos");
  } else if (worst === "warn" || healthVal < 50 || rulFrac < 0.3) {
    card.classList.add("warn");
  }
  return worst;
}

function rulText(frac) {
  /* RUL is expressed in simulation steps. rul_cap_steps * dt is only ~120 s of
   * simulated time, so the old wall-clock branches were unreachable and every
   * readout -- including a vehicle at 100% life -- rendered as
   * "CRITICAL - 120 s remaining". Report life fraction and steps, and convert
   * to operating hours only when config declares a step->hours mapping. */
  const pct = Math.max(0, Math.min(100, frac * 100));
  const steps = Math.round(frac * DASH.config.rul_cap_steps);
  const perStep = DASH.config.rul_step_hours;
  if (typeof perStep === "number" && perStep > 0) {
    const hours = steps * perStep;
    if (hours >= 2) return `≈ ${hours.toFixed(1)} h remaining  (${pct.toFixed(0)}% life)`;
    return `≈ ${(hours * 60).toFixed(0)} min remaining  (${pct.toFixed(0)}% life)`;
  }
  if (pct < 5) return `CRITICAL — ${steps} steps remaining  (${pct.toFixed(0)}% life)`;
  return `${steps} steps remaining  (${pct.toFixed(0)}% life)`;
}

/* ---------------- LSTM prediction ---------------- */
function normaliseFeatures(rec) {
  const feat = [];
  for (const key of DASH.config.input_features) {
    const s = DASH.config.scaler[key];
    const v = (rec && rec[key] !== undefined && !isNaN(Number(rec[key]))) ? Number(rec[key]) : (s ? (s.min + s.max) * 0.5 : 0.0);
    const min = s ? s.min : 0.0;
    const max = s ? s.max : 1.0;
    feat.push(Math.max(0, Math.min(1, (v - min) / Math.max(max - min, 1e-9))));
  }
  return feat;
}

/* ---------------- overall + status ---------------- */
function updateOverall(rec, reg, cls, overallHealth) {
  const overallFrac = reg[DASH.config.part_order.indexOf("overall")];

  Gauge.draw($("gauge_overall"), overallHealth, DASH.config.parts.overall.params[0],
             { label: "Fused Health Index" });
  $("rulOverall").textContent = rulText(overallFrac);
  $("rulOverallPct").className = "m-sub " +
    (overallFrac < 0.15 ? "crit" : (overallFrac < 0.4 ? "warn" : ""));

  const names = DASH.config.class_names;
  const order = names.map((n, i) => ({ n, p: cls[i] }))
    .filter(x => x.p > 0.01)
    .sort((a, b) => b.p - a.p).slice(0, 3);
  const bars = $("clsBars");
  bars.innerHTML = "";
  for (const o of order) {
    const row = document.createElement("div");
    row.className = "cls-row";
    row.innerHTML = `
      <span class="cls-name">${o.n.replace(/_/g, " ")}</span>
      <span class="cls-bar"><i style="width:${(o.p * 100).toFixed(0)}%"></i></span>
      <span class="cls-pct">${(o.p * 100).toFixed(0)}%</span>`;
    bars.appendChild(row);
  }

  $("v_rpm").textContent = rec.rpm.toFixed(0);
  $("v_load").textContent = (rec.load * 100).toFixed(0) + "%";
  $("v_terrain").textContent = (rec.terrain * 100).toFixed(0) + "%";
  const oilp = rec.oil_pressure * 1e-5, oilt = rec.oil_temp;
  const opEl = $("v_oilp");
  opEl.textContent = oilp.toFixed(1) + " bar";
  opEl.className = "digital small " + (oilp < 2 ? "crit" : (oilp < 3.2 ? "warn" : ""));
  const otEl = $("v_oilt");
  otEl.textContent = oilt.toFixed(0) + "°C";
  otEl.className = "digital small " + (oilt > 130 ? "crit" : (oilt > 110 ? "warn" : ""));
  const vrEl = $("v_vibrms");
  vrEl.textContent = rec.vib_rms.toFixed(2);
  vrEl.className = "digital small " + (rec.vib_rms > 1.2 ? "crit" : (rec.vib_rms > 0.75 ? "warn" : ""));
  /* Two new vitals (added in audit-remediation round 1, schema D=24 → 26):
     coolant level (%) and exhaust pressure (bar).  Thresholds mirror
     ml/parts.py: warn_lo=55 / crit_lo=35 for coolant level,
     warn_hi=2.2 / crit_hi=2.6 for exhaust pressure.

     The exhaust-pressure warn band is calibrated for industry figures
     (turbocharged diesel, post-DPF, with restrictive fault).  The
     synthetic stream tops out near 1.75 bar
     (``sim/physics/exhaust.py``), so the warn/crit class will not
     fire from the simulator.  This is intentional: it lets the
     operator distinguish "synthetic stream, healthy by design" from
     "real hardware, restricted exhaust".  The same display-only
     discipline as the NBC channels; see ``docs/PROVENANCE.md``. */
  const clEl = $("v_coolantlvl");
  if (rec.coolant_level != null) {
    const cl = rec.coolant_level * 100;
    clEl.textContent = cl.toFixed(0) + "%";
    clEl.className = "digital small " +
      (cl < 35 ? "crit" : (cl < 55 ? "warn" : ""));
  } else {
    clEl.textContent = "--";
    clEl.className = "digital small";
  }
  const epEl = $("v_exhaustp");
  if (rec.exhaust_pressure != null) {
    const ep = rec.exhaust_pressure * 1e-5;
    epEl.textContent = ep.toFixed(2) + " bar";
    epEl.className = "digital small " +
      (ep > 2.6 ? "crit" : (ep > 2.2 ? "warn" : ""));
  } else {
    epEl.textContent = "--";
    epEl.className = "digital small";
  }

  /* Guarded: a stream whose health arrays are shorter than its records (or
   * missing a part entirely) previously produced NaN here, and every
   * comparison against NaN is false, so the status silently fell through to
   * "healthy". Absent data must never read as green. */
  const partHealths = Object.keys(DASH.parts)
    .map(pid => healthAt(pid, LiveFeed.idx))
    .filter(v => typeof v === "number" && isFinite(v));
  const worstPart = partHealths.length ? Math.min(...partHealths) : 100;
  let status;
  if (overallHealth < DASH.config.fail_health || overallFrac < 0.03 || worstPart < DASH.config.fail_health) {
    status = "sos";
  } else if (overallHealth < 40 || overallFrac < 0.2) {
    status = "faulty";
  } else if (overallHealth < 60 || overallFrac < 0.45 || worstPart < 50) {
    status = "degraded";
  } else {
    status = "healthy";
  }
  setStatus(status, rec);
  return status;
}

function setStatus(status, rec) {
  const badge = $("statusBadge");
  const mode = $("overallMode");
  const cls = status === "sos" ? "sos" : status;
  badge.className = "badge " + cls;
  badge.textContent = status.toUpperCase();
  mode.className = "badge " + cls;
  mode.textContent = status.toUpperCase();

  const banner = $("sosBanner");
  const detail = [];
  if (status === "sos") {
    for (const pid of DASH.config.part_order) {
      if (pid === "overall") continue;
      const h = healthAt(pid, LiveFeed.idx);
      if (h !== null && h < DASH.config.fail_health) detail.push(DASH.config.parts[pid].label);
    }
    banner.classList.remove("hidden");
    $("sosDetail").textContent = detail.length
      ? " Modules: " + detail.join(", ") + " — immediate maintenance required."
      : " Overall health below failure threshold — immediate maintenance required.";
  } else {
    banner.classList.add("hidden");
  }
}

/* ---------------- module detail modal ---------------- */
function openModule(pid) {
  DASH.modalPid = pid;
  const part = DASH.config.parts[pid];
  $("modalTitle").textContent = part.label;
  $("modalSub").textContent = part.params.length + " monitored parameters · live feed";

  const gaugeParam = DASH.parts[pid].gaugeParam;
  Gauge.draw($("gauge_modal"), gaugeParam.min, gaugeParam, { label: gaugeParam.label });

  const list = $("modalParams");
  list.innerHTML = "";
  for (const p of part.params) {
    const item = document.createElement("div");
    // health_exclude parameters: same display-only discipline as the
    // part card.  The modal title bar is updated to call this out so
    // the operator does not mistake a placeholder for a measurement.
    const ex = p.health_exclude ? " display-only" : "";
    item.className = "spark-item" + ex;
    const labelSuffix = p.health_exclude
      ? ` <span class="display-only-tag" title="not a measurement; placeholder until a real sensor feeds the gateway">n/a</span>`
      : "";
    item.innerHTML = `
      <div class="spark-head">
        <span class="p-label">${p.label}${labelSuffix}</span>
        <span class="p-val" id="mspark_val_${p.key}">--</span>
        <span class="p-unit">${p.unit}</span>
      </div>
      <canvas class="spark" id="mspark_${p.key}" width="460" height="54"></canvas>`;
    list.appendChild(item);
  }
  renderModalEvents();
  $("moduleModal").classList.remove("hidden");
  updateModal();
}

function closeModule() {
  DASH.modalPid = null;
  $("moduleModal").classList.add("hidden");
}

function renderModalEvents() {
  const box = $("modalEvents");
  box.innerHTML = "";
  const evs = DASH.events.filter(e => e.pid === DASH.modalPid).slice(0, 30);
  if (!evs.length) {
    box.innerHTML = '<div class="ev-empty">No failure modes detected yet.</div>';
    return;
  }
  for (const ev of evs) prependEventRow(box, ev, false);
}

function drawSpark(canvas, data, p) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0a120e";
  ctx.fillRect(0, 0, W, H);
  if (!data || data.length < 2) return;

  const norm = v => Math.max(0, Math.min(1, (v - p.min) / (p.max - p.min)));
  const yOf = v => H - 4 - norm(v) * (H - 8);

  // threshold guides
  ctx.setLineDash([4, 4]);
  for (const [th, color] of [[p.warn_hi, "#f5b83d"], [p.warn_lo, "#f5b83d"],
                             [p.crit_hi, "#f03a3a"], [p.crit_lo, "#f03a3a"]]) {
    if (th === undefined || th < p.min || th > p.max) continue;
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    ctx.moveTo(0, yOf(th));
    ctx.lineTo(W, yOf(th));
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  // trace
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = i / (SPARK_N - 1) * W;
    const y = yOf(data[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#3dd6f0";
  ctx.lineWidth = 1.6;
  ctx.stroke();

  // endpoint
  const last = data[data.length - 1];
  ctx.beginPath();
  ctx.arc(W - 2, yOf(last), 3, 0, 2 * Math.PI);
  ctx.fillStyle = valueColor(last, p);
  ctx.fill();
}

function updateModal() {
  const pid = DASH.modalPid;
  if (!pid) return;
  const rec = LiveFeed.records[LiveFeed.idx];
  const idx = LiveFeed.idx;
  const health = healthAt(pid, idx, 100);
  const j = DASH.config.part_order.indexOf(pid);
  const rulFrac = DASH.lastReg ? DASH.lastReg[j] : 1;

  const hEl = $("modalHealth");
  hEl.textContent = health.toFixed(0);
  hEl.className = health < DASH.config.fail_health ? "crit" : (health < 50 ? "warn" : "");

  const rulPct = Math.max(0, Math.min(100, rulFrac * 100));
  const bar = $("modalRulBar");
  bar.style.width = rulPct + "%";
  bar.className = rulPct < 15 ? "red" : (rulPct < 40 ? "amber" : "green");
  $("modalRulText").textContent = rulText(rulFrac);
  const badge = $("modalRulBadge");
  badge.className = "badge " + (rulPct < 15 ? "sos" : (rulPct < 40 ? "degraded" : "healthy"));
  badge.textContent = "RUL " + (rulFrac * 100).toFixed(0) + "%";

  const st = DASH.parts[pid];
  const gp = st.gaugeParam;
  const gval = rec[gp.key] !== undefined ? scaledValue(gp, rec[gp.key]) : gp.min;
  Gauge.draw($("gauge_modal"), gval, gp, { label: gp.label });

  for (const p of DASH.config.parts[pid].params) {
    const raw = rec[p.key];
    if (raw === undefined) continue;
    const val = scaledValue(p, raw);
    const vEl = $("mspark_val_" + p.key);
    if (vEl) {
      vEl.textContent = fmtVal(val, p) + " " + p.unit;
      vEl.className = "p-val " + paramStatus(p, raw);
    }
    drawSpark($("mspark_" + p.key), st.hist[p.key], p);
  }
}

/* ---------------- live frame ---------------- */
function onFrame(rec, idx, cycle) {
  const vec = normaliseFeatures(rec);
  const win = DASH.windowBuf.push(vec);
  const { reg, cls } = LSTM.forward(DASH.model, win);
  DASH.lastReg = reg;
  DASH.lastCls = cls;

  const tstr = missionTimeString(rec, cycle);
  let worstOverall = "ok";
  for (const pid of DASH.config.part_order) {
    if (pid === "overall") continue;
    const j = DASH.config.part_order.indexOf(pid);
    const health = healthAt(pid, idx, 95.0);
    const w = updatePart(pid, rec, health, reg[j]);
    detectParamEvents(pid, rec, health, tstr);
    if (w === "crit") worstOverall = "crit"; else if (w === "warn" && worstOverall !== "crit") worstOverall = "warn";
  }
  const overallHealth = healthAt("overall", idx, 95.0);
  updateOverall(rec, reg, cls, overallHealth);
  detectAiEvent(cls, tstr);
  updateModal();

  $("missionClock").textContent = tstr;
  $("cycleInfo").textContent = `pass ${cycle + 1} · ${DASH.config.dt.toFixed(2)}s sampling · ${LiveFeed.speed}× speed`;
}

function onCycleWrap(cycle) {
  // restart LSTM window and detection latches for the new pass
  const rec0 = (LiveFeed.records && LiveFeed.records.length > 0) ? LiveFeed.records[0] : {};
  const first = normaliseFeatures(rec0);
  DASH.windowBuf = rollingWindow(DASH.config.window, DASH.config.input_features.length, first);
  DASH.clsEpisode = { name: null, armed: true };
  for (const pid of Object.keys(DASH.parts)) {
    DASH.parts[pid].prevStatus = {};
    DASH.parts[pid].statusCount = {};
    DASH.parts[pid].lastStatus = {};
    DASH.parts[pid].failLatched = false;
  }
}


/* ---------------- live hardware source ---------------- */
/* When the gateway is reachable, frames come from it and carry health, DTCs
 * and prognosis already computed by telemetry_gateway/pipeline.py. When it is
 * not, the recorded mission in LiveFeed drives the HUD instead. The badge must
 * always say which of the two the operator is looking at -- a HUD that cannot
 * distinguish live hardware from a replay is worse than one with no live mode
 * at all. */
const LiveSource = {
  mode: "replay",           // "live" | "replay"
  detail: "",

  setBadge(state, detail) {
    const el = $("sourceBadge");
    if (!el) return;
    const map = {
      live:        ["LIVE HARDWARE STREAM", "healthy"],
      connecting:  ["CONNECTING TO GATEWAY", "degraded"],
      stalled:     ["GATEWAY STALLED - REPLAY", "degraded"],
      disconnected:["GATEWAY DOWN - REPLAY", "degraded"],
      unavailable: ["SIMULATION REPLAY", "healthy"],
      replay:      ["SIMULATION REPLAY", "healthy"],
    };
    const [text, cls] = map[state] || map.replay;
    el.className = "badge " + cls;
    el.textContent = text;
    el.title = detail || "";
  },

  toReplay(state, detail) {
    if (this.mode === "live") {
      addEvent("overall", "warn",
               `live gateway lost (${detail || state}); falling back to replay`,
               $("missionClock").textContent || "--:--:--");
    }
    this.mode = "replay";
    this.setBadge(state, detail);
    if (!LiveFeed.playing) LiveFeed.play();
  },

  toLive(detail) {
    if (this.mode !== "live") {
      addEvent("overall", "ok", "live hardware stream acquired",
               $("missionClock").textContent || "--:--:--");
    }
    this.mode = "live";
    this.setBadge("live", detail);
    LiveFeed.pause();          // the gateway is driving now
  },
};

/* Render active SAE J1939-73 DM1 diagnostic trouble codes. */
function renderDTCs(dtcs) {
  const host = $("dtcChips");
  if (!host) return;
  if (!Array.isArray(dtcs) || dtcs.length === 0) {
    host.innerHTML = '<span class="m-sub">no active DTCs</span>';
    return;
  }
  /* textContent per field, not innerHTML on the payload: DTC descriptions
   * originate upstream of the browser and must not be able to inject markup. */
  host.innerHTML = "";
  dtcs.slice(0, 12).forEach(d => {
    const chip = document.createElement("span");
    const lamp = String(d.lamp_status || "").toUpperCase();
    chip.className = "chip dtc " +
      (lamp === "RED_STOP" ? "sos" : (lamp === "AMBER_WARNING" ? "degraded" : ""));
    chip.textContent = `SPN ${d.spn} FMI ${d.fmi}` +
      (d.oc != null ? ` ×${d.oc}` : "") +
      (d.description ? ` — ${d.description}` : "");
    host.appendChild(chip);
  });
}

/* One processed frame from the gateway. */
function onLiveFrame(frame) {
  LiveSource.toLive(frame.stream_badge);
  /* The HUD renders the sanitised view -- displaying unclamped raw values
   * would show the operator readings the FDIR gate has already rejected. */
  const rec = frame.telemetry_clean || frame.telemetry_raw || {};

  renderDTCs(frame.dtcs_active);

  (frame.sensor_faults || []).forEach(f => {
    addEvent("overall", "warn",
             `FDIR ${f.channel}: ${f.fault_type}` +
             (f.raw_value != null ? ` (raw ${Number(f.raw_value).toFixed(2)})` : ""),
             $("missionClock").textContent || "--:--:--");
  });

  /* Health comes from the gateway, already computed. Never synthesise a
   * number here: if the pipeline could not produce one, say so. */
  if (!frame.health_available || !frame.subsystem_health) {
    LiveSource.setBadge("live", "health assessment unavailable at the gateway");
    return;
  }

  const health = frame.subsystem_health;
  const order = DASH.config.part_order;
  /* RUL comes from the gateway's own LSTM pass when available. The browser
   * LSTM is for the replay path; running both on live frames would show two
   * different numbers for the same quantity. */
  const rulFrac = (frame.prognosis && frame.prognosis.rul_fraction) || null;
  const probs = (frame.prognosis && frame.prognosis.fault_probs) || null;

  const tstr = missionTimeString(rec, 0);
  let worst = "ok";
  for (const pid of order) {
    if (pid === "overall") continue;
    const h = Number(health[pid]);
    const r = rulFrac && rulFrac[pid] != null ? Number(rulFrac[pid]) : NaN;
    const w = updatePart(pid, rec, h, r);
    detectParamEvents(pid, rec, h, tstr);
    if (w === "crit") worst = "crit";
    else if (w === "warn" && worst !== "crit") worst = "warn";
  }

  const reg = order.map(pid => (rulFrac && rulFrac[pid] != null)
                               ? Number(rulFrac[pid]) : NaN);
  const cls = probs ? DASH.config.class_names.map(n => Number(probs[n] || 0)) : null;
  if (cls) { DASH.lastCls = cls; detectAiEvent(cls, tstr); }
  DASH.lastReg = reg;
  updateOverall(rec, reg, cls, Number(health.overall));
  updateModal();

  $("missionClock").textContent = tstr;
  $("cycleInfo").textContent =
    `gateway · ${LiveSocket.frameCount} frames · ` +
    `FDIR ${Number(frame.gate_ms || 0).toFixed(2)} ms` +
    (frame.inference_available ? "" : " · inference offline");
}

/* ---------------- init ---------------- */
async function init() {
  const [config, model] = await Promise.all([
    fetch("config.json").then(r => r.json()),
    fetch("model.json").then(r => r.json()),
  ]);
  DASH.config = config;
  DASH.model = model;
  await LiveFeed.init(config);

  DASH.windowBuf = rollingWindow(config.window, config.input_features.length,
                                 normaliseFeatures(LiveFeed.records[0]));

  buildGrid();
  $("missionName").textContent = "Mission: " + LiveFeed.stream.meta.name.replace(/_/g, " ");
  $("injectedFaults").innerHTML = (LiveFeed.stream.meta.faults || []).map(f =>
    `<span class="chip">&#9889; ${f.replace(/_/g, " ")}</span>`).join("");

  renderProvenance(LiveFeed.stream);
  // After everything else is wired, re-probe the artifacts with a
  // cache-busting fetch and flip the retrain chip to its real state.
  // This runs *after* the rest of init() so the chip cannot falsely
  // report "ok" before the page has actually consumed the artifacts.
  await verifyArtifacts();
  $("modelInfo").textContent =
    `LSTM · ${model.H} hidden units · input window ${config.window}×${config.input_features.length} · ` +
    `RUL cap ${config.rul_cap_steps} steps · trained offline on physics-simulated scenarios, inference runs live in-browser`;

  $("btnPlay").onclick = () => {
    if (LiveFeed.playing) { LiveFeed.pause(); $("btnPlay").textContent = "▶"; }
    else { LiveFeed.play(); $("btnPlay").textContent = "❚❚"; }
  };
  document.querySelectorAll(".speed-btn").forEach(b => {
    b.onclick = () => {
      LiveFeed.setSpeed(Number(b.dataset.speed));
      document.querySelectorAll(".speed-btn").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
    };
  });
  document.querySelectorAll(".stream-btn").forEach(b => {
    b.onclick = () => {
      const streamId = b.dataset.stream;
      LiveFeed.switchStream(streamId, (newStream) => {
        document.querySelectorAll(".stream-btn").forEach(x => x.classList.remove("active"));
        b.classList.add("active");
        onCycleWrap(0);
        const name = newStream.meta ? newStream.meta.name : (newStream.mission_name || streamId);
        $("missionName").textContent = name.replace(/_/g, " ");
        renderProvenance(newStream);
        const faults = newStream.meta ? (newStream.meta.faults || []) : [];
        $("injectedFaults").innerHTML = faults.map(f =>
          `<span class="chip">&#9889; ${f.replace(/_/g, " ")}</span>`).join("");
      });
    };
  });
  $("btnCloseModal").onclick = closeModule;
  $("modalBackdrop").onclick = closeModule;
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeModule(); });

  LiveFeed.onRecord(onFrame);
  LiveFeed.onCycle(onCycleWrap);
  LiveFeed.play();
  LiveSource.setBadge("replay", "recorded mission");

  /* Try the gateway. Everything above already works without it, so a failed
   * connection degrades to replay rather than breaking the HUD. */
  LiveSocket.onState((state, detail) => {
    if (state === "live") return;              // handled in onLiveFrame
    LiveSource.toReplay(state, detail);
  });
  LiveSocket.onFrame(onLiveFrame);
  LiveSocket.connect();
}

init().catch(err => {
  console.error("dashboard init failed:", err);
  // The chip is now an honest page-health indicator, not just an
  // artifact-reachability probe.  If init() rejected, the page is
  // half-broken regardless of whether the two files in docs/ parse,
  // so we force the chip to "error" and DO NOT re-call
  // verifyArtifacts() -- doing so would flip a green chip onto a
  // page whose own init has crashed.  If the chip element is gone
  // for some reason, fall back to a console error so the failure is
  // at least visible to the developer tools.
  const chip = $("chip_retrain");
  if (!chip) {
    console.error("init failed and #chip_retrain is missing; "
                  + "page is in an undefined state.");
    return;
  }
  chip.classList.remove("ok", "pending");
  chip.classList.add("error");
  chip.textContent = "PAGE INIT FAILED";
  chip.title = `dashboard init() rejected: ${err && err.message ? err.message : err}. `
             + "Open the browser console for the full trace; do not trust the readouts.";
});
