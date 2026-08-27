// Offline check for the verifyArtifacts() state machine.  We
// re-implement the chip-flip hook in isolation, stub out fetch and
// document, and exercise the four state transitions:
//
//   1. both 2xx+parseable JSON   -> ok       (MODEL RETRAIN COMPLETE)
//   2. both 4xx                  -> error    (ARTIFACTS UNREACHABLE)
//   3. one 4xx                   -> pending  (MODEL RETRAIN PENDING)
//   4. one 2xx+garbage body      -> error    (ARTIFACTS UNREACHABLE, build defect)
//
// This is the same logic dashboard.js::verifyArtifacts() runs in the
// browser; the test only differs in that the DOM is a hand-rolled
// stub instead of an HTML page.  Keep this file in sync with
// docs/dashboard.js:66-115 -- if the two diverge, this check will
// start passing while the browser fails (or vice versa).  The full
// Playwright suite (tests/test_hud_smoke.py) is the real gate.

"use strict";

const states = [];
const chip = {
  classList: { _set: new Set(), add(c) { this._set.add(c); }, remove(...cs) { cs.forEach(c => this._set.delete(c)); }, contains(c) { return this._set.has(c); } },
  dataset: { probed: undefined },
  textContent: "",
  title: "",
};

function makeFetch(map) {
  return async (url) => {
    const entry = map[url.endsWith("config.json") ? "config.json"
                   : url.endsWith("model.json")  ? "model.json"
                   : null];
    if (!entry) throw new Error("unexpected url " + url);
    return {
      ok: entry.status >= 200 && entry.status < 300,
      async json() { return JSON.parse(entry.body); },
    };
  };
}

async function verifyArtifacts(probe) {
  async function p(url) {
    try {
      const r = await probe(url, { cache: "no-store" });
      if (!r.ok) return "http";
      try { await r.json(); return "ok"; }
      catch (_) { return "parse"; }
    } catch (_) { return "http"; }
  }
  const [cfgR, mdlR] = await Promise.all([p("config.json"), p("model.json")]);
  chip.dataset.probed = "1";  // sync sentinel
  chip.classList.remove("ok", "pending", "error");
  const anyParse = (cfgR === "parse" || mdlR === "parse");
  if (cfgR === "ok" && mdlR === "ok") {
    chip.classList.add("ok");
    chip.textContent = "MODEL RETRAIN COMPLETE";
  } else if ((cfgR === "http" && mdlR === "http") || anyParse) {
    chip.classList.add("error");
    chip.textContent = "ARTIFACTS UNREACHABLE";
  } else {
    chip.classList.add("pending");
    chip.textContent = "MODEL RETRAIN PENDING";
  }
}

function assertEq(actual, expected, label) {
  if (actual !== expected) {
    console.error(`FAIL ${label}: got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`);
    process.exit(1);
  }
  console.log(`  ok  ${label}: ${JSON.stringify(actual)}`);
}

function resetChip() {
  chip.classList = { _set: new Set(), add(c){this._set.add(c);}, remove(...cs){cs.forEach(c=>this._set.delete(c));}, contains(c){return this._set.has(c);} };
  chip.dataset = { probed: undefined };
  chip.textContent = ""; chip.title = "";
}

(async () => {
  // 1. both 2xx+JSON
  resetChip();
  await verifyArtifacts(makeFetch({
    "config.json": { status: 200, body: '{"window":24}' },
    "model.json":  { status: 200, body: '{"D":26,"H":24}' },
  }));
  assertEq(chip.dataset.probed,    "1",            "1.both_2xx.sentinel");
  assertEq(chip.textContent,       "MODEL RETRAIN COMPLETE", "1.both_2xx");
  assertEq(chip.classList.contains("ok"), true,   "1.both_2xx.class");

  // 2. both 4xx
  resetChip();
  await verifyArtifacts(makeFetch({
    "config.json": { status: 404, body: "not found" },
    "model.json":  { status: 404, body: "not found" },
  }));
  assertEq(chip.textContent, "ARTIFACTS UNREACHABLE", "2.both_404");
  assertEq(chip.classList.contains("error"), true,   "2.both_404.class");

  // 3. one 4xx
  resetChip();
  await verifyArtifacts(makeFetch({
    "config.json": { status: 200, body: '{"window":24}' },
    "model.json":  { status: 404, body: "not found" },
  }));
  assertEq(chip.textContent, "MODEL RETRAIN PENDING", "3.one_404");
  assertEq(chip.classList.contains("pending"), true, "3.one_404.class");

  // 4. one 2xx+garbage (the C1 fix and the SFH #2 fix together)
  resetChip();
  await verifyArtifacts(makeFetch({
    "config.json": { status: 200, body: '{"window":24}' },
    "model.json":  { status: 200, body: "<html>oops</html>" },
  }));
  assertEq(chip.textContent, "ARTIFACTS UNREACHABLE", "4.one_garbage");
  assertEq(chip.classList.contains("error"), true,   "4.one_garbage.class");

  // 5. both 2xx+garbage
  resetChip();
  await verifyArtifacts(makeFetch({
    "config.json": { status: 200, body: "{ not json" },
    "model.json":  { status: 200, body: "[]" },
  }));
  assertEq(chip.textContent, "ARTIFACTS UNREACHABLE", "5.both_garbage");
  assertEq(chip.classList.contains("error"), true,   "5.both_garbage.class");

  console.log("all 5 state-machine checks passed");
})();
