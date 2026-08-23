/* Live telemetry feed.
 *
 * Two sources, one interface:
 *
 *   1. LiveSocket -- a real WebSocket client for telemetry_gateway/server.py.
 *      Frames arrive already processed by the PHM pipeline (FDIR-sanitised
 *      signals, computed subsystem health, LSTM prognosis, active DM1/DM2
 *      DTCs). A watchdog falls back to (2) when frames stop arriving.
 *
 *   2. LiveFeed -- replays a recorded mission from JSON, dt-scaled, looping.
 *
 * PROJECT.md described (1) -- "WebSocket live client with 2.0s watchdog
 * fallback" and a "[LIVE HARDWARE STREAM] badge" -- as feature 13/14, built.
 * It did not exist: this file contained only (2), with no WebSocket, no
 * watchdog and no badge. */

"use strict";

/* ------------------------------------------------------------------ */
/* Live hardware WebSocket client                                      */
/* ------------------------------------------------------------------ */
const LiveSocket = {
  ws: null,
  url: null,
  connected: false,
  lastFrameAt: 0,
  frameCount: 0,
  watchdogMs: 2000,
  _watchdog: null,
  _onFrame: null,
  _onState: null,
  _retryMs: 1000,
  _retryMax: 15000,
  _closedByUs: false,

  /* Default to the gateway on the same host. Overridable with
   * ?gateway=ws://host:port/ws/telemetry for a bench rig on another box. */
  defaultUrl() {
    const q = new URLSearchParams(location.search).get("gateway");
    if (q) return q;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const host = location.hostname || "localhost";
    return `${proto}//${host}:8000/ws/telemetry`;
  },

  onFrame(cb) { this._onFrame = cb; },
  onState(cb) { this._onState = cb; },

  _state(state, detail) {
    if (this._onState) this._onState(state, detail || "");
  },

  connect(url) {
    this.url = url || this.defaultUrl();
    this._closedByUs = false;
    let sock;
    try {
      sock = new WebSocket(this.url);
    } catch (e) {
      /* Bad URL, blocked scheme, or no WebSocket support. Report it rather
       * than leaving the HUD claiming to be connecting forever. */
      this._state("unavailable", String(e && e.message ? e.message : e));
      return false;
    }
    this.ws = sock;
    this._state("connecting", this.url);

    sock.onopen = () => {
      this.connected = true;
      this._retryMs = 1000;
      this.lastFrameAt = Date.now();
      this._startWatchdog();
      this._state("connecting", "handshake complete, awaiting first frame");
    };

    sock.onmessage = (ev) => {
      let frame;
      try {
        frame = JSON.parse(ev.data);
      } catch (e) {
        console.warn("[LiveSocket] non-JSON frame dropped", e);
        return;
      }
      this.lastFrameAt = Date.now();
      this.frameCount += 1;
      this._state("live", frame.stream_badge || "[LIVE]");
      if (this._onFrame) this._onFrame(frame);
    };

    sock.onerror = () => {
      /* onerror is always followed by onclose; let onclose do the work so the
       * fallback is not triggered twice. */
    };

    sock.onclose = () => {
      this.connected = false;
      this._stopWatchdog();
      this._state("disconnected", "");
      if (!this._closedByUs) this._scheduleRetry();
    };
    return true;
  },

  _scheduleRetry() {
    const delay = this._retryMs;
    this._retryMs = Math.min(this._retryMs * 2, this._retryMax);
    setTimeout(() => { if (!this._closedByUs) this.connect(this.url); }, delay);
  },

  _startWatchdog() {
    this._stopWatchdog();
    this._watchdog = setInterval(() => {
      if (!this.connected) return;
      if (Date.now() - this.lastFrameAt > this.watchdogMs) {
        /* The socket is open but the gateway has gone quiet. That is not the
         * same as a disconnect and must not look like one on the HUD. */
        this._state("stalled",
                    `no frame for ${((Date.now() - this.lastFrameAt) / 1000).toFixed(1)}s`);
      }
    }, Math.max(250, this.watchdogMs / 4));
  },

  _stopWatchdog() {
    if (this._watchdog) clearInterval(this._watchdog);
    this._watchdog = null;
  },

  close() {
    this._closedByUs = true;
    this._stopWatchdog();
    if (this.ws) { try { this.ws.close(); } catch (e) { /* already gone */ } }
    this.connected = false;
  },
};

const LiveFeed = {
  stream: null,
  multiStreams: null,
  activeStreamId: "sim_mbt",
  cfg: null,
  idx: 0,
  cycle: 0,
  playing: false,
  speed: 1,
  _timer: null,
  _cb: null,
  _cycleCb: null,

  async init(cfg) {
    this.cfg = cfg;
    try {
      this.multiStreams = await fetch("live_multi_streams.json").then(r => r.json());
      this.stream = this.multiStreams.streams[this.activeStreamId] || await fetch("live_stream.json").then(r => r.json());
    } catch (e) {
      this.stream = await fetch("live_stream.json").then(r => r.json());
    }
    return this;
  },

  get records() { return this.stream ? (this.stream.records || this.stream.data || []) : []; },
  get durationS() { return this.records.length > 0 ? (this.records.length - 1) * (this.cfg ? this.cfg.dt : 0.1) : 0; },

  onRecord(cb) { this._cb = cb; },
  onCycle(cb) { this._cycleCb = cb; },

  switchStream(streamId, onSwitchCb) {
    if (!this.multiStreams || !this.multiStreams.streams[streamId]) {
      /* Previously a silent no-op, so a button with no backing stream just
       * appeared to do nothing. Report it so the failure is visible. */
      console.warn(`[LiveFeed] no stream "${streamId}" in live_multi_streams.json`);
      if (typeof onSwitchCb === "function") onSwitchCb(null, `stream "${streamId}" unavailable`);
      return false;
    }
    const wasPlaying = this.playing;
    this.pause();
    this.activeStreamId = streamId;
    this.stream = this.multiStreams.streams[streamId];
    this.idx = 0;
    this.cycle = 0;
    if (onSwitchCb) onSwitchCb(this.stream, null);
    if (wasPlaying) this.play();
    return true;
  },

  play() {
    if (this.playing || !this.stream || this.records.length === 0) return;
    this.playing = true;
    if (this._cb) this._cb(this.records[this.idx], this.idx, this.cycle);
    const loop = () => {
      if (!this.playing) return;
      this.idx += 1;
      if (this.idx >= this.records.length) {
        this.idx = 0;
        this.cycle += 1;
        if (this._cycleCb) this._cycleCb(this.cycle);
      }
      if (this._cb) this._cb(this.records[this.idx], this.idx, this.cycle);
      const interval = Math.max(6, (this.cfg ? this.cfg.dt : 0.1) * 1000 / this.speed);
      this._timer = setTimeout(loop, interval);
    };
    const interval = Math.max(6, (this.cfg ? this.cfg.dt : 0.1) * 1000 / this.speed);
    this._timer = setTimeout(loop, interval);
  },

  pause() {
    this.playing = false;
    if (this._timer) clearTimeout(this._timer);
  },

  setSpeed(s) {
    const was = this.playing;
    this.pause();
    this.speed = s;
    if (was) this.play();
  },
};
