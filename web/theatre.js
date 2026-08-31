// The transmitter.
//
// Everything about WHAT is on air lives in schedule.js and is pure. This file
// is the part that cannot be pure: the DOM, the clock, the network, the
// audio element. Keeping the split honest is what let the schedule be tested
// without a browser -- see tests/schedule.test.js.
//
// The station is always running. Nothing here ever "starts" a film at frame
// zero; it asks the schedule what the wall clock says should be on screen and
// paints that. Tuning to a channel mid-film is the normal case, not an edge.

import { decodeFilm } from "./decoder.js";
import { frameAt, upNext, cycleSeconds } from "./schedule.js";
import { startSession, recordFlip, endSession, toAnalyticsPayload } from "./telemetry.js";
import { flickerIntensityForYear, gateWeaveActive } from "./period-fx.js";
import { wipeVariantFor, WIPE_VARIANTS } from "./wipe.js";
import { PRESENCE_POLL_MS, PRESENCE_JITTER_MS } from "./presence.js";
import { LIVE_TICK_MS, CYCLE_TICKS, epochIndexFor, simulateFrame } from "./live.js";
import { clipAt } from "./videochan.js";

const $ = (id) => document.getElementById(id);

const el = {
  poweron: $("poweron"), tune: $("tune"), set: $("set"),
  canvas: $("canvas"), screen: $("screen"), bug: $("bug"), tuning: $("tuning"),
  cue: $("cue"), vchan: $("vchan"),
  chno: $("m-chno"), chname: $("m-chname"), chtag: $("m-chtag"),
  clock: $("m-clock"), onair: $("m-onair"),
  nTitle: $("n-title"), nYear: $("n-year"), nProg: $("n-prog"),
  nBar: $("n-bar"), nHook: $("n-hook"), nSrc: $("n-src"), nFid: $("n-fid"),
  xTitle: $("x-title"),
  dial: $("dial"), wall: $("wall"), guide: $("guide"),
  btnGuide: $("btn-guide"), btnClose: $("guide-close"), btnMute: $("btn-mute"),
  curtains: $("curtains"), btnFull: $("btn-full"),
};

let M = null;                 // the manifest
let channel = null;           // the channel object now tuned
let audio = null;

// ---------------------------------------------------------------- telemetry
//
// Fire-and-forget: track() never blocks or throws if Analytics hasn't
// loaded (or is blocked by the viewer), because the picture must never
// depend on it -- the same principle the presence counter is built to.
// The dwell/flip/session MATH lives in telemetry.js and is pure and tested;
// this is only the DOM-lifecycle glue, same split as the rest of this file.
function track(event) {
  if (typeof window.va === "function") window.va("event", toAnalyticsPayload(event));
}

let telemetryState = null;    // null = no session open yet (or just flushed)
let poweredOn = false;        // true once the viewer has clicked through

/** Ends the open session (if any) and reports it. Idempotent: a
 *  visibilitychange->hidden and a later pagehide on the same departure
 *  must not double-count, so this clears telemetryState once sent. */
function flushSession() {
  if (!telemetryState) return;
  track(endSession(telemetryState, Date.now()));
  telemetryState = null;
}

function reopenIfDue() {
  if (poweredOn && !document.hidden && telemetryState === null) {
    // A fresh visit -- the session that just ended already reported its own
    // dwell and length.
    telemetryState = startSession(Date.now());
  }
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) flushSession();
  else reopenIfDue();
});
window.addEventListener("pagehide", flushSession);
// Some browsers (historically Safari) don't reliably fire visibilitychange
// on a back/forward-cache restore -- visibilitychange is the only path that
// reopens a session, so without this a bfcache return leaves telemetryState
// null for the rest of the page's life: every later flip and the eventual
// session end silently drop. `persisted` is true only for a real bfcache
// restore, never a fresh load (which already opens its own session at
// power-on).
window.addEventListener("pageshow", (e) => { if (e.persisted) reopenIfDue(); });
let tuningUntil = 0;          // wall-clock ms during which static is shown
let tuningTimer = 0;
let flipCount = 0;            // fed to wipeVariantFor -- see js/wipe.js
let lastPainted = "";         // avoid re-writing identical text every frame
let lastFilmIdx = -1;
let liveEpoch = -1;           // last CH00 epochIndex computed, so paint() at
let liveText = "";            // rAF rate does not re-run the sim every frame

// Period effects: JS only ever places a data attribute or a custom property
// at a boundary (film start / channel tune); theatre.css owns every frame of
// the actual motion. See css/theatre.css "period effects" and js/period-fx.js.
const CUE_MARK_MS = 400;      // matches @keyframes cueBurn's duration
const VHOLD_MS = 320;         // matches @keyframes vholdSlip's duration
let cueTimer = 0;
let vholdTimer = 0;

// Decoded films, keyed by index. A decoded film is ~400 KB of strings, so a
// handful is plenty; the single-file build kept 6 and that number held.
const decoded = new Map();
const inflight = new Map();
// A film whose fetch failed. paint() runs ~60x a second and asks for whatever
// is missing, and clearing `inflight` on rejection let it re-request the same
// file every frame -- 60 req/s for as long as the schedule kept that film on
// air. Back off instead, and keep showing snow.
const failedUntil = new Map();
const RETRY_MS = 4000;

// ------------------------------------------------------------------ loading

async function fetchFilm(i) {
  if (decoded.has(i)) return decoded.get(i);
  if (inflight.has(i)) return inflight.get(i);
  const until = failedUntil.get(i);
  if (until !== undefined && Date.now() < until) {
    return Promise.reject(new Error(`film ${i}: backing off`));
  }

  const p = (async () => {
    const r = await fetch(`data/films/${String(i).padStart(3, "0")}.json`);
    if (!r.ok) throw new Error(`film ${i}: HTTP ${r.status}`);
    const raw = await r.json();
    const frames = decodeFilm(raw.w, raw.c, raw.r);
    decoded.set(i, frames);
    // Keep the cache small, but never evict what is on screen right now, and
    // leave room for one warmed film per channel -- a cap of 6 would throw
    // away everything warmTheDial just paid for.
    for (const k of decoded.keys()) {
      if (decoded.size <= 14) break;
      if (k !== lastFilmIdx) decoded.delete(k);
    }
    inflight.delete(i);
    failedUntil.delete(i);
    return frames;
  })();

  inflight.set(i, p);
  p.catch(() => {
    inflight.delete(i);
    failedUntil.set(i, Date.now() + RETRY_MS);
  });
  return p;
}

// ------------------------------------------------------------------- static
//
// Between channels, a real set shows snow. It is also doing something useful:
// covering the fetch of the first film on the new channel, so a slow network
// reads as tuning rather than as a hang.
const SNOW = " ░▒▓█";
function snowFrame() {
  const { cols, rows } = M;
  let s = "";
  for (let r = 0; r < rows; r++) {
    if (r) s += "\n";
    for (let c = 0; c < cols; c++) {
      s += SNOW[(Math.random() * SNOW.length) | 0];
    }
  }
  return s;
}

// -------------------------------------------------------------------- paint

function paint(now) {
  if (now < tuningUntil) {
    el.canvas.textContent = snowFrame();
    el.canvas.style.setProperty("--glow", "#8d8376");
    lastPainted = "";
    return;
  }

  if (channel.live) {
    paintLive(now);
    return;
  }

  if (channel.video) {
    paintVideo(now);
    return;
  }

  const { filmIdx, frameIdx, slot } = frameAt(channel, now, M.films, M.epoch);
  const film = M.films[filmIdx];

  if (filmIdx !== lastFilmIdx) {
    lastFilmIdx = filmIdx;
    showNote(filmIdx, slot);
    // Pull the next film in while this one plays. A six-second film is ample
    // runway for a 131 KB fetch, so after the first one the station never
    // waits again.
    fetchFilm(upNext(channel, now, M.films, M.epoch)).catch(() => {});
  }

  const frames = decoded.get(filmIdx);
  if (!frames) {
    // Not here yet: hold snow rather than blanking the screen.
    el.canvas.textContent = snowFrame();
    fetchFilm(filmIdx).catch(() => {});   // backoff logged once, not per frame
    return;
  }

  // `lead` is dead air trimmed off the front of the BROADCAST by
  // tools/trim_lead.js. The payload still holds those frames -- deleting them
  // would break the byte-for-byte decoder parity proof -- so the manifest's
  // n is the broadcast length and we index past the blank opening here.
  const text = frames[Math.min(frameIdx + (film.lead || 0), frames.length - 1)];
  if (text !== lastPainted) {
    el.canvas.textContent = text;
    lastPainted = text;
  }

  // Shot colour: the film's own measured mean for the shot we are inside.
  // Shot ranges are in the film's ORIGINAL frame space, so they need the same
  // `lead` offset the picture got -- otherwise a trimmed film wears the wrong
  // shot's colour for its first second.
  const srcIdx = frameIdx + (film.lead || 0);
  const shot = (film.sh || []).find((s) => srcIdx >= s.a && srcIdx < s.b)
    || (film.sh || [])[0];
  if (shot) el.canvas.style.setProperty("--glow", shot.c);

  const dur = film.n / film.fps;
  el.nBar.style.width = `${((frameIdx / film.fps) / dur) * 100}%`;
  el.nProg.textContent =
    `${slot + 1} of ${channel.programme.length} · ${(frameIdx / film.fps).toFixed(1)}s / ${dur.toFixed(0)}s`;
}

// ---------------------------------------------------------------- CH00 live
//
// Not tape: what's on screen is computed, not looked up. schedule.js's
// frameAt/upNext/cycleSeconds are never called for this channel -- they
// require a `programme`, which a live channel does not have -- so this whole
// path is a sibling to paint()'s tape branch, not a modification of it.
//
// simulateFrame() is a pure function of epochIndex (see js/live.js) but it is
// not FREE: paint() runs at rAF (~60fps) while the simulation only advances
// once per LIVE_TICK_MS (250ms). Recomputing on every call would replay the
// same generation up to ~15 times for nothing, so the result is cached here,
// in the impure layer, keyed on the epochIndex it was computed for -- the
// same "what's on" (pure) / "the clock" (impure) split schedule.js draws.
function paintLive(now) {
  const epochIndex = epochIndexFor(now);
  if (epochIndex !== liveEpoch) {
    liveEpoch = epochIndex;
    liveText = simulateFrame(epochIndex, M.cols, M.rows);
  }
  if (liveText !== lastPainted) {
    el.canvas.textContent = liveText;
    lastPainted = liveText;
  }
  el.canvas.style.setProperty("--glow", "#59d68a"); // a life-green, not the archive amber

  const gen = ((epochIndex % CYCLE_TICKS) + CYCLE_TICKS) % CYCLE_TICKS;
  el.nBar.style.width = `${(gen / CYCLE_TICKS) * 100}%`;
  el.nProg.textContent = `generation ${gen + 1} of ${CYCLE_TICKS} · reseeds every ${(CYCLE_TICKS * LIVE_TICK_MS / 1000).toFixed(0)}s`;
}

/** The Now/Next panel for CH00, set once on tune-in rather than every frame --
 *  unlike a film's title, none of this changes between simulation ticks. */
function showLiveNote() {
  el.nTitle.textContent = "LIVE";
  el.nYear.textContent = "";
  el.nHook.textContent =
    "A predator/prey ecosystem, computed fresh from the wall clock -- not " +
    "archival footage. Every viewer looking now sees the same population at " +
    "the same instant, the same way every other channel on this station works.";
  el.nSrc.textContent =
    "Computed locally (js/live.js, a Wa-Tor-style simulation), rendered " +
    "through the vendored ascii-lab engine -- not sourced from archive.org.";
  el.nFid.innerHTML = '<span class="num">LIVE</span>';
  el.xTitle.textContent = "(continuous)";

  // showNote() is the only other place that ever writes --flicker-amt /
  // data-weave, and it runs once per FILM boundary -- a boundary CH00 does
  // not have, since paintLive() never calls it. Left alone, whatever the
  // previously-tuned tape film set (up to FLICKER_MAX, gate weave on for
  // anything pre-1920) stays on #screen and bleeds into this channel's
  // picture: a "LIVE" feed that is not archival footage would judder and
  // flicker like a 1902 reel, entirely depending on which channel a viewer
  // happened to flip in from. Reset both here, the same way an unknown-year
  // film would read -- CH00 has no year at all, let alone an old one.
  el.screen.style.setProperty("--flicker-amt", String(flickerIntensityForYear(undefined)));
  el.screen.dataset.weave = gateWeaveActive(undefined) ? "1" : "0";

  refreshDialNow();
  for (const c of el.wall.children) c.setAttribute("aria-current", "false");
}

// ---------------------------------------------------------------- CV LAB
//
// A channel of REAL video, not glyph frames: computer-vision subject/
// background splits (SAM2 matte + the fleet's ASCII engine). Sibling to
// paintLive() -- schedule.js's frameAt/upNext/cycleSeconds are never called
// here either, since this channel has no `programme`. clipAt() (js/
// videochan.js) is the pure "what's on and how far in" half; everything
// below is the impure video-element glue, same split as the rest of this
// file draws everywhere else.
const VIDEO_CLIP_MS = 6000; // every clip is a 6s cut, matching every tape film
let videoClipIdx = -1;      // last clip index loaded, so paint() does not

// re-set video.src (and restart playback) every frame once it is already
// showing the right clip.
function paintVideo(now) {
  if (!channel.clips || channel.clips.length === 0) return;
  const { index, offsetSec } = clipAt(channel.clips, VIDEO_CLIP_MS, M.epoch, now);
  const clip = channel.clips[index];
  if (index !== videoClipIdx) {
    videoClipIdx = index;
    el.vchan.src = clip.src;
    el.vchan.load();
    const seekAndPlay = () => {
      el.vchan.currentTime = offsetSec;
      el.vchan.play().catch(() => {});
    };
    el.vchan.addEventListener("loadedmetadata", seekAndPlay, { once: true });
    el.nTitle.textContent = clip.label || "CV LAB";
    el.nHook.textContent = clip.hook || "";
    const nx = channel.clips[(index + 1) % channel.clips.length];
    el.xTitle.textContent = nx.label || "CV LAB";
  } else if (Math.abs(el.vchan.currentTime - offsetSec) > 1.5 && el.vchan.readyState >= 1) {
    // Drift correction: <video>'s own clock is not locked to the station's
    // wall clock the way the glyph path is, so a paused/buffering/slow tab
    // can fall behind. Resync rather than let it silently keep drifting.
    el.vchan.currentTime = offsetSec;
  }
  const dur = VIDEO_CLIP_MS / 1000;
  el.nBar.style.width = `${(offsetSec / dur) * 100}%`;
  el.nProg.textContent = `${index + 1} of ${channel.clips.length} · ${offsetSec.toFixed(1)}s / ${dur.toFixed(0)}s`;
}

/** The Now/Next panel + screen-mode switch for the CV LAB channel, set once
 *  on tune-in -- mirrors showLiveNote(), but also has to show the <video>
 *  element and hide the glyph canvas, since this is the one channel type
 *  that is not text at all. */
function showVideoNote() {
  el.canvas.hidden = true;
  el.vchan.hidden = false;
  videoClipIdx = -1; // force paintVideo() to (re)load a clip on the next tick
  el.nYear.textContent = "";
  el.nSrc.textContent =
    "SAM2 (hiera-tiny) subject matte + the fleet's ASCII engine, real archive.org " +
    "footage -- not a filter, a real per-frame computer-vision split.";
  el.nFid.innerHTML = '<span class="num">CV</span>';
  refreshDialNow();
  for (const c of el.wall.children) c.setAttribute("aria-current", "false");
}

/** The inverse of showVideoNote(): back to the glyph canvas. Called from
 *  tuneTo() whenever the channel being LEFT was a video channel, so the
 *  <video> element does not keep playing/decoding in the background under
 *  a channel that no longer matches it. */
function leaveVideoIfNeeded(prevChannel) {
  if (!prevChannel || !prevChannel.video) return;
  el.vchan.pause();
  el.vchan.removeAttribute("src");
  el.vchan.load();
  el.canvas.hidden = false;
  el.vchan.hidden = true;
}

// Used in ATTRIBUTE contexts as well as text (style="--tile:${esc(...)}",
// href="https://${esc(...)}"), so quotes have to go too. No manifest value
// carries one today -- but `t` and `cr` come from archive.org metadata, so
// that is one data refresh away from mattering.
function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function showNote(filmIdx, slot) {
  const f = M.films[filmIdx];

  // Period effects, once per film boundary -- never per frame. The math
  // lives in period-fx.js and is unit-tested there; here it is only ever
  // written into a custom property / data attribute for theatre.css to draw.
  el.screen.style.setProperty("--flicker-amt", String(flickerIntensityForYear(f.y)));
  el.screen.dataset.weave = gateWeaveActive(f.y) ? "1" : "0";
  clearTimeout(cueTimer);
  el.cue.dataset.show = "1";
  cueTimer = setTimeout(() => { el.cue.dataset.show = "0"; }, CUE_MARK_MS);

  el.nTitle.textContent = f.t;
  el.nYear.textContent = f.y ? ` ${f.y}` : "";
  el.nHook.textContent = f.h ||
    (f.k === "kie-generated"
      ? "An original, generated for this programme."
      : "Six seconds from one of the most-watched public domain films on archive.org.");
  el.nSrc.innerHTML = String(f.cr).startsWith("archive.org")
    ? `<a href="https://${esc(f.cr)}" target="_blank" rel="noopener">${esc(f.cr)}</a>`
    : esc(f.cr);

  const g = f.g || {};
  const verdict = g.st === "pass" ? '<span class="num">PASS</span>'
    : g.st === "ungraded" ? '<span class="warn">UNGRADED</span>'
    : '<span class="bad">FAIL</span>';
  el.nFid.innerHTML =
    `${g.ra !== null && g.ra !== undefined ? g.ra + "&times;" : "not measurable"} ${verdict}`;

  const nx = M.films[upNext(channel, Date.now(), M.films, M.epoch)];
  el.xTitle.textContent = nx ? nx.t : "—";

  // The dial shows what is on every channel, not just this one.
  refreshDialNow();
  const cells = el.wall.children;
  for (const c of cells) {
    c.setAttribute("aria-current", c.dataset.i === String(filmIdx) ? "true" : "false");
  }
}

// --------------------------------------------------------------------- dial

function buildDial() {
  el.dial.innerHTML = "";
  for (const ch of M.channels) {
    const b = document.createElement("button");
    b.className = "ch";
    b.dataset.ch = ch.ch;
    b.setAttribute("aria-current", "false");
    b.innerHTML =
      `<span class="ch-no">${String(ch.ch).padStart(2, "0")}</span>` +
      `<span class="ch-name">${esc(ch.name)}</span>` +
      `<span class="ch-now" data-now="${ch.ch}"></span>`;
    b.onclick = () => tuneTo(ch.ch);
    el.dial.appendChild(b);
  }
}

function refreshDialNow() {
  const now = Date.now();
  for (const ch of M.channels) {
    const span = el.dial.querySelector(`[data-now="${ch.ch}"]`);
    if (!span) continue;
    if (ch.live) { span.textContent = "LIVE"; continue; }
    if (ch.video) { span.textContent = "CV LAB"; continue; }
    const { filmIdx } = frameAt(ch, now, M.films, M.epoch);
    span.textContent = M.films[filmIdx].t;
  }
  for (const b of el.dial.children) {
    b.setAttribute("aria-current",
      b.dataset.ch === String(channel.ch) ? "true" : "false");
  }
}

// --------------------------------------------------------------------- wall

function buildWall() {
  const chOf = new Map();
  for (const ch of M.channels) {
    if (ch.ch === 1 || ch.live || ch.video) continue;  // ch01 carries everything;
    for (const i of ch.programme) if (!chOf.has(i)) chOf.set(i, ch);  // live/video have no films
  }

  const frag = document.createDocumentFragment();
  M.films.forEach((f, i) => {
    const fig = document.createElement("figure");
    fig.className = "cell";
    fig.dataset.i = i;
    fig.tabIndex = 0;
    const home = chOf.get(i);
    const badge = f.g && f.g.st === "fail"
      ? '<span class="badge b-fail">FAIL</span>'
      : f.g && f.g.st === "ungraded"
        ? '<span class="badge b-ung">UNGRADED</span>' : "";
    fig.innerHTML =
      `<pre style="--tile:${esc(f.col)}">${esc(f.p)}</pre>` +
      (home ? `<span class="chip">${String(home.ch).padStart(2, "0")}</span>` : "") +
      badge +
      `<figcaption><span class="t">${esc(f.t)}</span>` +
      `<span class="y">${esc(f.y || "")}</span></figcaption>`;
    const go = () => {
      if (home) tuneTo(home.ch, i);
      closeGuide();
    };
    fig.onclick = go;
    fig.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } };
    frag.appendChild(fig);
  });
  el.wall.appendChild(frag);
}

// -------------------------------------------------------------------- tuner

function tuneTo(chNo, wantFilm) {
  const next = M.channels.find((c) => c.ch === chNo);
  if (!next || next === channel) return;

  // Only a real flip once a session is open. The internal tuneTo() call
  // boot() makes to pick the starting channel runs before the viewer has
  // clicked through power-on, when telemetryState is still null -- correctly
  // a no-op here, since no session has started yet.
  if (telemetryState) {
    const { state, event } = recordFlip(telemetryState, Date.now());
    telemetryState = state;
    track(event);
  }

  const prevChannel = channel;
  channel = next;
  lastFilmIdx = -1;
  lastPainted = "";
  liveEpoch = -1;   // force paintLive() to recompute rather than show stale text
  leaveVideoIfNeeded(prevChannel);

  // Which wipe plays for this flip -- deterministic, not random, so a
  // channel flip is still a testable, reproducible event (see js/wipe.js).
  // The rule: cycle through the three variants keyed on (destination
  // channel + a running flip counter), so two different channels don't
  // necessarily open on the same wipe even on their first-ever flip.
  const variant = wipeVariantFor(chNo, flipCount++);
  for (const v of WIPE_VARIANTS) el.screen.classList.remove(`wipe-${v}`);
  // Force a reflow before re-adding: two flips can land on the same variant
  // back to back (e.g. CH05->CH04 is a flipCount+1, chNo-1 -- net unchanged
  // mod 3), and removing+adding the identical class in the same tick would
  // not restart its CSS animation without one.
  void el.screen.offsetWidth;
  el.screen.classList.add(`wipe-${variant}`);

  // Snow for a beat. Long enough to read as a tuner, short enough not to be
  // in the way; it also covers the first fetch on the new channel. All three
  // wipe variants share this window -- only the CSS treatment of the
  // picture differs (css/theatre.css, .screen.wipe-*), so timing stays the
  // one thing this file has to reason about.
  tuningUntil = Date.now() + 340;
  el.tuning.hidden = false;

  // Vertical-hold slip: the raster rolls and settles during/just after the
  // snow, like a set that has just been switched to this channel.
  el.screen.dataset.vhold = "1";
  clearTimeout(vholdTimer);
  vholdTimer = setTimeout(() => { el.screen.dataset.vhold = "0"; }, VHOLD_MS);
  // Cancel the previous timer: two flips inside 340ms and the first one's
  // timer hides the overlay while paint() is still drawing snow.
  clearTimeout(tuningTimer);
  tuningTimer = setTimeout(() => { el.tuning.hidden = true; }, 340);

  el.chno.textContent = String(channel.ch).padStart(2, "0");
  el.chname.textContent = channel.name;
  el.chtag.textContent = channel.tagline;
  el.bug.textContent = String(channel.ch).padStart(2, "0");
  document.title = `CH${String(channel.ch).padStart(2, "0")} ${channel.name} — ASCII Cinema`;

  if (channel.live) {
    // Nothing to fetch -- the picture is computed, not tape -- and no bed to
    // play (there is no audio/ch00.mp3): stop whatever the previous channel
    // was playing rather than leave it looping under a channel it no longer
    // matches.
    stopAudio();
    showLiveNote();
  } else if (channel.video) {
    // No schedule.js film to fetch (video channels have no `programme`),
    // but setAudio() DOES apply here -- see its own ch.audioSrc comment.
    setAudio(channel);
    showVideoNote();
  } else {
    // Warm whatever is on air right now, plus the one after it.
    const now = Date.now() + 340;
    const { filmIdx } = frameAt(channel, now, M.films, M.epoch);
    fetchFilm(filmIdx).catch((e) => console.warn(e));
    if (wantFilm !== undefined && wantFilm !== filmIdx) fetchFilm(wantFilm).catch(() => {});

    setAudio(channel);
  }
  refreshDialNow();
  history.replaceState(null, "", `#ch${String(channel.ch).padStart(2, "0")}`);
}

function step(delta) {
  const i = M.channels.findIndex((c) => c.ch === channel.ch);
  const n = M.channels[(i + delta + M.channels.length) % M.channels.length];
  tuneTo(n.ch);
}

// -------------------------------------------------------------------- sound

function setAudio(ch) {
  if (!audio) return;
  // CH00 has no audio/ch00.mp3 -- guarded HERE, not just at tuneTo()'s call
  // site, because the power-on handler also calls setAudio(channel) directly
  // (via whenPainted) for whatever channel a #ch00 bookmark landed on before
  // the set was even turned on. One guard in the function closes every call
  // site at once instead of needing to remember it at each one.
  if (ch.live) { stopAudio(); return; }
  // A video channel's clips are 6s cuts with no score of their own; reusing
  // a real tape channel's bed (ch.audioSrc, set in data/index.json) beats
  // either silence or inventing a new one -- CV LAB's clips are all drawn
  // from CH02, so its own bed is the honest, thematically-matched choice.
  const src = ch.audioSrc || `audio/ch${String(ch.ch).padStart(2, "0")}.mp3`;
  if (audio.dataset.src === src) return;
  audio.dataset.src = src;
  audio.src = src;
  audio.loop = true;
  // A channel with no bed yet must not throw or leave the last one playing.
  audio.play().catch(() => {});
}

// CH00 has no audio/ch00.mp3 -- there is no bed to compose for a channel
// that isn't tape -- so tuning to it stops whatever was playing rather than
// leaving the previous channel's loop running under it, or requesting a file
// that 404s every time paint() reaches this channel.
function stopAudio() {
  if (!audio) return;
  audio.pause();
  audio.removeAttribute("src");
  audio.dataset.src = "";
}

function syncMuteButton() {
  // Read the ELEMENT, never a flag we set ourselves: the browser can refuse
  // to play for reasons this code does not know about, and a button that
  // reports our intention rather than the audio's state is a lie.
  const off = !audio || audio.muted || audio.paused;
  el.btnMute.textContent = off ? "Sound off" : "Sound on";
  el.btnMute.setAttribute("aria-pressed", off ? "true" : "false");
}

function toggleMute() {
  if (!audio) return;
  audio.muted = !audio.muted;
  if (!audio.muted && audio.paused) audio.play().catch(() => {});
  syncMuteButton();
}

// ----------------------------------------------------------------- presence
//
// A vanity signal, not analytics: "N watching" next to the ON AIR bug is the
// only place this station admits two people might be looking at the exact
// same frame right now (see README's "The one idea"). It rides on the
// project's first serverless endpoint, api/presence.js, and Vercel KV --
// which is NOT provisioned as of this writing, so every poll below is
// expected to fail in production today. That failure must be invisible:
// plain "● ON AIR" (already the element's default markup) is always a
// correct thing to be showing, so the fallback is just "leave it alone."
//
// Same lesson this file already learned the hard way with fetchFilm's
// RETRY_MS backoff: a dead endpoint that gets re-hit forever is a standing
// cost for nothing. Unlike a film (which might start succeeding once the
// backoff window passes), a 404/503 here won't change mid-session -- KV
// provisioning is an operator action in the dashboard, not something that
// flips while someone is watching -- so after a few consecutive failures
// this simply stops polling for the rest of the session instead of
// retrying forever.

const PRESENCE_SID_KEY = "ascii-cinema:presence:sid";
const PRESENCE_GIVE_UP_AFTER = 3;
let presenceFailures = 0;

function presenceSessionId() {
  try {
    let sid = sessionStorage.getItem(PRESENCE_SID_KEY);
    if (!sid) {
      sid = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
      sessionStorage.setItem(PRESENCE_SID_KEY, sid);
    }
    return sid;
  } catch {
    // Private browsing / storage disabled: fall back to an id that at least
    // survives this one poll, even though it can't dedupe across polls.
    // Not this feature's problem to solve -- see file header.
    return `${Date.now()}-${Math.random()}`;
  }
}

async function pollPresence() {
  try {
    const sid = presenceSessionId();
    const r = await fetch(`api/presence?sid=${encodeURIComponent(sid)}`);
    if (!r.ok) throw new Error(`presence: HTTP ${r.status}`);
    const body = await r.json();
    presenceFailures = 0;
    el.onair.textContent = typeof body.count === "number" && body.count > 0
      ? `● ON AIR · ${body.count} watching`
      : "● ON AIR";
  } catch {
    // 404 (route not deployed), KV not provisioned, offline, a timeout --
    // all the same outcome for the viewer: the plain indicator that was
    // already correct before this feature existed.
    el.onair.textContent = "● ON AIR";
    presenceFailures++;
    if (presenceFailures >= PRESENCE_GIVE_UP_AFTER) return;   // stop the chain
  }
  setTimeout(pollPresence, PRESENCE_POLL_MS + Math.random() * PRESENCE_JITTER_MS);
}

// -------------------------------------------------------------------- guide

function openGuide() { el.guide.hidden = false; }
function closeGuide() { el.guide.hidden = true; }

// -------------------------------------------------- the velvet + fullscreen
//
// The curtains are a WIPE, not a fixture: display:none between shows, so
// they never cover the picture, eat a double-click, or cost a paint. They run
// on the big transitions only -- the power-on reveal and entering/leaving
// fullscreen. Channel flips keep their 340 ms snow: tuning is a television
// beat, and burying it under a second of velvet would make the dial feel
// broken. Same reasoning as FUN.md F7b/F8: the fast path stays fast.

const CURTAIN_HOLD_MS = 260;   // closed velvet on screen, before the reveal
const CURTAIN_OPEN_MS = 1300;  // the drape; CSS keyframes run 1.2s, +100ms slack
let curtainBusy = false;

/** Close instantly, hold a beat, run mid() behind the drape, then open. */
function curtainWipe(mid) {
  if (curtainBusy) { mid && mid(); return; }
  curtainBusy = true;
  el.curtains.hidden = false;            // panels cover at once (closed state)
  setTimeout(() => {
    mid && mid();
    el.curtains.classList.add("opening"); // trigger the drape
    setTimeout(() => {
      el.curtains.classList.remove("opening");
      el.curtains.hidden = true;         // rig goes back to display:none
      curtainBusy = false;
    }, CURTAIN_OPEN_MS);
  }, CURTAIN_HOLD_MS);
}

let wasFs = false;

// Safari older than 16.4 only has the prefixed API. Without these fallbacks
// the F key is a silent no-op there -- a reported bug, not a guess.
const fsElement = () => document.fullscreenElement || document.webkitFullscreenElement;

function toggleFull() {
  if (fsElement()) {
    const exit = document.exitFullscreen || document.webkitExitFullscreen;
    const p = exit && exit.call(document);
    if (p && p.catch) p.catch(() => {});   // webkitExitFullscreen returns void
    return;
  }
  const req = el.screen.requestFullscreen || el.screen.webkitRequestFullscreen;
  if (!req) return;

  // The request must leave INSIDE this gesture task. Safari consumes
  // transient activation at task end, so a request deferred behind the
  // curtain hold is refused there and the .catch swallowed it -- "F does
  // nothing", measured 262ms late by tests/fullscreen.test.js. The drape
  // still happens, just around the request instead of before it: the velvet
  // covers the picture instantly (it appears closed, no transition needed),
  // the browser flips behind it, and the panels part on the big screen.
  if (!curtainBusy) {
    curtainBusy = true;
    el.curtains.hidden = false;
    setTimeout(() => {
      el.curtains.classList.add("opening");
      setTimeout(() => {
        el.curtains.classList.remove("opening");
        el.curtains.hidden = true;
        curtainBusy = false;
      }, CURTAIN_OPEN_MS);
    }, CURTAIN_HOLD_MS);
  }
  const p = req.call(el.screen);
  if (p && p.catch) p.catch(() => {});
}

function syncFullButton() {
  const fs = !!fsElement();
  el.btnFull.textContent = fs ? "Window" : "Full";
  el.btnFull.setAttribute("aria-pressed", fs ? "true" : "false");
  // Esc leaves fullscreen without asking us; the layout jump back is exactly
  // the kind of cut the velvet exists to cover.
  if (!fs && wasFs) curtainWipe();
  wasFs = fs;
}

// --------------------------------------------------------------------- loop

function loop() {
  const now = Date.now();
  try {
    paint(now);
  } catch (e) {
    console.error(e);
  }
  const d = new Date(now);
  el.clock.textContent =
    [d.getHours(), d.getMinutes(), d.getSeconds()]
      .map((x) => String(x).padStart(2, "0")).join(":");
  requestAnimationFrame(loop);
}

// --------------------------------------------------------------------- boot

async function boot() {
  const r = await fetch("data/index.json");
  if (!r.ok) throw new Error(`manifest: HTTP ${r.status}`);
  M = await r.json();

  buildDial();
  buildWall();

  // A #chNN in the URL is a bookmark to a channel, not to a film.
  const want = parseInt((location.hash.match(/^#ch(\d+)$/) || [])[1], 10);
  const start = M.channels.find((c) => c.ch === want) || M.channels[0];
  tuneTo(start.ch);

  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (k === "Escape") { closeGuide(); return; }
    if (k === "g" || k === "G") {
      e.preventDefault();
      el.guide.hidden ? openGuide() : closeGuide();
      return;
    }
    if (k === "m" || k === "M") { e.preventDefault(); toggleMute(); return; }
    if (k === "f" || k === "F") { e.preventDefault(); toggleFull(); return; }
    if (k === "ArrowDown") { e.preventDefault(); step(1); return; }
    if (k === "ArrowUp") { e.preventDefault(); step(-1); return; }
    if (/^[0-9]$/.test(k)) {
      e.preventDefault();
      tuneTo(k === "0" ? 10 : parseInt(k, 10));
    }
  });

  el.btnGuide.onclick = openGuide;
  el.btnClose.onclick = closeGuide;
  el.btnMute.onclick = toggleMute;
  el.btnFull.onclick = toggleFull;
  el.screen.ondblclick = toggleFull;
  document.addEventListener("fullscreenchange", syncFullButton);
  document.addEventListener("webkitfullscreenchange", syncFullButton);
  // No Fullscreen API (iOS Safari) = no button. An absent control is honest;
  // a dead one is not.
  if (!document.fullscreenEnabled && !document.webkitFullscreenEnabled) {
    el.btnFull.hidden = true;
  }
  setInterval(refreshDialNow, 2000);

  requestAnimationFrame(loop);
}

// The set turns on with a gesture, because audio needs one. Boot the data
// immediately so the wait happens behind the button rather than after it.
const booted = boot().catch((e) => {
  console.error(e);
  el.poweron.querySelector(".po-sub").textContent =
    "The station could not be reached. " + e.message;
  throw e;
});

el.tune.onclick = async () => {
  el.tune.disabled = true;
  el.tune.textContent = "Warming up…";
  try {
    await booted;
  } catch {
    return;
  }
  audio = new Audio();
  audio.loop = true;
  audio.volume = 0.55;
  // NOT preload="none": that makes the download wait on play() succeeding, and
  // where autoplay is refused the bed then never loads at all. Deferring the
  // setAudio CALL already keeps the bed out of the first film's way.
  el.poweron.hidden = true;
  el.set.hidden = false;
  poweredOn = true;
  // Usually starts the session immediately. But `await booted` above can
  // finish while the tab is backgrounded (e.g. the viewer switched away
  // mid-load), and unconditionally opening here would fold that hidden
  // interval into the session's dwell/length once it later ends. Leave
  // telemetryState null in that case -- the visibilitychange handler opens
  // it for real once the tab is actually visible again.
  telemetryState = document.hidden ? null : startSession(Date.now());
  curtainWipe();  // the house opens: velvet parts on the first broadcast

  // Fire-and-forget: never awaited, never on the path to first paint. Its
  // own try/catch means a hung or failing endpoint cannot delay or break
  // anything downstream of it.
  pollPresence();

  // The score bed is 3.7 MB and the first film is 131 KB. Starting them
  // together let the bed win the bandwidth and pushed first picture past
  // seven seconds (FUN.md F9, measured: 7279 ms -> 1664 ms once deferred).
  // Picture first; sound and speculation queue up behind it.
  whenPainted(() => {
    setAudio(channel);
    syncMuteButton();
    setInterval(syncMuteButton, 1000);
    warmTheDial();
  });
};

/** Run fn once a real frame is on screen, or give up after ~10s. */
function whenPainted(fn) {
  let tries = 0;
  const poll = () => {
    if (lastPainted || ++tries > 200) { fn(); return; }
    setTimeout(poll, 50);
  };
  poll();
}

// Flipping to a cold channel meant waiting on its film mid-snow (FUN.md F8).
// Warming the other channels is the fix -- but the first attempt made it
// WORSE, one slow flip in six becoming three, because decodeFilm turns a
// 131 KB payload into ~400 KB of strings SYNCHRONOUSLY. Nine of those back to
// back hold the main thread and block the very paint a flip is waiting for.
//
// So: one film at a time, on a timer, standing aside whenever someone has just
// touched the dial. A warm cache is worth nothing if earning it is what makes
// the picture late.
let warmQueue = [];

function warmTheDial() {
  // CH00 has no film to warm -- frameAt() would throw on its missing
  // programme, and warmStep()'s try/catch would just burn a queue slot on it
  // every pass.
  warmQueue = M.channels.filter((c) => c.ch !== channel.ch && !c.live && !c.video).map((c) => c.ch);
  setTimeout(warmStep, 900);
}

function warmStep() {
  if (!warmQueue.length) return;
  // Someone is tuning. Their film outranks our speculation.
  if (Date.now() < tuningUntil + 1500) { setTimeout(warmStep, 900); return; }
  // shift() must happen ONCE, not once per candidate examined. Calling it
  // inside the find predicate drained the whole queue on the first step and
  // matched nothing, so this warmed 0 of 9 channels and died silently in the
  // try/catch below. Found by review, not by any test.
  const chNo = warmQueue.shift();
  const ch = M.channels.find((c) => c.ch === chNo);
  try {
    // Aim slightly ahead, so we fetch what that channel will be showing by
    // the time anyone gets there rather than what it is showing now.
    if (!ch) { setTimeout(warmStep, 60); return; }
    const { filmIdx } = frameAt(ch, Date.now() + 1200, M.films, M.epoch);
    if (decoded.has(filmIdx)) { setTimeout(warmStep, 60); return; }
    fetchFilm(filmIdx).catch(() => {}).finally(() => setTimeout(warmStep, 700));
  } catch {
    setTimeout(warmStep, 700);
  }
}
