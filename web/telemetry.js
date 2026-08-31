// Dwell/flip/session math for the station's viewership telemetry.
//
// Same split as schedule.js: pure functions of (state, wall clock), testable
// with fake timestamps, no window.va, no DOM. theatre.js is the only caller
// and the only place these events reach the network -- this file never
// touches it. That split is what lets FUN.md's standing objection ("no
// human has watched an hour") become something measurable instead of an
// assertion that a hook was called.

/** Called once, when the set is switched on. */
export function startSession(now) {
  return { sessionStart: now, tuneStart: now, flips: 0 };
}

/**
 * Called every time the viewer actually changes channel (never on the
 * initial tune-in, which has no prior dwell to report). Returns the new
 * state and the dwell event for the channel just LEFT -- dwell is measured
 * from the previous flip (or session start), never from session start
 * itself, so a binge session's Nth flip reports that channel's own dwell,
 * not the whole visit so far.
 */
export function recordFlip(state, now) {
  const flips = state.flips + 1;
  const event = {
    name: "channel_dwell",
    dwellMs: now - state.tuneStart,
    flips,
  };
  return { state: { sessionStart: state.sessionStart, tuneStart: now, flips }, event };
}

/**
 * Called once, on visibilitychange->hidden or beforeunload. dwellMs is the
 * channel still on screen when the viewer left (measured from the last
 * flip, or session start if they never touched the dial); sessionMs is the
 * whole visit, from arrival.
 */
export function endSession(state, now) {
  return {
    name: "session_end",
    dwellMs: now - state.tuneStart,
    sessionMs: now - state.sessionStart,
    flips: state.flips,
  };
}

/**
 * Vercel Analytics' documented custom-event contract is
 * va('event', { name, data: {...} }) -- everything but the event name has
 * to nest under `data`, or it fires and carries no usable numbers.
 */
export function toAnalyticsPayload(event) {
  const { name, ...data } = event;
  return { name, data };
}
