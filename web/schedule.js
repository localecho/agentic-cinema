// The broadcast scheduler.
//
// Pure by design: no DOM, no requestAnimationFrame, no Date.now(). What is on
// screen is a function of (channel, wall clock) and nothing else. Two people
// who tune to CALIGARI HALL at the same instant see the same frame, and the
// station is already running when you arrive -- which is the single thing
// separating a television channel from a <video> playlist.
//
// Keeping the clock OUT of this file is also what makes it testable. Chrome
// freezes rAF outright in a hidden tab, so a frame counter read by an
// automated browser measures the throttle rather than the page; every claim
// about scheduling is settled against fake timestamps in tests/schedule.test.js
// instead of against a screenshot.

/** Running time of one full pass through a channel's programme, in seconds. */
export function cycleSeconds(channel, films) {
  let total = 0;
  for (const i of channel.programme) total += films[i].n / films[i].fps;
  return total;
}

/**
 * What this channel is showing at `nowMs`.
 * @returns {{filmIdx:number, frameIdx:number, slot:number}}
 *   slot is the position within the programme, so the UI can say "3 of 12"
 *   without searching for the film (a channel may play a film twice).
 */
export function frameAt(channel, nowMs, films, epochMs) {
  const programme = channel.programme;
  if (!programme || programme.length === 0) {
    // Returning a blank would paint `undefined` into the screen and read as a
    // rendering bug three layers away from the empty channel that caused it.
    throw new Error("channel has an empty programme");
  }

  const cycle = cycleSeconds(channel, films);
  // `%` in JavaScript keeps the sign of the left operand, so a clock earlier
  // than the epoch yields a negative offset and indexes off the front of the
  // programme. Anyone west of the epoch, or with a slow clock, hits this.
  let t = ((nowMs - epochMs) / 1000) % cycle;
  if (t < 0) t += cycle;

  for (let slot = 0; slot < programme.length; slot++) {
    const filmIdx = programme[slot];
    const film = films[filmIdx];
    const dur = film.n / film.fps;
    if (t < dur) {
      // Math.floor, not round: at 5.999s of a 6s film we are still on its
      // last frame, and rounding up would index one past the end.
      const frameIdx = Math.min(Math.floor(t * film.fps), film.n - 1);
      return { filmIdx, frameIdx, slot };
    }
    t -= dur;
  }

  // Only reachable through floating-point drift at the very last boundary;
  // land on the final frame rather than falling out with nothing.
  const slot = programme.length - 1;
  const filmIdx = programme[slot];
  return { filmIdx, frameIdx: films[filmIdx].n - 1, slot };
}

/** The film that follows whatever is on now, wrapping at the end. */
export function upNext(channel, nowMs, films, epochMs) {
  const { slot } = frameAt(channel, nowMs, films, epochMs);
  return channel.programme[(slot + 1) % channel.programme.length];
}
