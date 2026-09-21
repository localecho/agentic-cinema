// Smoke tests for web/schedule.js -- the pure broadcast scheduler.
//
// schedule.js's own header comment says "every claim about scheduling is
// settled against fake timestamps in tests/schedule.test.js instead of
// against a screenshot" -- this file is that promise, made real. No DOM, no
// network, no clock: frameAt/upNext/cycleSeconds are pure functions of
// (channel, wall clock, films, epoch), so they can be exercised with plain
// fake numbers.

import { test } from "node:test";
import assert from "node:assert/strict";
import { cycleSeconds, frameAt, upNext } from "../web/schedule.js";

// Two films, three seconds each at 1fps (3 frames), on a channel that plays
// film 0 then film 1 then loops. Cycle length: 6 seconds.
const films = [
  { n: 3, fps: 1 }, // film 0: frames 0,1,2 at t=[0,3)
  { n: 3, fps: 1 }, // film 1: frames 0,1,2 at t=[3,6)
];
const channel = { programme: [0, 1] };
const EPOCH = 1_000_000; // arbitrary epoch in ms

test("cycleSeconds sums every film's runtime in the programme", () => {
  assert.equal(cycleSeconds(channel, films), 6);
});

test("frameAt lands on the right film and frame at the start of the cycle", () => {
  const { filmIdx, frameIdx, slot } = frameAt(channel, EPOCH, films, EPOCH);
  assert.equal(filmIdx, 0);
  assert.equal(frameIdx, 0);
  assert.equal(slot, 0);
});

test("frameAt crosses into the second film once the first film's runtime elapses", () => {
  // 3.5s in: film 0 (3s) has finished, we're 0.5s into film 1 -> frame 0.
  const { filmIdx, frameIdx, slot } = frameAt(channel, EPOCH + 3500, films, EPOCH);
  assert.equal(filmIdx, 1);
  assert.equal(frameIdx, 0);
  assert.equal(slot, 1);
});

test("frameAt wraps around the cycle back to the first film", () => {
  // One full cycle (6s) plus 1.2s -> back at film 0, frame 1.
  const { filmIdx, frameIdx, slot } = frameAt(channel, EPOCH + 6000 + 1200, films, EPOCH);
  assert.equal(filmIdx, 0);
  assert.equal(frameIdx, 1);
  assert.equal(slot, 0);
});

test("frameAt does not index past the end of a film at its very last instant", () => {
  // 2.999s into film 0 (3s @ 1fps): last valid frame is index 2, not 3.
  const { filmIdx, frameIdx } = frameAt(channel, EPOCH + 2999, films, EPOCH);
  assert.equal(filmIdx, 0);
  assert.equal(frameIdx, 2);
});

test("frameAt handles a wall clock earlier than the epoch (negative offset)", () => {
  // 1.5s before epoch, on a 6s cycle -> equivalent to t=4.5s -> film 1, frame 1.
  const { filmIdx, frameIdx } = frameAt(channel, EPOCH - 1500, films, EPOCH);
  assert.equal(filmIdx, 1);
  assert.equal(frameIdx, 1);
});

test("frameAt throws on an empty programme rather than painting undefined", () => {
  assert.throws(() => frameAt({ programme: [] }, EPOCH, films, EPOCH), /empty programme/);
});

test("upNext reports the following film in the programme", () => {
  assert.equal(upNext(channel, EPOCH, films, EPOCH), 1);
});

test("upNext wraps from the last slot back to the first", () => {
  // 3.5s in: currently on slot 1 (film 1) -> next wraps to slot 0 (film 0).
  assert.equal(upNext(channel, EPOCH + 3500, films, EPOCH), 0);
});
