/**
 * `dream.trained_inputs` against the Python functions' own answers (scripts/gen_fixtures.py, trained.json): difflib's
 * ratio and closest match, the prompt as compared, the JPEG frame size, and notes() line for line.
 */

import { describe, expect, it } from "vitest";
import { b64decode, casefold, checkedTrainedInputs, closestMatch, fixed2, jpegSize, plainPrompt, pyQuote, pyRepr, sequenceRatio, trainedNotes, trainingDataPath } from "../src/internal.js";
import { fixture } from "./helpers.js";

const trained = fixture("trained.json");

describe("trained inputs, as dream.trained_inputs", () => {
  it("difflib's SequenceMatcher ratio", () => {
    for (const { a, b, ratio } of trained.ratios) expect(sequenceRatio(a, b), `${JSON.stringify(a)} vs ${JSON.stringify(b)}`).toBe(ratio);
  });

  it("difflib's get_close_matches(n=1, cutoff=0.6)", () => {
    for (const { word, possibilities, match } of trained.closest) expect(closestMatch(word, possibilities, 0.6), word).toBe(match[0] ?? null);
  });

  it("the prompt as compared (_plain)", () => {
    for (const { text, plain } of trained.plain) expect(plainPrompt(text), JSON.stringify(text)).toBe(plain);
  });

  it("str.casefold() for every code point Python folds, and no other", () => {
    const folded = new Map<number, string>(trained.folds);
    for (const [cp, text] of folded) expect(casefold(String.fromCodePoint(cp)), cp.toString(16)).toBe(text);
    for (let cp = 0; cp < 0x110000; cp++) {
      if (folded.has(cp) || (cp >= 0xd800 && cp <= 0xdfff)) continue;
      const c = String.fromCodePoint(cp);
      if (casefold(c) !== c) expect.fail(`U+${cp.toString(16)} folds here but not in Python`);
    }
  });

  it("base64 as Python's b64decode(validate=False)", () => {
    for (const { text, bytes } of trained.decodes) {
      const got = b64decode(text);
      expect(got === null ? null : Array.from(got), JSON.stringify(text)).toEqual(bytes);
    }
  });

  it("leaves malformed training sizes and prompts out, where Python fails", () => {
    expect(trained.malformed.python).toBe("TypeError");
    const frames = { "video.front": ["/9j/wAARCAHgAoADAAAAAAAAAAAA/9lqcGVnLTk5"], "video.wrist_r": ["/9j/wAARCAHgAoADAAAAAAAAAAAA/9lqcGVnLTk5"] };
    expect(checkedTrainedInputs(trained.malformed.trained)).toEqual({ cameras: { wrist_r: [[640, 360]] }, prompts: ["wave"] });
    const notes = trainedNotes(trained.malformed.trained, "dream-1-ft:0a1b2c3d-x", frames, "wave");
    expect(notes).toHaveLength(1);
    expect(notes[0]).toMatch(/^video\.wrist_r frames are 640 x 480 \(aspect 1\.33\)/);
  });

  it("a JPEG's size from its start of frame, and none from a frame that has none", () => {
    for (const { frame, size } of trained.sizes) expect(jpegSize(frame), frame.slice(0, 24)).toEqual(size);
  });

  it("notes() line for line", () => {
    for (const { trained: inputs, frames, prompt, notes } of trained.notes) {
      expect(trainedNotes(inputs, "dream-1-ft:0a1b2c3d-x", frames, prompt ?? undefined)).toEqual(notes);
    }
  });

  it("Python's repr, .2f and quote(safe='')", () => {
    for (const { text, repr } of trained.reprs) expect(pyRepr(text)).toBe(repr);
    for (const { x, text } of trained.formats) expect(fixed2(x), String(x)).toBe(text);
    for (const { text, quoted } of trained.quotes) expect(pyQuote(text)).toBe(quoted);
    expect(trainingDataPath("dream-1-ft:0a1b2c3d-towels")).toBe("/v1/fine_tuning/models/dream-1-ft%3A0a1b2c3d-towels/training_data");
  });
});

describe("float() as dream.arm_file reads URDF numbers", () => {
  it("matches Python on Unicode digits and whitespace, underscores and specials", async () => {
    const { pyFloat } = await import("../src/pyfloat.js");
    for (const entry of trained.floats) {
      const got = pyFloat(entry.text);
      const label = JSON.stringify(entry.text);
      if (entry.error) expect(got, label).toBeNaN();
      else if (entry.value === null) expect(got, label).toBeNaN();
      else if (entry.value === "inf" || entry.value === "-inf") expect(got, label).toBe(entry.value === "inf" ? Infinity : -Infinity);
      else expect(got, label).toBe(entry.value);
    }
  });

  it("reads in linear time, a long invalid attribute included", async () => {
    const { pyFloat } = await import("../src/pyfloat.js");
    const started = performance.now();
    for (const text of ["1_".repeat(200_000) + "x", "1".repeat(400_000) + "e", "1".repeat(200_000) + ".1e+_"]) expect(pyFloat(text)).toBeNaN();
    // As in Python, a valid spelling too large for a double is an infinity (then refused as not finite).
    expect(pyFloat("\u0661".repeat(200_000) + ".")).toBe(Infinity);
    expect(performance.now() - started).toBeLessThan(2000);
  });
});
