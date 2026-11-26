/**
 * Python's `float()` of a string, as `dream.arm_file` reads a URDF's numbers: in one pass, in linear time.
 *
 * As CPython does (`_PyUnicode_TransformDecimalAndSpaceToASCII`), every non-ASCII `str.isspace()` character becomes a
 * space and every non-ASCII decimal digit (full-width, Arabic-Indic, ...) its ASCII digit; any other non-ASCII character
 * makes the text no number, and ASCII is left as it is. The ASCII whitespace at the ends (space, tab, line feed,
 * carriage return, vertical tab, form feed; not U+001C to U+001F) is dropped, and what is left must be a sign, digits with single
 * underscores between digits, a point, more digits, an exponent; or `inf`, `infinity`, `nan` in any case. A leading
 * U+FEFF is not whitespace, so it makes the text no number, as in Python.
 */

import { DIGIT_ZEROS, SPACES } from "./unicode.data.js";

const SPACE_SET = new Set(SPACES);

/** The decimal value of `cp` (0 to 9), or -1 when it is not a decimal digit. */
function decimal(cp: number): number {
  let lo = 0;
  let hi = DIGIT_ZEROS.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const zero = DIGIT_ZEROS[mid]!;
    if (cp < zero) hi = mid - 1;
    else if (cp > zero + 9) lo = mid + 1;
    else return cp - zero;
  }
  return -1;
}

/**
 * `float(text)` in Python, or NaN for text it refuses (and for `nan` itself).
 *
 * @param work Counts the characters stepped over, for tests: at most a few per character of `text`.
 */
export function pyFloat(text: string, work?: { steps: number }): number {
  let ascii = "";
  for (const c of text) {
    if (work) work.steps++;
    const cp = c.codePointAt(0)!;
    if (cp < 0x80) ascii += c;
    else if (SPACE_SET.has(cp)) ascii += " ";
    else {
      const digit = decimal(cp);
      if (digit < 0) return NaN;
      ascii += String(digit);
    }
  }
  let start = 0;
  let end = ascii.length;
  const space = (c: string | undefined) => c === " " || c === "\t" || c === "\n" || c === "\r" || c === "\v" || c === "\f";
  while (start < end && space(ascii[start])) start++;
  while (end > start && space(ascii[end - 1])) end--;
  if (work) work.steps += ascii.length - (end - start);
  const body = ascii.slice(start, end);
  let i = 0;
  let sign = 1;
  if (body[i] === "+" || body[i] === "-") sign = body[i++] === "-" ? -1 : 1;
  const word = body.slice(i).toLowerCase();
  if (word === "inf" || word === "infinity") return sign * Infinity;
  if (word === "nan") return NaN;
  const isDigit = (c: string | undefined) => c !== undefined && c >= "0" && c <= "9";
  /** Digits with single underscores between them, from `i`; the number of digits read (0 for none). */
  const digits = (): number => {
    let count = 0;
    while (isDigit(body[i])) {
      if (work) work.steps++;
      i++;
      count++;
      if (body[i] === "_" && isDigit(body[i + 1])) i++;
    }
    return count;
  };
  const whole = digits();
  let fraction = 0;
  if (body[i] === ".") {
    i++;
    fraction = digits();
  }
  if (whole + fraction === 0) return NaN;
  if (body[i] === "e" || body[i] === "E") {
    i++;
    if (body[i] === "+" || body[i] === "-") i++;
    if (digits() === 0) return NaN;
  }
  if (i !== body.length) return NaN;
  return Number(body.replace(/_/g, ""));
}

/** Python's `str.split()` with no separator: the runs of non-`str.isspace()` characters, in order. */
export function pySplit(text: string): string[] {
  const out: string[] = [];
  let word = "";
  for (const c of text) {
    if (SPACE_SET.has(c.codePointAt(0)!)) {
      if (word) out.push(word);
      word = "";
    } else word += c;
  }
  if (word) out.push(word);
  return out;
}
