/**
 * JSON read so that every number keeps the text it arrived as.
 *
 * `JSON.parse` turns `9007199254740993` into 9007199254740992 and `1e400` into Infinity, and `JSON.stringify` writes
 * -0 as `0`: a reply re-serialised that way is not the reply. {@link parseRaw} reads the same grammar as `JSON.parse`
 * (strict: no comments, no trailing commas, nothing after the value), without recursion, keeping each number as a
 * {@link RawNumber}: its source token and the double it reads as. {@link writeRaw} writes such a tree back with every
 * number byte-identical to its source; {@link unwrap} turns it into plain values. Object keys, `__proto__` included,
 * are own properties, as `JSON.parse` makes them.
 */

/** A number as it arrived: its JSON token, and the double the token reads as (Infinity for `1e400`). */
export class RawNumber {
  constructor(
    readonly raw: string,
    readonly value: number,
  ) {}

  /** Whether the double is the token's value exactly: finite, and for an integer token, that very integer. */
  get exact(): boolean {
    if (!Number.isFinite(this.value)) return false;
    if (/[.eE]/.test(this.raw) || Number.isSafeInteger(this.value)) return true;
    return BigInt(this.raw) === BigInt(this.value);
  }
}

/** A value read by {@link parseRaw}. */
export type RawJson = null | boolean | string | RawNumber | RawJson[] | { [key: string]: RawJson };

/** Sets `key` on `target` as an own data property: a key named `__proto__` stays a key, never the prototype. */
export function setOwn(target: Record<string, unknown>, key: string, value: unknown): void {
  if (key === "__proto__") Object.defineProperty(target, key, { value, enumerable: true, writable: true, configurable: true });
  else target[key] = value;
}

const NUMBER = /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/y;
const STRING = /"(?:[^"\\\u0000-\u001f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"/y;
const SPACE = /[ \t\n\r]*/y;

/**
 * Reads JSON `text` as `JSON.parse` does, keeping every number as a {@link RawNumber}.
 *
 * @throws SyntaxError `text` is not one JSON value (the message names the offset only, never the text).
 */
export function parseRaw(text: string): RawJson {
  let i = 0;
  const fail = (): never => {
    throw new SyntaxError(`not JSON at offset ${i}`);
  };
  const space = () => {
    SPACE.lastIndex = i;
    SPACE.exec(text);
    i = SPACE.lastIndex;
  };
  const scalar = (): RawJson => {
    const c = text[i];
    if (c === '"') {
      STRING.lastIndex = i;
      const match = STRING.exec(text) ?? fail();
      i = STRING.lastIndex;
      return JSON.parse(match[0]) as string;
    }
    if (c === "-" || (c !== undefined && c >= "0" && c <= "9")) {
      NUMBER.lastIndex = i;
      const match = NUMBER.exec(text) ?? fail();
      i = NUMBER.lastIndex;
      return new RawNumber(match[0], Number(match[0]));
    }
    for (const [word, value] of [["true", true], ["false", false], ["null", null]] as const) {
      if (text.startsWith(word, i)) {
        i += word.length;
        return value;
      }
    }
    return fail();
  };
  type Frame = { kind: "array"; out: RawJson[] } | { kind: "object"; out: Record<string, RawJson>; key: string | null };
  const stack: Frame[] = [];
  let result: RawJson = null;
  let done = false;
  // Hands a finished value to the container being built, or makes it the result.
  const put = (value: RawJson) => {
    const top = stack.at(-1);
    if (!top) {
      result = value;
      done = true;
    } else if (top.kind === "array") top.out.push(value);
    else {
      setOwn(top.out, top.key!, value);
      top.key = null;
    }
  };
  const open = (): boolean => {
    space();
    if (text[i] === "[") {
      i++;
      stack.push({ kind: "array", out: [] });
      space();
      if (text[i] === "]") {
        i++;
        put((stack.pop() as { out: RawJson[] }).out);
        return false;
      }
      return true;
    }
    if (text[i] === "{") {
      i++;
      stack.push({ kind: "object", out: {}, key: null });
      space();
      if (text[i] === "}") {
        i++;
        put((stack.pop() as { out: Record<string, RawJson> }).out);
        return false;
      }
      readKey();
      return true;
    }
    put(scalar());
    return false;
  };
  const readKey = () => {
    space();
    if (text[i] !== '"') fail();
    const key = scalar() as string;
    space();
    if (text[i] !== ":") fail();
    i++;
    (stack.at(-1) as { key: string | null }).key = key;
  };
  // A value is expected next; containers stay open on the stack until their closing bracket.
  let expectValue = true;
  while (!done) {
    if (expectValue) {
      expectValue = open();
      if (expectValue) continue;
      if (done) break;
    }
    space();
    const top = stack.at(-1)!;
    const close = top.kind === "array" ? "]" : "}";
    if (text[i] === ",") {
      i++;
      if (top.kind === "object") readKey();
      expectValue = true;
    } else if (text[i] === close) {
      i++;
      stack.pop();
      put(top.out);
    } else fail();
  }
  space();
  if (i !== text.length) fail();
  return result;
}

/**
 * A {@link RawJson} tree as plain values, each number its double; without recursion, however deep.
 *
 * @param text Applied to every string and object key (a redaction).
 */
export function unwrap(value: RawJson, text: (s: string) => string = (s) => s): unknown {
  const plain = (node: RawJson): unknown => (node instanceof RawNumber ? node.value : typeof node === "string" ? text(node) : node);
  if (typeof value !== "object" || value === null || value instanceof RawNumber) return plain(value);
  const root: unknown = Array.isArray(value) ? [] : {};
  const stack: [RawJson[] | Record<string, RawJson>, unknown[] | Record<string, unknown>][] = [[value as RawJson[] | Record<string, RawJson>, root as unknown[] | Record<string, unknown>]];
  while (stack.length) {
    const [from, to] = stack.pop()!;
    const keys = Array.isArray(from) ? from.map((_, i) => i) : Object.keys(from);
    for (const key of keys) {
      const node = (from as Record<string | number, RawJson>)[key]!;
      let out: unknown = plain(node);
      if (typeof node === "object" && node !== null && !(node instanceof RawNumber)) {
        out = Array.isArray(node) ? [] : {};
        stack.push([node as RawJson[] | Record<string, RawJson>, out as unknown[] | Record<string, unknown>]);
      }
      if (Array.isArray(to)) to.push(out);
      else setOwn(to, text(key as string), out);
    }
  }
  return root;
}

/** JSON text of a {@link RawJson} tree, every number written as its source token; for trees of bounded depth. */
export function writeRaw(value: RawJson): string {
  if (value instanceof RawNumber) return value.raw;
  if (Array.isArray(value)) return `[${value.map(writeRaw).join(",")}]`;
  if (typeof value === "object" && value !== null) return `{${Object.keys(value).map((key) => `${JSON.stringify(key)}:${writeRaw(value[key]!)}`).join(",")}}`;
  return JSON.stringify(value);
}

/** The first number token under `value` that is finite but not the integer it spells (`9007199254740993`), or null. */
export function firstInexact(value: RawJson): RawNumber | null {
  const stack: RawJson[] = [value];
  while (stack.length) {
    const node = stack.pop()!;
    if (node instanceof RawNumber) {
      if (Number.isFinite(node.value) && !node.exact) return node;
    } else if (Array.isArray(node)) stack.push(...node);
    else if (typeof node === "object" && node !== null) for (const key of Object.keys(node)) stack.push(node[key]!);
  }
  return null;
}
