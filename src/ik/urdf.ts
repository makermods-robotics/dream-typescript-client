/**
 * An arm of your own (`dream.arm_file`): dream_ik's model of it from its URDF and the keys of an arm file.
 *
 * The URDF is cut to the chain from `base` to `flange`; each joint on it is fixed, revolute or prismatic with limits,
 * or continuous (taken as revolute within [-pi, pi]); a joint `fixed` names is cut as fixed at its URDF zero. The
 * arm's joints are the chain's other moving joints in order from `base`, radians (metres for a prismatic one), kept 2
 * degrees inside their limits and under their URDF velocity, at most 3 rad/s (a velocity of 0 takes 3).
 *
 * Python reads the arm file and the URDF from disk; here you pass their contents, so it runs in a browser too.
 */

import { DreamConfigError } from "../errors.js";
import { every, isNumbers } from "../finite.js";
import { cross3, dot3, norm3, scale3, sub3, type Vec3 } from "../linalg.js";
import { Embodiment, type ArmData, type JointData } from "./arms.js";

/** The keys of an arm file. */
export interface ArmFile {
  /** In Python, the URDF's path relative to the arm file; ignored here (pass the URDF's text). */
  urdf?: string;
  /** The link the hand's pose is measured from: x forward, y to the arm's left, z up. */
  base: string;
  /** The link the gripper hangs off. */
  flange: string;
  /** `[x, y, z]`, metres in the flange's frame: the point between the closed fingertips. */
  tcp: number[];
  /** `[x, y, z]` in the flange's frame: the way the fingers point. */
  approach: number[];
  /** `[x, y, z]` in the flange's frame: the way the fingers close (only its part across `approach` counts). */
  jaw: number[];
  /** The arm's joints in a working pose, radians: the pose the solve leans toward. */
  rest: number[];
  /** Joints on the chain the robot holds still, cut as fixed joints at their URDF zero. */
  fixed?: string[];
  /** The setup tools' record of the tool point; not checked here. */
  tcp_calibration?: unknown;
}

/** The joint speed cap, rad/s, where the URDF's is higher or unset. */
export const MAX_SPEED = 3.0;
const REST_WEIGHTS = [0.15, 0.3];
const LIMIT_MARGIN = (2 * Math.PI) / 180;
const WEIGHTS = { pos: 50, ori: 10, limit: 1000, last_pose: 1, collision: 1000, collision_margin: 0.02 };

/** One XML element: its tag, attributes and child elements. */
export interface XmlElement {
  tag: string;
  attributes: Record<string, string>;
  children: XmlElement[];
}

/**
 * A strict XML reader for URDF, as strict as Python's (expat): the whole text must be one well-formed `<robot>`
 * element, or it throws. A lenient reader skips what it cannot read, so a typo such as an unquoted `xyz=-0.28` would
 * drop that origin and quietly move the arm. Elements and attributes are kept (entities decoded); text content,
 * comments, processing instructions, CDATA and an external DOCTYPE are checked and dropped.
 *
 * Stricter than XML in two places: a bare `>` in text (legal XML, but in a URDF it is a tag's stray `>`; write
 * `&gt;`), and a DOCTYPE's internal subset, whose entity declarations it does not expand.
 *
 * @throws DreamConfigError The text is not well-formed (with the line and column, and what was found there), or its
 *   root element is not `<robot>`.
 */
export function parseXml(text: string): XmlElement {
  const start = text.charCodeAt(0) === 0xfeff ? 1 : 0;
  let pos = start;
  const where = (at: number) => `line ${text.slice(0, at).split("\n").length}, column ${at - (at ? text.lastIndexOf("\n", at - 1) : -1)}`;
  const found = (at = pos) => (at >= text.length ? "the end of the text" : JSON.stringify(text.slice(at, at + 20)));
  function fail(what: string, at = pos): never {
    throw new DreamConfigError(`the URDF does not parse at ${where(at)}: ${what}`);
  }
  const take = (pattern: RegExp): string | undefined => {
    pattern.lastIndex = pos;
    const match = pattern.exec(text)?.[0];
    if (match !== undefined) pos += match.length;
    return match;
  };
  const space = () => take(SPACE) !== undefined;
  const closeAt = (end: string, what: string) => {
    const at = text.indexOf(end, pos);
    if (at < 0) fail(`${what} is never closed`);
    return at;
  };
  /** `raw`, found at `at`, with its entity references decoded and (in a value) its literal tabs and newlines as spaces. */
  const decode = (raw: string, at: number, value: boolean) => {
    const literal = (s: string) => (value ? s.replace(/\r\n?|[\n\t]/g, " ") : s);
    let out = "";
    let i = 0;
    for (let amp = raw.indexOf("&"); amp >= 0; amp = raw.indexOf("&", i)) {
      out += literal(raw.slice(i, amp));
      REFERENCE.lastIndex = amp;
      const [ref, hex, dec, name] = REFERENCE.exec(raw) ?? fail(`"&" starts no entity reference (&lt; &gt; &amp; &quot; &apos; &#...;), found ${found(at + amp)}`, at + amp);
      const code = hex ? parseInt(hex, 16) : dec ? Number(dec) : undefined;
      if (code === undefined) out += ENTITIES.get(name!) ?? fail(`unknown entity ${ref}`, at + amp);
      else if (xmlChar(code)) out += String.fromCodePoint(code);
      else fail(`${ref} names no XML character`, at + amp);
      i = amp + ref.length;
    }
    return out + literal(raw.slice(i));
  };

  const bad = text.search(NOT_XML_CHAR);
  if (bad >= 0) fail(`character U+${text.codePointAt(bad)!.toString(16).toUpperCase().padStart(4, "0")} is not allowed in XML`, bad);
  const stack: XmlElement[] = [];
  const opened: number[] = [];
  let root: XmlElement | undefined;
  let prolog = true;
  while (pos < text.length) {
    const from = pos;
    if (text.startsWith("<!--", pos)) {
      pos += 4;
      const end = closeAt("--", "a comment");
      if (text[end + 2] !== ">") fail(`"--" inside a comment`, end);
      pos = end + 3;
    } else if (text.startsWith("<?", pos)) {
      pos += 2;
      const target = take(NAME) ?? fail(`expected a processing instruction's name after "<?", found ${found()}`);
      if (/^xml$/i.test(target) && from !== start) fail("the XML declaration is not at the start of the text", from);
      if (!space() && !text.startsWith("?>", pos)) fail(`expected whitespace or "?>" after <?${target}, found ${found()}`);
      pos = closeAt("?>", `<?${target}`) + 2;
    } else if (text.startsWith("<![CDATA[", pos)) {
      if (!stack.length) fail("CDATA outside the root element");
      pos += 9;
      pos = closeAt("]]>", "a CDATA section") + 3;
    } else if (text.startsWith("<!DOCTYPE", pos)) {
      if (!prolog) fail("a DOCTYPE after the root element or another DOCTYPE");
      pos += 9;
      if (!space() || !take(NAME)) fail(`expected whitespace and the root's name after <!DOCTYPE, found ${found()}`);
      for (;;) {
        space();
        if (text[pos] === ">") break;
        if (text[pos] === "[") fail("a DOCTYPE with an internal subset (entity declarations), which this reader does not expand");
        if (text[pos] === '"' || text[pos] === "'") pos = closeAt(text[pos++]!, "a quoted DOCTYPE literal") + 1;
        else if (!take(/(?:SYSTEM|PUBLIC)(?=[\s"'])/y)) fail(`expected SYSTEM, PUBLIC, a quoted literal or ">" in <!DOCTYPE, found ${found()}`);
      }
      pos++;
      prolog = false;
    } else if (text.startsWith("</", pos)) {
      pos += 2;
      const name = take(NAME) ?? fail(`expected a tag name after "</", found ${found()}`);
      space();
      if (text[pos++] !== ">") fail(`expected ">" to end </${name}>, found ${found(pos - 1)}`, pos - 1);
      const open = stack.pop() ?? fail(`</${name}> closes nothing`, from);
      const openedAt = opened.pop()!;
      if (open.tag !== name) fail(`</${name}> closes <${open.tag}> (opened at ${where(openedAt)})`, from);
    } else if (text[pos] === "<") {
      pos++;
      const tag = take(NAME) ?? fail(`expected a tag name after "<", found ${found()}`);
      const attributes = new Map<string, string>();
      let selfClosing = false;
      for (;;) {
        const spaced = space();
        if (text.startsWith("/>", pos) || text[pos] === ">") {
          selfClosing = text[pos] === "/";
          pos += selfClosing ? 2 : 1;
          break;
        }
        if (!spaced) fail(`expected whitespace, ">" or "/>" in <${tag}>, found ${found()}`);
        const keyAt = pos;
        const key = take(NAME) ?? fail(`expected an attribute name, ">" or "/>" in <${tag}>, found ${found()}`);
        if (attributes.has(key)) fail(`<${tag}> has attribute ${key} twice`, keyAt);
        space();
        if (text[pos] !== "=") fail(`expected "=" after attribute ${key} of <${tag}>, found ${found()}`);
        pos++;
        space();
        const quote = text[pos];
        if (quote !== '"' && quote !== "'") fail(`expected a quoted value for attribute ${key} of <${tag}>, found ${found()}`);
        pos++;
        const end = closeAt(quote, `the value of attribute ${key} of <${tag}>`);
        const raw = text.slice(pos, end);
        if (raw.includes("<")) fail(`"<" in the value of attribute ${key} of <${tag}>`, pos + raw.indexOf("<"));
        attributes.set(key, decode(raw, pos, true));
        pos = end + 1;
      }
      // fromEntries defines each key as an own property, even one named __proto__.
      const element: XmlElement = { tag, attributes: Object.fromEntries(attributes), children: [] };
      if (stack.length) stack.at(-1)!.children.push(element);
      else if (root) fail(`a second root element <${tag}>; the URDF must be one <robot> element`, from);
      else root = element;
      prolog = false;
      if (!selfClosing) {
        stack.push(element);
        opened.push(from);
      }
    } else {
      const next = text.indexOf("<", pos);
      const end = next < 0 ? text.length : next;
      const run = text.slice(pos, end);
      if (!stack.length) {
        const junk = run.search(/[^ \t\r\n]/);
        if (junk >= 0) fail(`${root ? "text after" : "text before"} the root element, found ${found(pos + junk)}`, pos + junk);
      } else {
        // Legal XML (bar "]]>"), but in a URDF a ">" in text is a tag's stray ">".
        if (run.includes(">")) fail(`a bare ">" in the text of <${stack.at(-1)!.tag}> (a stray ">" in a tag? write &gt;)`, pos + run.indexOf(">"));
        decode(run, pos, false);
      }
      pos = end;
    }
  }
  if (stack.length) fail(`<${stack.at(-1)!.tag}> is never closed`, opened.at(-1));
  if (!root) throw new DreamConfigError("the URDF does not parse: it has no root element (no <robot>)");
  if (root.tag !== "robot") throw new DreamConfigError(`the URDF does not parse: its root element is <${root.tag}>, not <robot>`);
  return root;
}

const NAME = /[A-Za-z_:\u00c0-\uffff][\w:.\-\u00b7\u00c0-\uffff]*/y;
const SPACE = /[ \t\r\n]+/y;
const REFERENCE = /&(?:#x([0-9A-Fa-f]+)|#([0-9]+)|([A-Za-z]+));/y;
const ENTITIES = new Map([["lt", "<"], ["gt", ">"], ["amp", "&"], ["quot", '"'], ["apos", "'"]]);
/** A control character XML forbids, U+FFFE or U+FFFF, or half a surrogate pair. */
const NOT_XML_CHAR = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\ufffe\uffff]|[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/;
const xmlChar = (c: number) => c === 0x9 || c === 0xa || c === 0xd || (c >= 0x20 && c <= 0xd7ff) || (c >= 0xe000 && c <= 0xfffd) || (c >= 0x10000 && c <= 0x10ffff);
const child = (element: XmlElement, tag: string) => element.children.find((c) => c.tag === tag);
/** A URDF number: NaN for an empty attribute, which Number would read as 0. */
const number = (text: string) => (text.trim() ? Number(text) : NaN);

/**
 * dream_ik's spec of an arm from its URDF's text and an arm file's keys (`arm_file.register`): one arm, run on both
 * sides, each from its own base.
 *
 * @throws DreamConfigError The arm file's keys or the URDF are malformed, or the chain cannot be solved for.
 */
export function defineArm(file: ArmFile, urdfText: string, name = "file"): Embodiment {
  for (const key of ["base", "flange"] as const) {
    if (typeof file[key] !== "string" || !file[key]) throw new DreamConfigError(`arm file: ${key} must be a name, got ${JSON.stringify(file[key])}`);
  }
  for (const key of ["tcp", "approach", "jaw"] as const) {
    if (!finiteList(file[key]) || file[key].length !== 3) throw new DreamConfigError(`arm file: ${key} must be 3 numbers [x, y, z], got ${JSON.stringify(file[key])}`);
  }
  if (!finiteList(file.rest) || !file.rest.length) throw new DreamConfigError(`arm file: rest must be the arm's joints in radians, got ${JSON.stringify(file.rest)}`);
  const fixed = file.fixed ?? [];
  if (!Array.isArray(fixed) || !every(fixed, (n) => typeof n === "string")) throw new DreamConfigError(`arm file: fixed must be a list of joint names, got ${JSON.stringify(fixed)}`);
  const robot = parseXml(urdfText);
  const links = new Set(robot.children.filter((c) => c.tag === "link").map((c) => attribute(c, "name")));
  for (const [key, link] of [["base", file.base], ["flange", file.flange]] as const) {
    if (!links.has(link)) throw new DreamConfigError(`arm file: ${key} '${link}' is not a link of the URDF: ${JSON.stringify([...links].sort())}`);
  }
  const parentJoint = new Map<string, XmlElement>();
  for (const joint of robot.children.filter((c) => c.tag === "joint")) {
    const link = linkOf(joint, "child");
    const other = parentJoint.get(link);
    if (other) throw new DreamConfigError(`arm file: link '${link}' is the child of joints '${other.attributes.name}' and '${joint.attributes.name}' in the URDF; a link hangs from one joint`);
    parentJoint.set(link, joint);
  }
  const path: XmlElement[] = [];
  const seen = new Set<string>();
  for (let link = file.flange; link !== file.base; ) {
    // Python walks a loop of joints forever; here it is refused.
    if (seen.has(link)) throw new DreamConfigError(`arm file: the joints above flange '${file.flange}' loop back to link '${link}' in the URDF`);
    seen.add(link);
    const joint = parentJoint.get(link);
    if (!joint) throw new DreamConfigError(`arm file: flange '${file.flange}' does not hang below base '${file.base}' in the URDF`);
    path.unshift(joint);
    link = linkOf(joint, "parent");
  }
  const chain: JointData[] = [];
  const joints: string[] = [];
  const limits: number[][] = [];
  const speeds: number[] = [];
  const held: string[] = [];
  for (const joint of path) {
    const jointName = attribute(joint, "name");
    let kind = joint.attributes.type;
    const entry: JointData = { name: jointName, type: "fixed", xyz: vector(joint, "origin", "xyz", [0, 0, 0]), rpy: vector(joint, "origin", "rpy", [0, 0, 0]) };
    chain.push(entry);
    if (kind === "fixed") continue;
    if (fixed.includes(jointName)) {
      held.push(jointName);
      continue;
    }
    const mimic = child(joint, "mimic") !== undefined;
    let limit = child(joint, "limit")?.attributes ?? null;
    if (kind === "continuous" && !mimic) {
      limit = { ...(limit ?? {}), lower: String(-Math.PI), upper: String(Math.PI) };
      kind = "revolute";
    }
    if ((kind !== "revolute" && kind !== "prismatic") || mimic) {
      throw new DreamConfigError(
        `arm file: joint '${jointName}' between '${file.base}' and '${file.flange}' is ${kind}${mimic ? " with a mimic" : ""}; the chain takes ` +
          "fixed joints, revolute and prismatic joints with limits, and continuous joints, none with a mimic",
      );
    }
    if (!limit || limit.lower === undefined || limit.upper === undefined) throw new DreamConfigError(`arm file: joint '${jointName}' has no lower and upper limit in the URDF`);
    const lo = number(limit.lower);
    const hi = number(limit.upper);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) throw new DreamConfigError(`arm file: joint '${jointName}' has limits ${JSON.stringify([limit.lower, limit.upper])}, not finite numbers`);
    if (!(lo < hi)) throw new DreamConfigError(`arm file: joint '${jointName}' has limits [${lo}, ${hi}], an empty range`);
    // The solve keeps inside the margined limits (Arm.lo/hi); Python lets them cross and the solve then clips to garbage.
    if (!(lo + LIMIT_MARGIN < hi - LIMIT_MARGIN)) throw new DreamConfigError(`arm file: joint '${jointName}' has limits [${lo}, ${hi}], no range left 2 degrees inside them`);
    // No velocity, or an empty one, takes MAX_SPEED as in Python, as does 0 below.
    const velocity = limit.velocity === undefined || !limit.velocity.trim() ? 0 : Number(limit.velocity);
    if (!(Number.isFinite(velocity) && velocity >= 0)) throw new DreamConfigError(`arm file: joint '${jointName}' has velocity ${JSON.stringify(limit.velocity)}, not a finite number of at least 0`);
    const axis = vector(joint, "axis", "xyz", [1, 0, 0]);
    const length = Math.hypot(...axis);
    if (!(length > 0)) throw new DreamConfigError(`arm file: joint '${jointName}' has a zero axis`);
    entry.type = kind;
    entry.axis = axis.map((v) => v / length);
    joints.push(jointName);
    limits.push([lo, hi]);
    speeds.push(velocity <= 0 ? MAX_SPEED : Math.min(velocity, MAX_SPEED));
  }
  const unknown = fixed.filter((n) => !held.includes(n));
  if (unknown.length) throw new DreamConfigError(`arm file: fixed names '${unknown[0]}', which is not a moving joint on the chain from '${file.base}' to '${file.flange}': ${JSON.stringify([...joints, ...held])}`);
  if (!joints.length) throw new DreamConfigError(`arm file: the chain from '${file.base}' to '${file.flange}' has no moving joint`);
  if (file.rest.length !== joints.length) {
    throw new DreamConfigError(`arm file: rest has ${file.rest.length} joints, the chain from '${file.base}' to '${file.flange}' has ${joints.length}: ${JSON.stringify(joints)}`);
  }
  file.rest.forEach((value, i) => {
    const [lo, hi] = limits[i]!;
    if (!(lo! <= value && value <= hi!)) throw new DreamConfigError(`arm file: rest puts ${joints[i]} at ${value}, outside [${lo}, ${hi}]`);
  });
  if (!(norm3(cross3(file.approach as Vec3, file.jaw as Vec3)) > 1e-6)) {
    throw new DreamConfigError(`arm file: approach ${JSON.stringify(file.approach)} and jaw ${JSON.stringify(file.jaw)} must be nonzero and not parallel`);
  }
  const data: ArmData = {
    urdf: file.urdf ?? "",
    joints,
    heldJoints: [],
    flange: file.flange,
    root: file.base,
    chain,
    limits,
    tool: { offset: [...file.tcp], rotation: toolAxes(file.approach as Vec3, file.jaw as Vec3) },
    restPose: [...file.rest],
    restWeight: joints.map((_, i) => (i === 0 ? REST_WEIGHTS[0]! : REST_WEIGHTS[1]!)),
    maxSpeed: speeds,
    limitMargin: LIMIT_MARGIN,
    jaw: null,
    jointUnits: null,
    weights: { ...WEIGHTS },
  };
  return new Embodiment(name, { sides: ["right", "left"], jawInArm: false, arm: data });
}

/**
 * A tool's axes in the flange frame from the way the fingers point and close (`specs._tool_axes`): z = approach,
 * x = jaw made orthogonal to it, y = z cross x; the rows of the matrix whose columns are those axes.
 */
export function toolAxes(approach: Vec3, jaw: Vec3): number[][] {
  const z = scale3(approach, 1 / norm3(approach));
  const xRaw = sub3(jaw, scale3(z, dot3(jaw, z)));
  const x = scale3(xRaw, 1 / norm3(xRaw));
  const y = cross3(z, x);
  return [0, 1, 2].map((i) => [x[i]!, y[i]!, z[i]!]);
}

function attribute(element: XmlElement, key: string): string {
  const value = element.attributes[key];
  if (value === undefined) throw new DreamConfigError(`a <${element.tag}> of the URDF has no ${key}`);
  return value;
}

function linkOf(joint: XmlElement, end: "parent" | "child"): string {
  const tag = child(joint, end);
  if (!tag) throw new DreamConfigError(`joint '${joint.attributes.name}' of the URDF has no <${end}> link`);
  return attribute(tag, "link");
}

/** A joint's `<origin>` xyz or rpy, or its `<axis>` xyz: 3 finite numbers, `fallback` when the element or attribute is absent. */
function vector(joint: XmlElement, tag: "origin" | "axis", key: "xyz" | "rpy", fallback: number[]): number[] {
  const text = child(joint, tag)?.attributes[key];
  if (text === undefined) return fallback;
  const values = text.trim().split(/\s+/).map(number);
  if (values.length !== 3 || !values.every(Number.isFinite)) throw new DreamConfigError(`arm file: joint '${joint.attributes.name}' has ${tag} ${key} ${JSON.stringify(text)}, not 3 finite numbers`);
  return values;
}

/** A plain array of finite numbers with no hole (every() would skip one). */
function finiteList(value: unknown): value is number[] {
  return Array.isArray(value) && isNumbers(value, true);
}
