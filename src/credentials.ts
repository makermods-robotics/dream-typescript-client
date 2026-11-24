/**
 * Credentials (`dream.credentials`), the part that runs anywhere: the production URLs, the names, the file format and
 * the header the key travels in. Reading the file and the environment is Node's job: `@makermods/dreamlabs/node`.
 *
 * In a browser, never ship an API key: point `baseUrl` (and `platformUrl`) at your own server, which adds the
 * `Authorization` header, as the MakerMods console's proxy does.
 */

import { DreamConfigError } from "./errors.js";

/** The model's URL when nothing names one: production's. */
export const DEFAULT_GATEWAY_URL = "https://api.dreamlabs.ai";
/** The platform (models list, docs) when nothing names one: production's. */
export const DEFAULT_PLATFORM_URL = "https://platform.dreamlabs.ai";
/** The data service when nothing names one: production's. */
export const DEFAULT_DATA_URL = "https://data.dreamlabs.ai";
/** The credentials file unless {@link FILE_ENV} names another. */
export const DEFAULT_FILE = "~/.dreamlabs/credentials";
/** Environment variable naming another credentials file. */
export const FILE_ENV = "DREAM_CREDENTIALS";
export const GATEWAY_URL_ENV = "DREAM_GATEWAY_URL";
export const PLATFORM_URL_ENV = "DREAM_API_URL";
export const DATA_URL_ENV = "DREAM_DATA_URL";
export const API_KEY_ENV = "DREAM_API_KEY";
/** The four names the file and the environment are read for, in the file's order. */
export const NAMES = [GATEWAY_URL_ENV, PLATFORM_URL_ENV, DATA_URL_ENV, API_KEY_ENV] as const;

/** How every refused key's message ends (`credentials.KEY_PER_SITE`). */
export const KEY_PER_SITE =
  "keys work only on the site that made them (dev, staging and production are separate), so this key may be from " +
  "another site, mistyped or revoked";

/** What the credentials resolve to: the three URLs, always set, and the key, when anything set it. */
export interface Credentials {
  gatewayUrl: string;
  platformUrl: string;
  dataUrl: string;
  apiKey: string | null;
  /** The file read, or looked for and not found; null when none was read (the browser). */
  path: string | null;
}

const ASSIGNMENT = /^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/;

/**
 * The assignments of a credentials file (`credentials._read_file`): blank lines, `#` comments, `NAME=value` and
 * `export NAME=value`, the value optionally in matching single or double quotes; a later line for a name wins.
 */
export function parseCredentialsFile(text: string): Record<string, string> {
  const found: Record<string, string> = {};
  for (const line of text.split(/\r\n|\r|\n/)) {
    const match = ASSIGNMENT.exec(line);
    if (!match) continue;
    const name = match[1]!;
    let value = match[2]!;
    if (value.length >= 2 && value[0] === value.at(-1) && (value[0] === "'" || value[0] === '"')) value = value.slice(1, -1);
    found[name] = value;
  }
  return found;
}

/**
 * Resolves the credentials name by name: the environment over the file over the production defaults
 * (`credentials.load`). Throws {@link DreamConfigError} for a URL that is set but is not `http(s)://<host>[:<port>]`.
 *
 * @param env The environment (`process.env`, `Deno.env.toObject()`, ...).
 * @param file The file's text and path, when one was read.
 */
export function resolveCredentials(
  env: Readonly<Record<string, string | undefined>>,
  file?: { text: string; path: string } | null,
): Credentials {
  const fromFile = file ? parseCredentialsFile(file.text) : {};
  const values: Record<string, [string, string]> = {};
  for (const name of NAMES) {
    const own = env[name]?.trim();
    const filed = fromFile[name]?.trim();
    if (own) values[name] = [own, "the environment"];
    else if (filed) values[name] = [filed, file!.path];
  }
  for (const name of [GATEWAY_URL_ENV, PLATFORM_URL_ENV, DATA_URL_ENV]) {
    const value = values[name];
    if (value) value[0] = checkServiceUrl(value[0], `${name} (from ${value[1]})`, false);
  }
  return {
    gatewayUrl: values[GATEWAY_URL_ENV]?.[0] ?? DEFAULT_GATEWAY_URL,
    platformUrl: values[PLATFORM_URL_ENV]?.[0] ?? DEFAULT_PLATFORM_URL,
    dataUrl: values[DATA_URL_ENV]?.[0] ?? DEFAULT_DATA_URL,
    apiKey: values[API_KEY_ENV]?.[0] ?? null,
    path: file?.path ?? null,
  };
}

/** A path segment as written: unreserved, sub-delims, `:`, `@` and percent-escapes. */
const SEGMENT = /^(?:[A-Za-z0-9._~!$&'()*+,;=:@-]|%[0-9A-Fa-f]{2})*$/;

/** A URL for a message: anything before an `@` in its authority (a user name, a password) left out. */
export function redactUrl(url: string): string {
  const text = String(url);
  const scheme = /^([A-Za-z][A-Za-z0-9+.-]*:)?\/\//.exec(text);
  if (!scheme) return text;
  const rest = text.slice(scheme[0].length);
  const end = rest.search(/[/?#]/);
  const authority = end < 0 ? rest : rest.slice(0, end);
  const at = authority.lastIndexOf("@");
  return at < 0 ? text : `${scheme[0]}<userinfo>@${authority.slice(at + 1)}${end < 0 ? "" : rest.slice(end)}`;
}

/** Whether a path, as written, has only valid segments and no dot segment in any spelling (`.`, `%2e`, `.%2E`, ...). */
function pathOk(path: string): boolean {
  return path
    .split("/")
    .slice(1)
    .every((segment, i, all) => {
      const dots = segment.replace(/%2e/gi, ".");
      return SEGMENT.test(segment) && dots !== "." && dots !== ".." && (segment !== "" || i === all.length - 1);
    });
}

/**
 * Checks a service URL. Credentials name `http(s)://<host>[:<port>]` with no path (`credentials._check_url`); a
 * client's `baseUrl` may also carry a path (a proxy mount such as `/api/dream`) or be relative to the page
 * (`allowPath`; `/` and `""` are the page's root), and the requests go to `<baseUrl>/v1/...`. The URL is checked as
 * written, before any normalisation: it must start `http://` or `https://` and a host; a user name or password, a
 * query or fragment marker (even an empty `?` or `#`), a dot segment in any spelling (`.`, `..`, `%2e`, `.%2E`, ...),
 * or a port outside 1..65535 is refused. The host is read by the URL standard's host parser, so a DNS name with a
 * trailing dot and an internationalised name (as punycode) are taken. A message names the rule broken, never the URL's
 * authority.
 *
 * @returns The URL as parsed: its origin (and, with `allowPath`, its path), without a trailing slash; a relative path
 *   as given, without a trailing slash (`""` for the root).
 */
export function checkServiceUrl(url: string, what: string, allowPath: boolean): string {
  const fail = (why: string) =>
    new DreamConfigError(`${what} must be http(s)://<host>[:<port>]${allowPath ? "[/<path>], or a path such as /api/dream" : " with no path"}: ${why}`);
  if (typeof url !== "string") throw fail("it is not a string");
  if (!url && !allowPath) throw fail("it is empty");
  if (/[?#\s\\]/.test(url)) throw fail("it holds a query or fragment marker, a space or a backslash");
  if (allowPath && (url === "" || (url.startsWith("/") && !url.startsWith("//")))) {
    if (url !== "" && !pathOk(url)) throw fail("its path has a dot segment or a character a path cannot hold");
    return url.replace(/\/+$/, "");
  }
  const scheme = /^(https?):\/\//i.exec(url);
  if (!scheme) throw fail("it does not start with http:// or https:// and a host");
  const rest = url.slice(scheme[0].length);
  const slash = rest.indexOf("/");
  const authority = slash < 0 ? rest : rest.slice(0, slash);
  const path = slash < 0 ? "" : rest.slice(slash);
  if (!authority) throw fail("it has no host");
  if (authority.includes("@")) throw fail("it holds a user name or password, which must never be in a service URL");
  const port = /:(\d*)$/.exec(authority.startsWith("[") ? authority.slice(authority.indexOf("]") + 1) : authority);
  if (port && !(/^\d{1,5}$/.test(port[1]!) && Number(port[1]) >= 1 && Number(port[1]) <= 65535)) throw fail("its port is not 1 to 65535");
  if (allowPath ? path !== "" && !pathOk(path) : path !== "" && path !== "/") {
    throw fail(allowPath ? "its path has a dot segment or a character a path cannot hold" : "it has a path");
  }
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw fail("its host is not a valid host name or address");
  }
  if (!parsed.hostname) throw fail("it has no host");
  return allowPath ? `${parsed.origin}${parsed.pathname.replace(/\/+$/, "")}` : parsed.origin;
}

/**
 * The `Authorization` header that carries the key (`credentials._authorization`). Throws {@link DreamConfigError},
 * never showing the key, when it holds a line break or a character outside Latin-1 (a typographic quote).
 */
export function authorization(apiKey: string): Record<string, string> {
  if (typeof apiKey !== "string" || !apiKey) throw new DreamConfigError(`the api key must be a non-empty string; set ${API_KEY_ENV} or the apiKey option`);
  // fetch strips a header value's surrounding spaces and tabs: the key sent would differ from the one held, and from the
  // one every redaction looks for.
  if (/^[ \t]|[ \t]$/.test(apiKey)) {
    throw new DreamConfigError(`the api key starts or ends with a space or tab; copy the key again, whole, into ${API_KEY_ENV} or the apiKey option`);
  }
  if (/[\x00-\x1f\x7f]/.test(apiKey) || /[^\x00-\xff]/.test(apiKey)) {
    throw new DreamConfigError(
      "the api key holds a character an HTTP header cannot carry (a line break or another control character, or a character such as a " +
        `typographic quote); copy the key again, whole, into ${API_KEY_ENV} or the apiKey option`,
    );
  }
  return { Authorization: `Bearer ${apiKey}` };
}

/** `text` with the key cut out: a service that echoes the request must not put the key in a message or a log. */
export function scrub(text: string, secret: string | null | undefined): string {
  return secret ? text.split(secret).join("<api key>") : text;
}

/** A decoded JSON value with the key cut out of every object key and string ({@link scrub}). */
export function scrubJson<T>(value: T, secret: string | null | undefined): T {
  if (!secret) return value;
  const walk = (node: unknown): unknown => {
    if (typeof node === "string") return scrub(node, secret);
    if (Array.isArray(node)) return node.map(walk);
    if (typeof node === "object" && node !== null) return Object.fromEntries(Object.entries(node).map(([key, item]) => [scrub(key, secret), walk(item)]));
    return node;
  };
  return walk(value) as T;
}

/**
 * The keys a proxy takes (createDreamProxy's `apiKey`): at least {@link PROXY_KEY_MIN} characters of letters, digits and
 * `._~+/=-`, as Dream API keys are. Its answers are withheld when they carry this exact key, so a short one (such as a
 * field name) would withhold everything, and one with surrounding spaces would not be the key sent.
 */
export const PROXY_KEY = /^[A-Za-z0-9._~+/=-]+$/;
/** The shortest key a proxy takes. */
export const PROXY_KEY_MIN = 16;

/**
 * Whether `hostname` (a URL's, as the URL standard writes it) is this computer: `localhost` and any name under
 * `.localhost` (trusted by spelling, as browsers treat it), 127.0.0.0/8, `[::1]`, and the IPv4-mapped 127.0.0.0/8
 * (`[::ffff:127.0.0.1]`).
 */
export function isLoopback(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/\.$/, "");
  return (
    host === "localhost" ||
    host.endsWith(".localhost") ||
    /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(host) ||
    host === "[::1]" ||
    /^\[::ffff:7f[0-9a-f]{2}:[0-9a-f]{1,4}\]$/.test(host)
  );
}

/**
 * @throws DreamConfigError `url`, resolved where the request will go (a relative one against the page, when there is
 *   one), is plain `http:` to another computer, which would carry the key in the clear, and `allowInsecureHttp` is not
 *   literally `true`.
 */
export function requireHttps(url: string, what: string, allowInsecureHttp: unknown): void {
  if (allowInsecureHttp === true) return;
  let resolved: URL;
  try {
    const page = (globalThis as { location?: { href?: unknown } }).location?.href;
    resolved = typeof page === "string" ? new URL(url, page) : new URL(url);
  } catch {
    // A relative URL with no page to resolve it against: no request can go anywhere with it.
    return;
  }
  if (resolved.protocol !== "http:" || isLoopback(resolved.hostname)) return;
  throw new DreamConfigError(
    `${what} is plain http:// to another computer, so the API key would cross the network in the clear: use https://, or set allowInsecureHttp: true on a network you trust`,
    { kind: "unsupported", key: what, row: null },
  );
}

/** A message for any thrown value that never throws itself: an object with no prototype, a throwing getter. */
export function safeMessage(error: unknown): string {
  try {
    if (error instanceof Error) return String(error.message);
    if (typeof error === "string") return error;
    return String(error);
  } catch {
    return "(a value that cannot be shown)";
  }
}

/**
 * A new error standing for `error`, built from allowlisted fields only: its name, its message and its `code`, `errno`
 * and `syscall` when they are strings or numbers, `secret` cut from each, and its cause the same way, `depth` deep.
 * Never the original object, which may carry the request that failed (its headers, the key among them).
 */
export function safeCause(error: unknown, secret: string | null | undefined, depth = 3): Error {
  const read = (key: string): unknown => {
    try {
      return typeof error === "object" && error !== null ? (error as Record<string, unknown>)[key] : undefined;
    } catch {
      return undefined;
    }
  };
  let cause: unknown;
  try {
    cause = depth > 1 && error instanceof Error ? error.cause : undefined;
  } catch {
    cause = undefined;
  }
  const out = new Error(scrub(safeMessage(error), secret), cause === undefined ? undefined : { cause: safeCause(cause, secret, depth - 1) });
  const name = read("name");
  if (typeof name === "string") out.name = scrub(name, secret);
  for (const key of ["code", "errno", "syscall"]) {
    const value = read(key);
    if (typeof value === "string" || typeof value === "number") (out as unknown as Record<string, unknown>)[key] = typeof value === "string" ? scrub(value, secret) : value;
  }
  return out;
}
