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

/** An absolute service URL as written: scheme, host (a DNS name, IPv4, or bracketed IPv6), optional port, path. */
const SERVICE_URL = /^(https?):\/\/([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*|\[[0-9A-Fa-f:.]+\])(?::(\d{1,5}))?(\/.*)?$/i;
/** A path segment: unreserved, sub-delims, `:`, `@` and percent-escapes; never `.` or `..`. */
const SEGMENT = /^(?:[A-Za-z0-9._~!$&'()*+,;=:@-]|%[0-9A-Fa-f]{2})*$/;

/** A URL for a message: anything before an `@` in its authority (a user name, a password) left out. */
export function redactUrl(url: string): string {
  return String(url).replace(/^([A-Za-z][A-Za-z0-9+.-]*:\/*)[^/?#@]*@/, "$1<userinfo>@");
}

/**
 * Checks a service URL. Credentials name `http(s)://<host>[:<port>]` with no path (`credentials._check_url`); a
 * client's `baseUrl` may also carry a path (a proxy mount such as `/api/dream`) or be relative to the page
 * (`allowPath`; `/` and `""` are the page's root), and the requests go to `<baseUrl>/v1/...`. The URL is checked as
 * written, before any normalisation: a query or fragment marker (even an empty `?` or `#`), a user name or password,
 * a missing `//` or host (`https:host`, `https:///host`), a `.` or `..` segment, a port outside 1..65535, or any scheme
 * but http(s) is refused. Messages never show a user name or password.
 *
 * @returns The URL as parsed: its origin (and, with `allowPath`, its path), without a trailing slash; a relative path
 *   as given, without a trailing slash (`""` for the root).
 */
export function checkServiceUrl(url: string, what: string, allowPath: boolean): string {
  if (typeof url !== "string" || (!url && !allowPath)) throw new DreamConfigError(`${what} must be a URL, got ${JSON.stringify(typeof url === "string" ? url : String(url))}`);
  const shape = () =>
    new DreamConfigError(
      `${what} must be http(s)://<host>[:<port>]${allowPath ? "[/<path>] or a path such as /api/dream" : " with no path"}, with no query, fragment, ` +
        `user name or dot segments, e.g. https://api.example.com, got ${JSON.stringify(redactUrl(url))}`,
    );
  const pathOk = (path: string) => path.split("/").slice(1).every((segment, i, all) => SEGMENT.test(segment) && segment !== "." && segment !== ".." && (segment !== "" || i === all.length - 1));
  if (allowPath && (url === "" || url.startsWith("/"))) {
    if (url !== "" && (url.startsWith("//") || !pathOk(url))) throw shape();
    return url.replace(/\/+$/, "");
  }
  const match = SERVICE_URL.exec(url);
  if (!match) throw shape();
  const [, , , port, path = ""] = match;
  if (port !== undefined && !(Number(port) >= 1 && Number(port) <= 65535)) throw shape();
  if (allowPath ? path !== "" && !pathOk(path) : path !== "" && path !== "/") throw shape();
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw shape();
  }
  return allowPath ? `${parsed.origin}${parsed.pathname.replace(/\/+$/, "")}` : parsed.origin;
}

/**
 * The `Authorization` header that carries the key (`credentials._authorization`). Throws {@link DreamConfigError},
 * never showing the key, when it holds a line break or a character outside Latin-1 (a typographic quote).
 */
export function authorization(apiKey: string): Record<string, string> {
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
