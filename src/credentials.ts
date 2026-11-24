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

/**
 * Checks a service URL. Credentials name `http(s)://<host>[:<port>]` with no path (`credentials._check_url`); a
 * client's `baseUrl` may also carry a path (a proxy mount such as `/api/dream`) or be relative to the page
 * (`allowPath`), and the requests go to `<baseUrl>/v1/...`. A query or fragment marker (even an empty `?` or `#`),
 * a user name or password in the URL, or any scheme but http(s) is refused: appending `/v1/...` must stay in the path.
 *
 * @returns The URL as parsed: its origin (and, with `allowPath`, its path), without a trailing slash; a relative
 *   path as given, without a trailing slash.
 */
export function checkServiceUrl(url: string, what: string, allowPath: boolean): string {
  if (typeof url !== "string" || !url) throw new DreamConfigError(`${what} must be a URL, got ${JSON.stringify(url)}`);
  const shape = `${what} must be http(s)://<host>[:<port>]${allowPath ? "[/<path>]" : " with no path"}, with no query, fragment or user name, e.g. https://api.example.com, got ${JSON.stringify(url)}`;
  if (/[?#\s\\]/.test(url)) throw new DreamConfigError(shape);
  const trimmed = url.replace(/\/+$/, "");
  if (allowPath && (trimmed === "" || (trimmed.startsWith("/") && !trimmed.startsWith("//")))) return trimmed;
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new DreamConfigError(`${what} is not a URL: ${JSON.stringify(url)}`);
  }
  const pathless = parsed.pathname === "" || parsed.pathname === "/";
  if (!["http:", "https:"].includes(parsed.protocol) || !parsed.hostname || parsed.username || parsed.password || (!allowPath && !pathless)) {
    throw new DreamConfigError(shape);
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
