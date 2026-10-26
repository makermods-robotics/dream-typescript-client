/**
 * `@makermods/dreamlabs/node`: the credentials as the Python client reads them (`dream.credentials.load`), for Node,
 * Bun and Deno's Node compatibility. Kept out of the main entry so browser bundles never see `node:` modules.
 *
 * Each of `DREAM_GATEWAY_URL`, `DREAM_API_URL`, `DREAM_DATA_URL` and `DREAM_API_KEY` comes from the environment first,
 * then from `~/.dreamlabs/credentials` (or the file `DREAM_CREDENTIALS` names), then from the production default.
 */

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { DreamClient, type DreamClientOptions } from "../facade.js";
import { DEFAULT_FILE, FILE_ENV, resolveCredentials, type Credentials } from "../credentials.js";
import { DreamConfigError } from "../errors.js";

type Env = Readonly<Record<string, string | undefined>>;

/** The credentials file: the one `DREAM_CREDENTIALS` names, else `~/.dreamlabs/credentials`, `~` expanded. */
function credentialsPath(env: Env = process.env): string {
  const named = env[FILE_ENV]?.trim() || DEFAULT_FILE;
  return named === "~" ? homedir() : named.startsWith("~/") ? join(homedir(), named.slice(2)) : named;
}

/**
 * Resolves the credentials: the environment over the file over the defaults, name by name. A missing file is not an
 * error.
 *
 * @throws DreamConfigError The file exists but cannot be read, or a URL is set but is not `http(s)://<host>[:<port>]`.
 */
export function loadCredentials(env: Env = process.env): Credentials {
  const path = credentialsPath(env);
  let text: string | null = null;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code !== "ENOENT" && code !== "ENOTDIR") {
      throw new DreamConfigError(`cannot read the credentials file ${path}: ${(error as Error).message}`);
    }
  }
  const credentials = resolveCredentials(env, text === null ? null : { text, path });
  return { ...credentials, path };
}

/**
 * A {@link DreamClient} with the key and the URLs the credentials name, as Python's `dream.DreamClient()` builds
 * itself; `options` override them.
 */
export function createClient(options: DreamClientOptions = {}, env: Env = process.env): DreamClient {
  const credentials = loadCredentials(env);
  return new DreamClient({
    apiKey: credentials.apiKey,
    baseUrl: credentials.gatewayUrl,
    platformUrl: credentials.platformUrl,
    ...options,
  });
}

export type { Credentials };
