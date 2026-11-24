/**
 * `@makermods/dreamlabs/node`: the credentials as the Python client reads them (`dream.credentials.load`), for Node,
 * Bun and Deno's Node compatibility, and {@link nodeHandler}, which serves a fetch handler (the proxy's) from
 * `node:http`. Kept out of the main entry so browser bundles never see `node:` modules.
 *
 * Each of `DREAM_GATEWAY_URL`, `DREAM_API_URL`, `DREAM_DATA_URL` and `DREAM_API_KEY` comes from the environment first,
 * then from `~/.dreamlabs/credentials` (or the file `DREAM_CREDENTIALS` names), then from the production default.
 */

import { readFileSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import { homedir } from "node:os";
import { join } from "node:path";
import type { TLSSocket } from "node:tls";
import { DreamClient, type DreamClientOptions } from "../facade.js";
import { DEFAULT_FILE, FILE_ENV, resolveCredentials, type Credentials } from "../credentials.js";
import { DreamConfigError } from "../errors.js";
import { PROXY_MAX_OPTION, readAll } from "../proxy.js";

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
 * itself: the gateway, the platform and the data service; `options` override them. With `options.proxyUrl` the
 * client goes through your proxy, which holds the key: nothing is taken from the credentials then.
 */
export function createClient(options: DreamClientOptions = {}, env: Env = process.env): DreamClient {
  if (options.proxyUrl !== undefined) return new DreamClient(options);
  const credentials = loadCredentials(env);
  // Each one is the option's, else the credentials': an option given as undefined is left out, never a reset to the
  // production default (which would send the staging key there). `apiKey: null` asks for no key.
  return new DreamClient({
    ...options,
    apiKey: options.apiKey === undefined ? credentials.apiKey : options.apiKey,
    baseUrl: options.baseUrl ?? credentials.gatewayUrl,
    platformUrl: options.platformUrl ?? credentials.platformUrl,
    dataUrl: options.dataUrl ?? credentials.dataUrl,
  });
}

/**
 * A `node:http` request listener that serves a standard fetch handler, such as the proxy's
 * (`createServer(nodeHandler(createDreamProxy({ ... })))`). The request's URL is built from its `Host` header; its body
 * is streamed to the handler. A body the handler did not read to its end (the proxy's size limit) is never buffered:
 * after the answer is sent, what is left of it is read and thrown away, so a client still uploading finishes its write
 * and reads the answer (closing on it instead makes the client's write fail, with EPIPE or ECONNRESET, before it sees
 * the answer, as macOS and Windows do). At most `drainBytes` (16 MiB) more, for at most `drainMs` (5 s): past either
 * the connection is closed. A client that disconnects before the answer aborts the request's `signal`,
 * which the proxy passes on to its upstream call. Nothing it does becomes an unhandled rejection: a failure before the
 * answer is a 500, one while sending it closes the connection.
 *
 * @param options.protocol The scheme of the URL the handler sees; `https` on a TLS socket, else `http`. Set it behind
 *   a TLS-terminating proxy.
 * @param options.drainBytes The most of an unread body thrown away after the answer, bytes.
 * @param options.drainMs The longest an unread body is waited for after the answer, milliseconds.
 */
export function nodeHandler(
  handler: (request: Request) => Promise<Response>,
  options: { protocol?: "http" | "https"; drainBytes?: number; drainMs?: number } = {},
): (request: IncomingMessage, response: ServerResponse) => Promise<void> {
  return async (request, response) => {
    const answer = (status: number, error: string) => {
      if (response.destroyed) return;
      if (response.headersSent) return void response.destroy();
      response.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store" });
      response.end(JSON.stringify({ proxy_error: error }));
    };
    const disconnected = new AbortController();
    // Closed before the answer was sent whole: the client is gone.
    response.on("close", () => response.writableFinished || disconnected.abort(new DOMException("the client disconnected", "AbortError")));
    let fetchRequest: Request;
    let abandoned = false;
    // The URL comes from the request's own Host, never a made-up one: a request without one (HTTP/1.0), with a
    // malformed one, or with an absolute-form target is refused before the handler sees it.
    // Exactly one Host field: Node's `headers.host` keeps the first of two and hides the other.
    const hostFields = request.headersDistinct?.host ?? rawHostFields(request.rawHeaders);
    const host = hostFields.length === 1 ? hostFields[0]! : undefined;
    if (typeof host !== "string" || !HOST.test(host)) return answer(400, "missing, duplicate or malformed Host");
    if (!request.url?.startsWith("/")) return answer(400, "request target must be a path");
    try {
      const protocol = options.protocol ?? ((request.socket as TLSSocket).encrypted ? "https" : "http");
      const headers = new Headers();
      for (const [name, value] of Object.entries(request.headers)) {
        if (value === undefined) continue;
        for (const item of Array.isArray(value) ? value : [value]) headers.append(name, item);
      }
      const method = request.method ?? "GET";
      let body: ReadableStream<Uint8Array> | undefined;
      if (method !== "GET" && method !== "HEAD") {
        body = new ReadableStream<Uint8Array>({
          start(controller) {
            request.on("data", (chunk: Buffer) => {
              if (abandoned) return;
              controller.enqueue(new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength));
              if ((controller.desiredSize ?? 1) <= 0) request.pause();
            });
            request.on("end", () => abandoned || controller.close());
            request.on("error", (error) => abandoned || controller.error(error));
            disconnected.signal.addEventListener("abort", () => abandoned || request.complete || controller.error(disconnected.signal.reason), { once: true });
          },
          pull() {
            request.resume();
          },
          cancel() {
            abandoned = true;
            request.pause();
          },
        });
      }
      fetchRequest = new Request(`${protocol}://${host}${request.url}`, {
        method,
        headers,
        body,
        duplex: "half",
        signal: disconnected.signal,
      } as RequestInit);
    } catch {
      return answer(400, "bad request");
    }
    try {
      const reply = await handler(fetchRequest);
      // Read under the disconnect: a client gone while the answer's body is still open cancels it, never waits on it.
      let bytes: Buffer | null = null;
      if (reply.body) {
        try {
          bytes = Buffer.from(await readAll(reply.body, PROXY_MAX_OPTION, disconnected.signal, 500));
        } catch (error) {
          if (disconnected.signal.aborted) return;
          throw error;
        }
      }
      const headers = Object.fromEntries(reply.headers);
      const unread = abandoned || !request.complete;
      if (response.destroyed) return;
      response.writeHead(reply.status, headers);
      response.end(bytes ?? undefined);
      if (unread) {
        abandoned = true;
        drain(request, options.drainBytes ?? DRAIN_BYTES, options.drainMs ?? DRAIN_MS);
      }
    } catch {
      answer(500, "internal");
    }
  };
}

/** The values of every Host field in `raw` (name, value, name, value, ...). */
function rawHostFields(raw: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i + 1 < raw.length; i += 2) if (raw[i]!.toLowerCase() === "host") out.push(raw[i + 1]!);
  return out;
}

/** A Host header: a DNS name or IPv4 address, or a bracketed IPv6 literal, and an optional port. */
const HOST = /^(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*\.?|\[[0-9A-Fa-f:.]+\])(?::\d{1,5})?$/;

/** The most of an unread request body nodeHandler() throws away after the answer, bytes. */
export const DRAIN_BYTES = 16 * 1024 * 1024;
/** The longest nodeHandler() waits for an unread request body after the answer, milliseconds. */
export const DRAIN_MS = 5000;

/** Reads what is left of `request` and throws it away; past `limit` bytes or `ms`, closes the connection. */
function drain(request: IncomingMessage, limit: number, ms: number): void {
  if (request.complete) return;
  let left = limit;
  const stop = () => {
    clearTimeout(timer);
    request.destroy();
  };
  const timer = setTimeout(stop, ms);
  timer.unref();
  request.on("data", (chunk: Buffer) => {
    left -= chunk.byteLength;
    if (left < 0) stop();
  });
  request.on("end", () => clearTimeout(timer));
  request.on("close", () => clearTimeout(timer));
  request.resume();
}

export type { Credentials };
