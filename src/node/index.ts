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
  return new DreamClient({
    apiKey: credentials.apiKey,
    baseUrl: credentials.gatewayUrl,
    platformUrl: credentials.platformUrl,
    dataUrl: credentials.dataUrl,
    ...options,
  });
}

/**
 * A `node:http` request listener that serves a standard fetch handler, such as the proxy's
 * (`createServer(nodeHandler(createDreamProxy({ ... })))`). The request's URL is built from its `Host` header; its body
 * is streamed to the handler, and a body the handler stops reading (the proxy's size limit) is not read further: the
 * connection is closed after the answer. A client that disconnects before the answer aborts the request's `signal`,
 * which the proxy passes on to its upstream call. Nothing it does becomes an unhandled rejection: a failure before the
 * answer is a 500, one while sending it closes the connection.
 *
 * @param options.protocol The scheme of the URL the handler sees; `https` on a TLS socket, else `http`. Set it behind
 *   a TLS-terminating proxy.
 */
export function nodeHandler(
  handler: (request: Request) => Promise<Response>,
  options: { protocol?: "http" | "https" } = {},
): (request: IncomingMessage, response: ServerResponse) => Promise<void> {
  return async (request, response) => {
    const answer = (status: number, error: string) => {
      if (response.headersSent) return void response.destroy();
      response.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store" });
      response.end(JSON.stringify({ proxy_error: error }));
    };
    const disconnected = new AbortController();
    // Closed before the answer was sent whole: the client is gone.
    response.on("close", () => response.writableFinished || disconnected.abort(new DOMException("the client disconnected", "AbortError")));
    let fetchRequest: Request;
    let abandoned = false;
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
      fetchRequest = new Request(`${protocol}://${request.headers.host ?? "localhost"}${request.url ?? "/"}`, {
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
      const bytes = reply.body ? Buffer.from(await reply.arrayBuffer()) : null;
      const headers = Object.fromEntries(reply.headers);
      // A body the handler did not read to its end is not read further: the connection closes after the answer.
      const unread = abandoned || !request.complete;
      if (unread) headers.connection = "close";
      if (response.destroyed) return;
      response.writeHead(reply.status, headers);
      response.end(bytes ?? undefined, () => unread && request.destroy());
    } catch {
      answer(500, "internal");
    }
  };
}

export type { Credentials };
