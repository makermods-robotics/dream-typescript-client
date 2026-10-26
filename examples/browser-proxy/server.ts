/**
 * A minimal Dream proxy for a browser app, as the MakerMods console's server runs: the page posts the same JSON the
 * client sends, and this server adds the API key, which never reaches the browser. Bodies pass through unchanged
 * (DreamAPI reports every outcome in the body's status envelope).
 *
 *   node examples/browser-proxy/server.ts      (Node 23.6+; the key from ~/.dreamlabs/credentials or DREAM_API_KEY)
 *
 * It serves the page on http://127.0.0.1:8787/ and forwards
 *   POST /api/dream/v1/get_action, /api/dream/v1/end_session  -> the gateway (DREAM_GATEWAY_URL)
 *   GET  /api/dream-platform/v1/models                         -> the platform (DREAM_API_URL)
 *
 * The page's client runs with `proxyHeaders: true`, so it sends `Dream-Client-Version` and `Dream-Client-Instance`
 * (8 hex digits); this proxy checks both and builds the gateway's `User-Agent: dream-client/<version> <instance>`, as
 * the Python client sends it. It answers only requests addressed to itself (`Host`) from its own page (`Origin`), and
 * refuses bodies over 8 MiB.
 *
 * A deployed proxy needs your own sign-in in front of it: whoever reaches it spends your key. The Host and Origin
 * checks only keep other web pages, and DNS-rebinding tricks, from using it through a visitor's browser.
 */

import { readFile } from "node:fs/promises";
import { createServer, type IncomingMessage } from "node:http";
import { loadCredentials } from "@makermods/dreamlabs/node";

const PORT = 8787;
/** The names this server answers to; add your own when it runs behind another host name. */
const HOSTS = new Set([`127.0.0.1:${PORT}`, `localhost:${PORT}`]);
const MAX_BODY_BYTES = 8 * 1024 * 1024;
const VERSION = /^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,64}$/;
const INSTANCE = /^[0-9a-f]{8}$/;

const credentials = loadCredentials();
if (!credentials.apiKey) throw new Error("no DREAM_API_KEY: set it, or write it to ~/.dreamlabs/credentials");

const ROUTES: Record<string, { method: string; target: string }> = {
  "/api/dream/v1/get_action": { method: "POST", target: `${credentials.gatewayUrl}/v1/get_action` },
  "/api/dream/v1/end_session": { method: "POST", target: `${credentials.gatewayUrl}/v1/end_session` },
  "/api/dream-platform/v1/models": { method: "GET", target: `${credentials.platformUrl}/v1/models` },
};
const STATIC: Record<string, [string, string]> = {
  "/": ["index.html", "text/html"],
  "/app.js": ["app.js", "text/javascript"],
};

/** The body, or null once it passes `MAX_BODY_BYTES` (the rest is not read). */
async function readBody(request: IncomingMessage): Promise<Buffer | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    size += (chunk as Buffer).length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks);
}

const header = (request: IncomingMessage, name: string) => {
  const value = request.headers[name];
  return typeof value === "string" ? value : undefined;
};

createServer(async (request, response) => {
  const fail = (status: number, error: string) => void response.writeHead(status, { "Content-Type": "application/json" }).end(JSON.stringify({ proxy_error: error }));
  // Only requests to this server by name, from its own page: no other site's page, no rebound DNS name.
  const host = header(request, "host") ?? "";
  if (!HOSTS.has(host)) return fail(421, "unknown host");
  const origin = header(request, "origin");
  if (origin !== undefined && origin !== `http://${host}`) return fail(403, "foreign origin");
  const path = (request.url ?? "/").split("?")[0]!;
  const page = STATIC[path];
  if (page && request.method === "GET") {
    try {
      const body = await readFile(new URL(page[0], import.meta.url));
      response.writeHead(200, { "Content-Type": page[1] }).end(body);
    } catch {
      response.writeHead(404).end(`build the page first: ${page[0]} is missing (see README.md)`);
    }
    return;
  }
  const route = ROUTES[path];
  if (!route || route.method !== request.method) return fail(404, "no such route");
  if (Number(header(request, "content-length") ?? 0) > MAX_BODY_BYTES) return fail(413, "too large");
  const body = await readBody(request);
  if (body === null) return fail(413, "too large");
  const headers: Record<string, string> = { Authorization: `Bearer ${credentials.apiKey}`, Accept: "application/json" };
  if (route.method === "POST") {
    const version = header(request, "dream-client-version");
    const instance = header(request, "dream-client-instance");
    if (!version || !VERSION.test(version) || !instance || !INSTANCE.test(instance)) {
      return fail(400, "send Dream-Client-Version and an 8-hex Dream-Client-Instance (DreamClient({ proxyHeaders: true }))");
    }
    headers["Dream-Client-Version"] = version;
    headers["User-Agent"] = `dream-client/${version} ${instance}`;
    headers["Content-Type"] = "application/json";
  }
  try {
    const reply = await fetch(route.target, {
      method: route.method,
      headers,
      body: route.method === "POST" ? new Uint8Array(body) : undefined,
      redirect: "manual", // the key never follows a redirect to another host
      signal: AbortSignal.timeout(70_000), // above the client's own timeout: the browser gives up first
    });
    response.writeHead(reply.status, { "Content-Type": "application/json" }).end(await reply.text());
  } catch (error) {
    const timeout = error instanceof Error && error.name === "TimeoutError";
    fail(timeout ? 504 : 502, timeout ? "timeout" : "unreachable");
  }
}).listen(PORT, "127.0.0.1", () => console.log(`http://127.0.0.1:${PORT}/ -> ${credentials.gatewayUrl}`));
