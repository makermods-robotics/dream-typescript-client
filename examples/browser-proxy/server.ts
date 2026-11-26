/**
 * A Dream proxy for a browser app: the page's client calls `/api/dream/...` on this server, and createDreamProxy()
 * forwards those calls to Dream Labs with the API key added, which never reaches the browser.
 *
 *   node examples/browser-proxy/server.ts      (Node 23.6+; the key from ~/.dreamlabs/credentials or DREAM_API_KEY)
 *
 * It serves the page on http://127.0.0.1:8787/, and the proxy under /api/dream (see @makermods/dreamlabs/proxy for
 * what it forwards and checks). It answers only requests addressed to itself (`allowedHosts`) and, from a page, only
 * from its own (`allowedOrigins`).
 *
 * Its sign-in is a session cookie the page gets when it is served (SameSite=Strict); a deployed proxy needs your own
 * sign-in in its place (createDreamProxy's `authorize`): whoever reaches it spends your key. The host and origin checks only keep other web pages, and DNS-rebinding tricks, from using it through a
 * visitor's browser.
 */

import { createServer } from "node:http";
import { loadCredentials, nodeHandler } from "@makermods/dreamlabs/node";
import { createDreamProxy } from "@makermods/dreamlabs/proxy";
import { randomBytes } from "node:crypto";
import { hasSession, PAGES, SESSION_COOKIE, servePage } from "./static.ts";

const PORT = 8787;
const HOSTS = [`127.0.0.1:${PORT}`, `localhost:${PORT}`];

const credentials = loadCredentials();
if (!credentials.apiKey) throw new Error("no DREAM_API_KEY: set it, or write it to ~/.dreamlabs/credentials");

// The sign-in: serving the page sets a random session cookie (SameSite=Strict, so no other site's page or link sends
// it), and the proxy forwards only requests that carry it. Replace it with your own sign-in before others reach this.
const session = randomBytes(16).toString("hex");

const dream = nodeHandler(
  createDreamProxy({
    apiKey: credentials.apiKey,
    basePath: "/api/dream",
    gatewayUrl: credentials.gatewayUrl,
    platformUrl: credentials.platformUrl,
    dataUrl: credentials.dataUrl,
    allowedOrigins: HOSTS.map((host) => `http://${host}`),
    allowedHosts: HOSTS,
    authorize: (request) => hasSession(request.headers.get("cookie"), session),
  }),
);

createServer(async (request, response) => {
  const path = (request.url ?? "/").split("?")[0]!;
  if (!(path in PAGES) || request.method !== "GET") return dream(request, response);
  if (!HOSTS.includes(request.headers.host ?? "")) return void response.writeHead(421).end();
  await servePage(path, response, undefined, { "Set-Cookie": `${SESSION_COOKIE}=${session}; HttpOnly; SameSite=Strict; Path=/` });
}).listen(PORT, "127.0.0.1", () => console.log(`http://127.0.0.1:${PORT}/ -> ${credentials.gatewayUrl}`));
