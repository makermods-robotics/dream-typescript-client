/** The example's page and bundle, served from this directory (or `root`). */

import { readFile } from "node:fs/promises";
import type { ServerResponse } from "node:http";

export const PAGES: Record<string, [string, string]> = {
  "/": ["index.html", "text/html"],
  "/app.js": ["app.js", "text/javascript"],
};

/** Answers one of {@link PAGES}: the file is read before the status is written, so a missing one is a plain 404. */
export async function servePage(path: string, response: ServerResponse, root: URL = new URL(".", import.meta.url), headers: Record<string, string> = {}): Promise<void> {
  const page = PAGES[path]!;
  let body: Buffer;
  try {
    body = await readFile(new URL(page[0], root));
  } catch {
    response.writeHead(404, { "Content-Type": "text/plain" }).end(`build the page first: ${page[0]} is missing (see README.md)`);
    return;
  }
  // Never framed by another page, which could click Start for the visitor.
  response.writeHead(200, { ...headers, "Content-Type": page[1], "Content-Security-Policy": "frame-ancestors 'none'", "X-Frame-Options": "DENY" }).end(body);
}

/** The sign-in cookie's name: the page gets it, and only requests that carry it are forwarded. */
export const SESSION_COOKIE = "dream_proxy_session";

/** Whether `cookieHeader` carries the session cookie with `token`. */
export function hasSession(cookieHeader: string | null, token: string): boolean {
  return (cookieHeader ?? "").split(";").some((part) => part.trim() === `${SESSION_COOKIE}=${token}`);
}
