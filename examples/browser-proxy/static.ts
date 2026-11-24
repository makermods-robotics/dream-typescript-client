/** The example's page and bundle, served from this directory (or `root`). */

import { readFile } from "node:fs/promises";
import type { ServerResponse } from "node:http";

export const PAGES: Record<string, [string, string]> = {
  "/": ["index.html", "text/html"],
  "/app.js": ["app.js", "text/javascript"],
};

/** Answers one of {@link PAGES}: the file is read before the status is written, so a missing one is a plain 404. */
export async function servePage(path: string, response: ServerResponse, root: URL = new URL(".", import.meta.url)): Promise<void> {
  const page = PAGES[path]!;
  let body: Buffer;
  try {
    body = await readFile(new URL(page[0], root));
  } catch {
    response.writeHead(404, { "Content-Type": "text/plain" }).end(`build the page first: ${page[0]} is missing (see README.md)`);
    return;
  }
  response.writeHead(200, { "Content-Type": page[1] }).end(body);
}
