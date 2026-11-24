/** The browser-proxy example's page serving: a missing bundle is a plain 404, not a crash. */

import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { servePage } from "../examples/browser-proxy/static.ts";

let server: Server | null = null;
afterEach(() => new Promise<void>((resolve) => (server ? server.close(() => resolve()) : resolve())));

describe("examples/browser-proxy", () => {
  it("answers 404 for a bundle not built yet, and serves the page", async () => {
    const errors: unknown[] = [];
    server = createServer((request, response) => void servePage(request.url!, response, new URL("../examples/browser-proxy/", import.meta.url)).catch((e) => errors.push(e)));
    await new Promise<void>((resolve) => server!.listen(0, "127.0.0.1", resolve));
    const url = `http://127.0.0.1:${(server!.address() as AddressInfo).port}`;
    const empty = createServer((request, response) => void servePage(request.url!, response, new URL("./fixtures/no-such-dir/", import.meta.url)).catch((e) => errors.push(e)));
    await new Promise<void>((resolve) => empty.listen(0, "127.0.0.1", resolve));
    try {
      const missing = await fetch(`http://127.0.0.1:${(empty.address() as AddressInfo).port}/app.js`);
      expect(missing.status).toBe(404);
      expect(await missing.text()).toMatch(/build the page first: app\.js is missing/);
      const page = await fetch(`${url}/`);
      expect(page.status).toBe(200);
      expect(page.headers.get("content-type")).toBe("text/html");
      expect(errors).toEqual([]);
    } finally {
      await new Promise<void>((resolve) => empty.close(() => resolve()));
    }
  });
});
