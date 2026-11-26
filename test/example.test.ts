/** The browser-proxy example's page serving: a missing bundle is a plain 404, not a crash. */

import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { hasSession, SESSION_COOKIE, servePage } from "../examples/browser-proxy/static.ts";

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
      expect(page.headers.get("content-security-policy")).toBe("frame-ancestors 'none'");
      expect(await page.text()).toContain('<button id="start"');
      expect(errors).toEqual([]);
    } finally {
      await new Promise<void>((resolve) => empty.close(() => resolve()));
    }
  });

  it("signs a request in only with the session cookie the page was served", () => {
    expect(hasSession(`a=1; ${SESSION_COOKIE}=abc`, "abc")).toBe(true);
    expect(hasSession(`${SESSION_COOKIE}=abcd`, "abc")).toBe(false);
    expect(hasSession(null, "abc")).toBe(false);
  });

  it("starts the camera and the inference loop only on the Start button", async () => {
    const { readFile } = await import("node:fs/promises");
    const app = await readFile(new URL("../examples/browser-proxy/app.ts", import.meta.url), "utf8");
    // The only call of main() is inside the button's click handler.
    expect(app.match(/\bmain\(\)(?!: )/g)).toHaveLength(1);
    expect(app).toMatch(/start\.addEventListener\("click", \(\) => \{\n\s+start\.disabled = true;\n\s+main\(\)/);
    expect(app.indexOf("getUserMedia")).toBeLessThan(app.indexOf("start.addEventListener"));
    expect(app.slice(0, app.indexOf("async function main")).includes("getUserMedia")).toBe(false);
  });
});
