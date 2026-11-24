/**
 * test/fixtures/manifest.json: a check recorded on a fixture (`checks`) names the upstream packages and versions it was
 * measured with, and they are the ones that wrote the fixture (scripts/manifest.py drops a check measured elsewhere).
 */

import { describe, expect, it } from "vitest";
import { fixture } from "./helpers.js";

const manifest = fixture("manifest.json");

describe("the fixture manifest", () => {
  it("keeps a check only with the environment that wrote its fixture", () => {
    let checks = 0;
    for (const section of Object.values<any>(manifest)) {
      for (const [name, entry] of Object.entries<any>(section.files)) {
        for (const [key, check] of Object.entries<any>(entry.checks ?? {})) {
          checks++;
          expect(check.environment, `${name} ${key}`).toEqual({ upstream: entry.environment.upstream, versions: entry.environment.versions });
        }
      }
    }
    expect(checks).toBeGreaterThan(0);
  });
});
