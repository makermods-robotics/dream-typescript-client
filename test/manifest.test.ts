/**
 * Fixtures presented as current are what the current upstream wrote: every file scripts/manifest.py records has the
 * bytes its sha256 names and was written by the upstream versions this package tracks (`UPSTREAM`), and a check on it
 * names the environment that wrote it. limit_stop.json, recorded on hardware, has its float64 answer replayed against
 * the same versions (scripts/replay_limit_stop.py).
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { UPSTREAM } from "../src/index.js";
import { fixture } from "./helpers.js";

const manifest = fixture("manifest.json");
const upstream = { dreamlabs: UPSTREAM.dreamlabs, "dreamlabs-client": UPSTREAM.dreamlabsClient };
const sha256 = (name: string) => createHash("sha256").update(readFileSync(new URL(`../${name}`, import.meta.url))).digest("hex");

describe("the fixture manifest", () => {
  it("names the bytes and the upstream versions of every recorded file", () => {
    let files = 0;
    for (const section of Object.values<any>(manifest)) {
      for (const [name, entry] of Object.entries<any>(section.files)) {
        files++;
        expect(sha256(name), `${name} bytes`).toBe(entry.sha256);
        const versions = Object.fromEntries(Object.entries<any>(entry.environment.upstream).map(([pkg, info]) => [pkg, info.version]));
        expect(versions, `${name} upstream`).toEqual(upstream);
      }
    }
    expect(files).toBeGreaterThanOrEqual(10);
  });

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

  it("has limit_stop.json's converged answer replayed against the tracked versions, and recorded by name", () => {
    const entry = manifest.replay_limit_stop?.files?.["test/fixtures/limit_stop.json"];
    expect(entry, "limit_stop.json's manifest entry").toBeDefined();
    expect(sha256("test/fixtures/limit_stop.json")).toBe(entry.sha256);
    const limit = fixture("limit_stop.json");
    expect({ dreamlabs: limit.versions.dreamlabs, "dreamlabs-client": limit.versions["dreamlabs-client"] }).toEqual(upstream);
    expect(limit.recorded_versions).toMatchObject({ dreamlabs: "1.0.17", "dreamlabs-client": "1.0.8" });
    expect(limit.replayed.converged_moved_rad).toBeLessThanOrEqual(limit.replayed.bound_rad);
  });
});
