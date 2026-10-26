import { mkdtempSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  DreamActionError,
  DreamBadRequest,
  DreamClient,
  DreamConfigError,
  DreamError,
  DreamObservationError,
  DreamServerError,
} from "../src/index.js";
import { authorization, check, checkPerKey, compileChecks, encode, errorFor, isEmbodimentConfig, parseCredentialsFile, parseEmbodiment, resolveCredentials } from "../src/internal.js";
import { toModel } from "../src/models.js";
import { createClient, loadCredentials } from "../src/node/index.js";

describe("errors", () => {
  it("a DreamBadRequest is a DreamConfigError and a DreamServerError, as in Python", () => {
    const error = errorFor({ code: "DA-4000", name: "bad_request", reason: "too_large" }, "https://api.dreamlabs.ai", "c");
    expect(error).toBeInstanceOf(DreamBadRequest);
    expect(error).toBeInstanceOf(DreamServerError);
    expect(error).toBeInstanceOf(DreamConfigError);
    expect(error).toBeInstanceOf(DreamError);
    expect(new DreamConfigError("x")).not.toBeInstanceOf(DreamServerError);
    expect(new DreamServerError("x")).not.toBeInstanceOf(DreamConfigError);
  });

  it("keeps DreamAPI's wording and the retry fields", () => {
    const error = errorFor({ code: "DA-2000", name: "unavailable", reason: "pool_warming", retryable: true, retry_after_s: 4, detail: "starting", trace_id: "t1" }, "u", "c");
    expect(error.message).toBe("DreamAPI refused the inference request: DA-2000 unavailable (pool_warming): starting");
    expect([error.retryable, error.retryAfterS, error.traceId]).toEqual([true, 4, "t1"]);
  });
});

describe("the wire encoding", () => {
  it("writes typed arrays as lists, compactly", () => {
    expect(encode({ a: Float32Array.of(0.5, 1), b: [1, "x", null, true] })).toBe('{"a":[0.5,1],"b":[1,"x",null,true]}');
  });

  it("refuses what JSON cannot carry instead of changing it", () => {
    for (const value of [NaN, Infinity, undefined, 1n, () => 1, new ArrayBuffer(2), new Map(), new Set(), new Date()]) {
      expect(() => encode({ key: value }), String(value)).toThrow(TypeError);
    }
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    expect(() => encode(circular)).toThrow(/circular/);
  });
});

describe("credentials (dream.credentials)", () => {
  it("parses the file as Python does", () => {
    const text = "# comment\nDREAM_API_KEY='dk_1'\nexport DREAM_GATEWAY_URL=\"http://127.0.0.1:8000\"\n  OTHER = x \nDREAM_API_KEY=dk_2\nbroken line\n";
    expect(parseCredentialsFile(text)).toEqual({ DREAM_API_KEY: "dk_2", DREAM_GATEWAY_URL: "http://127.0.0.1:8000", OTHER: "x" });
  });

  it("takes the environment over the file over production", () => {
    const resolved = resolveCredentials({ DREAM_API_KEY: "env" }, { text: "DREAM_API_KEY=file\nDREAM_API_URL=https://staging.platform.dreamlabs.ai", path: "/f" });
    expect(resolved).toEqual({
      gatewayUrl: "https://api.dreamlabs.ai",
      platformUrl: "https://staging.platform.dreamlabs.ai",
      dataUrl: "https://data.dreamlabs.ai",
      apiKey: "env",
      path: "/f",
    });
    expect(() => resolveCredentials({ DREAM_GATEWAY_URL: "https://api.dreamlabs.ai/v1" })).toThrow(/no path/);
    expect(() => resolveCredentials({ DREAM_GATEWAY_URL: "ftp://x" })).toThrow(DreamConfigError);
  });

  it("refuses a key an HTTP header cannot carry, without showing it", () => {
    expect(authorization("dk_abc")).toEqual({ Authorization: "Bearer dk_abc" });
    for (const key of ["dk\nsecret", "dk“secret"]) {
      expect(() => authorization(key)).toThrow(DreamConfigError);
      expect(() => authorization(key)).not.toThrow(/secret/);
    }
  });

  it("reads ~/.dreamlabs/credentials or DREAM_CREDENTIALS in Node", () => {
    const dir = mkdtempSync(join(tmpdir(), "dream-"));
    const file = join(dir, "credentials");
    writeFileSync(file, "DREAM_GATEWAY_URL=http://127.0.0.1:9\nDREAM_API_KEY=dk_file\n");
    const loaded = loadCredentials({ DREAM_CREDENTIALS: file });
    expect(loaded).toMatchObject({ gatewayUrl: "http://127.0.0.1:9", apiKey: "dk_file", path: file });
    expect(loadCredentials({ DREAM_CREDENTIALS: join(dir, "missing") }).apiKey).toBeNull();
    const client = createClient({}, { DREAM_CREDENTIALS: file });
    expect(client.baseUrl).toBe("http://127.0.0.1:9");
  });
});

describe("embodiments and the strict checks", () => {
  const config = {
    state: { right_arm: [0, 6], right_gripper: [6, 7] },
    action: { right_arm: [0, 6], right_gripper: [6, 7] },
    video: { overhead: "observation.images.top" },
  };

  it("tells an embodiment config from a shape spec", () => {
    expect(isEmbodimentConfig(config)).toBe(true);
    expect(isEmbodimentConfig({ state: { a: {} }, action: { b: {} }, video: { c: {} } })).toBe(false);
    expect(isEmbodimentConfig({ state: { "arm_r.joint_pos": { shape: [6] } } })).toBe(false);
  });

  it("parses spans that tile and refuses ones that do not", () => {
    expect(parseEmbodiment(config).state.right_gripper).toEqual([6, 7]);
    expect(() => parseEmbodiment({ ...config, state: { a: [0, 3], b: [4, 6] } })).toThrow(/nothing covers \[3, 4\)/);
    expect(() => parseEmbodiment({ ...config, state: { a: [0, 3], b: [2, 6] } })).toThrow(/overlapping/);
    expect(() => parseEmbodiment({ ...config, video: { overhead: "a/b" } })).toThrow(/not a path/);
    expect(() => parseEmbodiment({ ...config, parts: {} })).toThrow(/does not convert parts/);
    expect(() => parseEmbodiment({ ...config, extra: 1 })).toThrow(/Unknown key 'extra'/);
  });

  it("checks observations and per_key replies of an embodiment config", () => {
    const checks = compileChecks(config)!;
    const good = { "state.right_arm": [0, 0, 0, 0, 0, 0], "state.right_gripper": Float32Array.of(1), "video.overhead": ["Zg=="], prompt: "x" };
    check(good, checks.observation, "observation", { frameWindowGroup: checks.frameWindowGroup });
    expect(() => check({ ...good, "state.right_arm": [0] }, checks.observation, "observation")).toThrow(/has shape \(1,\), but the configured embodiment declares \(6,\)/);
    expect(() => check({ ...good, "video.overhead": "Zg==" }, checks.observation, "observation", { frameWindowGroup: "video" })).toThrow(DreamObservationError);
    expect(() => check({ ...good, "state.extra": 1 }, checks.observation, "observation")).toThrow(/not declared/);
    const rows = Array.from({ length: 48 }, () => [0, 0, 0, 0, 0, 0]);
    checkPerKey({ per_key: { "action.right_arm": rows, "action.right_gripper": rows.map(() => 0.5) } }, checks.perKeyWidths);
    expect(() => checkPerKey({ per_key: { "action.right_arm": rows } }, checks.perKeyWidths)).toThrow(DreamActionError);
    expect(() => checkPerKey({ per_key: { "action.right_arm": rows, "action.right_gripper": [0.5] } }, checks.perKeyWidths)).toThrow(/1 rows where the other parts have 48/);
  });

  it("checks typed-array dtypes against a shape spec", () => {
    const checks = compileChecks({ state: { "arm_r.joint_pos": { shape: [2], dtype: "float32" } } })!;
    check({ "state.arm_r.joint_pos": Float32Array.of(1, 2) }, checks.observation, "observation");
    check({ "state.arm_r.joint_pos": [1, 2] }, checks.observation, "observation");
    expect(() => check({ "state.arm_r.joint_pos": Float64Array.of(1, 2) }, checks.observation, "observation")).toThrow(/declares float32/);
    expect(() => compileChecks({ state: { a: { dtype: "float7" } } })).toThrow(DreamConfigError);
  });
});

describe("models (dream.models)", () => {
  it("reads a model as the platform answers it", () => {
    const model = toModel({ id: "dream-1-ft:abc-x", owner: "mine", base_model: "dreamlabs/dream-1-beta", embodiment: "emb_1", servable: true, created_at: "2026-10-01T12:00:00Z", cameras: ["front"], embodiment_config: null, parts_withheld: true });
    expect(model).toMatchObject({ id: "dream-1-ft:abc-x", baseModel: "dreamlabs/dream-1-beta", partsWithheld: true, cameras: ["front"] });
    expect(model.createdAt?.toISOString()).toBe("2026-10-01T12:00:00.000Z");
    expect(() => toModel({ id: "x", owner: "public" })).toThrow(/older than this library/);
  });

  it("needs a key to call Dream Labs directly, but not a proxy", async () => {
    const client = new DreamClient();
    await expect(client.getModels()).rejects.toThrow(/no api key/);
    const seen: string[] = [];
    const proxied = new DreamClient({ platformUrl: "http://localhost:1/api/platform", fetch: async (url) => (seen.push(url), new Response("[]")) });
    expect(await proxied.getModels()).toEqual([]);
    expect(seen).toEqual(["http://localhost:1/api/platform/v1/models"]);
  });
});

describe("the package", () => {
  it("keeps Node built-ins out of every entry but ./node", () => {
    const files: string[] = [];
    const walk = (dir: string) => {
      for (const name of readdirSync(dir)) {
        const path = join(dir, name);
        if (statSync(path).isDirectory()) walk(path);
        else if (path.endsWith(".ts")) files.push(path);
      }
    };
    walk(new URL("../src", import.meta.url).pathname);
    for (const file of files) {
      if (file.includes("/src/node/")) continue;
      const code = readFileSync(file, "utf8").replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
      expect(code, file).not.toMatch(/from "node:|\brequire\(|(^|[^\w.])(process|Buffer|__dirname)\b/);
    }
  });
});

describe("review fixes", () => {
  it("never shows the key when a client is logged or serialized", async () => {
    const { inspect } = await import("node:util");
    const client = new DreamClient({ apiKey: "dk_secret_123", baseUrl: "http://127.0.0.1:9" });
    await client.policy({ checkpoint: "c", strict: false });
    for (const text of [inspect(client, { depth: 10, showHidden: true }), JSON.stringify(client), String(client)]) expect(text).not.toContain("dk_secret_123");
    await client.close();
  });

  it("refuses a key with a NUL or other control character", () => {
    for (const key of ["dk\u0000x", "dk\tx", "dk\u007fx"]) expect(() => authorization(key)).toThrow(DreamConfigError);
  });

  it("refuses a listed embodiment config it cannot read instead of running unchecked", async () => {
    const record = { id: "dream-1-ft:x-y", owner: "mine", base_model: "b", embodiment: "emb_1", servable: true, created_at: null, cameras: ["front"], embodiment_config: {} };
    const client = new DreamClient({ baseUrl: "http://127.0.0.1:9", platformUrl: "http://127.0.0.1:9", fetch: async () => new Response(JSON.stringify([record])) });
    await expect(client.policy({ checkpoint: "dream-1-ft:x-y" })).rejects.toThrow(DreamConfigError);
  });
});
