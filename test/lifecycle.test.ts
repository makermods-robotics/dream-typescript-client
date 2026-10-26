/**
 * The lifecycle (close, replace, cancel), the client identity on the wire, and `processing: "raw"`, against a stub
 * `fetch` whose replies the test releases by hand.
 */

import { describe, expect, it } from "vitest";
import {
  CLIENT_VERSION,
  DreamCancelled,
  DreamClient,
  DreamConfigError,
  DreamProtocolError,
  DreamTimeout,
  PolicyClient,
  type Diagnostics,
} from "../src/index.js";
import { Transport } from "../src/transport.js";

/** A promise the test resolves by hand. */
function gate(): { promise: Promise<void>; open: () => void } {
  let open!: () => void;
  const promise = new Promise<void>((resolve) => (open = resolve));
  return { promise, open };
}

const json = (body: unknown) => new Response(JSON.stringify(body));
const chunk = (value: number) => Array.from({ length: 48 }, (_, k) => [value + k * 0.001, 0, 0, 0, 0, 0, 0.5]);
const okReply = (session = "ses_a") => ({ status: { code: "DA-0000" }, session_id: session, actions: { q_r_chunk: chunk(0), q_l_chunk: chunk(0) } });
const record = (id: string, extra: Record<string, unknown> = {}) => ({ id, owner: "mine", base_model: "b", embodiment: "emb_1", servable: true, created_at: null, cameras: ["front"], embodiment_config: null, ...extra });
const observation = { "state.arm_r.joint_pos": [0, 0, 0, 0, 0, 0], "state.arm_l.joint_pos": [0, 0, 0, 0, 0, 0], prompt: "x" };

interface Seen {
  url: string;
  headers: Record<string, string>;
  body: any;
}

/**
 * A stub fetch: `/v1/models` answers `models`, `/v1/get_action` the reply, `/v1/end_session` ok, each once `hold` (if
 * set) is opened. `honour` false makes it ignore the abort signal, as a careless custom fetch might.
 */
function stub(options: { models?: unknown[]; hold?: Promise<void> | null; honour?: boolean; reply?: () => unknown } = {}) {
  const seen: Seen[] = [];
  const fetch = async (url: string, init: RequestInit): Promise<Response> => {
    seen.push({ url, headers: Object.fromEntries(new Headers(init.headers).entries()), body: init.body ? JSON.parse(String(init.body)) : null });
    const signal = init.signal;
    const wait = options.hold ?? Promise.resolve();
    await (options.honour === false || !signal
      ? wait
      : Promise.race([wait, new Promise((_, reject) => signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")), { once: true }))]));
    if (url.endsWith("/v1/models")) return json(options.models ?? []);
    if (url.endsWith("/v1/end_session")) return json({ status: { code: "DA-0000" } });
    return json(options.reply?.() ?? okReply());
  };
  return { fetch, seen };
}

const urls = { baseUrl: "http://gateway.test", platformUrl: "http://platform.test" };

describe("the lifecycle", () => {
  it("a policy() whose model lookup is pending when close() runs opens nothing", async () => {
    const hold = gate();
    const { fetch } = stub({ models: [record("dream-1-ft:a-x")], hold: hold.promise, honour: false });
    const client = new DreamClient({ ...urls, fetch });
    const opening = client.policy({ checkpoint: "dream-1-ft:a-x" });
    await client.close();
    hold.open();
    await expect(opening).rejects.toBeInstanceOf(DreamCancelled);
    expect(client.checkpoint).toBeNull();
    expect(() => client.mode).toThrow(DreamConfigError);
  });

  it("close() aborts the model lookup of a policy() in progress", async () => {
    const { fetch } = stub({ models: [record("dream-1-ft:a-x")], hold: new Promise(() => {}) });
    const client = new DreamClient({ ...urls, fetch });
    const opening = client.policy({ checkpoint: "dream-1-ft:a-x" });
    await Promise.resolve();
    void client.close();
    await expect(opening).rejects.toThrow(/the client was closed/);
  });

  it("a getAction() in flight when close() runs returns nothing, even from a fetch that ignores its signal", async () => {
    for (const honour of [true, false]) {
      const hold = gate();
      const { fetch, seen } = stub({ hold: hold.promise, honour });
      const client = new DreamClient({ ...urls, fetch });
      await client.policy({ checkpoint: "c", strict: false });
      client.setObservation({ "video.top": "Zg==" }, observation, { timestamp: 0 });
      const pending = client.getAction();
      await Promise.resolve();
      const closing = client.close();
      hold.open();
      await expect(pending, `honour ${honour}`).rejects.toBeInstanceOf(DreamCancelled);
      await closing;
      expect(seen.filter((s) => s.url.endsWith("/v1/get_action"))).toHaveLength(1);
    }
  });

  it("a policy() called while another is pending supersedes it; only the newer one opens", async () => {
    const hold = gate();
    const { fetch } = stub({ models: [record("dream-1-ft:a-x"), record("dream-1-ft:b-x")], hold: hold.promise, honour: false });
    const client = new DreamClient({ ...urls, fetch });
    const first = client.policy({ checkpoint: "dream-1-ft:a-x" });
    const second = client.policy({ checkpoint: "dream-1-ft:b-x" });
    hold.open();
    await expect(first).rejects.toThrow(/a newer policy\(\) call started/);
    await second;
    expect(client.checkpoint).toBe("dream-1-ft:b-x");
    await client.close();
  });

  it("replacing an open policy closes it: its session is released and its getAction() in flight cancelled", async () => {
    const hold = gate();
    let held = false;
    const { fetch, seen } = stub({ reply: () => okReply("ses_old") });
    const slow = async (url: string, init: RequestInit) => {
      if (held && url.endsWith("/v1/get_action")) await hold.promise;
      return fetch(url, init);
    };
    const client = new DreamClient({ ...urls, fetch: slow });
    await client.policy({ checkpoint: "c", strict: false });
    client.setObservation({ "video.top": "Zg==" }, observation, { timestamp: 0 });
    await client.getAction();
    held = true;
    const pending = client.getAction();
    const replacing = client.policy({ checkpoint: "d", strict: false });
    hold.open();
    await expect(pending).rejects.toBeInstanceOf(DreamCancelled);
    await replacing;
    expect(seen.some((s) => s.url.endsWith("/v1/end_session") && s.body.session_id === "ses_old")).toBe(true);
    expect(client.checkpoint).toBe("d");
    await client.close();
  });

  it("concurrent close() calls share one promise", async () => {
    const client = new DreamClient({ ...urls, fetch: stub().fetch });
    await client.policy({ checkpoint: "c", strict: false });
    const a = client.close();
    expect(client.close()).toBe(a);
    await a;
    const p = new PolicyClient({ checkpoint: "c", ...urls, fetch: stub().fetch });
    expect(p.close()).toBe(p.close());
  });

  it("a caller's signal cancels getAction(), getModels() and policy(), distinctly from a timeout", async () => {
    const { fetch } = stub({ hold: new Promise(() => {}), models: [record("dream-1-ft:a-x")] });
    const policy = new PolicyClient({ checkpoint: "c", ...urls, fetch, timeoutMs: 50 });
    await expect(policy.getAction(observation)).rejects.toBeInstanceOf(DreamTimeout);
    const abort = new AbortController();
    const pending = policy.getAction(observation, { signal: abort.signal });
    abort.abort("stop");
    const error = await pending.catch((e) => e);
    expect(error).toBeInstanceOf(DreamCancelled);
    expect(error).not.toBeInstanceOf(DreamProtocolError);
    expect([error.retryable, error.cause]).toEqual([false, "stop"]);
    const client = new DreamClient({ ...urls, fetch });
    const models = new AbortController();
    const listing = client.getModels({ signal: models.signal });
    models.abort();
    await expect(listing).rejects.toBeInstanceOf(DreamCancelled);
    const opening = new AbortController();
    const opened = client.policy({ checkpoint: "dream-1-ft:a-x" }, { signal: opening.signal });
    opening.abort();
    await expect(opened).rejects.toBeInstanceOf(DreamCancelled);
    expect(client.checkpoint).toBeNull();
    await expect(policy.getAction(observation, { signal: AbortSignal.abort() })).rejects.toBeInstanceOf(DreamCancelled);
  });

  it("a call cancelled or closed from onDiagnostics returns nothing and keeps no plan", async () => {
    // Codex review round 2: abort() and PolicyClient.close() inside the callback still returned the action.
    const abort = new AbortController();
    const client = new DreamClient({ ...urls, fetch: stub().fetch, onDiagnostics: () => abort.abort("stop") });
    await client.policy({ checkpoint: "c", strict: false });
    client.setObservation({ "video.top": "Zg==" }, observation, { timestamp: 0 });
    await expect(client.getAction({ signal: abort.signal })).rejects.toBeInstanceOf(DreamCancelled);
    await client.close();
    let closing: Promise<void> | null = null;
    const policy: PolicyClient = new PolicyClient({ checkpoint: "c", ...urls, fetch: stub().fetch, onDiagnostics: () => void (closing = policy.close()) });
    await expect(policy.getAction(observation)).rejects.toBeInstanceOf(DreamCancelled);
    await closing;
    // The plan was not kept: a later client's first chunk would be filtered only; here the raw path is checked too.
    const raw = new PolicyClient({ checkpoint: "c", ...urls, processing: "raw", fetch: stub().fetch, onDiagnostics: () => void raw.close() });
    await expect(raw.getAction(observation)).rejects.toBeInstanceOf(DreamCancelled);
  });

  it("keeps the first abort's cause: a cancel then a late rejection is DreamCancelled, a late reply after the timeout DreamTimeout", async () => {
    // A custom fetch that rejects 30 ms after its abort: the timer (10 ms) fires meanwhile and must not win.
    const slowReject = async (_url: string, init: RequestInit): Promise<Response> =>
      new Promise((_, reject) => init.signal!.addEventListener("abort", () => setTimeout(() => reject(init.signal!.reason), 30), { once: true }));
    const cancelledFirst = new PolicyClient({ checkpoint: "c", ...urls, timeoutMs: 10, fetch: slowReject });
    const abort = new AbortController();
    const pending = cancelledFirst.getAction(observation, { signal: abort.signal }).catch((e) => e);
    abort.abort("user stop");
    const error = await pending;
    expect(error).toBeInstanceOf(DreamCancelled);
    expect(error.cause).toBe("user stop");
    // A fetch that ignores its abort and answers after the timeout: the answer is refused, as a timeout.
    const ignoring = new PolicyClient({
      checkpoint: "c",
      ...urls,
      timeoutMs: 10,
      fetch: async () => (await new Promise((resolve) => setTimeout(resolve, 35)), json(okReply())),
    });
    await expect(ignoring.getAction(observation)).rejects.toBeInstanceOf(DreamTimeout);
    // And the timeout first, then the caller: still a timeout.
    const timedFirst = new PolicyClient({ checkpoint: "c", ...urls, timeoutMs: 10, fetch: slowReject });
    const late = new AbortController();
    const outcome = timedFirst.getAction(observation, { signal: late.signal }).catch((e) => e);
    setTimeout(() => late.abort("too late"), 20);
    expect(await outcome).toBeInstanceOf(DreamTimeout);
  });

  it("keeps the first cancellation's cause: a caller's abort then close() stays the caller's, close() then an abort stays close()'s", async () => {
    // Codex final check: the transport read the caller and closed state when the fetch rejected, not what fired first.
    const slowReject = async (_url: string, init: RequestInit): Promise<Response> =>
      new Promise((_, reject) => init.signal!.addEventListener("abort", () => setTimeout(() => reject(init.signal!.reason), 30), { once: true }));
    const caller = (error: DreamCancelled) => ({ message: error.message, cause: error.cause });
    // The transport alone.
    for (const order of ["caller-first", "close-first"] as const) {
      const transport = new Transport({ baseUrl: "http://gateway.test", timeoutMs: 10_000, apiKey: null, fetch: slowReject });
      const abort = new AbortController();
      const pending = transport.request({}, "/v1/get_action", { signal: abort.signal }).then(() => null, (e: DreamCancelled) => e);
      if (order === "caller-first") (abort.abort("user stop"), transport.close());
      else (transport.close(), abort.abort("user stop"));
      const error = await pending;
      expect(error, order).toBeInstanceOf(DreamCancelled);
      expect(caller(error!), order).toEqual(
        order === "caller-first" ? { message: "the request was cancelled by the caller's signal", cause: "user stop" } : { message: "the request was cancelled: the client was closed", cause: undefined },
      );
    }
    // And through the client, whose own close() aborts the call.
    for (const order of ["caller-first", "close-first"] as const) {
      const client = new PolicyClient({ checkpoint: "c", ...urls, timeoutMs: 10_000, fetch: slowReject });
      const abort = new AbortController();
      const pending = client.getAction(observation, { signal: abort.signal }).catch((e) => e);
      await new Promise((resolve) => setTimeout(resolve, 5));
      let closing: Promise<void>;
      if (order === "caller-first") (abort.abort("user stop"), (closing = client.close()));
      else ((closing = client.close()), abort.abort("user stop"));
      const error = await pending;
      await closing;
      expect(error, order).toBeInstanceOf(DreamCancelled);
      if (order === "caller-first") expect(caller(error), order).toEqual({ message: "the request was cancelled by the caller's signal", cause: "user stop" });
      else expect(caller(error), order).toEqual({ message: "the call was cancelled: the client was closed", cause: undefined });
    }
  });

  it("close() during resetEpisode() still releases the session, on a fresh signal", async () => {
    const paths: string[] = [];
    let releases = 0;
    const fetch = async (url: string, init: RequestInit): Promise<Response> => {
      paths.push(url);
      // The first release hangs until aborted; the retry is answered.
      if (url.endsWith("/v1/end_session") && releases++ === 0) {
        return new Promise((_, reject) => init.signal!.addEventListener("abort", () => reject(init.signal!.reason), { once: true }));
      }
      return json(url.endsWith("/v1/end_session") ? { status: { code: "DA-0000" } } : okReply("ses_r"));
    };
    const client = new PolicyClient({ checkpoint: "c", ...urls, fetch });
    await client.getAction(observation);
    const reset = client.resetEpisode();
    await Promise.resolve();
    await client.close();
    await reset;
    expect(paths.filter((u) => u.endsWith("/v1/end_session"))).toHaveLength(2);
    expect(releases).toBe(2);
  });

  it("keeps the cause of a transport error, without the key", async () => {
    const failing = async () => {
      throw new TypeError("connect ECONNREFUSED", { cause: new Error("socket hang up") });
    };
    const policy = new PolicyClient({ checkpoint: "c", ...urls, fetch: failing });
    const error = await policy.getAction(observation).catch((e) => e);
    expect(error).toBeInstanceOf(DreamProtocolError);
    expect(error.cause).toBeInstanceOf(TypeError);
    const leaking = async () => {
      throw new TypeError("bad header value Bearer dk_secret_1");
    };
    const keyed = new PolicyClient({ checkpoint: "c", ...urls, apiKey: "dk_secret_1", fetch: leaking });
    const scrubbed = await keyed.getAction(observation).catch((e) => e);
    expect(String(scrubbed.message)).not.toContain("dk_secret_1");
    expect(String(scrubbed.cause?.message)).not.toContain("dk_secret_1");
  });
});

describe("the client identity", () => {
  it("names this package, deliberately not a semver, and takes an override", async () => {
    expect(CLIENT_VERSION).toMatch(/^makermods-dreamlabs-ts-\d+\.\d+\.\d+\+dream-1\.0\.8$/);
    expect(CLIENT_VERSION).not.toMatch(/^\d+\.\d+\.\d+$/);
    const { fetch, seen } = stub();
    await new PolicyClient({ checkpoint: "c", ...urls, fetch }).getAction(observation);
    expect(seen[0]!.headers["dream-client-version"]).toBe(CLIENT_VERSION);
    expect(seen[0]!.headers["user-agent"]).toMatch(new RegExp(`^dream-client/${CLIENT_VERSION.replace(/[.+]/g, "\\$&")} [0-9a-f]{8}$`));
    expect(seen[0]!.headers["dream-client-instance"]).toBeUndefined();
    await new PolicyClient({ checkpoint: "c", ...urls, fetch, clientVersion: "makermods-console-1.0.8" }).getAction(observation);
    expect(seen[1]!.headers["dream-client-version"]).toBe("makermods-console-1.0.8");
    expect(() => new DreamClient({ clientVersion: "1.0 8" })).toThrow(DreamConfigError);
  });

  it("sends Dream-Client-Instance, not a User-Agent, in proxy mode", async () => {
    const { fetch, seen } = stub();
    const client = new DreamClient({ ...urls, fetch, proxyHeaders: true });
    await client.policy({ checkpoint: "c", strict: false });
    client.setObservation({ "video.top": "Zg==" }, observation, { timestamp: 0 });
    await client.getAction();
    await client.getAction();
    const [first, second] = seen.filter((s) => s.url.endsWith("/v1/get_action"));
    expect(first!.headers["dream-client-instance"]).toMatch(/^[0-9a-f]{8}$/);
    expect(second!.headers["dream-client-instance"]).toBe(first!.headers["dream-client-instance"]);
    expect(first!.headers["dream-client-version"]).toBe(CLIENT_VERSION);
    expect(first!.headers["user-agent"]).toBeUndefined();
    await client.close();
  });

  it("refuses a checkpoint whose robot has parts whatever strict says, when its record is known", async () => {
    const client = new DreamClient({ ...urls, fetch: stub().fetch });
    const model = { id: "dream-1-ft:p-x", owner: "mine", baseModel: "b", embodiment: "emb_p", servable: true, createdAt: null, cameras: null, embodimentConfig: null, partsWithheld: true };
    for (const strict of [true, false]) {
      const error = await client.policy({ checkpoint: model.id, strict, model }).catch((e) => e);
      expect(error, `strict ${strict}`).toBeInstanceOf(DreamConfigError);
      expect(error.detail).toEqual({ kind: "unsupported", key: "checkpoint", row: null });
    }
    const parts = { state: { a: [0, 1] }, action: { a: [0, 1] }, video: { front: "observation.images.front" }, parts: {} };
    await expect(client.policy({ checkpoint: "c", strict: false, embodiment: parts as never })).rejects.toThrow(/does not convert parts/);
  });
});

describe("processing: raw", () => {
  it("returns the checked reply as the model answered it, with no join and no pacing state", async () => {
    let value = 0;
    const { fetch } = stub({ reply: () => okReply(`ses_${(value += 1)}`) });
    let clock = 0;
    const diagnostics: Diagnostics[] = [];
    const client = new PolicyClient({ checkpoint: "c", ...urls, fetch, processing: "raw", now: () => (clock += 900), onDiagnostics: (d) => diagnostics.push(d) });
    for (let i = 0; i < 3; i++) expect(await client.getAction(observation)).toEqual(okReply().actions);
    expect(client.mode).toBe("async");
    expect(diagnostics).toHaveLength(3);
    expect(diagnostics[0]).toMatchObject({ processing: "raw", ik: null, raw: okReply().actions });
    expect(client.lastTiming).toMatchObject({ ikMs: null });
    expect(client.lastTiming!.totalMs).toBeGreaterThan(0);
  });

  it("is refused with ik or the EEF seam join", async () => {
    const client = new DreamClient({ ...urls, fetch: stub().fetch });
    await expect(client.policy({ checkpoint: "c", processing: "raw", ik: { arm: "metal", head: "eef" } })).rejects.toThrow(/'raw'/);
    await expect(client.policy({ checkpoint: "c", processing: "bent" as never })).rejects.toThrow(DreamConfigError);
  });

  it("still refuses a reply holding a non-finite number (JSON's 1e400), naming the key and row", async () => {
    const reply = '{"status":{"code":"DA-0000"},"session_id":"s","actions":{"per_key":{"action.right_arm":[[0,0],[0,1e400]]}}}';
    const client = new PolicyClient({ checkpoint: "c", ...urls, processing: "raw", fetch: async () => new Response(reply) });
    const error = await client.getAction(observation).catch((e) => e);
    expect(error.name).toBe("DreamActionError");
    expect(error.detail).toEqual({ kind: "non_finite", key: "per_key.action.right_arm", row: 1 });
  });
});
