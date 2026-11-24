# @makermods/dreamlabs

A TypeScript SDK for Dream Labs' Dream API, so you can run Dream-1 policies from web and Node apps. It is the
TypeScript counterpart of the Python `dreamlabs-client` package (module `dream`), plus the client-side parts of
`dreamlabs`: `dream_ik`'s arm models and chunk solve.

- **Runtime-agnostic.** Browsers, Node 22+, Bun and Deno. The main entry uses standard Web APIs only (`fetch`,
  `AbortController`, `TextEncoder`, `crypto.getRandomValues`, `performance`). It has no runtime dependencies.
- **Python's behaviour, where the tests show it.** The wire protocol, the frame window, the pacing, the seam
  smoothing and the IK are ports of the upstream code, tested against golden fixtures from the Python packages and
  an end-to-end replay of what `dream.DreamClient` sends and receives. See [Fidelity](#fidelity) for what exactly is
  covered, and [Where this differs from Python](#where-this-differs-from-python) for the deliberate departures.
- **Tracks** `dreamlabs` **1.0.42** and `dreamlabs-client` **1.1.7** (`UPSTREAM` in `src/version.ts`).
- **Honest client identity.** Every request names `makermods-dreamlabs-ts-<version>+dream-1.1.7` in
  `Dream-Client-Version` (`CLIENT_VERSION`), deliberately not a semver. See [Client identity](#client-identity).
- **A proxy for browser apps.** `@makermods/dreamlabs/proxy` forwards the client's calls with your key added, as a
  standard fetch handler for Node, Cloudflare Workers, Deno and Bun. See [The proxy](#the-proxy).

> Status: 0.1.0, local and unpublished (`"private": true` in package.json). Remove `private` before publishing.
> The license is TBD (see LICENSE).

## Install

```sh
npm install @makermods/dreamlabs          # once published
npm install /path/to/dreamlabs-ts          # today: from a checkout (run `npm run build` in it first)
```

ESM only. The 0.x public API is five entry points; a sixth holds the internals:

| Import                          | What                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| `@makermods/dreamlabs`          | `DreamClient`, `PolicyClient`, `ObservationWindow`, `Pacer`, `SeamSmoother`, the errors, the types. No Node built-ins. |
| `@makermods/dreamlabs/ik`       | `arm`, `embodiment`, `defineArm`, `solveChunk`, `ClientIK` (FK/IK for the dream_ik arms), `matToRot6`/`rot6ToMat`. Arm data bundled. |
| `@makermods/dreamlabs/node`     | `loadCredentials()` / `createClient()`: `~/.dreamlabs/credentials` and `DREAM_*` variables (the gateway, platform and data service, and the key; with `proxyUrl`, none of them); `nodeHandler()`: a fetch handler on `node:http`. |
| `@makermods/dreamlabs/browser`  | `encodeJpegFrame()`, `bytesToBase64()`, `CONSOLE_FRAME`: camera frames as base64 JPEG.     |
| `@makermods/dreamlabs/proxy`    | `createDreamProxy()`: the server side of a browser app, a fetch handler that adds the key. No Node built-ins. |
| `@makermods/dreamlabs/internal` | Everything else (the wire, `Transport`, the envelope codes, the strict checks, the seam join's helpers, the solver, `Wire`, `parseXml`, ...). **No stability promise**: for tests and experiments. |

`sideEffects: false`, and the IK is imported lazily (a dynamic `import()`) only when a policy names `ik`, so a
bundler splits it out. With rolldown (minified, ESM), the main entry is about 64 KB (two chunks), the IK chunk about
59 KB and the proxy entry about 6 KB; no chunk imports a `node:` module.

## Quick start: Node

```ts
import { DREAM_1_BETA } from "@makermods/dreamlabs";
import { createClient } from "@makermods/dreamlabs/node";

const client = createClient(); // key and URLs from ~/.dreamlabs/credentials, as dream.DreamClient()

// A pair of Metal arms on the YAM-trained base model, with client-side IK (the client adds each hand's pose,
// sends the model what it was trained on, and solves its hand poses back to Metal joints).
await client.policy({ checkpoint: DREAM_1_BETA, embodiment, ik: { arm: "metal", head: "eef" }, actionHz: 20 });

// Every camera tick: one base64 JPEG per camera, and the latest state (joints in radians, grip 0 closed..1 open).
client.setObservation(
  { "video.top": top, "video.wrist_l": wristL, "video.wrist_r": wristR },
  { "state.right_arm": right, "state.right_grip": [0.8], "state.left_arm": left, "state.left_grip": [0.8], prompt: "pick up the cube" },
);

// executed: rows of the current chunk run since it was installed (not its row counted from the observation, which
// counts the rows skipped at the install too); remaining: its rows still queued. As upstream's robots.control.
if (client.refillDue(executed, remaining)) {
  const chunk = await client.getAction({ commanded: { q_r_chunk: lastRight, q_l_chunk: lastLeft } });
  // chunk.q_r_chunk / q_l_chunk: [48][7], the arm's joints (rad) then the grip, already joined onto the last chunk
}
await client.close();
```

See [`examples/node-basic.ts`](examples/node-basic.ts) for the whole loop: refill pacing, takeover, retries, and
the embodiment config.

## Quick start: browser, through your own proxy

Never ship an API key to a browser. Point the page's client at a proxy of yours with one option, and run the proxy
on your server:

```ts
// the page
import { DreamClient } from "@makermods/dreamlabs";
import { encodeJpegFrame } from "@makermods/dreamlabs/browser";

const client = new DreamClient({ proxyUrl: "/api/dream" }); // no apiKey
await client.policy({ checkpoint: "dreamlabs/dream-1-beta", embodiment, ik: { arm: "so101_faas", head: "eef" } });
client.setObservation({ "video.front": await encodeJpegFrame(video), /* ... */ }, observation);
const chunk = await client.getAction();
```

```ts
// the server (a Cloudflare Worker here; see The proxy for Node, Deno and Bun)
import { createDreamProxy } from "@makermods/dreamlabs/proxy";

export default {
  fetch: (request: Request, env: { DREAM_API_KEY: string }) =>
    createDreamProxy({
      apiKey: env.DREAM_API_KEY,
      allowedHosts: "any", // the Worker's routes decide which names reach it
      allowedOrigins: ["https://app.example.com"],
      basePath: "/api/dream",
      authorize: yourSignIn,
    })(request),
};
```

`proxyUrl` is the default of `baseUrl`, `platformUrl` and `dataUrl`, and turns `proxyHeaders` on: a browser cannot
set `User-Agent`, so the client sends `Dream-Client-Version` and `Dream-Client-Instance` (8 random hex digits per
policy), and the proxy builds `User-Agent: dream-client/<version> <instance>` upstream, as the Python client sends it.
[`examples/browser-proxy/`](examples/browser-proxy/) has a Node server and a page.

If your proxy does not forward `/v1/models`, pass the model record yourself (`policy({ ..., model })`) or give
`embodiment` and `strict: false`. `policy()` reads the record to learn what the model was trained on, which `ik`
needs and strict mode uses.

## The proxy

`createDreamProxy(options)` from `@makermods/dreamlabs/proxy` returns a standard fetch handler,
`(request: Request) => Promise<Response>`: `export default { fetch }` on Cloudflare Workers and Bun,
`Deno.serve(handler)`, and on Node 22 `createServer(nodeHandler(handler))` with `nodeHandler` from
`@makermods/dreamlabs/node`. Every option is read and checked when it is created; changing the options object later
changes nothing. It forwards only what the client calls, under `basePath`:

| Request                                                  | Goes to                                     |
| -------------------------------------------------------- | ------------------------------------------- |
| `POST /v1/get_action`, `POST /v1/end_session`             | the gateway (`gatewayUrl`, `DREAM_GATEWAY_URL`) |
| `GET /v1/models`                                          | the platform (`platformUrl`, `DREAM_API_URL`) |
| `GET /v1/fine_tuning/models/<fine-tune>/training_data`    | the data service (`dataUrl`, `DREAM_DATA_URL`), only for a published fine-tune id, `dream-1-ft:<8 of a-z0-9>-<1 to 40 of a-z0-9->`; any other `dream-1-ft:` id is answered `404 not_found` without a lookup |

**Runtimes.** Two platform limits sit in front of the proxy's own `timeoutMs` (70 s), which a valid inference behind a
cold model can use: Bun's server closes a request idle for 10 s by default, so serve it with `Bun.serve({ fetch:
handler, idleTimeout: 0 })` (or above 70); and a Cloudflare Worker sees a client disconnect (the request's `signal`,
which stops the upstream call) only with the `enable_request_signal` compatibility flag, or a compatibility date recent
enough to turn it on by default (Cloudflare's compatibility-flags page names it). On Node, `nodeHandler` aborts the signal itself when the client disconnects.

```ts
createDreamProxy({
  apiKey,              // required: added as Authorization: Bearer on every forwarded request
  allowedHosts,        // required: the Host names answered (421 otherwise), or "any" behind a router (below)
  allowedOrigins?,     // page origins answered (CORS headers and preflight); none by default, no wildcard
  basePath?,           // where it is mounted, e.g. "/api/dream"
  gatewayUrl?, platformUrl?, dataUrl?,  // production by default; reduced to their origin
  clientVersion?,      // the client version sent upstream; CLIENT_VERSION by default
  maxBodyBytes?,       // 8 MiB by default (413 above it, declared or streamed)
  uploadTimeoutMs?,    // 30 s by default for the body to arrive (408 after it)
  maxResponseBytes?,   // 16 MiB by default (502 above it)
  timeoutMs?,          // 70 s by default for the upstream call, answer included (504 after it; 502 unreachable)
  allowedHeaders?,     // extra request headers a page may send (your sign-in header), for the preflight
  authorize?,          // your sign-in, given the request without its body: true | false (401) | a Response
  fetch?,              // the fetch that reaches Dream Labs
})
```

Upstream statuses pass through unchanged (DreamAPI reports every outcome in the body's status envelope; a 204, 205
or 304 has no body), and the answer carries only `Content-Type`, `Cache-Control: no-store` and the CORS headers. **The
answer is parsed and written anew, never passed through**, and only in the shape its route produces: for get_action
and end_session the `status` envelope (`code`, `name`, `reason`, `retryable`, `retry_after_s`, `detail`, `trace_id`),
`session_id`, `actions` and `timing`; for the models list each model's fields the client reads; for training data
`cameras` and `prompts`; for a platform or data-service refusal `{ error: { code, message } }`. An answer that is not
JSON, or not of that shape, is withheld (502, `upstream answer withheld`), and so is one in which any object key or
string carries the key (and so `Bearer <key>`), as written or decoded from `\uXXXX` escapes, percent-encoding, or any
run of base64, base64url or hex, at every alignment, two levels deep. That is the guarantee: the key does not leave in
those forms; a key split across strings or otherwise transformed is not detected. **The client identity sent upstream is the proxy's,
never the caller's:** `Dream-Client-Version: <clientVersion>` and `User-Agent: dream-client/<clientVersion>
<instance>` on the POSTs (the instance from the client's `Dream-Client-Instance`, or the one a non-browser client's
`User-Agent` names), and `User-Agent: dream-client/<clientVersion>` on the GETs. The version decides whether the
gateway serves checkpoints whose robot has parts, so a page must not choose it: set `clientVersion` only on a server
whose clients convert parts. A caller's `Authorization`, cookies and other headers are dropped. A request with an
`Origin` not in `allowedOrigins` is refused (403), and a page served from the proxy's own origin must be listed too.
So is one whose Fetch Metadata says `Sec-Fetch-Site: cross-site` without an allowed `Origin` (a navigation or a no-CORS
load from another site, which carries no `Origin`), and a GET without `Dream-Client-Version` (400): the client always
sends it through a proxy, and another site's page cannot add it to a link, an image or a form.
Redirects are not followed, and the key is never logged or put in an error. A caller that disconnects or cancels
(the request's `signal`) stops the upstream call, and nothing is sent for one that cancelled during `authorize`.

**Host check.** A page served from an attacker's name that resolves to your proxy (DNS rebinding) is same-origin to
it, and its GETs carry no `Origin`; only the `Host` check stops it, so `allowedHosts` is required. List the names the
proxy is reached by (`["127.0.0.1:8787", "localhost:8787"]` for a local one). Pass `"any"` only where something in
front already routes by name: a Cloudflare Worker on its routes or custom domain, a reverse proxy that matches
`server_name` / `Host` and drops the rest. That router is then where the check lives.

**Sign-in.** **Whoever reaches the proxy spends your key:** give it `authorize`, or put your own sign-in in front of
it; the origin and host checks are not a sign-in. `authorize` sees the request's URL, method, headers and signal but
not its body, before anything is read or forwarded; a Response it returns gets the CORS headers of an allowed origin.
From a page on another origin, sign in with a header, not a cookie: send it with the client's `headers` option (e.g.
`{ "X-App-Auth": token }`), list it in `allowedHeaders`, and check it in `authorize`. Credentialed CORS (cookies) is
not supported.

## API overview

### `DreamClient` (`dream.DreamClient`)

```ts
new DreamClient({ apiKey?, baseUrl?, platformUrl?, dataUrl?, proxyUrl?, fetch?, headers?, clientVersion?, proxyHeaders?, onWarning?, onDiagnostics?, now? })
client.getModels({ signal? }): Promise<Model[]>
client.policy(config: PolicyConfig, { signal? }): Promise<void>
client.setObservation(frames, observation, { timestamp? }): void
client.getAction({ commanded?, signal?, execution?: { observationAgeMs?, elapsedExecutionS? } }): Promise<ActionDict>
client.onRoundTrip(fn: (trip: RoundTrip) => unknown): () => void
client.waitForIk(timeoutS): Promise<boolean>
client.resetEpisode(): Promise<void>
client.refillDue(executed, remaining): boolean
client.takeoverAt(action, row, commanded): ActionDict
client.mode / executionHorizon / actionHz / lastTiming / lastDiagnostics / checkpoint / sessionId
client.close(): Promise<void>
```

**Round trips.** `onRoundTrip(fn)` calls `fn` after every getAction() that built its observation from the window
(not before the first setObservation() of an episode), returned or threw, a local refusal before sending included, with a
frozen `RoundTrip`: `sentS` (epoch seconds), `frames` (per frame key, the base64 JPEGs sent, oldest first),
`observation` (every other key sent), `action` (null when it threw), `timing` (`lastTiming` after the call), `error`
(the `DreamError`, null when it returned) and `trained` (below). Callbacks are synchronous: as in Python, what `fn`
throws is thrown out of getAction(); a promise it returns is not waited for, and its rejection goes to `onWarning` as a
`DreamCallbackWarning`. It returns a function that removes that registration (TS-only; idempotent, and a function added twice
stays registered once).

**What a fine-tune was trained on.** For a checkpoint that is one of your fine-tunes (`dream-1-ft:...`), policy()
reads once per checkpoint each camera's training frame sizes and the dataset's tasks
(`GET <dataUrl>/v1/fine_tuning/models/<id>/training_data`); a lookup that fails is one `DreamTrainingWarning` (the
key cut from its message) and never refuses. Each episode's first getAction() then holds its frames and prompt to them: a camera whose aspect is
more than 5% off every training size it had, and a prompt that is none of the tasks (case, spacing and end
punctuation aside; the closest task by difflib's ratio is named), are each one line, warned once a policy through
`onWarning` as a `DreamTrainingWarning` and handed to every round trip of the episode as `trained`. Nothing is
refused.

**`waitForIk(timeoutS)`** resolves true at once with a policy open (DreamConfigError without one): Python compiles
its `traj` solve on a thread for up to 20 s and a robot program waits for it before powering arms that would starve
meanwhile; here there is nothing to compile.

`PolicyConfig` takes the Python keys in camelCase: `checkpoint`, `embodiment`, `strict`, `timeoutMs`, `obsWindow`,
`obsFps`, `executionHorizon`, `actionHz`, `syncAboveMs`, `kinematics`, `robot` and `ik`. Three keys are TS-only:
`model` (the platform's record, which skips the lookup), `processing` and `ikLeadMs` (both below). Any other key is
a `DreamConfigError`.

**Lifecycle and cancellation.** One policy at a time, with a generation bumped synchronously by `close()` and by
each `policy()`: a `policy()` still looking its model up when `close()` or a newer `policy()` runs opens nothing and
throws `DreamCancelled`; a `getAction()` in flight when its policy is closed or replaced is aborted and returns
nothing, even if its reply was already on the way (and even from a custom `fetch` that ignores its signal).
Concurrent `close()` calls share one promise. A `signal` on `getModels()`, `policy()` and `getAction()` cancels
the call (`DreamCancelled`, with the signal's reason as `cause`); it composes with `timeoutMs`, which is still a
`DreamTimeout`. Whichever fires first decides: a cancel the timer then overtakes stays a `DreamCancelled`, and a
reply a custom `fetch` delivers after the timeout is a `DreamTimeout`, never an answer. Cancellation is checked again
at the return: a call aborted or closed from `onDiagnostics` returns nothing and keeps no plan for the next seam.
`close()` during `resetEpisode()` still releases the session, sending the release again on its own signal.

**`processing`.** `"default"`: the client IK, the seam join and the final projection, with the pacing measured.
`"raw"`: the reply as the model answered it, checked (strict mode, finite numbers), with the session and the errors
handled, but no IK, seam join, projection or pacing state (`ik` and `kinematics` are refused with it). For an app
with its own post-processing that adopts the transport first.

**Timing and diagnostics.** `lastTiming` is `{ rttMs, serveInferMs, gatewayMs, networkMs, ikMs, totalMs,
requestBytes, responseBytes }`; the pacing measures the round trip before the IK, as upstream does.
`lastDiagnostics` (and the `onDiagnostics` option, called on every successful `getAction()`) adds the raw reply,
the IK trace per side (`clock`, `speedCap`, `posMiss`, `angMiss`, `atCap`) and the seconds the seam took as in
flight (`inFlightS`, of which `ikLeadS` is the IK lead below). `execution.observationAgeMs` adds to the round trip;
`execution.elapsedExecutionS` replaces it, and the IK lead with it.

**The IK's own time (`ikLeadMs`, TS-only).** A loop keeps playing rows while the client IK solves (220-800 ms in JS
on a Jetson), so a chunk is installed that much later than its reply arrives. Upstream's seam takes the round trip
only (its solve is fast); here, with `ik`, the seam's time in flight also counts the solve: `"auto"` (default) the
largest `ikMs` of the last three solves, a number of milliseconds, or `0` for upstream's behaviour. `"auto"` leaves
out each episode's first solve (cold: no plan to continue, and the first in a fresh process, often several times
slower), which it uses only until a later solve is measured. Overestimating is safe (the old plan is continued a
few rows longer); underestimating made seams jump up to 0.377 rad on an SO-101.

**`takeoverAt(action, row, commanded)` (TS-only).** For an install later than the seam predicted (a slower solve, a
hand-off from a worker): pass the chunk's row due now (rows run since the request) and the joints the loop commanded
last. From `row` on, each arm's rows are run through the final projection again from `commanded`, so the step into
the chunk stays within the arm's limits and speed caps however late it comes (the first chunk of an episode under
`startupSpeedCap`, from the measured pose), and the grip within `gripSpeedCap` when set (from `commanded`'s grip when
it is the written row). When `action` is the latest reply, the next seam continues the re-anchored rows. Every arm's
chunk in `action`, whether `commanded` names the arm or not, must be rows of finite numbers with no holes (a sparse
array's missing row or number is refused like a NaN; for the latest reply, still the shape it was returned in),
before and after the projection; otherwise it is a `DreamActionError` and, as for getAction(), no arm's plan changes. Without `ik`
(no projection) or with `commanded` null, the chunk comes back as it is.

### `PolicyClient` and the rest of the main entry

- `PolicyClient`: one checkpoint's inference client without the window (`dream.client.PolicyClient`). You send it
  observations you built yourself; it takes the same options (`processing`, `proxyHeaders`, `clientVersion`,
  `onDiagnostics`, ...), and `getAction(observation, { commanded?, signal?, execution? })`.
- `ObservationWindow` (`_window.Window`), `Pacer` (`pacing.Pacer`), `SeamSmoother` (`smoothing.SeamSmoother`).
- `DREAM_1_BETA`, `DREAM_1_BETA_FLASH`, `FINE_TUNE_PREFIX`, `VERSION`, `UPSTREAM`, `CLIENT_VERSION`, `REASONS` (per
  envelope code, the reasons the gateway answers it with, as `shared.contracts.envelope` lists them), and the types
  (`Model`, `PolicyConfig`, `ActionDict`, `Commanded`, `Timing`, `Diagnostics`, `RoundTrip`, `TrainedInputs`,
  `GetActionOptions`, `IkConfig`, `DreamReason`, `DreamWarning`, ...).
- Errors: `DreamError`, `DreamConfigError`, `DreamObservationError`, `DreamActionError`, `DreamTimeout`,
  `DreamProtocolError`, `DreamCancelled`, `DreamServerError` (`code`, `reason`, `retryable`, `retryAfterS`,
  `envelope`, `traceId`), `DreamAuthError`, `DreamBadRequest` (which is also `instanceof DreamConfigError`, as in
  Python), `DreamSessionExpired`, and `DreamSlowNetworkWarning` and `DreamTrainingWarning` (passed to `onWarning`,
  never thrown).
  `DreamConfigError` and `DreamActionError` carry `detail: { kind, key, row }` for the checks that name one
  (`kind`: `non_finite`, `shape`, `range`, `degenerate_rotation`, `unsupported`, `model_mismatch`). Transport
  errors keep the underlying error as `cause` (with the key scrubbed from it).

### `@makermods/dreamlabs/ik`

- `arm(name, side?)` / `embodiment(name)`: the built-in arms, which are `dream.client_ik.ARMS` minus `yam`:
  `metal`, `makers`, `so101_faas`, `trossen_wb_faas`, `ai_worker_faas`, `mantis_faas`, `robo_t_faas`,
  `openarm2_faas`, `bracketbot_faas`, `galaxea_faas`, `seeed_faas`, `unitree_r1_faas`. An `Arm` has `tool(q)` (FK,
  float64), `jacobian(q)`, `frames(q)` (for drawing) and `kinematics()` (for the `kinematics` config key: an EEF-space
  seam join). Its `data`, `lo`, `hi` and `free` are frozen.
- `solveChunk(arm, xyz, rot6, qFrom, { hz, speedCap?, serve?, qBefore? })` is `dream_ik.decode.solve_chunk`:
  Savitzky-Golay over the hand path, then the `tick` or `traj` solve under the speed caps, then the seam blend. Bad
  options are a `DreamConfigError`, bad targets (non-finite, 0 or more than 512 rows, a degenerate rotation_6d) a
  `DreamActionError`; its answer is always finite.
- `ClientIK`: what `policy({ ik })` runs (`dream.client_ik.ClientIK`).
- `defineArm(armFile, urdfText)`: your own arm from its URDF and an arm file's keys (`dream.arm_file`). You pass
  the contents, not paths, so it works in a browser. The arm file takes `base`, `flange`, `tcp`, `approach`, `jaw`,
  `rest`, optionally `urdf` (ignored) and `tcp_calibration`, and nothing else; the chain takes fixed joints, and
  revolute and prismatic joints with limits (no continuous joint, no mimic). It refuses a URDF with a loop or a link
  with two parents, non-finite or malformed vectors, limits or velocities, and a range that is empty 2 degrees inside
  the limits.
- `matToRot6` / `rot6ToMat` (null for a degenerate rotation_6d): `tool()`'s matrices to the poses' rotation_6d.

The `ik` config key takes four TS-only keys: `speedCap` (rad/s, every chunk, on top of the arm's caps),
`startupSpeedCap` (an arm's first chunk of an episode; the MakerMods console runs 0.5), `gripSpeedCap` (grip units,
0 closed to 1 open, per second; off by default, as upstream) and `embodimentId` (below).

**The final projection.** With `ik`, the last step on every returned chunk clamps each joint to the arm's margined
limits and keeps each row within one row's speed cap of the row before, starting from `commanded` (else the measured
joints), the step's direction kept. The seam join and the reach hold run after the solve and could otherwise step
past what it was capped to. It clamps the grip to [0, 1] and, with `gripSpeedCap`, keeps each row's grip within one
row's grip cap of the row before, from the grip commanded last (`commanded` as the written row, its grip last), else
the measured `state.<side>_grip`. What is returned is what the next seam continues from.

**Which models run on which arm.** A joint head runs the model's joints as the arm's, so it needs a model trained
on that arm: never a YAM-trained one, and always the `embodimentId` you give (the arm's embodiment id on the
platform), which must be the model's embodiment; an embodiment config, when known, must also answer a `<side>_arm`
action as wide as each side's joints. Matching widths alone are never enough: they cannot tell another robot's joint
order, units or identity from this arm's. An EEF head is
cross-embodiment by design (tool poses, solved here); a fine-tune's embodiment config, when known, must answer
each side's tool pose and grip. Anything else is a `DreamConfigError` with `detail.kind` `model_mismatch`.

## Python → TypeScript

| Python (`dream`, `dream_ik`)                          | TypeScript                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------------ |
| `dream.DreamClient(api_key=None)`                     | `new DreamClient({ apiKey, baseUrl, platformUrl })`; Node: `createClient()` |
| `client.policy({"checkpoint": ..., "timeout_ms": 5000})` | `await client.policy({ checkpoint, timeoutMs: 5000 })`          |
| `client.set_observation(frames, obs, timestamp=t)`    | `client.setObservation(frames, obs, { timestamp: t })`             |
| `client.get_action(commanded=...)`                    | `await client.getAction({ commanded })`                            |
| `client.on_round_trip(fn)`, `dream.RoundTrip(sent_s, frames, observation, action, timing, error, trained)` | `client.onRoundTrip(fn)`, `RoundTrip { sentS, frames, observation, action, timing, error, trained }` |
| `client.wait_for_ik(timeout_s)`                       | `await client.waitForIk(timeoutS)`                                 |
| `client.reset_episode()` / `client.close()`           | `await client.resetEpisode()` / `await client.close()`             |
| `client.refill_due(executed, remaining)`              | `client.refillDue(executed, remaining)`                            |
| `client.mode`, `execution_horizon`, `action_hz`       | `client.mode`, `executionHorizon`, `actionHz`                      |
| `client.last_timing["rtt_ms"]`                        | `client.lastTiming.rttMs` (also `serveInferMs`, `gatewayMs`, `networkMs`, `requestBytes`, `responseBytes`; TS-only `ikMs`, `totalMs`) |
| `client.get_models()` → `Model(id, base_model, ...)`  | `await client.getModels()` → `Model { id, baseModel, ... }`        |
| config `host` / `port` / `scheme` (refused; from credentials) | constructor `baseUrl` (gateway), `platformUrl` (platform), `dataUrl` (data service), or `proxyUrl` for all three |
| config `obs_window`, `obs_fps`, `execution_horizon`, `action_hz`, `sync_above_ms`, `strict`, `robot`, `embodiment`, `kinematics` | `obsWindow`, `obsFps`, `executionHorizon`, `actionHz`, `syncAboveMs`, `strict`, `robot`, `embodiment`, `kinematics` |
| config `ik: {"arm": "metal", "head": "eef", "solve": "traj", "reach": ...}` | `ik: { arm: "metal", head: "eef", solve: "traj", reach }` |
| config `ik: {"arm": "/path/arm.json", ...}`           | `ik: { arm: defineArm(armFile, urdfText), ... }`                   |
| `dream.errors.DreamServerError.retry_after_s`         | `DreamServerError.retryAfterS`                                     |
| `warnings` `DreamSlowNetworkWarning`, the `UserWarning` lines of `dream.trained_inputs` | `onWarning(warning)` option (default `console.warn`): `DreamSlowNetworkWarning`, `DreamTrainingWarning` |
| `dream.trained_inputs.fetch`, `notes`                 | `fetchTrainedInputs`, `trainedNotes` (`/internal`)                 |
| `dream.credentials.load()`                            | `loadCredentials()` (`/node`); `resolveCredentials(env, file)` (`/internal`) elsewhere |
| `dream._window.Window`, `dream.pacing.Pacer`          | `ObservationWindow`, `Pacer`                                       |
| `dream.smoothing.SeamSmoother`                        | `SeamSmoother`                                                     |
| `dream.smoothing.savgol_filter`, `filter_from`, `anchor_row`, ... | `savgolFilter`, `filterFrom`, `anchorRow`, ... (`/internal`) |
| `dream.seam_join.Kinematics`                          | `Kinematics` (type); `velocityBlend`, `eefWindow`, ... (`/internal`) |
| `dream.transport.Transport`, `encode`, `client._error_for` | `Transport`, `encode`, `errorFor` (`/internal`)               |
| `dream_ik.specs.EMBODIMENTS["metal"].side_arm("right")` | `embodiment("metal").sideArm("right")` / `arm("metal")`          |
| `ChunkIK(...).fk(q, side)`                            | `arm(name, side).tool(q)`                                          |
| `dream_ik.decode.solve_chunk(ik, xyz, rot6, q_from, cap, serve, q_before)` | `solveChunk(arm, xyz, rot6, qFrom, { speedCap, hz, serve, qBefore })` (one arm) |
| `dream_ik.specs.ServeSpec`, `SERVE`                   | `ServeSpec` (type, `/ik`); `SERVE` (`/internal`)                   |
| `dream_ik.solver.ChunkIK.solve` / `solve_traj`        | `solve` / `solveTraj` (`/internal`)                                |
| `dream.client_ik.ClientIK`, `_Wire`, `camera_slots`   | `ClientIK` (`/ik`); `Wire`, `cameraSlots`, `clampToReach`, `reachFence` (`/internal`) |
| `dream.arm_file.register(path)`                       | `defineArm(file, urdfText)`                                        |

Observation keys, action keys and wire bodies stay exactly as Python and DreamAPI name them: `state.right_arm`,
`video.overhead`, `q_r_chunk`, `per_key`, `session_id`, and so on.

## The wire, as the Python client speaks it

- `POST <gateway>/v1/get_action` with `{"checkpoint", "session_id" (null mints one), "reset": false,
  "observation"}`, compact JSON. The headers are `Content-Type: application/json`, `Dream-Client-Version:
  <CLIENT_VERSION>`, `Authorization: Bearer <key>` (unless a proxy adds it), and outside browsers `User-Agent:
  dream-client/<CLIENT_VERSION> <8 hex>` (random per policy); with `proxyHeaders`, `Dream-Client-Instance: <8 hex>`
  instead of the `User-Agent`.
- The HTTP status is ignored: the body's `status` envelope decides. `DA-0000` is ok. `DA-0001` is a new session:
  fine on the first call or with reason `new`, otherwise `DreamSessionExpired` (the new session is adopted).
  `DA-1000` is `DreamAuthError`, `DA-4000` is `DreamBadRequest`, and everything else is `DreamServerError` with
  `retryable` / `retryAfterS`.
- A timeout aborts the request, as does your `signal` or closing the client. Redirects are not followed, so the key
  never goes to another host.
- `POST <gateway>/v1/end_session` with `{"checkpoint", "session_id"}` on resetEpisode()/close(). It is best effort,
  waits at most 2 s, and never throws.
- `GET <platform>/v1/models` (policy()'s lookup) and, for one of your fine-tunes, `GET
  <data>/v1/fine_tuning/models/<id>/training_data` (once per checkpoint, not retried; `not_found` means not yours),
  with `Authorization`, `Accept: application/json` and, outside browsers, `User-Agent: dream-client/<CLIENT_VERSION>`
  (with `proxyHeaders`, also `Dream-Client-Version`).
- With `robot` (or `ik` on a model trained on YAM arms), the observation carries `"embodiment": "other_embodiment"`.

## Client identity

`Dream-Client-Version` is `makermods-dreamlabs-ts-<VERSION>+dream-1.1.7` (`CLIENT_VERSION`), not `1.1.7`. The
gateway (per `shared.contracts.envelope.BadRequestReason`, dreamlabs 1.0.42) refuses a checkpoint whose robot has
parts unless the header is a `MAJOR.MINOR.PATCH` version no older than the first library that converts parts; any
other shape passes its ordinary major-version check (a major below the oldest it serves, which is 0, so none yet).
This package does not convert parts, so claiming a semver could get such a checkpoint served to a client that cannot
read it; with this identity the gateway refuses it instead (`DA-4000 client_outdated`). The platform reads the
`User-Agent: dream-client/<version>` of `GET /v1/models` the same way: it lists a body's parts only to a client that
converts them. The MakerMods console names itself `makermods-console-<version>` for the same reason. `clientVersion`
overrides it (an HTTP token, at most 64 characters); claim a semver only if you convert parts.

Independently of `strict`, the client itself refuses what it cannot run whenever it knows: an `embodiment` with
`parts`, a model record (looked up, or `config.model`) whose robot has parts, the `yam` arm, `keep_out`, `region`
and `ServeSpec.fences`. With `strict: false` and neither `embodiment` nor `model`, nothing is looked up, and the
gateway's refusal above is the guard.

## Where this differs from Python

These differences are deliberate:

- **Async.** `policy`, `getAction`, `resetEpisode` and `close` return promises. A second `getAction()` while one is
  in flight throws `DreamError`, where Python's thread lock does the same. `close()` aborts a call in flight (Python
  waits for it); a policy() superseded or closed mid-lookup opens nothing; `DreamCancelled` is TS-only.
- **Numbers are checked.** Every number of a reply must be finite (JSON's `1e400` reads as Infinity), and so must
  every returned row, the IK's targets, its answer and its diagnostics; a rotation_6d with a near-zero or parallel
  column is refused (Python divides by zero and returns NaN). A chunk extended from a single remaining pose holds it
  (Python's `extend_poses` divides by a zero span). `hz`, the speed caps and the horizon are range-checked.
- **The final projection** (with `ik`, see above) is not in Python: Python's seam join after the IK can step a few
  percent past the IK's caps and limits. Against the end-to-end fixtures the returned joints still agree with
  Python's to the tolerance below. It also clamps the grip to [0, 1] (the model's grip reached -0.024 on an SO-101,
  and a loop maps it straight onto the gripper's drive range); Python passes the grip through. `ik.gripSpeedCap`
  (off by default) also rate-limits the grip, which Python never does.
- **The IK lead** (`ikLeadMs`, above): the seam counts the client IK's solve time as run; Python's counts the round
  trip only. `ikLeadMs: 0` restores Python's.
- **`takeoverAt()`** (above) has no Python counterpart; it checks what it is given as getAction() checks a reply.
- **Seam overflow.** A finite reply whose filtered rows overflow (rows alternating near ±1.7e308) is refused as a
  `DreamActionError`, before any arm's plan is kept; Python returns the infinities.
- **Snapshots.** `setObservation()` and `PolicyClient.getAction()` deep-copy the observation's numbers, so a sensor
  buffer you reuse cannot change what was sent or what the reply is joined onto (a Node `Buffer`'s bytes too, whose own
  `slice()` is a view). Built-in arm data is frozen.
- **Atomic seams.** The smoother prepares and checks every arm before it keeps any arm's plan.
- **Strict URDF.** `defineArm()` reads the URDF with a strict XML tokenizer: an unquoted attribute, an unclosed or
  mismatched tag, or anything left over is a `DreamConfigError` with its line and column, where a lenient reader
  would drop the element and quietly change the arm's geometry. Stricter than XML only for a bare `>` in text and a
  DOCTYPE's internal subset.
- **Where the services are** comes from the constructor (`baseUrl`, `platformUrl`, `dataUrl`, or `proxyUrl`), not
  the credentials file, because a browser has no file. In Node, `createClient()` reads the file.
- **Training-input lines** list the cameras in the order the episode's frames were first set; Python iterates a set,
  whose order varies from run to run.
- **URDF numbers** are read as Python's `float()` reads them (`0x10` is refused, `1_0` is 10), then must be finite. A
  negative joint velocity is refused; Python takes it as the 3 rad/s cap, as it does 0.
- **Malformed training records** (a camera size that is not a finite positive width and height, a prompt that is not
  a string) are left out of the comparison; Python fails getAction() on them.
- **JSON safety.** `NaN`, `Infinity`, `undefined`, bigints, binary data, `Map`, `Set` and class instances in an
  observation are a `DreamObservationError` naming the key. `JSON.stringify` would silently send `null`, or drop the
  value.
- **dtype checks** apply to typed arrays (`Float32Array` for `float32`, ...). Plain arrays are only shape-checked,
  as Python checks only numpy arrays.
- **`lastTiming`** has no TCP fields (`up_mbps`, `tcp_*`): `fetch` does not expose `TCP_INFO`.
- **The IK runs the `traj` solve from the first chunk.** Python solves with `tick` while JAX compiles `traj` on a
  thread. Here there is nothing to compile.
- **The IK is float64 with central-difference Jacobians.** dream_ik is float32 (pyroki/jaxls) with analytic ones.
  The Levenberg-Marquardt loop follows jaxls (column scaling, step-quality acceptance, its termination tolerances).
  See [Fidelity](#fidelity) for how closely they agree.
- **At a joint limit, the `traj` solve can end apart from dream_ik's.** With a joint at its margined limit, the stiff
  limit cost keeps each step short, and jaxls can stop on its relative cost change (under 1e-5) while the solve is
  still descending. Where it stops then turns on float rounding: on a recorded two-arm SO-101 chunk (the left lift
  at its lower limit) dream_ik's answer stopped at iteration 30 with its gradient at 1.9, 0.0201 rad from the same
  solve run to convergence, and moved by as much between two runs of the same input. This package's solve reaches the
  converged answer there (2.73e-4 rad from it; [Fidelity](#fidelity), `limit_stop.json`), so it differs from
  dream_ik's by up to 0.02 rad on such a chunk. Both answers stay within the margined limits and the speed caps.
- **A failed `policy()`** leaves no policy open once its model lookup has succeeded, as in Python: the previous
  policy is closed before the new one is built.
- `setObservation()` without a timestamp uses `performance.now()`.

## Not ported (gaps)

- **Recording** (`dream._recorder`). Python records every `set_observation()` tick to the platform's live service
  in the background by default. This package records nothing.
- **Live View** (`dream.LiveViewClient`, `dream.live`). It needs on-device face blurring and VP8 encoding before
  any frame leaves the robot. Port it only with both.
- **The page of what the model sees and the checks** (`dream.DebugView`, `python -m dream.check`, `check_loop`,
  `check_model`, `check_arm`, `dream.input_check` and the reminder policy() prints until a person records that check).
  They serve a local page or a local stand-in for the gateway, read recordings from disk or move the arms.
  `onRoundTrip()` hands an app the same round trips `DebugView` draws. The `x-dream-*` headers a client sends a check
  server on 127.0.0.1 are not sent either.
- **Datasets, embodiments and fine-tuning** (`add_embodiment`, `upload_dataset`, `validate_dataset`,
  `get_datasets`, `finetune` and its `steps`, `resume_from` and `base_share`, ...), and `delete_dataset`'s terminal
  prompt. Only `getModels()` and the training-input lookup are here.
- **Embodiment configs with `parts`** (a body whose fine-tune trains other slots than its own parts) are refused
  with `DreamConfigError`, as are models whose parts the platform withholds. The part conversion
  (`shared.datasets.conversion`) is not ported.
- **IK:** the `yam` arm, `keep_out` obstacles, `region`, the two-arm cost and the fences (`ServeSpec.fences`) need
  the collision-sphere costs, which are not ported. They are refused with `DreamConfigError`. The arm file's
  tool-point warning (no `dreamcli tcp` record in `tcp_calibration`, or one older than the file) is not ported:
  `defineArm()` takes the file's contents, not its path. The zero-shot simulator robots in `dream_ik.specs` are left
  out (they are sim-only).
- **Logging** (`dream.logs`): none.

## Fidelity

What the tests hold this package to, and no more. Every number below is measured from the committed fixtures by
`npm run fidelity` (`scripts/fidelity.ts`), which prints this table; the tests take their ceilings from the same file,
and `test/fidelity.test.ts` fails when this table is not what the script prints. Each ceiling is a margin above the
measured value. Lengths are tool (TCP) positions; angles are joint differences or rotation angles.

- **FK**: every built-in arm and side (12 arms, 24 arm-sides) against float64 forward kinematics of its URDF
  (yourdfpy), and against dream_ik's own float32 FK. The Jacobian is checked against finite differences on three
  arms.
- **IK** (`solveChunk` vs `dream_ik.decode.solve_chunk`, `test/ik.test.ts`): 7 arms (`metal`, `makers`,
  `so101_faas`, `trossen_wb_faas`, `bracketbot_faas`, `seeed_faas`, `ai_worker_faas`), 30 cases of tick or traj
  (`metal`, `makers` and `so101_faas` both, `seeed_faas` tick only, the other three traj only), each without a speed
  cap and with a 0.5 rad/s one. Per case: the commanded tool path, the joints, the rotation reached, and our
  orientation miss of the target row by row against dream_ik's own miss (dream_ik misses the target rotation itself:
  it weighs orientation 10 to position's 50). The one wide exception is `ai_worker_faas`'s traj solve: it has seven
  joints, a null space the two optimizers (pyroki/jaxls float32, ours float64) settle in differently, so the joints
  differ by up to 0.0712 rad uncapped and 0.0653 rad capped; uncapped the tool paths still agree to 0.143 mm, but
  capped the different joints move the hand differently, by up to 6.71 mm. `bracketbot_faas`, seven joints too,
  is held to the same ceilings as the other arms (its capped tool path within 0.131 mm). The five-joint `so101_faas` cannot reach
  every orientation, so its own orientation miss has no ceiling. The other five built-in arms (`mantis_faas`,
  `robo_t_faas`, `openarm2_faas`, `galaxea_faas`, `unitree_r1_faas`) have FK fixtures but no IK fixture.
- **IK invariants** (`test/invariants.test.ts`, on Metal): a reachable path from rest (a 0.15 rad swing of every
  joint, 24 rows and 2, no smoothing) is tracked in position and orientation; the traj solve trades tracking for
  smoothness, so it trails by up to 8.01 mm. Near a limit and for an unreachable target the answer stays finite,
  within the margined limits and the speed caps; one- and two-row horizons solve; a one-row remainder after a seam
  holds; and every returned row of a `DreamClient` run stays within the caps and limits across seams, the next seam
  continuing from what was returned.
- **IK at a joint limit** (`test/ik.test.ts`, `limit_stop.json`): a two-arm SO-101 chunk recorded on hardware with
  `dreamlabs` 1.0.17 / `dreamlabs-client` 1.0.8, the left lift at its margined lower limit, where dream_ik's own traj
  solve stops early (see [Where this differs from Python](#where-this-differs-from-python)). Both arms are held to the
  same solve run to convergence in float64 (2.73e-4 rad), which `npm run limit-stop` replays offline against the
  installed `dreamlabs` (1.0.42: within 2.2e-14 rad of the recording); dream_ik's own answer, 0.0201 rad from it, is
  kept as recorded, since where it stops turns on float rounding.
- **End to end** (`test/wire.test.ts`): the requests of `dream.DreamClient` 1.1.7 replayed: path, headers (all but
  `Dream-Client-Version` and the `User-Agent`'s version, by design), and every number of the JSON body (Python's tool
  poses are float32); the training-data lookups; its returned actions, and where client IK runs, the arms' joints
  (dream_ik's float32 solve against ours); every round trip it handed `on_round_trip()` (its frames, observation,
  timing, error and lines, and that its action is the one returned) and every training-input line it warned; the
  models listings per call (one where Python makes one or more: this client reuses one listing within a policy()).
- **Training inputs** (`test/trained-inputs.test.ts`): difflib's ratio and closest match (ties by code point), the
  prompt as compared (`str.casefold()` for every code point), the base64 decode and JPEG frame size, and `notes()`
  against `dream.trained_inputs` on drawn prompts and frames, exactly.
- **The rest** (window, pacing, smoothing and seam join helpers, `client_ik` helpers) against the Python functions'
  own answers on seeded inputs (`scripts/gen_fixtures.py`).

The last three rows are not this package against Python: they are what `npm run fixtures:check` last recorded in
`test/fixtures/manifest.json`, how far dream_ik's own answers move without the jaxls sort patch the fixtures are
generated with (see [Development](#development)).

<!-- fidelity:start (npm run fidelity) -->
| Quantity | Conditions | Measured | Ceiling |
| --- | --- | ---: | ---: |
| FK position vs yourdfpy (float64) | 24 arm-sides | 2.22e-16 m | 1.00e-12 m |
| FK rotation entries vs yourdfpy | 24 arm-sides | 6.66e-16 | 1.00e-12 |
| FK position vs dream_ik (float32) | 24 arm-sides | 0.200 µm | 1.00 µm |
| Jacobian vs finite differences | metal, bracketbot_faas, ai_worker_faas | 2.02e-10 | 1.00e-8 |
| IK tool path vs dream_ik | tick, every case | 16.8 µm | 50.0 µm |
| IK joints vs dream_ik | tick, every case | 1.75e-4 rad | 5.00e-4 rad |
| IK tool path vs dream_ik | traj, all but ai_worker_faas capped | 0.143 mm | 0.300 mm |
| IK joints vs dream_ik | traj, all but ai_worker_faas | 1.79e-3 rad | 5.00e-3 rad |
| IK tool path vs dream_ik | bracketbot_faas traj, uncapped | 4.34 µm | 0.300 mm |
| IK tool path vs dream_ik | bracketbot_faas traj, capped 0.5 rad/s | 0.131 mm | 0.300 mm |
| IK joints vs dream_ik | bracketbot_faas traj, uncapped | 1.79e-3 rad | 5.00e-3 rad |
| IK joints vs dream_ik | bracketbot_faas traj, capped 0.5 rad/s | 3.21e-4 rad | 5.00e-3 rad |
| IK tool path vs dream_ik | ai_worker_faas traj, uncapped | 0.143 mm | 0.300 mm |
| IK tool path vs dream_ik | ai_worker_faas traj, capped 0.5 rad/s | 6.71 mm | 10.0 mm |
| IK joints vs dream_ik | ai_worker_faas traj, uncapped | 0.0712 rad | 0.100 rad |
| IK joints vs dream_ik | ai_worker_faas traj, capped 0.5 rad/s | 0.0653 rad | 0.100 rad |
| IK orientation miss, ours less dream_ik's | every case, row by row | 4.68e-4 rad | 1.00e-3 rad |
| IK orientation miss, ours less dream_ik's | five and six joints | 8.74e-5 rad | 1.00e-3 rad |
| IK rotation reached vs dream_ik's | five and six joints | 8.79e-5 rad | 2.00e-4 rad |
| IK rotation reached vs dream_ik's | seven joints, all but ai_worker_faas capped traj | 7.34e-4 rad | 2.00e-3 rad |
| IK rotation reached vs dream_ik's | ai_worker_faas capped traj | 0.0404 rad | 0.100 rad |
| IK orientation miss of the target | six joints | 0.148 rad | 0.200 rad |
| IK orientation miss of the target | seven joints | 0.397 rad | 0.500 rad |
| IK joints at a limit vs dream_ik converged | so101_faas traj, recorded, 2 arms | 2.73e-4 rad | 2.00e-3 rad |
| IK joints at a limit, dream_ik's vs converged | same, dream_ik's own stop | 0.0201 rad | - |
| Metal tracking a reachable path | tick, position | 63.4 µm | 0.200 mm |
| Metal tracking a reachable path | tick, orientation | 5.68e-4 rad | 2.00e-3 rad |
| Metal tracking a reachable path | traj, position | 8.01 mm | 12.0 mm |
| Metal tracking a reachable path | traj, orientation | 0.0306 rad | 0.0400 rad |
| Wire: request bodies vs Python's | 4 scenarios, every number | 4.05e-7 | 1.00e-6 |
| Wire: returned actions vs Python's | no client IK | 1.22e-15 | 1.00e-12 |
| Wire: returned joints vs Python's | client IK, worst of 16 chunks | 2.82e-4 rad | 1.00e-3 rad |
| Wire: returned joints vs Python's | client IK, median chunk | 1.36e-4 rad | - |
| fixtures:check (recorded), joints | jaxls unpatched vs patched | 1.63e-3 rad | 0.0100 rad |
| fixtures:check (recorded), tool | same, all but seven-joint capped traj | 16.3 µm | 0.250 mm |
| fixtures:check (recorded), tool | same, seven-joint capped traj | 0.232 mm | 2.00 mm |
<!-- fidelity:end -->

## Development

```sh
npm install
npm run typecheck     # tsc on src, tests, examples; then src without Node types
npm test              # vitest
npm run build         # tsc -> dist/ with .d.ts and source maps
npm run arms          # regenerate src/ik/arms.data.ts from dream_ik
npm run limit-stop    # replay limit_stop.json's converged solve against the installed dreamlabs, offline
npm run unicode       # regenerate src/unicode.data.ts, Python's casefold() and isprintable() tables
npm run fixtures      # regenerate test/fixtures/*.json from the Python packages (a few minutes; JAX compiles)
npm run fixtures:check # bound what the jaxls sort patch changes in ik.json (a few minutes)
```

The Python scripts run with `$DREAM_PYTHON`, default `~/metal-arm-sdk/.venv/bin/python`: an environment with
`dreamlabs` 1.0.42 and `dreamlabs-client` 1.1.7 installed. They read nothing from `~/.dreamlabs` and reach no network:
the wire fixture runs the client against a local server, with the credentials pointed at it.

`scripts/gen_fixtures.py` writes the golden fixtures. Each generator draws from its own RNG, seeded from one base
seed and its name, and runs in its own process with XLA and BLAS on one thread (and jaxls's cost order made
independent of memory addresses), so a full run and `--only <name>` write byte-identical files. A non-finite number
is an error, never a `null`. Both scripts record `test/fixtures/manifest.json`, per written file (so a partial
rerun never misdescribes the others): its sha256, seed and URDFs' sha256, and its own environment, the upstream
wheels' and installed RECORDs' sha256 (wheels from `$DREAM_WHEELS`, default `../dreamlabs-versions/dl`), the Python,
numpy, scipy, jax and yourdfpy versions (jaxls and pyroki are vendored in dreamlabs), the thread and platform
settings in force, and whether jaxls's sort patch was applied. `npm run fixtures:check` (`--check-unpatched`)
re-solves every `ik.json` case with and without that patch, each in a fresh process: with it the answers must be
exact, and without it they must stay within recorded bounds, showing the patch only changes floating-point order:
1e-2 rad in the joints and 0.25 mm of tool position (recorded 1.63e-3 rad and 16.3 µm), except a seven-joint arm's
capped traj solve (`ai_worker_faas`, `bracketbot_faas`), whose tool position may move 2 mm along the arm's null space
(recorded 0.232 mm). That is dream_ik against itself, patched or not; how far this package's solve is from dream_ik's
is the Fidelity table above (6.71 mm at most, on `ai_worker_faas` capped). `ArmName` is generated from the arm keys. Each one is the upstream code's own answer, computed with a
fixed seed: the window, pacing, smoothing and seam join, FK of every built-in arm (float64 through yourdfpy,
cross-checked against dream_ik's own), `solve_chunk` on seven arms in tick and traj, `client_ik`'s helpers,
`trained_inputs`' helpers and `wire.json`. `wire.json` comes from driving the real `dream.DreamClient` against a local
HTTP server, recording every request it sent, every answer it returned, every round trip and every training-input
warning. Two things a policy does on the computer rather than on the wire are turned off in memory there: the
`x-dream-*` headers for a check server on 127.0.0.1 (where the recorder listens) and the input-check reminder.
`test/wire.test.ts` replays that against a Node mock server, and checks that this client sends the same requests
(path, headers, body), returns the same actions, hands over the same round trips and warns the same lines.
`test/fixtures/metal.urdf` is dream_ik's `metal.urdf`, used to test `defineArm`.
