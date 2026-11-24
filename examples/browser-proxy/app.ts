/**
 * The page's side of the proxy example: the webcam as the scene camera, and the chunks Dream-1 answers for a pair of
 * SO-101 arms, printed rather than sent to motors. No key here: the proxy (server.ts) adds it.
 *
 * Bundle it for the page, e.g. `npx esbuild examples/browser-proxy/app.ts --bundle --format=esm
 * --outfile=examples/browser-proxy/app.js`.
 */

import { DREAM_1_BETA, DreamClient, type Model } from "@makermods/dreamlabs";
import { encodeJpegFrame } from "@makermods/dreamlabs/browser";

const log = (text: string) => (document.querySelector("#log")!.textContent = text);

// Every call goes to the proxy (server.ts), which adds the key; the page sends Dream-Client-Version and
// Dream-Client-Instance, and the proxy builds the User-Agent the page cannot set.
const client = new DreamClient({ proxyUrl: "/api/dream" });

/** The body the cameras and arms are, for the checks and for sending the YAM-trained model its camera slots. */
const parts = { arm: 6, gripper: 1, eef_xyz: 3, eef_rot6: 6, grip: 1 };
const spans: Record<string, [number, number]> = {};
let at = 0;
for (const side of ["right", "left"]) for (const [part, width] of Object.entries(parts)) spans[`${side}_${part}`] = [at, (at += width)];
const embodiment = { state: spans, action: spans, video: { front: "observation.images.front", wrist_l: "observation.images.wrist_l", wrist_r: "observation.images.wrist_r" } };

async function main(): Promise<void> {
  const video = document.querySelector("video")!;
  video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 } });
  await video.play();

  // The platform's record of the model, through the proxy: what it was trained on decides how the arms are sent.
  const models: Model[] = await client.getModels();
  log(`models: ${models.map((m) => m.id).join(", ")}`);
  await client.policy({ checkpoint: DREAM_1_BETA, embodiment, ik: { arm: "so101_faas", head: "eef" }, actionHz: 30 });

  // Stand-ins for the arms: five joints each, radians, at the SO-101's rest pose.
  const right = [-0.3722, -1.6763, 1.459, 1.0564, -1.4099];
  const left = [0.3409, -1.7104, 1.5285, 1.1724, -1.4119];
  let rows: number[][] = [];
  let executed = 0;
  for (let tick = 0; tick < 30 * 10; tick++) {
    const frame = await encodeJpegFrame(video);
    client.setObservation(
      { "video.front": frame, "video.wrist_l": frame, "video.wrist_r": frame },
      { "state.right_arm": right, "state.right_grip": [1], "state.left_arm": left, "state.left_grip": [1], prompt: "wave" },
    );
    if (client.refillDue(executed, rows.length - executed)) {
      // Awaited here for brevity (the arms would hold meanwhile): a real loop keeps playing rows while it waits.
      rows = (await client.getAction()).q_r_chunk!;
      executed = 0;
    }
    const row = rows[executed++];
    if (row) log(`${client.mode}: row ${executed} of ${rows.length}, right arm ${row.map((v) => v.toFixed(3)).join(" ")}`);
    await new Promise((resolve) => setTimeout(resolve, 1000 / 30));
  }
  await client.close();
}

main().catch((error: Error) => log(`${error.name}: ${error.message}`));
