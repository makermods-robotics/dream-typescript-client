/**
 * Run Dream-1 on a pair of Metal arms from Node, with client-side IK, as the MakerMods console does.
 *
 *   node examples/node-basic.ts            (Node 23.6+ runs TypeScript directly; or `npx tsx`)
 *
 * The key and the URLs come from ~/.dreamlabs/credentials and DREAM_* variables. The robot here is a stand-in:
 * readArms(), readCameras() and command() are where your own hardware code goes. Nothing in this file moves an arm.
 */

import { DreamServerError, DREAM_1_BETA, type ActionDict } from "@makermods/dreamlabs";
import { createClient } from "@makermods/dreamlabs/node";

const HZ = 20;
const PROMPT = "pick up the cube";

// ---- your robot (stand-ins) ----
const joints = { right: [0, -1.2, 1.5, 0, 0, 0], left: [0, -1.2, 1.5, 0, 0, 0] };
const readArms = () => ({ right: [...joints.right], left: [...joints.left], grip: { right: 0.8, left: 0.8 } });
const readCameras = (): Record<string, string> => ({ top: "", wrist_l: "", wrist_r: "" }); // base64 JPEGs, 320x176
const command = (side: "right" | "left", row: number[]) => {
  joints[side] = row.slice(0, 6); // a real loop writes the joints (radians) and the grip (row[6], 0..1) to the motors
};

/** The body as Dream's platform knows it: each arm's joints, gripper and tool pose, and the three cameras. */
const embodiment = (() => {
  const parts = { arm: 6, gripper: 1, eef_xyz: 3, eef_rot6: 6, grip: 1 };
  const spans: Record<string, [number, number]> = {};
  let at = 0;
  for (const side of ["right", "left"]) for (const [part, width] of Object.entries(parts)) spans[`${side}_${part}`] = [at, (at += width)];
  return { state: spans, action: spans, video: { top: "observation.images.top", wrist_l: "observation.images.wrist_l", wrist_r: "observation.images.wrist_r" } };
})();

const client = createClient();
await client.policy({ checkpoint: DREAM_1_BETA, embodiment, ik: { arm: "metal", head: "eef" }, actionHz: HZ });

let chunk = null as ActionDict | null; // set from the request callback below
let row = 0; // the chunk's row to command next, counted from its observation
let installedAt = 0; // the row it was installed at: refillDue() takes the rows run since then, as upstream's loop does
let pending: Promise<ActionDict> | null = null;
let failure: unknown = null;
let ranSinceRequest = 0;
const last: Record<string, number[]> = {};

for (let tick = 0; tick < 20 * HZ; tick++) {
  if (failure) throw failure;
  const arms = readArms();
  const cameras = readCameras();
  client.setObservation(
    { "video.top": cameras.top!, "video.wrist_l": cameras.wrist_l!, "video.wrist_r": cameras.wrist_r! },
    {
      // Joints in radians and the grip from 0 closed to 1 open; the client adds each hand's pose (FK) and, for a model
      // trained on YAM arms such as Dream-1 Beta, sends zeros for the joints and the grip as the gripper.
      "state.right_arm": arms.right, "state.right_grip": [arms.grip.right],
      "state.left_arm": arms.left, "state.left_grip": [arms.grip.left],
      prompt: PROMPT,
    },
  );
  const remaining = chunk ? (chunk.q_r_chunk as number[][]).length - row : 0;
  if (!pending && client.refillDue(row - installedAt, remaining)) {
    ranSinceRequest = 0;
    pending = client.getAction({ commanded: Object.keys(last).length ? last : null });
    void pending.then(
      // The new chunk takes over at the row due now, the rows run since the request (the round trip and the IK, which
      // the client leads by): the rows before it continue the old plan. takeoverAt() re-anchors it there from the last
      // command, so a later install than the client predicted steps in within the arm's speed caps instead of jumping.
      (next) => {
        const commanded = Object.keys(last).length ? last : null;
        chunk = client.takeoverAt(next, ranSinceRequest, commanded);
        row = installedAt = Math.min(ranSinceRequest, (chunk.q_r_chunk as number[][]).length);
        pending = null;
      },
      (error) => {
        pending = null; // a retryable refusal (pool_warming, rate_limited) is asked again on a later tick
        if (!(error instanceof DreamServerError && error.retryable)) failure = error;
      },
    );
  }
  if (chunk && row < (chunk.q_r_chunk as number[][]).length) {
    command("right", (last.q_r_chunk = (chunk.q_r_chunk as number[][])[row]!));
    command("left", (last.q_l_chunk = (chunk.q_l_chunk as number[][])[row]!));
    row++;
    ranSinceRequest++;
  }
  await new Promise((resolve) => setTimeout(resolve, 1000 / HZ));
}

console.log("pacing", client.mode, "last round trip", client.lastTiming);
await client.close();
