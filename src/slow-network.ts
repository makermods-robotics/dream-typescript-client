/**
 * The slow-network warning (`dream.slow_network`): one {@link DreamSlowNetworkWarning} when a client's recent round
 * trips stay slow. After every successful getAction() the client hands its `lastTiming` to its {@link Watch}; when the
 * median of the last {@link WINDOW} round trips is at or above {@link SLOW_MS} the watch words one warning, then stays
 * quiet for {@link QUIET_S} while they stay slow, unless their median first recovers under {@link RECOVERED_MS}.
 */

import { roundHalfEven } from "./linalg.js";
import { median } from "./window.js";
import type { Timing } from "./client.js";

export const SLOW_MS = 1000.0;
export const RECOVERED_MS = 800.0;
export const WINDOW = 5;
export const QUIET_S = 300.0;

/** One client's last round trips, and whether a slow median among them is worth a warning now. */
export class Watch {
  isArmed = true;
  warnedAtS: number | null = null;
  private recent: Timing[] = [];

  /** Takes one successful getAction()'s timing; returns the warning's text when one is due now. */
  record(timing: Timing | null, nowS: number): string | null {
    if (timing === null) return null;
    this.recent.push(timing);
    if (this.recent.length > WINDOW) this.recent.shift();
    if (this.recent.length < WINDOW) return null;
    const rttMs = median(this.recent.map((t) => t.rttMs));
    if (rttMs < RECOVERED_MS) {
      this.isArmed = true;
      return null;
    }
    if (rttMs < SLOW_MS) return null;
    if (!this.isArmed && this.warnedAtS !== null && nowS - this.warnedAtS < QUIET_S) return null;
    this.isArmed = false;
    this.warnedAtS = nowS;
    return message(this.recent);
  }
}

const about = (ms: number) => roundHalfEven(ms / 10) * 10;
const medianOf = (values: (number | null)[]): number | null => {
  const known = values.filter((v): v is number => typeof v === "number");
  return known.length ? median(known) : null;
};

/** Words the warning: the network when its median part is larger than each other part's, the model otherwise. */
export function message(timings: readonly Timing[]): string {
  const rttMs = median(timings.map((t) => t.rttMs));
  const networkMs = medianOf(timings.map((t) => t.networkMs));
  const oursMs = medianOf(timings.map((t) => t.gatewayMs));
  const modelMs = medianOf(timings.map((t) => t.serveInferMs));
  const serversMs = medianOf(timings.map((t) => (t.gatewayMs !== null && t.serveInferMs !== null ? t.gatewayMs - t.serveInferMs : null)));
  const smooth = "the robot's motion may be less smooth. See client.lastTiming.";
  if (networkMs === null) return `get_action round trips are slow: about ${about(rttMs)} ms; ${smooth}`;
  if ([serversMs, modelMs].every((part) => part === null || networkMs > part)) {
    return `Network is slow: get_action round trips are about ${about(rttMs)} ms (network ${about(networkMs)} ms); ${smooth}`;
  }
  const ours = oursMs !== null ? `the model and our servers ${about(oursMs)} ms, ` : "";
  return `The model is responding slowly: get_action round trips are about ${about(rttMs)} ms (${ours}network ${about(networkMs)} ms); ${smooth}`;
}
