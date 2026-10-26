/**
 * Pacing (`dream.pacing`): when the next chunk is asked for, and sync or async by latency.
 *
 * Three numbers, each a `policy()` config key: `executionHorizon` (30 rows of each chunk run before the next takes
 * over, in async), `actionHz` (20, the loop's row rate) and `syncAboveMs` (700, the round trip past which the client
 * paces synchronously). The mode is the median of the last {@link RTT_WINDOW} round trips against `syncAboveMs`:
 *
 * - async: the next request goes out while the current chunk runs, timed so its reply lands as the arms finish
 *   `executionHorizon` rows of it;
 * - sync: the arms run the whole chunk, and the next request goes out only once they have.
 *
 * The async lead is sized from the shortest recent round trip, the one no queue added to.
 */

import { DreamConfigError, detail } from "./errors.js";
import { median } from "./window.js";

/** Rows of each chunk the arms execute before the next one takes over. */
export const EXECUTION_HORIZON = 30;
/** The loop's row rate, rows per second. */
export const ACTION_HZ = 20.0;
/** Round trip, milliseconds, past which the client paces synchronously. */
export const SYNC_ABOVE_MS = 700.0;
/** Round trips the mode and the async lead are taken from. */
export const RTT_WINDOW = 5;
/** Seconds of queued rows the async path never lets the buffer drop below without a request in flight. */
export const RUNWAY_MARGIN_S = 0.25;

export type PacingMode = "async" | "sync";

/** Measures round trips, picks the mode from them, and answers the question a control loop asks: refill now? */
export class Pacer {
  private modeValue: PacingMode = "async";
  private recentRttS: number[] = [];

  /** @throws DreamConfigError A value is out of range: see {@link DreamClient.policy}'s config keys of the same names. */
  constructor(
    readonly executionHorizon: number = EXECUTION_HORIZON,
    readonly actionHz: number = ACTION_HZ,
    readonly syncAboveMs: number = SYNC_ABOVE_MS,
  ) {
    if (!(Number.isInteger(executionHorizon) && executionHorizon >= 1)) {
      throw new DreamConfigError(`config key 'executionHorizon' must be an int of rows, at least 1, got ${JSON.stringify(executionHorizon)}`, detail("range", "executionHorizon"));
    }
    if (!(typeof actionHz === "number" && actionHz > 0 && Number.isFinite(actionHz))) {
      throw new DreamConfigError(`config key 'actionHz' must be a positive, finite number, got ${JSON.stringify(actionHz)}`, detail("range", "actionHz"));
    }
    if (!(typeof syncAboveMs === "number" && syncAboveMs > 0)) {
      throw new DreamConfigError(`config key 'syncAboveMs' must be a positive number (Infinity: never sync), got ${JSON.stringify(syncAboveMs)}`, detail("range", "syncAboveMs"));
    }
  }

  /** `"async"` or `"sync"`, as the median of the recent round trips decided; `"async"` before the first. */
  get mode(): PacingMode {
    return this.modeValue;
  }

  /** Takes one successful getAction()'s round trip, seconds, and sets the mode from the recent median. */
  record(rttS: number): void {
    this.recentRttS.push(rttS);
    if (this.recentRttS.length > RTT_WINDOW) this.recentRttS.shift();
    this.modeValue = median(this.recentRttS) * 1e3 > this.syncAboveMs ? "sync" : "async";
  }

  /**
   * Whether the next request should go out now, given none is in flight.
   *
   * @param executed Rows of the current chunk the arms have executed since it was installed: for a chunk installed
   *   at its row k (the rows run during the round trip skipped), its row now less k, as robots.control passes it.
   * @param remaining Rows of it still queued.
   */
  refillDue(executed: number, remaining: number): boolean {
    if (remaining <= 0) return true;
    if (this.modeValue === "sync") return false;
    const rttS = this.recentRttS.length ? Math.min(...this.recentRttS) : 0;
    const leadRows = Math.ceil(rttS * this.actionHz);
    return executed + leadRows >= this.executionHorizon || remaining / this.actionHz <= rttS + RUNWAY_MARGIN_S;
  }
}
