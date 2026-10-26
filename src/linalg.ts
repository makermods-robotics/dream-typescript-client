/** Small dense linear algebra and rotations, float64, for the seam join and the IK. Matrices are row-major. */

export type Vec3 = [number, number, number];
/** A 3x3 rotation, row-major: `[r00, r01, r02, r10, ..., r22]`. */
export type Mat3 = [number, number, number, number, number, number, number, number, number];

export const IDENTITY3: Readonly<Mat3> = [1, 0, 0, 0, 1, 0, 0, 0, 1];

export function mul3(a: Readonly<Mat3>, b: Readonly<Mat3>): Mat3 {
  return [
    a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
    a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
    a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
    a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
    a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
    a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
    a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
    a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
    a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
  ];
}

export function transpose3(r: Readonly<Mat3>): Mat3 {
  return [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];
}

export function apply3(r: Readonly<Mat3>, v: Readonly<Vec3>): Vec3 {
  return [r[0] * v[0] + r[1] * v[1] + r[2] * v[2], r[3] * v[0] + r[4] * v[1] + r[5] * v[2], r[6] * v[0] + r[7] * v[1] + r[8] * v[2]];
}

export const add3 = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub3 = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale3 = (a: Readonly<Vec3>, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot3 = (a: Readonly<Vec3>, b: Readonly<Vec3>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm3 = (a: Readonly<Vec3>): number => Math.hypot(a[0], a[1], a[2]);
export const cross3 = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

/** The rotation of a rotation vector (axis times angle), by Rodrigues' formula (`seam_join.rotation_matrix`). */
export function rotationMatrix(v: Readonly<Vec3>): Mat3 {
  const angle = norm3(v);
  const [x, y, z] = v;
  const k: Mat3 = [0, -z, y, z, 0, -x, -y, x, 0];
  const k2 = mul3(k, k);
  const s = angle < 1e-8 ? 1 : Math.sin(angle) / angle;
  const c = angle < 1e-8 ? 0.5 : (1 - Math.cos(angle)) / (angle * angle);
  return k.map((value, i) => (i % 4 === 0 ? 1 : 0) + s * value + c * k2[i]!) as Mat3;
}

/** `[w, x, y, z]` of a rotation, by the branch on its largest diagonal term (Shepperd's method). */
export function quaternion(m: Readonly<Mat3>): [number, number, number, number] {
  const trace = m[0] + m[4] + m[8];
  let q: [number, number, number, number];
  if (trace > 0) {
    const s = 2 * Math.sqrt(trace + 1);
    q = [0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s];
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const s = 2 * Math.sqrt(1 + m[0] - m[4] - m[8]);
    q = [(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s];
  } else if (m[4] > m[8]) {
    const s = 2 * Math.sqrt(1 + m[4] - m[0] - m[8]);
    q = [(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s];
  } else {
    const s = 2 * Math.sqrt(1 + m[8] - m[0] - m[4]);
    q = [(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s];
  }
  const n = Math.hypot(...q);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/** The rotation vector (axis times angle, angle in `[0, pi]`) of a rotation, through its unit quaternion. */
export function rotvec(m: Readonly<Mat3>): Vec3 {
  let q = quaternion(m);
  if (q[0] < 0) q = [-q[0], -q[1], -q[2], -q[3]];
  const sine = Math.hypot(q[1], q[2], q[3]);
  const scale = sine < 1e-12 ? 2 : (2 * Math.atan2(sine, q[0])) / sine;
  return [scale * q[1], scale * q[2], scale * q[3]];
}

/** A turn of `angle` about the unit `axis`. */
export function axisAngle(axis: Readonly<Vec3>, angle: number): Mat3 {
  const [x, y, z] = axis;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;
  return [t * x * x + c, t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c, t * y * z - s * x, t * x * z - s * y, t * y * z + s * x, t * z * z + c];
}

/** URDF's fixed-axis roll-pitch-yaw: `Rz(yaw) Ry(pitch) Rx(roll)`. */
export function rpyMatrix([r, p, y]: Readonly<Vec3>): Mat3 {
  const [cr, sr, cp, sp, cy, sy] = [Math.cos(r), Math.sin(r), Math.cos(p), Math.sin(p), Math.cos(y), Math.sin(y)];
  return [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, -sp, cp * sr, cp * cr];
}

/** Below this norm a rotation_6d column (or the second's part across the first) has no direction. */
export const ROT6_MIN_NORM = 1e-6;

/**
 * rotation_6d `[R00, R10, R20, R01, R11, R21]` back onto SO(3) by Gram-Schmidt (`dream_ik.decode.rot6d_to_mat`); null
 * when it names no rotation: a non-finite value, a column of near-zero norm, or two near-parallel columns (where
 * Python divides by zero and answers NaN).
 */
export function rot6ToMat(x: ArrayLike<number>): Mat3 | null {
  for (let i = 0; i < 6; i++) if (!Number.isFinite(x[i])) return null;
  const a1: Vec3 = [x[0]!, x[1]!, x[2]!];
  if (!(norm3(a1) >= ROT6_MIN_NORM)) return null;
  const b1 = scale3(a1, 1 / norm3(a1));
  const a2: Vec3 = [x[3]!, x[4]!, x[5]!];
  const u = sub3(a2, scale3(b1, dot3(b1, a2)));
  if (!(norm3(u) >= ROT6_MIN_NORM)) return null;
  const b2 = scale3(u, 1 / norm3(u));
  const b3 = cross3(b1, b2);
  return [b1[0], b2[0], b3[0], b1[1], b2[1], b3[1], b1[2], b2[2], b3[2]];
}

/** The first two columns of a rotation, `[R00, R10, R20, R01, R11, R21]` (`dream_ik.decode.mat_to_rot6d`). */
export function matToRot6(r: Readonly<Mat3>): [number, number, number, number, number, number] {
  return [r[0], r[3], r[6], r[1], r[4], r[7]];
}

/** Solves `A x = b` by Gaussian elimination with partial pivoting; null when `A` is singular. `A` is n x n. */
export function solveLinear(a: readonly (readonly number[])[], b: readonly number[]): number[] | null {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r]![c]!) > Math.abs(m[p]![c]!)) p = r;
    if (!(Math.abs(m[p]![c]!) > 0)) return null;
    [m[c], m[p]] = [m[p]!, m[c]!];
    for (let r = c + 1; r < n; r++) {
      const f = m[r]![c]! / m[c]![c]!;
      if (f === 0) continue;
      for (let k = c; k <= n; k++) m[r]![k]! -= f * m[c]![k]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = m[r]![n]!;
    for (let k = r + 1; k < n; k++) s -= m[r]![k]! * x[k]!;
    x[r] = s / m[r]![r]!;
  }
  return x;
}

/** The inverse of a small square matrix (Gauss-Jordan with partial pivoting). */
export function invert(m: readonly (readonly number[])[]): number[][] {
  const n = m.length;
  const a = m.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r]![col]!) > Math.abs(a[pivot]![col]!)) pivot = r;
    [a[col], a[pivot]] = [a[pivot]!, a[col]!];
    const p = a[col]![col]!;
    for (let k = 0; k < 2 * n; k++) a[col]![k]! /= p;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = a[r]![col]!;
      if (f === 0) continue;
      for (let k = 0; k < 2 * n; k++) a[r]![k]! -= f * a[col]![k]!;
    }
  }
  return a.map((row) => row.slice(n));
}

/** Python's `round()`: half to even. */
export function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}
