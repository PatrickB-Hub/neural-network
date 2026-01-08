/**
 * Shape data: synthetic "hand-drawn" training shapes + the feature extractor.
 *
 * A drawing is always a list of strokes, each stroke a list of [x, y] points
 * (canvas coordinates, y pointing down). Training samples and the user's own
 * drawing go through the *same* `featurize()` so the network sees identical
 * input statistics in both cases.
 */

export type Point = [number, number];
export type Stroke = Point[];
export type Rng = () => number;

/** One labelled drawing */
export interface Sample {
  strokes: Stroke[];
  extent: number;
  x: Float32Array;
  y: number;
}

export const GRID = 10; // input image is GRID × GRID pixels → 100 input neurons
export const CLASSES = ["Dreieck", "Rechteck", "Kreis"];
export const CLASS_ICONS = ["▲", "■", "●"];

/** German number formatting */
export const num = (v: number, digits = 2): string =>
  v.toLocaleString("de-DE", { minimumFractionDigits: digits, maximumFractionDigits: digits });
export const pct = (p: number, digits = 0): string => `${num(p * 100, digits)} %`;

/** Small seedable PRNG (mulberry32) so datasets are reproducible. */
export function makeRng(seed = 1): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const range = (rng: Rng, lo: number, hi: number): number => lo + (hi - lo) * rng();

/**
 * Walk around a closed polygon and add a low-frequency sideways wobble plus a bit of
 * pen jitter, so edges look hand-drawn. The end either overshoots or stops a little
 * short of the start, like a real pen stroke.
 */
function handDrawnPolygon(verts: Point[], rng: Rng, wobble: number): Point[] {
  const pts: Point[] = [];
  const phase = rng() * Math.PI * 2;
  const freq = range(rng, 0.8, 2.2);
  let travelled = 0;
  const close = range(rng, -0.12, 0.2);
  const edges = verts.length + (close > 0 ? 1 : 0);

  for (let e = 0; e < edges; e++) {
    const a = verts[e % verts.length];
    const b = verts[(e + 1) % verts.length];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len = Math.hypot(dx, dy);
    const nx = -dy / len; // unit normal, for the sideways wobble
    const ny = dx / len;
    const isLast = e === edges - 1;
    const limit = isLast ? (close > 0 ? close : 1 + close) : 1;
    const steps = Math.max(4, Math.round(len * 18));
    for (let s = 0; s < steps * limit; s++) {
      const u = s / steps;
      const off = wobble * Math.sin(phase + (travelled + u * len) * freq * 3) + (rng() - 0.5) * wobble * 0.4;
      pts.push([a[0] + dx * u + nx * off, a[1] + dy * u + ny * off]);
    }
    travelled += len;
  }
  return pts;
}

function rotate(pts: Point[], angle: number): Point[] {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return pts.map(([x, y]) => [x * cos - y * sin, x * sin + y * cos]);
}

function triangleVerts(rng: Rng): Point[] {
  if (rng() < 0.25) {
    // right triangle with random flips
    const w = range(rng, 0.8, 1.2);
    const h = range(rng, 0.8, 1.2);
    const fx = rng() < 0.5 ? 1 : -1;
    const fy = rng() < 0.5 ? 1 : -1;
    return [
      [-w * fx, -h * fy],
      [w * fx, h * fy],
      [-w * fx, h * fy],
    ];
  }
  // apex up (y is down, so -90°), then perturb every corner
  return [-90, 30, 150].map((deg): Point => {
    const a = ((deg + range(rng, -22, 22)) * Math.PI) / 180;
    const r = range(rng, 0.8, 1.15);
    return [Math.cos(a) * r, Math.sin(a) * r];
  });
}

function rectangleVerts(rng: Rng): Point[] {
  const aspect = Math.exp(range(rng, -0.8, 0.8));
  const w = Math.sqrt(aspect);
  const h = 1 / Math.sqrt(aspect);
  const corners: Point[] = [
    [-w, -h],
    [w, -h],
    [w, h],
    [-w, h],
  ];
  // perturb corners slightly (sloppy drawing)
  const out = corners.map(([x, y]): Point => [x + range(rng, -0.08, 0.08), y + range(rng, -0.08, 0.08)]);
  const start = Math.floor(rng() * 4);
  const ordered = [...out.slice(start), ...out.slice(0, start)];
  return rng() < 0.5 ? ordered : ordered.reverse();
}

function circlePoints(rng: Rng): Point[] {
  const aspect = Math.exp(range(rng, -0.35, 0.35));
  const rx = Math.sqrt(aspect);
  const ry = 1 / Math.sqrt(aspect);
  const start = rng() * Math.PI * 2;
  const sweep = Math.PI * 2 + range(rng, -0.35, 0.5);
  const dir = rng() < 0.5 ? 1 : -1;
  // two slow harmonics make the outline slightly lumpy
  const a1 = range(rng, 0, 0.07);
  const a2 = range(rng, 0, 0.05);
  const p1 = rng() * Math.PI * 2;
  const p2 = rng() * Math.PI * 2;
  const drift = range(rng, -0.08, 0.08); // slight spiral
  const n = 64;
  const pts: Point[] = [];
  for (let i = 0; i <= n; i++) {
    const u = i / n;
    const t = start + dir * sweep * u;
    const r = 1 + a1 * Math.sin(2 * t + p1) + a2 * Math.sin(3 * t + p2) + drift * u + (rng() - 0.5) * 0.02;
    pts.push([Math.cos(t) * rx * r, Math.sin(t) * ry * r]);
  }
  return pts;
}

/**
 * Generate one hand-drawn-looking shape (0 = triangle, 1 = rectangle,
 * 2 = circle) inside a canvas
 */
export function generateShape(cls: number, rng: Rng = Math.random, extent = 100): Stroke[] {
  let pts: Point[];
  if (cls === 0) {
    const upright = rng() < 0.65;
    pts = rotate(
      handDrawnPolygon(triangleVerts(rng), rng, range(rng, 0.01, 0.045)),
      upright ? range(rng, -0.25, 0.25) : rng() * Math.PI * 2,
    );
  } else if (cls === 1) {
    const tilt = rng() < 0.75 ? range(rng, -0.14, 0.14) : range(rng, -0.35, 0.35);
    pts = rotate(handDrawnPolygon(rectangleVerts(rng), rng, range(rng, 0.01, 0.04)), tilt);
  } else {
    pts = rotate(circlePoints(rng), rng() * Math.PI * 2);
  }
  // place into the canvas at a random size / position
  const size = range(rng, 0.25, 0.4) * extent;
  const cx = extent / 2 + range(rng, -0.08, 0.08) * extent;
  const cy = extent / 2 + range(rng, -0.08, 0.08) * extent;
  return [pts.map(([x, y]): Point => [cx + x * size, cy + y * size])];
}

/**
 * Turn strokes into GRID×GRID pixel intensities in [0, 1].
 *
 * 1. Crop to the drawing's bounding box and stretch it to fill the grid, so position
 *    and size of the drawing don't matter
 * 2. "Stamp" a round pen along every stroke on a 4× finer grid.
 * 3. Average each 4×4 block → anti-aliased GRID×GRID image.
 */
export function featurize(strokes: Stroke[], grid = GRID): Float32Array {
  const SUB = 4;
  const fine = grid * SUB;
  const hi = new Uint8Array(fine * fine);
  const all = strokes.flat();
  const out = new Float32Array(grid * grid);
  if (all.length === 0) return out;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of all) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const w = maxX - minX;
  const h = maxY - minY;
  const m = Math.max(w, h) || 1;
  const sx = 0.8 / Math.max(w, m / 3); // 0.8 → leaves a 10% margin on each side
  const sy = 0.8 / Math.max(h, m / 3);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const toUnit = ([x, y]: Point): Point => [0.5 + (x - cx) * sx, 0.5 + (y - cy) * sy];

  const radius = 0.055 * fine; // pen radius
  const r2 = radius * radius;
  const stamp = (ux: number, uy: number): void => {
    const fx = ux * fine;
    const fy = uy * fine;
    const x0 = Math.max(0, Math.floor(fx - radius));
    const x1 = Math.min(fine - 1, Math.ceil(fx + radius));
    const y0 = Math.max(0, Math.floor(fy - radius));
    const y1 = Math.min(fine - 1, Math.ceil(fy + radius));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dx = x + 0.5 - fx;
        const dy = y + 0.5 - fy;
        if (dx * dx + dy * dy <= r2) hi[y * fine + x] = 1;
      }
    }
  };

  for (const stroke of strokes) {
    const pts = stroke.map(toUnit);
    if (pts.length === 1) stamp(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) {
      const [ax, ay] = pts[i - 1];
      const [bx, by] = pts[i];
      const steps = Math.max(1, Math.ceil((Math.hypot(bx - ax, by - ay) * fine) / 0.5));
      for (let s = 0; s <= steps; s++) stamp(ax + ((bx - ax) * s) / steps, ay + ((by - ay) * s) / steps);
    }
  }

  let max = 0;
  for (let gy = 0; gy < grid; gy++) {
    for (let gx = 0; gx < grid; gx++) {
      let sum = 0;
      for (let y = 0; y < SUB; y++) {
        for (let x = 0; x < SUB; x++) sum += hi[(gy * SUB + y) * fine + gx * SUB + x];
      }
      const v = sum / (SUB * SUB);
      out[gy * grid + gx] = v;
      if (v > max) max = v;
    }
  }
  if (max > 0) for (let i = 0; i < out.length; i++) out[i] /= max;
  return out;
}

/** Build a shuffled dataset of { strokes, x, y } samples */
export function makeDataset(perClass: number, rng: Rng): Sample[] {
  const data: Sample[] = [];
  for (let i = 0; i < perClass; i++) {
    for (let cls = 0; cls < CLASSES.length; cls++) {
      const strokes = generateShape(cls, rng);
      data.push({ strokes, extent: 100, x: featurize(strokes), y: cls });
    }
  }
  for (let i = data.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [data[i], data[j]] = [data[j], data[i]];
  }
  return data;
}
