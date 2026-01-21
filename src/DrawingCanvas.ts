/**
 * A simple pointer-driven sketch pad
 * It records the drawing as strokes (lists of [x, y] points in CSS pixels)
 */
import type { Point, Stroke } from "./shapes.ts";

export class DrawingCanvas {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private onChange?: (strokes: Stroke[]) => void;
  strokes: Stroke[] = [];
  private drawing = false;

  constructor(canvas: HTMLCanvasElement, { onChange }: { onChange?: (strokes: Stroke[]) => void } = {}) {
    this.canvas = canvas;
    this.onChange = onChange;
    this.ctx = canvas.getContext("2d")!;
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);

    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture(e.pointerId);
      this.drawing = true;
      this.strokes.push([this.point(e)]);
      this.redraw();
      this.onChange?.(this.strokes);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!this.drawing) return;
      const stroke = this.strokes[this.strokes.length - 1];
      const p = this.point(e);
      const last = stroke[stroke.length - 1];
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) < 2) return;
      stroke.push(p);
      this.redraw();
      this.onChange?.(this.strokes);
    });
    const end = () => {
      this.drawing = false;
    };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
  }

  /** Canvas size in CSS pixels (the coordinate space of the strokes) */
  get size(): number {
    return this.canvas.clientWidth;
  }

  private point(e: PointerEvent): Point {
    const rect = this.canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  }

  resize(): void {
    const dpr = Math.min(window.devicePixelRatio, 2);
    this.canvas.width = this.canvas.clientWidth * dpr;
    this.canvas.height = this.canvas.clientHeight * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.redraw();
  }

  clear(): void {
    this.strokes = [];
    this.redraw();
    this.onChange?.(this.strokes);
  }

  /** Replace the drawing with strokes given in a `extent`×`extent` coordinate space */
  setStrokes(strokes: Stroke[], extent: number): void {
    const scale = this.size / extent;
    this.strokes = strokes.map((stroke) => stroke.map(([x, y]): Point => [x * scale, y * scale]));
    this.redraw();
    this.onChange?.(this.strokes);
  }

  private redraw(): void {
    const ctx = this.ctx;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "#f2f6ff";
    ctx.lineWidth = Math.max(6, w * 0.028);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const stroke of this.strokes) {
      ctx.beginPath();
      stroke.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      if (stroke.length === 1) ctx.lineTo(stroke[0][0] + 0.1, stroke[0][1]);
      ctx.stroke();
    }
  }
}

/** Paint a GRID×GRID feature vector as a pixel preview */
export function drawPixels(canvas: HTMLCanvasElement, pixels: Float32Array, grid: number): void {
  const ctx = canvas.getContext("2d")!;
  const cell = canvas.width / grid;
  ctx.fillStyle = "#0b1226";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < pixels.length; i++) {
    const v = pixels[i];
    ctx.fillStyle = `rgba(77, 163, 255, ${0.08 + 0.92 * v})`;
    ctx.fillRect((i % grid) * cell + 1, Math.floor(i / grid) * cell + 1, cell - 2, cell - 2);
  }
}
