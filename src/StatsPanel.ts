/** Epoch progress, accuracy and loss (each with a sparkline) and the sample currently shown */
import { CLASSES, CLASS_ICONS, num, pct, type Sample } from "./shapes.ts";
import type { HistoryPoint, TrainingStats } from "./TrainingController.ts";
import { $ } from "./dom.ts";

interface Spark {
  canvas: HTMLCanvasElement;
  readout: HTMLElement;
  key: "accuracy" | "loss";
  fmt: (v: number) => string;
  hover: number;
}

export class StatsPanel {
  private history: HistoryPoint[] = [];
  private sparks: Spark[];

  constructor() {
    this.sparks = [
      {
        canvas: $<HTMLCanvasElement>("spark-acc"),
        readout: $("spark-acc-readout"),
        key: "accuracy",
        fmt: (v) => pct(v),
        hover: -1,
      },
      {
        canvas: $<HTMLCanvasElement>("spark-loss"),
        readout: $("spark-loss-readout"),
        key: "loss",
        fmt: (v) => num(v, 3),
        hover: -1,
      },
    ];
    for (const spark of this.sparks) {
      spark.canvas.addEventListener("pointermove", (e) => {
        const rect = spark.canvas.getBoundingClientRect();
        const n = this.history.length;
        spark.hover = n ? Math.round(((e.clientX - rect.left) / rect.width) * (n - 1)) : -1;
        this.drawSpark(spark);
      });
      spark.canvas.addEventListener("pointerleave", () => {
        spark.hover = -1;
        this.drawSpark(spark);
      });
    }
  }

  update(stats: TrainingStats, history: HistoryPoint[]): void {
    this.history = history;
    $("stat-epoch").textContent =
      `${num(Math.min(stats.epoch, stats.targetEpochs), 1)} / ${stats.targetEpochs}`;
    $("epoch-bar").style.width = `${Math.min(100, (stats.epoch / stats.targetEpochs) * 100)}%`;
    $("stat-acc").textContent = pct(stats.accuracy, 1);
    $("stat-loss").textContent = stats.loss == null ? "—" : num(stats.loss, 3);
    this.sparks.forEach((spark) => this.drawSpark(spark));
  }

  showSample(sample: Sample): void {
    $("sample-label").innerHTML = `<i>${CLASS_ICONS[sample.y]}</i> ${CLASSES[sample.y]}`;
    $("sample-guess").textContent = "…";
    $("sample-guess").className = "";
  }

  showGuess(sample: Sample, probs: Float32Array, predicted: number): void {
    const ok = predicted === sample.y;
    $("sample-guess").innerHTML =
      `${CLASS_ICONS[predicted]} ${CLASSES[predicted]} · ${pct(probs[predicted])} ${ok ? "✓" : "✗"}`;
    $("sample-guess").className = ok ? "ok" : "bad";
  }

  /** 2px line, recessive baseline, hover crosshair with readout */
  private drawSpark(spark: Spark): void {
    const canvas = spark.canvas;
    const dpr = Math.min(window.devicePixelRatio, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(150,175,255,0.14)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h - 0.5);
    ctx.lineTo(w, h - 0.5);
    ctx.stroke();

    const data = this.history.map((point) => point[spark.key]);
    if (data.length < 2) {
      spark.readout.textContent = "";
      return;
    }
    const max = spark.key === "accuracy" ? 1 : Math.max(...data) * 1.05;
    const x = (i: number) => (i / (data.length - 1)) * (w - 4) + 2;
    const y = (v: number) => h - 3 - (v / max) * (h - 6);
    ctx.strokeStyle = "#8fb4ff";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    data.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.stroke();

    const i = spark.hover >= 0 ? Math.min(spark.hover, data.length - 1) : -1;
    if (i >= 0) {
      ctx.strokeStyle = "rgba(233,238,255,0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x(i), 0);
      ctx.lineTo(x(i), h);
      ctx.stroke();
      ctx.fillStyle = "#e9eeff";
      ctx.beginPath();
      ctx.arc(x(i), y(data[i]), 4, 0, Math.PI * 2);
      ctx.fill();
      spark.readout.textContent = `Epoch ${num(this.history[i].epoch, 1)} · ${spark.fmt(data[i])}`;
    } else {
      spark.readout.textContent = "";
    }
  }
}
