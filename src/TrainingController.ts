/**
 * Runs training as a sequence of visual *phases*
 *
 * One cycle = one mini-batch update:
 *   input -> forward-1 -> forward-2 -> forward-3 -> loss -> backprop -> update

 *   onPhase(id, ctx)   a new phase started (ctx = sample, pass, deltas, grads, …)
 *   onStats(stats)     epoch / accuracy / loss changed
 *   onState(ctrl)      running / finished state changed
 *   onComplete()       target epoch reached
 */
import {
  argmax,
  crossEntropy,
  type ForwardPass,
  type Gradients,
  type NeuralNetworkModel,
} from "./NeuralNetworkModel.ts";
import type { Sample } from "./shapes.ts";

export type PhaseId = "input" | "forward-1" | "forward-2" | "forward-3" | "loss" | "backprop" | "update";

export interface Phase {
  id: PhaseId;
  duration: number; // sim seconds
}

export interface Speed {
  label: string;
  time: number; // sim seconds per real second
  batches: number; // mini-batches trained per cycle
}

/** Everything known about the current cycle */
export interface CycleContext {
  batch: Sample[];
  sample: Sample; // the one sample shown in detail
  pass: ForwardPass;
  predicted: number;
  loss?: number;
  deltas?: Float32Array[];
  grads?: Gradients;
}

export interface TrainingStats {
  epoch: number;
  targetEpochs: number;
  accuracy: number;
  loss: number | null;
}

export interface HistoryPoint {
  epoch: number;
  accuracy: number;
  loss: number;
}

export interface TrainingHooks {
  onPhase?: (id: PhaseId, ctx: CycleContext) => void;
  onStats?: (stats: TrainingStats, history: HistoryPoint[]) => void;
  onState?: (ctrl: TrainingController) => void;
  onComplete?: () => void;
}

export const PHASES: Phase[] = [
  { id: "input", duration: 1.0 },
  { id: "forward-1", duration: 0.9 },
  { id: "forward-2", duration: 0.7 },
  { id: "forward-3", duration: 0.8 },
  { id: "loss", duration: 1.0 },
  { id: "backprop", duration: 1.5 },
  { id: "update", duration: 0.9 },
];

export const SPEEDS: Speed[] = [
  { label: "0,25×", time: 0.25, batches: 1 },
  { label: "0,5×", time: 0.5, batches: 1 },
  { label: "1×", time: 1, batches: 1 },
  { label: "2×", time: 2, batches: 2 },
  { label: "5×", time: 5, batches: 5 },
  { label: "15×", time: 15, batches: 12 },
  { label: "40×", time: 40, batches: 30 },
];

export class TrainingController {
  private readonly model: NeuralNetworkModel;
  private readonly trainSet: Sample[];
  private readonly valSet: Sample[];
  private readonly batchSize: number;
  private readonly targetEpochs: number;
  private readonly hooks: TrainingHooks;
  speedIndex: number;
  running = false;
  finished = false;
  phaseIndex = -1; // -1 = idle until first play
  stats!: TrainingStats;
  private stepping = false;
  private phaseTime = 0;
  private samplesSeen = 0;
  private cursor = 0;
  private order: number[] = [];
  private lossEma: number | null = null;
  private history: HistoryPoint[] = [];
  private ctx: CycleContext | null = null;

  constructor(
    model: NeuralNetworkModel,
    trainSet: Sample[],
    valSet: Sample[],
    { batchSize = 8, targetEpochs = 5, hooks = {} as TrainingHooks } = {},
  ) {
    this.model = model;
    this.trainSet = trainSet;
    this.valSet = valSet;
    this.batchSize = batchSize;
    this.targetEpochs = targetEpochs;
    this.hooks = hooks;
    this.speedIndex = 3;
    this.reset();
  }

  private get speed(): Speed {
    return SPEEDS[this.speedIndex];
  }

  private get phase(): Phase {
    return PHASES[this.phaseIndex];
  }

  private get epoch(): number {
    return this.samplesSeen / this.trainSet.length;
  }

  reset(): void {
    this.model.reset();
    this.running = false;
    this.stepping = false;
    this.finished = false;
    this.phaseIndex = -1;
    this.phaseTime = 0;
    this.samplesSeen = 0;
    this.cursor = 0;
    this.order = [];
    this.lossEma = null;
    this.history = [];
    this.ctx = null;
    this.shuffle();
    this.emitStats(this.model.evaluate(this.valSet).accuracy);
    this.hooks.onState?.(this);
  }

  private shuffle(): void {
    this.order = this.trainSet.map((_, i) => i);
    for (let i = this.order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.order[i], this.order[j]] = [this.order[j], this.order[i]];
    }
    this.cursor = 0;
  }

  private nextBatch(): Sample[] {
    if (this.cursor >= this.order.length) this.shuffle();
    const batch = this.order.slice(this.cursor, this.cursor + this.batchSize).map((i) => this.trainSet[i]);
    this.cursor += this.batchSize;
    this.samplesSeen += batch.length;
    return batch;
  }

  play(): void {
    if (this.finished) return;
    this.running = true;
    this.stepping = false;
    this.hooks.onState?.(this);
  }

  pause(): void {
    this.running = false;
    this.stepping = false;
    this.hooks.onState?.(this);
  }

  /** Advance to the start of the next phase, then pause again */
  step(): void {
    if (this.finished) return;
    if (this.phaseIndex === -1) {
      this.enterPhase(0);
      this.hooks.onState?.(this);
      return;
    }
    this.running = true;
    this.stepping = true;
    this.hooks.onState?.(this);
  }

  setSpeed(index: number): void {
    this.speedIndex = Math.max(0, Math.min(SPEEDS.length - 1, index));
  }

  /** Called every frame with real seconds. Returns the sim-time delta used */
  update(dt: number): number {
    if (!this.running) return 0;
    const simDt = dt * (this.stepping ? Math.max(1, this.speed.time) : this.speed.time);
    if (this.phaseIndex === -1) {
      this.enterPhase(0);
      return simDt;
    }
    this.phaseTime += simDt;
    while (this.running && this.phaseTime >= this.phase.duration) {
      this.phaseTime -= this.phase.duration;
      this.enterPhase((this.phaseIndex + 1) % PHASES.length);
      if (this.stepping) {
        this.phaseTime = 0;
        this.pause();
      }
    }
    return simDt;
  }

  private enterPhase(index: number): void {
    this.phaseIndex = index;
    const id = this.phase.id;
    const model = this.model;

    if (id === "input") {
      // Draw a batch, show its first sample, run the forward pass
      // with the current (not yet updated) weights
      const batch = this.nextBatch();
      const sample = batch[0];
      const pass = model.forward(sample.x);
      this.ctx = { batch, sample, pass, predicted: argmax(pass.probs) };
    }
    const ctx = this.ctx!; // set by the 'input' phase, which always runs first

    if (id === "loss") {
      ctx.loss = crossEntropy(ctx.pass.probs, ctx.sample.y);
    } else if (id === "backprop") {
      // Error signals of the shown sample (for neuron glow) + batch gradients (for lines)
      const scratchW = model.W.map((W) => new Float32Array(W.length));
      const scratchB = model.b.map((b) => new Float32Array(b.length));
      ctx.deltas = model.backward(ctx.pass, ctx.sample.y, scratchW, scratchB);
      ctx.grads = model.computeGradients(ctx.batch);
    } else if (id === "update") {
      const grads = ctx.grads!;
      model.applyGradients(grads);
      let loss = grads.loss;
      // at higher speeds, train a few extra batches in the background
      for (let k = 1; k < this.speed.batches; k++) {
        loss = model.trainBatch(this.nextBatch());
      }
      this.trackLoss(loss);
      this.emitStats(model.evaluate(this.valSet).accuracy);
    }

    this.hooks.onPhase?.(id, ctx);

    if (id === "update" && this.epoch >= this.targetEpochs) this.complete();
  }

  /** Smoothed loss for the display */
  private trackLoss(loss: number): void {
    this.lossEma = this.lossEma == null ? loss : this.lossEma * 0.8 + loss * 0.2;
  }

  private emitStats(accuracy: number): void {
    const stats: TrainingStats = {
      epoch: this.epoch,
      targetEpochs: this.targetEpochs,
      accuracy,
      loss: this.lossEma,
    };
    if (this.lossEma != null) this.history.push({ epoch: this.epoch, accuracy, loss: this.lossEma });
    this.stats = stats;
    this.hooks.onStats?.(stats, this.history);
  }

  /** Train all remaining batches instantly. */
  finishNow(): void {
    if (this.finished) return;
    let n = 0;
    while (this.epoch < this.targetEpochs) {
      this.trackLoss(this.model.trainBatch(this.nextBatch()));
      if (++n % 6 === 0) this.emitStats(this.model.evaluate(this.valSet).accuracy);
    }
    this.emitStats(this.model.evaluate(this.valSet).accuracy);
    this.complete();
  }

  private complete(): void {
    this.finished = true;
    this.running = false;
    this.stepping = false;
    this.hooks.onState?.(this);
    this.hooks.onComplete?.();
  }
}
