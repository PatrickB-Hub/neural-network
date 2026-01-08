/**
 * A simple but fully-connected neural network (multi-layer perceptron) in Typescript.
 *
 * input (100 pixels) -> hidden 1 (ReLU) -> hidden 2 (ReLU) -> output (softmax, 3 classes)
 *
 * Every neuron j in layer l+1 computes   z_j = Σ_i w_ji · a_i + b_j   and then
 * a_j = ReLU(z_j) (hidden) or softmax(z)_j (output). Training minimises the
 * cross-entropy loss  L = −log p_target  with backpropagation + the Adam optimizer.
 *
 * Weights are stored per layer as a flat Float32Array, row-major: W[l][j * nIn + i]
 * is the weight from neuron i (layer l) to neuron j (layer l+1).
 */
import type { Rng } from "./shapes.ts";

/** Result of a forward pass: pre-activations z and activations a for every layer */
export interface ForwardPass {
  acts: Float32Array[]; // acts[0] = input, acts[L] = output probabilities
  zs: Float32Array[]; // zs[l] belongs to layer l+1
  probs: Float32Array;
}

/** Mini-batch gradients ∂L/∂W and ∂L/∂b (one array per weight layer) plus the mean loss */
export interface Gradients {
  gW: Float32Array[];
  gb: Float32Array[];
  loss: number;
}

export interface LabeledInput {
  x: ArrayLike<number>;
  y: number;
}

export class NeuralNetworkModel {
  sizes: number[];
  learningRate: number;
  rng: Rng;
  W: Float32Array[] = [];
  b: Float32Array[] = [];
  // Adam moment estimates
  mW: Float32Array[] = [];
  vW: Float32Array[] = [];
  mb: Float32Array[] = [];
  vb: Float32Array[] = [];
  steps = 0;

  constructor(sizes = [100, 16, 10, 3], { learningRate = 0.002, rng = Math.random as Rng } = {}) {
    this.sizes = sizes;
    this.learningRate = learningRate;
    this.rng = rng;
    this.reset();
  }

  /** Fresh random weights with zero biases */
  reset(): void {
    this.W = [];
    this.b = [];
    this.mW = [];
    this.vW = [];
    this.mb = [];
    this.vb = [];
    this.steps = 0;
    for (let l = 0; l < this.sizes.length - 1; l++) {
      const nIn = this.sizes[l];
      const nOut = this.sizes[l + 1];
      const std = Math.sqrt(2 / nIn);
      const W = new Float32Array(nIn * nOut);
      for (let k = 0; k < W.length; k++) W[k] = this.gaussian() * std;
      this.W.push(W);
      this.b.push(new Float32Array(nOut));
      this.mW.push(new Float32Array(W.length));
      this.vW.push(new Float32Array(W.length));
      this.mb.push(new Float32Array(nOut));
      this.vb.push(new Float32Array(nOut));
    }
  }

  gaussian(): number {
    const u = 1 - this.rng();
    const v = this.rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Forward pass. Returns every layer's pre-activations (zs) and activations (acts) */
  forward(x: ArrayLike<number>): ForwardPass {
    const acts: Float32Array[] = [Float32Array.from(x)];
    const zs: Float32Array[] = [];
    const last = this.W.length - 1;
    for (let l = 0; l <= last; l++) {
      const nIn = this.sizes[l];
      const nOut = this.sizes[l + 1];
      const W = this.W[l];
      const a = acts[l];
      const z = new Float32Array(nOut);
      for (let j = 0; j < nOut; j++) {
        let s = this.b[l][j];
        const row = j * nIn;
        for (let i = 0; i < nIn; i++) s += W[row + i] * a[i];
        z[j] = s;
      }
      zs.push(z);
      acts.push(l === last ? softmax(z) : z.map((v) => (v > 0 ? v : 0))); // ReLU
    }
    return { acts, zs, probs: acts[acts.length - 1] };
  }

  /**
   * Backpropagation for one sample. Adds the gradients ∂L/∂w and ∂L/∂b into gW / gb
   * and returns the per-layer error signals δ (deltas[l] belongs to layer l+1).
   */
  backward(pass: ForwardPass, target: number, gW: Float32Array[], gb: Float32Array[]): Float32Array[] {
    const L = this.W.length;
    const deltas = new Array<Float32Array>(L);
    // softmax + cross-entropy gives the output error: δ = p − onehot
    let delta = Float32Array.from(pass.probs);
    delta[target] -= 1;
    for (let l = L - 1; l >= 0; l--) {
      deltas[l] = delta;
      const nIn = this.sizes[l];
      const nOut = this.sizes[l + 1];
      const a = pass.acts[l];
      for (let j = 0; j < nOut; j++) {
        const d = delta[j];
        if (d === 0) continue;
        const row = j * nIn;
        for (let i = 0; i < nIn; i++) gW[l][row + i] += d * a[i];
        gb[l][j] += d;
      }
      if (l > 0) {
        // push the error back through the weights and the ReLU derivative
        const prev = new Float32Array(nIn);
        const z = pass.zs[l - 1];
        for (let i = 0; i < nIn; i++) {
          if (z[i] <= 0) continue;
          let s = 0;
          for (let j = 0; j < nOut; j++) s += this.W[l][j * nIn + i] * delta[j];
          prev[i] = s;
        }
        delta = prev;
      }
    }
    return deltas;
  }

  /** Average gradients over a mini-batch (does not change the weights yet) */
  computeGradients(batch: LabeledInput[]): Gradients {
    const gW = this.W.map((W) => new Float32Array(W.length));
    const gb = this.b.map((b) => new Float32Array(b.length));
    let loss = 0;
    for (const s of batch) {
      const pass = this.forward(s.x);
      loss += crossEntropy(pass.probs, s.y);
      this.backward(pass, s.y, gW, gb);
    }
    const n = batch.length;
    for (const g of [...gW, ...gb]) for (let k = 0; k < g.length; k++) g[k] /= n;
    return { gW, gb, loss: loss / n };
  }

  /** Adam update: w ← w − η · m̂ / (√v̂ + ε), a smoothed version of w ← w − η·∂L/∂w */
  applyGradients({ gW, gb }: Gradients): void {
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    this.steps++;
    const c1 = 1 - b1 ** this.steps;
    const c2 = 1 - b2 ** this.steps;
    const lr = this.learningRate;
    const step = (p: Float32Array, g: Float32Array, m: Float32Array, v: Float32Array) => {
      for (let k = 0; k < p.length; k++) {
        m[k] = b1 * m[k] + (1 - b1) * g[k];
        v[k] = b2 * v[k] + (1 - b2) * g[k] * g[k];
        p[k] -= (lr * (m[k] / c1)) / (Math.sqrt(v[k] / c2) + eps);
      }
    };
    for (let l = 0; l < this.W.length; l++) {
      step(this.W[l], gW[l], this.mW[l], this.vW[l]);
      step(this.b[l], gb[l], this.mb[l], this.vb[l]);
    }
  }

  trainBatch(batch: LabeledInput[]): number {
    const grads = this.computeGradients(batch);
    this.applyGradients(grads);
    return grads.loss;
  }

  evaluate(samples: LabeledInput[]): { accuracy: number; loss: number } {
    let correct = 0;
    let loss = 0;
    for (const s of samples) {
      const p = this.forward(s.x).probs;
      loss += crossEntropy(p, s.y);
      if (argmax(p) === s.y) correct++;
    }
    return { accuracy: correct / samples.length, loss: loss / samples.length };
  }
}

export function softmax(z: Float32Array): Float32Array {
  const max = Math.max(...z);
  const e = z.map((v) => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

/** L = −log p_target, clamped so a zero probability doesn't give Infinity */
export function crossEntropy(probs: ArrayLike<number>, target: number): number {
  return -Math.log(Math.max(probs[target], 1e-9));
}

export function argmax(arr: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}
