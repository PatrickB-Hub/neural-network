// Sanity check: gradients are correct and the network actually learns the shapes.
// Run with `npm test`.
import assert from "node:assert/strict";
import { NeuralNetworkModel } from "../src/NeuralNetworkModel.ts";
import { makeDataset, makeRng, featurize, type Stroke } from "../src/shapes.ts";

// 1. Backprop matches a numerical gradient.
{
  const rng = makeRng(7);
  const net = new NeuralNetworkModel([6, 5, 4, 3], { rng });
  const batch = [{ x: [0.2, 0.9, 0, 0.4, 1, 0.3], y: 1 }];
  const { gW } = net.computeGradients(batch);
  const lossAt = () => net.evaluate(batch).loss;
  for (const [l, k] of [
    [0, 3],
    [1, 7],
    [2, 5],
  ]) {
    const w = net.W[l][k];
    const h = 1e-3;
    net.W[l][k] = w + h;
    const up = lossAt();
    net.W[l][k] = w - h;
    const down = lossAt();
    net.W[l][k] = w;
    assert.ok(Math.abs((up - down) / (2 * h) - gW[l][k]) < 1e-3, `gradient mismatch at W[${l}][${k}]`);
  }
}

// 2. Featurize is position/size invariant.
{
  const sq = (o: number, s: number): Stroke[] => [
    [
      [o, o],
      [o + s, o],
      [o + s, o + s],
      [o, o + s],
      [o, o],
    ],
  ];
  const a = featurize(sq(10, 50));
  const b = featurize(sq(100, 200));
  assert.ok(
    a.every((v, i) => Math.abs(v - b[i]) < 0.15),
    "featurize should ignore position and scale",
  );
}

// 3. Training reaches good validation accuracy.
{
  const rng = makeRng(42);
  const train = makeDataset(90, rng);
  const val = makeDataset(40, rng);
  const net = new NeuralNetworkModel(undefined, { rng });
  const t0 = performance.now();
  for (let epoch = 1; epoch <= 5; epoch++) {
    for (let i = 0; i < train.length; i += 8) net.trainBatch(train.slice(i, i + 8));
    const { accuracy, loss } = net.evaluate(val);
    console.log(
      `epoch ${String(epoch).padStart(2)}  val acc ${(accuracy * 100).toFixed(1)}%  loss ${loss.toFixed(3)}`,
    );
  }
  console.log(`trained in ${(performance.now() - t0).toFixed(0)} ms`);
  assert.ok(net.evaluate(val).accuracy > 0.9, "validation accuracy should exceed 90%");
}
console.log("✓ all checks passed");
