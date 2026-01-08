/**
 *  Renders the untrained network. Hover a neuron, weight or bias node for details.
 */
import "./style.scss";
import { NeuralNetworkModel } from "./NeuralNetworkModel.ts";
import { GRID } from "./shapes.ts";
import { NetworkScene } from "./Scene.ts";
import { $ } from "./dom.ts";

// model
const model = new NeuralNetworkModel([GRID * GRID, 16, 10, 3]);

const scene = new NetworkScene($("stage"), model, $("tooltip"));

// main render loop
let last = performance.now();
function frame(now: number): void {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  scene.update(dt, dt);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
