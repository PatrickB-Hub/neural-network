/**
 *  training:  TrainingController steps through phases; each phase triggers a scene
 *             animation, a caption and a highlighted formula.
 *  inference: the user draws, presses Predict, and the same forward animation runs
 *             on their drawing with the trained weights.
 */
import "./style.scss";
import { NeuralNetworkModel, argmax } from "./NeuralNetworkModel.ts";
import {
  makeDataset,
  makeRng,
  featurize,
  generateShape,
  GRID,
  CLASSES,
  CLASS_ICONS,
  num,
  pct,
} from "./shapes.ts";
import { TrainingController, PHASES, type CycleContext, type PhaseId } from "./TrainingController.ts";
import { NetworkScene, type Mode } from "./Scene.ts";
import { DrawingCanvas, drawPixels } from "./DrawingCanvas.ts";
import { UIControls } from "./UIControls.ts";
import { StatsPanel } from "./StatsPanel.ts";
import { PredictionPanel } from "./PredictionPanel.ts";
import { $ } from "./dom.ts";

const BATCH_SIZE = 8;

// data + model
const rng = makeRng(2024);
const trainSet = makeDataset(90, rng); // 270 synthetic drawings
const valSet = makeDataset(30, rng); // 90 held-out drawings → accuracy
const model = new NeuralNetworkModel([GRID * GRID, 16, 10, 3]);

const scene = new NetworkScene($("stage"), model, $("tooltip"));
const stats = new StatsPanel();
const prediction = new PredictionPanel();
let mode: Mode = "training";

function clearScene(): void {
  scene.clearActivity({ immediate: true });
  scene.clearDrawing();
}

// captions, step indicator, formulas
type StepId = "input" | "forward" | "output" | "loss" | "backprop" | "update" | "prediction";
type FormulaId = "x" | "z" | "softmax" | "loss" | "grad" | "update";

const TRAIN_STEPS: [StepId, string][] = [
  ["input", "Eingabe"],
  ["forward", "Vorwärts"],
  ["output", "Ausgabe"],
  ["loss", "Verlust"],
  ["backprop", "Backprop"],
  ["update", "Anpassung"],
];
const INFER_STEPS: [StepId, string][] = [
  ["input", "Eingabe"],
  ["forward", "Vorwärts"],
  ["output", "Ausgabe"],
  ["prediction", "Vorhersage"],
];
const STEP_OF: Record<PhaseId, StepId> = {
  input: "input",
  "forward-1": "forward",
  "forward-2": "forward",
  "forward-3": "output",
  loss: "loss",
  backprop: "backprop",
  update: "update",
};
const FORMULA_OF: Record<PhaseId, FormulaId> = {
  input: "x",
  "forward-1": "z",
  "forward-2": "z",
  "forward-3": "softmax",
  loss: "loss",
  backprop: "grad",
  update: "update",
};

function renderSteps(steps: [StepId, string][]): void {
  $("steps").innerHTML = steps
    .map(([id, name], i) => `<li data-step="${id}"><span>${i + 1}</span>${name}</li>`)
    .join("");
}

function setCaption(stepId: StepId | null, html: string, formula: FormulaId | null): void {
  document
    .querySelectorAll<HTMLElement>("#steps li")
    .forEach((li) => li.classList.toggle("active", li.dataset.step === stepId));
  $("caption-text").innerHTML = html;
  document
    .querySelectorAll<HTMLElement>("#formulas [data-f]")
    .forEach((el) =>
      el.classList.toggle("active", formula != null && el.dataset.f!.split(" ").includes(formula)),
    );
}

const probsLine = (probs: Float32Array): string =>
  Array.from(probs, (p, i) => `${CLASS_ICONS[i]} ${pct(p)}`).join(" &nbsp; ");

function trainingCaption(id: PhaseId, ctx: CycleContext): string {
  const { sample, pass, predicted } = ctx;
  const name = CLASSES[sample.y];
  switch (id) {
    case "input":
      return `<b>Eingabe.</b> Eine Trainingszeichnung (Klasse <b>${name}</b>) wird auf ${GRID}×${GRID} Pixel verkleinert. Die Helligkeit jedes Pixels (0–1) wird zu einem Eingabeneuron.`;
    case "forward-1":
      return "<b>Vorwärtsdurchlauf.</b> Jedes verborgene Neuron addiert seine gewichteten Eingaben plus einen Bias – <code>z = w·x + b</code> – und wendet dann <code>ReLU</code> an: Negative Werte werden zu 0.";
    case "forward-2":
      return "<b>Vorwärtsdurchlauf.</b> Die zweite verborgene Schicht kombiniert die einfachen Muster der ersten Schicht zu formähnlicheren Merkmalen.";
    case "forward-3":
      return `<b>Ausgabe.</b> Softmax macht aus den drei Werten Wahrscheinlichkeiten, die zusammen 100 % ergeben: &nbsp;${probsLine(pass.probs)}`;
    case "loss": {
      const ok = predicted === sample.y;
      return `<b>Verlust (Loss).</b> Richtig ist <b>${name}</b>; das Netz gibt dafür ${pct(pass.probs[sample.y])} ${ok ? "✓" : `und tippt auf ${CLASSES[predicted]} ✗`}. Loss = −log(${num(pass.probs[sample.y])}) = <b>${num(ctx.loss!)}</b> – klein, wenn die Antwort richtig und sicher ist.`;
    }
    case "backprop":
      return "<b>Backpropagation.</b> Der Fehler fließt rückwärts (violett). Jedes Gewicht erfährt, wie viel es zum Fehler beigetragen hat – seinen Gradienten <code>∂L/∂w</code>.";
    case "update":
      return `<b>Anpassung.</b> Jedes Gewicht macht einen kleinen Schritt entgegen seinem Gradienten, gemittelt über einen Mini-Batch aus ${ctx.batch.length} Zeichnungen. Aufblitzende Linien haben sich am stärksten verändert.`;
  }
}

// training
const controls = new UIControls(BATCH_SIZE, {
  play: () => controller.play(),
  pause: () => controller.pause(),
  step: () => controller.step(),
  reset: () => resetTraining(),
  skip: () => controller.finishNow(),
  speed: (i) => controller.setSpeed(i),
  toggleLabels: (on) => {
    scene.setLabelsVisible(on);
    document.body.classList.toggle("no-labels", !on);
  },
  toggleFormulas: (on) => {
    $("formulas").hidden = !on;
  },
});

const controller = new TrainingController(model, trainSet, valSet, {
  batchSize: BATCH_SIZE,
  targetEpochs: 5,
  hooks: {
    onPhase(id, ctx) {
      const duration = PHASES.find((p) => p.id === id)!.duration;
      if (id === "input") {
        scene.showInput(ctx.sample.strokes, ctx.sample.extent, ctx.sample.x, duration);
        stats.showSample(ctx.sample);
      } else if (id.startsWith("forward")) {
        const layer = Number(id.slice(-1)) - 1;
        scene.forward(layer, ctx.pass, duration);
        if (layer === 2) stats.showGuess(ctx.sample, ctx.pass.probs, ctx.predicted);
      } else if (id === "loss") {
        scene.setRings({ target: ctx.sample.y, winner: ctx.predicted });
      } else if (id === "backprop") {
        scene.setRings({ target: ctx.sample.y });
        scene.backward(ctx.deltas!, ctx.grads!, duration);
      } else if (id === "update") {
        scene.setRings();
        scene.applyUpdate(model);
      }
      setCaption(STEP_OF[id], trainingCaption(id, ctx), FORMULA_OF[id]);
    },
    onStats: (values, history) => stats.update(values, history),
    onState: (ctrl) =>
      controls.setState({ running: ctrl.running, started: ctrl.phaseIndex >= 0, finished: ctrl.finished }),
    onComplete: () => {
      scene.setWeights(model, true);
      setCaption(
        "update",
        "<b>Training abgeschlossen!</b> Die Gewichte sind jetzt so eingestellt, dass das Netz Dreiecke, Rechtecke und Kreise unterscheiden kann.",
        "update",
      );
      setTimeout(enterInference, 1400);
    },
  },
});
controls.showSpeed(controller.speedIndex);

function idleCaption(): void {
  setCaption(
    null,
    "Klicke auf <b>Training starten</b> und sieh zu, wie das Netz aus Beispielzeichnungen lernt. Fahre mit der Maus über ein Neuron, eine Linie oder einen Bias-Knoten, um zu sehen, was es tut.",
    null,
  );
}

function resetTraining(): void {
  controller.reset();
  scene.setWeights(model, false);
  clearScene();
  idleCaption();
}

// inference
const drawing = new DrawingCanvas($<HTMLCanvasElement>("pad"), {
  onChange(strokes) {
    const pixels = featurize(strokes);
    drawPixels($<HTMLCanvasElement>("pixels"), pixels, GRID);
    $<HTMLButtonElement>("btn-predict").disabled = strokes.length === 0;
  },
});

function enterInference(): void {
  if (mode === "inference") return;
  mode = "inference";
  controller.pause();
  document.body.classList.replace("mode-training", "mode-inference");
  scene.setMode("inference");
  clearScene();
  prediction.setModelInfo(controller.stats.accuracy);
  prediction.setWaiting();
  renderSteps(INFER_STEPS);
  setCaption(
    null,
    "<b>Du bist dran.</b> Zeichne ein Dreieck, Rechteck oder einen Kreis und klicke auf <b>Erkennen</b>, um dem Netz beim Denken zuzusehen.",
    null,
  );
  drawing.resize(); // pad is visible
}

function leaveInference(): void {
  mode = "training";
  document.body.classList.replace("mode-inference", "mode-training");
  scene.setMode("training");
  renderSteps(TRAIN_STEPS);
  drawing.clear();
  resetTraining();
}

/** Run the drawing through the trained network with the full forward animation */
function predict(): void {
  if (!drawing.strokes.length) return;
  const pixels = featurize(drawing.strokes);
  const pass = model.forward(pixels);
  const winner = argmax(pass.probs);
  $<HTMLButtonElement>("btn-predict").disabled = true;
  prediction.setThinking();
  scene.clearActivity({ immediate: true });
  scene.showInput(drawing.strokes, drawing.size, pixels, 1.0);
  setCaption(
    "input",
    "<b>Eingabe.</b> Deine Zeichnung wird zugeschnitten, gestreckt und auf 10×10 Pixel verkleinert – 100 Zahlen fließen in die Eingabeschicht.",
    "x",
  );
  scene.schedule(1.0, () => {
    scene.forward(0, pass, 0.9);
    setCaption(
      "forward",
      "<b>Vorwärtsdurchlauf.</b> Jedes verborgene Neuron berechnet <code>z = w·x + b</code> und danach <code>ReLU</code>. Hellere Neuronen reagieren stärker auf deine Zeichnung.",
      "z",
    );
  });
  scene.schedule(1.9, () => scene.forward(1, pass, 0.7));
  scene.schedule(2.6, () => {
    scene.forward(2, pass, 0.8);
    setCaption(
      "output",
      `<b>Ausgabe.</b> Softmax macht aus den Werten Wahrscheinlichkeiten: &nbsp;${probsLine(pass.probs)}`,
      "softmax",
    );
  });
  scene.schedule(3.4, () => {
    scene.setRings({ winner });
    prediction.show(pass.probs);
    setCaption(
      "prediction",
      `<b>Vorhersage: ${CLASS_ICONS[winner]} ${CLASSES[winner]}</b> – höchste Aktivierung im Ausgabeneuron „${CLASSES[winner]}“ (${pct(pass.probs[winner])}).`,
      "softmax",
    );
    $<HTMLButtonElement>("btn-predict").disabled = false;
  });
}

$("btn-predict").addEventListener("click", predict);
$("btn-clear").addEventListener("click", () => {
  drawing.clear();
  clearScene();
  prediction.setWaiting();
});
$("btn-retrain").addEventListener("click", leaveInference);
document.querySelectorAll<HTMLElement>("[data-example]").forEach((btn) => {
  btn.addEventListener("click", () => drawing.setStrokes(generateShape(Number(btn.dataset.example)), 100));
});

// main render loop
renderSteps(TRAIN_STEPS);
idleCaption();
drawPixels($<HTMLCanvasElement>("pixels"), new Float32Array(GRID * GRID), GRID);

let last = performance.now();
function frame(now: number): void {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  // Sim time follows the training clock
  let simDt = controller.update(dt);
  if (mode === "inference" || controller.finished) simDt = dt;
  scene.update(dt, simDt);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
