/** The network's answer for the user's drawing. Shows winning class, confidence bars and explanation */
import { CLASSES, CLASS_ICONS, pct } from "./shapes.ts";
import { argmax } from "./NeuralNetworkModel.ts";
import { $ } from "./dom.ts";

export class PredictionPanel {
  private bars = $("pred-bars");

  constructor() {
    this.bars.innerHTML = CLASSES.map(
      (name, i) => `
      <div class="bar-row" data-i="${i}">
        <span class="bar-name"><i>${CLASS_ICONS[i]}</i>${name}</span>
        <span class="bar-track"><span class="bar-fill"></span></span>
        <span class="bar-val">0 %</span>
      </div>`,
    ).join("");
  }

  setModelInfo(accuracy: number): void {
    $("pred-model").textContent = `Trainiertes Netz · ${pct(accuracy)} Genauigkeit auf Validierungsdaten`;
  }

  setWaiting(text = "Zeichne ein Dreieck, Rechteck oder einen Kreis und klicke auf „Erkennen“."): void {
    $("pred-result").innerHTML = `<span class="muted">${text}</span>`;
    $("pred-explain").textContent = "";
    this.setBars([0, 0, 0], -1);
  }

  setThinking(): void {
    $("pred-result").innerHTML = '<span class="muted">Signale fließen durch das Netz …</span>';
    $("pred-explain").textContent = "";
    this.setBars([0, 0, 0], -1);
  }

  show(probs: Float32Array): void {
    const winner = argmax(probs);
    const confidence = probs[winner];
    $("pred-result").innerHTML =
      `<span class="pred-icon">${CLASS_ICONS[winner]}</span><b>${CLASSES[winner]}</b>`;
    const certainty = confidence > 0.9 ? "sehr sicher" : confidence > 0.7 ? "ziemlich sicher" : "unsicher";
    let text = `Höchste Aktivierung im Ausgabeneuron „${CLASSES[winner]}“ (${pct(confidence)}). Das Netz ist ${certainty}.`;
    if (confidence <= 0.7) text += " Tipp: Zeichne einen geschlossenen Umriss, nicht zu klein.";
    $("pred-explain").textContent = text;
    this.setBars(probs, winner);
  }

  private setBars(probs: ArrayLike<number>, winner: number): void {
    this.bars.querySelectorAll(".bar-row").forEach((row, i) => {
      row.querySelector<HTMLElement>(".bar-fill")!.style.width = `${probs[i] * 100}%`;
      row.querySelector(".bar-val")!.textContent = pct(probs[i]);
      row.classList.toggle("win", i === winner);
    });
  }
}
