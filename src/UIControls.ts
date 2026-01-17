import { SPEEDS } from "./TrainingController.ts";
import { $ } from "./dom.ts";

export interface ControlActions {
  play(): void;
  pause(): void;
  step(): void;
  reset(): void;
  skip(): void;
  speed(index: number): void;
  toggleLabels(on: boolean): void;
  toggleFormulas(on: boolean): void;
}

export class UIControls {
  private playButton = $<HTMLButtonElement>("btn-play");
  private playIcon = this.playButton.querySelector<HTMLElement>(".btn-icon")!;
  private playLabel = this.playButton.querySelector<HTMLElement>(".btn-label")!;
  private stepButton = $<HTMLButtonElement>("btn-step");
  private speedSlider = $<HTMLInputElement>("speed");
  private running = false;
  private readonly batchSize: number;

  constructor(batchSize: number, actions: ControlActions) {
    this.batchSize = batchSize;
    this.speedSlider.max = String(SPEEDS.length - 1);

    this.playButton.addEventListener("click", () => (this.running ? actions.pause() : actions.play()));
    this.stepButton.addEventListener("click", () => actions.step());
    $("btn-reset").addEventListener("click", () => actions.reset());
    $("btn-skip").addEventListener("click", () => actions.skip());
    this.speedSlider.addEventListener("input", () => {
      const index = Number(this.speedSlider.value);
      actions.speed(index);
      this.showSpeed(index);
    });
    $<HTMLInputElement>("toggle-labels").addEventListener("change", (e) =>
      actions.toggleLabels((e.target as HTMLInputElement).checked),
    );
    $<HTMLInputElement>("toggle-formulas").addEventListener("change", (e) =>
      actions.toggleFormulas((e.target as HTMLInputElement).checked),
    );

    // Key-Controls space = play/pause, -> = step
    window.addEventListener("keydown", (e) => {
      const target = e.target as Element;
      if (document.body.classList.contains("mode-inference") || target.matches("input")) return;
      if (e.code === "Space" && !target.matches("button")) {
        e.preventDefault();
        this.playButton.click();
      }
      if (e.code === "ArrowRight") {
        e.preventDefault();
        this.stepButton.click();
      }
    });
  }

  showSpeed(index: number): void {
    const speed = SPEEDS[index];
    this.speedSlider.value = String(index);
    $("speed-label").textContent = speed.label;
    $("speed-hint").textContent =
      `${speed.batches} Mini-Batch${speed.batches > 1 ? "es" : ""} à ${this.batchSize} Beispiele pro Durchlauf`;
  }

  setState({ running, started, finished }: { running: boolean; started: boolean; finished: boolean }): void {
    this.running = running;
    this.playIcon.textContent = running ? "❚❚" : "▶";
    this.playIcon.dataset.icon = running ? "pause" : "play";
    this.playLabel.textContent = running ? "Pause" : started ? "Weiter" : "Training starten";
    this.playButton.setAttribute("aria-pressed", String(running));
    this.playButton.disabled = finished;
    this.stepButton.disabled = finished;
  }
}
