/**
 *  - Neurons: instanced glowing spheres. Brightness/size = current activation.
 *  - Bias nodes: amber octahedra (a constant +1 input) below each layer.
 *  - Weights: instanced thin cylinders. Cyan = positive, orange = negative,
 *    thickness/brightness = |w|. They re-tint smoothly whenever weights change.
 *  - Pulses: additive glowing points travelling along connections — white-cyan for the
 *    forward pass, violet for backpropagation (flowing right → left).
 *
 * The scene animates on two clocks:
 *  - real time -> idle motion, hover, camera transitions (never pauses)
 *  - sim time -> everything that belongs to training (pauses with the training)
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { GRID, CLASSES, CLASS_ICONS, num, pct, type Stroke } from "./shapes.ts";
import type { ForwardPass, Gradients, NeuralNetworkModel } from "./NeuralNetworkModel.ts";
import { $ } from "./dom.ts";

export type Mode = "training" | "inference";

/** One cylinder in the scene: a weight (i ≥ 0) or a bias (i = -1) of weight layer l. */
interface Connection {
  bundle: number; // 0-2 = weights between layers, 3-5 = bias → layer 1..3
  l: number;
  i: number;
  j: number;
  a: THREE.Vector3;
  b: THREE.Vector3;
}

interface HoverTarget {
  kind: "neuron" | "bias" | "conn";
  k: number;
}

interface Hover extends HoverTarget {
  conns: Set<number>; // connections to highlight
}

interface Ring {
  mesh: THREE.Mesh<THREE.TorusGeometry, THREE.MeshBasicMaterial>;
  index: number;
  opacity: number;
}

interface CameraView {
  offset: THREE.Vector3;
  target: THREE.Vector3;
}

interface PulseOptions {
  delay?: number;
  dur?: number;
  color: THREE.Color;
  strength?: number;
  conn?: number;
  back?: boolean;
}

const C = (hex: string) => new THREE.Color(hex);
const COLORS = {
  layers: [C("#4da3ff"), C("#35e39b"), C("#35e39b"), C("#ff5c8a")], // input, hidden, hidden, output
  bias: C("#ffc24d"),
  positive: C("#5ce1ff"),
  negative: C("#ff9440"),
  forward: C("#eafcff"),
  backward: C("#c77dff"),
  white: C("#ffffff"),
};

const LAYER_X = [-5.6, -1.2, 2.6, 6.2];
const LAYER_NAMES = ["Input", "Hidden Layer 1", "Hidden Layer 2", "Output"];
const CARD_X = -8.9;
const CARD_Y = 0.5;
const CARD_SIZE = 2.6;
const CARD_PX = 256; // resolution of the card's canvas texture
const BIAS_Y = -3.3;
const NEURON_RADIUS = [0.13, 0.2, 0.2, 0.38];

const BUNDLE_RADIUS = [0.007, 0.02, 0.026, 0.012, 0.014, 0.018];
const BUNDLE_BRIGHT = [0.07, 0.4, 0.5, 0.3, 0.35, 0.4];
const BUNDLE_HIGHLIGHT = [0.3, 0.8, 1, 0.8, 0.8, 1]; // keep the glow subtle because there is 1600 lines in the input bundle
const FORWARD_PULSES = [90, 60, 30];
const BACKWARD_PULSES = [80, 60, 30];

const CAMERA_VIEWS: Record<Mode, CameraView> = {
  training: {
    offset: new THREE.Vector3(-15.9, 8.3, 20.1),
    target: new THREE.Vector3(-1.7, -0.6, 0),
  },
  inference: {
    offset: new THREE.Vector3(-13.8, 7.4, 27.8),
    target: new THREE.Vector3(-1.6, -0.5, 0),
  },
};

/** Camera position for a view. Camera moves back für narrow windows */
function viewPos(view: CameraView, aspect: number): THREE.Vector3 {
  return view.offset
    .clone()
    .multiplyScalar(Math.max(1, 1.6 / aspect))
    .add(view.target);
}

/** Unlit glowing-orb shader with a fresnel rim */
function glowMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: `
      varying vec3 vColor; varying vec3 vN; varying vec3 vV;
      void main() {
        vColor = instanceColor;
        vec4 mv = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
        vN = normalize(normalMatrix * mat3(instanceMatrix) * normal);
        vV = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vColor; varying vec3 vN; varying vec3 vV;
      void main() {
        float ndv = max(dot(normalize(vN), normalize(vV)), 0.0);
        float rim = pow(1.0 - ndv, 2.2);
        vec3 col = vColor * (0.45 + 0.7 * ndv) + (vColor * 1.2 + 0.08) * rim;
        gl_FragColor = vec4(col, 1.0);
      }`,
  });
}

function glowTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.25, "rgba(255,255,255,0.65)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(canvas);
}

function label(html: string, className: string, center: [number, number] = [0.5, 0.5]): CSS2DObject {
  const el = document.createElement("div");
  el.className = `label ${className}`;
  el.innerHTML = html;
  const obj = new CSS2DObject(el);
  obj.center.set(...center);
  return obj;
}

const formatSigned = (v: number | undefined): string =>
  v == null ? "—" : (v > -0.005 ? "+" : "−") + num(Math.abs(v));

export class NetworkScene {
  private container: HTMLElement;
  private model: NeuralNetworkModel;
  private sizes: number[];
  private tooltipEl: HTMLElement;
  private time = 0;
  private simClock = 0;
  private timers: { at: number; fn: () => void }[] = [];
  private pass: ForwardPass | null = null;
  private inputPixels: Float32Array | null = null;
  private hover: Hover | null = null;
  private pointer = new THREE.Vector2();
  private pointerClient: [number, number] = [0, 0];
  private pointerDirty = false;
  private cameraTween: {
    t: number;
    fromPos: THREE.Vector3;
    fromTarget: THREE.Vector3;
    toPos: THREE.Vector3;
    toTarget: THREE.Vector3;
  } | null = null;

  // renderer & scene graph
  private renderer!: THREE.WebGLRenderer;
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private controls!: OrbitControls;
  private labelRenderer!: CSS2DRenderer;
  private composer!: EffectComposer;
  private group: THREE.Group;
  private labels!: THREE.Group;
  private raycaster = new THREE.Raycaster();

  // layout
  private offsets!: number[];
  private neuronCount!: number;
  private neuronPos!: THREE.Vector3[];
  private neuronLayer!: number[];
  private neuronIndex!: number[];
  private biasPos!: THREE.Vector3[];

  // neurons
  private neurons!: THREE.InstancedMesh<THREE.SphereGeometry, THREE.ShaderMaterial>;
  private biasNodes!: THREE.InstancedMesh<THREE.OctahedronGeometry, THREE.ShaderMaterial>;
  private nTarget!: Float32Array;
  private nDisp!: Float32Array;
  private eTarget!: Float32Array;
  private eDisp!: Float32Array;

  // connections
  private conns!: Connection[];
  private connMesh!: THREE.InstancedMesh<THREE.CylinderGeometry, THREE.MeshBasicMaterial>;
  private connBasis!: Float32Array;
  private wTarget!: Float32Array;
  private wDisp!: Float32Array;
  private hlF!: Float32Array;
  private hlB!: Float32Array;
  private flash!: Float32Array;
  private bundleScale!: Float32Array;

  // pulses
  private pulseCount!: number;
  private pulseNext!: number;
  private pulse!: {
    from: Float32Array;
    to: Float32Array;
    rgb: Float32Array;
    t: Float32Array;
    dur: Float32Array;
    strength: Float32Array;
    conn: Int32Array;
    back: Uint8Array;
    active: Uint8Array;
  };
  private pulsePoints!: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>;

  // misc
  private cardCanvas!: HTMLCanvasElement;
  private cardTexture!: THREE.CanvasTexture;
  private card!: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  private targetRing!: Ring;
  private winnerRing!: Ring;
  private outputLabels!: { obj: CSS2DObject; pct: HTMLElement; last: number }[];
  private stars!: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>;

  constructor(container: HTMLElement, model: NeuralNetworkModel, tooltipEl: HTMLElement) {
    this.container = container;
    this.model = model;
    this.sizes = model.sizes;
    this.tooltipEl = tooltipEl;

    this.initRenderer();
    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.buildLayout();
    this.buildNeurons();
    this.buildConnections();
    this.buildPulses();
    this.buildCard();
    this.buildRings();
    this.buildLabels();
    this.buildStars();
    this.bindPointer();

    this.setWeights(model, false);
    this.clearDrawing();
  }

  // ───────────────────────────── setup ─────────────────────────────

  private initRenderer(): void {
    const { clientWidth: w, clientHeight: h } = this.container;
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    this.container.appendChild(renderer.domElement);
    this.renderer = renderer;

    this.scene = new THREE.Scene();
    this.scene.background = C("#070b18");
    this.scene.fog = new THREE.FogExp2("#070b18", 0.011);

    this.camera = new THREE.PerspectiveCamera(38, w / h, 0.1, 300);
    this.camera.position.copy(viewPos(CAMERA_VIEWS.training, w / h));

    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.target.copy(CAMERA_VIEWS.training.target);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.rotateSpeed = 0.6;
    this.controls.minDistance = 6;
    this.controls.maxDistance = 45;

    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.setSize(w, h);
    this.labelRenderer.domElement.className = "label-layer";
    this.container.appendChild(this.labelRenderer.domElement);

    this.composer = new EffectComposer(renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new UnrealBloomPass(new THREE.Vector2(w, h), 0.5, 0.45, 0.32));
    this.composer.addPass(new OutputPass());

    window.addEventListener("resize", () => this.resize());
  }

  private resize(): void {
    const { clientWidth: w, clientHeight: h } = this.container;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
    this.labelRenderer.setSize(w, h);
  }

  /** Positions of every neuron, laid out as planes perpendicular to the x axis */
  private buildLayout(): void {
    const [nIn, nH1, nH2, nOut] = this.sizes;
    this.offsets = [0, nIn, nIn + nH1, nIn + nH1 + nH2];
    this.neuronCount = nIn + nH1 + nH2 + nOut;
    this.neuronPos = [];
    this.neuronLayer = [];
    this.neuronIndex = [];
    const add = (l: number, i: number, y: number, z: number) => {
      this.neuronPos.push(new THREE.Vector3(LAYER_X[l], y, z));
      this.neuronLayer.push(l);
      this.neuronIndex.push(i);
    };
    // input: the 10×10 pixel image, row 0 at the top, columns running toward +z
    for (let i = 0; i < nIn; i++)
      add(0, i, ((GRID - 1) / 2 - Math.floor(i / GRID)) * 0.5, ((i % GRID) - (GRID - 1) / 2) * 0.5);
    // hidden layer #1: 4×4 grid, hidden layer #2: 5×2 grid
    for (let i = 0; i < nH1; i++) add(1, i, (1.5 - Math.floor(i / 4)) * 0.95, ((i % 4) - 1.5) * 0.95);
    for (let i = 0; i < nH2; i++) add(2, i, (2 - Math.floor(i / 2)) * 0.95, ((i % 2) - 0.5) * 0.95);
    // output: vertical column
    for (let i = 0; i < nOut; i++) add(3, i, (1 - i) * 1.5, 0);
    this.biasPos = [0, 1, 2].map((l) => new THREE.Vector3(LAYER_X[l], BIAS_Y, 0));
  }

  private buildNeurons(): void {
    const n = this.neuronCount;
    this.nTarget = new Float32Array(n); // activation shown (0..1)
    this.nDisp = new Float32Array(n);
    this.eTarget = new Float32Array(n); // backprop error glow (0..1)
    this.eDisp = new Float32Array(n);

    this.neurons = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 28, 18), glowMaterial(), n);
    this.biasNodes = new THREE.InstancedMesh(new THREE.OctahedronGeometry(1), glowMaterial(), 3);
    const m = new THREE.Matrix4();
    for (let k = 0; k < n; k++) {
      m.makeScale(1.3, 1.3, 1.3)
        .scale(new THREE.Vector3().setScalar(NEURON_RADIUS[this.neuronLayer[k]]))
        .setPosition(this.neuronPos[k]);
      this.neurons.setMatrixAt(k, m);
      this.neurons.setColorAt(k, COLORS.layers[this.neuronLayer[k]]);
    }
    for (let l = 0; l < 3; l++) {
      this.biasNodes.setMatrixAt(l, m.makeScale(0.22, 0.22, 0.22).setPosition(this.biasPos[l]));
      this.biasNodes.setColorAt(l, COLORS.bias);
    }
    this.neurons.computeBoundingSphere();
    this.neurons.frustumCulled = this.biasNodes.frustumCulled = false;
    this.group.add(this.neurons, this.biasNodes);
  }

  /** One instanced cylinder per weight and per bias */
  private buildConnections(): void {
    const conns: Connection[] = [];
    for (let l = 0; l < 3; l++) {
      const nIn = this.sizes[l];
      const nOut = this.sizes[l + 1];
      for (let j = 0; j < nOut; j++) {
        for (let i = 0; i < nIn; i++)
          conns.push({
            bundle: l,
            l,
            i,
            j,
            a: this.neuronPos[this.offsets[l] + i],
            b: this.neuronPos[this.offsets[l + 1] + j],
          });
      }
    }
    for (let l = 0; l < 3; l++) {
      for (let j = 0; j < this.sizes[l + 1]; j++)
        conns.push({
          bundle: 3 + l,
          l,
          i: -1,
          j,
          a: this.biasPos[l],
          b: this.neuronPos[this.offsets[l + 1] + j],
        });
    }
    this.conns = conns;
    const n = conns.length;
    this.wTarget = new Float32Array(n);
    this.wDisp = new Float32Array(n);
    this.hlF = new Float32Array(n); // forward-pass highlight
    this.hlB = new Float32Array(n); // backprop highlight
    this.flash = new Float32Array(n); // weight-update flash
    this.bundleScale = new Float32Array(6).fill(1);

    // Precompute each cylinder's rotation basis, midpoint and length
    this.connBasis = new Float32Array(n * 13);
    const q = new THREE.Quaternion();
    const rot = new THREE.Matrix4();
    const dir = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    conns.forEach((c, k) => {
      dir.subVectors(c.b, c.a);
      const len = dir.length();
      rot.makeRotationFromQuaternion(q.setFromUnitVectors(up, dir.normalize()));
      const e = rot.elements;
      const o = k * 13;
      for (let t = 0; t < 3; t++) {
        this.connBasis[o + t] = e[t];
        this.connBasis[o + 3 + t] = e[4 + t];
        this.connBasis[o + 6 + t] = e[8 + t];
      }
      this.connBasis[o + 9] = (c.a.x + c.b.x) / 2;
      this.connBasis[o + 10] = (c.a.y + c.b.y) / 2;
      this.connBasis[o + 11] = (c.a.z + c.b.z) / 2;
      this.connBasis[o + 12] = len;
    });

    const mat = new THREE.MeshBasicMaterial({
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.connMesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 6, 1, true), mat, n);
    this.connMesh.setColorAt(0, COLORS.white);
    this.connMesh.frustumCulled = false;
    this.writeConnections(0);
    this.connMesh.computeBoundingSphere();
    this.group.add(this.connMesh);
  }

  private buildPulses(): void {
    const N = 1400;
    this.pulseCount = N;
    this.pulseNext = 0;
    this.pulse = {
      from: new Float32Array(N * 3),
      to: new Float32Array(N * 3),
      rgb: new Float32Array(N * 3),
      t: new Float32Array(N),
      dur: new Float32Array(N),
      strength: new Float32Array(N),
      conn: new Int32Array(N).fill(-1),
      back: new Uint8Array(N),
      active: new Uint8Array(N),
    };
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
    this.pulsePoints = new THREE.Points(
      geo,
      new THREE.PointsMaterial({
        size: 0.26,
        map: glowTexture(),
        vertexColors: true,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true,
      }),
    );
    this.pulsePoints.frustumCulled = false;
    this.group.add(this.pulsePoints);
  }

  /** The floating "drawing" card left of the input layer */
  private buildCard(): void {
    this.cardCanvas = document.createElement("canvas");
    this.cardCanvas.width = this.cardCanvas.height = CARD_PX;
    this.cardTexture = new THREE.CanvasTexture(this.cardCanvas);
    this.cardTexture.colorSpace = THREE.SRGBColorSpace;
    this.card = new THREE.Mesh(
      new THREE.PlaneGeometry(CARD_SIZE, CARD_SIZE),
      new THREE.MeshBasicMaterial({
        map: this.cardTexture,
        color: C("#b8c2dd"),
        transparent: true,
      }),
    );
    this.card.position.set(CARD_X, CARD_Y, 0);
    this.group.add(this.card);
  }

  private buildRings(): void {
    const ring = (color: THREE.Color): Ring => {
      const mesh = new THREE.Mesh(
        new THREE.TorusGeometry(0.62, 0.028, 8, 64),
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity: 0,
          depthWrite: false,
        }),
      );
      this.group.add(mesh);
      return { mesh, index: -1, opacity: 0 };
    };
    this.targetRing = ring(C("#ffffff").multiplyScalar(1.2));
    this.winnerRing = ring(C("#ff7aa2").multiplyScalar(2.2));
  }

  private buildLabels(): void {
    const titles = [
      ["Input", `${this.sizes[0]} Pixel<br>${GRID}×${GRID}-Bild`],
      ["Hidden Layer 1", `${this.sizes[1]} Neuronen<br>Aktivierung: ReLU`],
      ["Hidden Layer 2", `${this.sizes[2]} Neuronen<br>Aktivierung: ReLU`],
      ["Output", `${this.sizes[3]} Klassen<br>Softmax → Vorhersage`],
    ];
    this.labels = new THREE.Group();
    this.group.add(this.labels);
    const cls = ["l-input", "l-hidden", "l-hidden", "l-output"];
    titles.forEach(([t, sub], l) => {
      const o = label(`<b>${t}</b><span>${sub}</span>`, `layer-title ${cls[l]}`, [0.5, 1]);
      o.position.set(LAYER_X[l], l === 3 ? 4.7 : 3.6, 0);
      this.labels.add(o);
    });

    const card = label(
      '<b>Zeichnung</b><span id="card-caption">Trainingsbeispiel</span>',
      "layer-title l-card",
      [0.5, 1],
    );
    card.position.set(CARD_X, CARD_Y + CARD_SIZE / 2 + 0.35, 0);
    this.labels.add(card);

    this.biasPos.forEach((p, l) => {
      const o = label(`Bias <span>+1 → ${LAYER_NAMES[l + 1]}</span>`, "bias-label", [0.5, 0]);
      o.position.set(p.x, p.y - 0.35, p.z);
      this.labels.add(o);
    });

    // educational callouts
    const neuron = label("Neuron", "callout", [0, 1]);
    neuron.position.copy(this.neuronPos[this.offsets[1]]).add(new THREE.Vector3(0, 0.3, -0.4));
    const weights = label("Gewichte", "callout", [0.5, 1]);
    weights.position
      .lerpVectors(this.neuronPos[this.offsets[1] + 0], this.neuronPos[this.offsets[2] + 0], 0.5)
      .add(new THREE.Vector3(0, 0.35, 0));
    this.labels.add(neuron, weights);

    // output class labels with live probabilities
    this.outputLabels = CLASSES.map((name, i) => {
      const o = label(`<i>${CLASS_ICONS[i]}</i><b>${name}</b><em>0 %</em>`, "output-label", [0, 0.5]);
      o.position.copy(this.neuronPos[this.offsets[3] + i]).add(new THREE.Vector3(0, 0, 0.62));
      this.group.add(o); // always visible, even when labels are toggled off
      return { obj: o, pct: o.element.querySelector("em")!, last: -1 };
    });
  }

  private buildStars(): void {
    const N = 1200;
    const pos = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const v = new THREE.Vector3().randomDirection().multiplyScalar(45 + Math.random() * 60);
      pos.set([v.x, v.y, v.z], i * 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.stars = new THREE.Points(
      geo,
      new THREE.PointsMaterial({
        color: C("#7f8fc0"),
        size: 0.25,
        transparent: true,
        opacity: 0.55,
        depthWrite: false,
      }),
    );
    this.scene.add(this.stars);
  }

  private bindPointer(): void {
    const el = this.renderer.domElement;
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect();
      this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.pointerClient = [e.clientX, e.clientY];
      this.pointerDirty = true;
    });
    el.addEventListener("pointerleave", () => this.setHover(null));
  }

  // ───────────────────────────── public API ─────────────────────────────

  schedule(delay: number, fn: () => void): void {
    this.timers.push({ at: this.simClock + delay, fn });
  }

  setLabelsVisible(visible: boolean): void {
    this.labels.visible = visible;
  }

  setMode(mode: Mode): void {
    const view = CAMERA_VIEWS[mode];
    this.cameraTween = {
      t: 0,
      fromPos: this.camera.position.clone(),
      fromTarget: this.controls.target.clone(),
      toPos: viewPos(view, this.camera.aspect),
      toTarget: view.target,
    };
    $("card-caption").textContent = mode === "inference" ? "deine Zeichnung" : "Trainingsbeispiel";
  }

  /** Copy weights from the model. With `animate`, lines fade to the new values and flash by |Δw| */
  setWeights(model: NeuralNetworkModel, animate = true): void {
    const sums = new Float32Array(6);
    const counts = new Float32Array(6);
    const maxDelta = new Float32Array(6);
    this.conns.forEach((c, k) => {
      const w = c.i < 0 ? model.b[c.l][c.j] : model.W[c.l][c.j * this.sizes[c.l] + c.i];
      const d = Math.abs(w - this.wTarget[k]);
      if (animate) this.flash[k] = d;
      else this.wDisp[k] = w;
      this.wTarget[k] = w;
      sums[c.bundle] += Math.abs(w);
      counts[c.bundle]++;
      if (d > maxDelta[c.bundle]) maxDelta[c.bundle] = d;
    });
    for (let b = 0; b < 6; b++)
      this.bundleScale[b] = Math.max(b < 3 ? 0.05 : 0.02, (sums[b] / counts[b]) * 2.2);
    if (animate)
      this.conns.forEach((c, k) => {
        this.flash[k] = maxDelta[c.bundle] > 0 ? (this.flash[k] / maxDelta[c.bundle]) ** 1.5 : 0;
      });
  }

  clearActivity({ immediate = false } = {}): void {
    this.nTarget.fill(0);
    this.eTarget.fill(0);
    if (immediate) {
      this.nDisp.set(this.nTarget);
      this.eDisp.fill(0);
      this.hlF.fill(0);
      this.hlB.fill(0);
      this.flash.fill(0);
      this.pulse.active.fill(0);
      this.timers = [];
    }
    this.targetRing.index = this.winnerRing.index = -1;
  }

  clearDrawing(): void {
    this.drawCard(null);
  }

  private drawCard(strokes: Stroke[] | null, extent = 100): void {
    const ctx = this.cardCanvas.getContext("2d")!;
    const S = CARD_PX;
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = "rgba(14, 22, 44, 0.92)";
    ctx.strokeStyle = "rgba(120, 170, 255, 0.55)";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.roundRect(4, 4, S - 8, S - 8, 22);
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = "rgba(120,170,255,0.08)";
    ctx.lineWidth = 1;
    for (let i = 1; i < GRID; i++) {
      const p = 8 + (i * (S - 16)) / GRID;
      ctx.beginPath();
      ctx.moveTo(p, 8);
      ctx.lineTo(p, S - 8);
      ctx.moveTo(8, p);
      ctx.lineTo(S - 8, p);
      ctx.stroke();
    }
    if (strokes?.length) {
      const s = (S - 24) / extent;
      ctx.strokeStyle = "#f2f6ff";
      ctx.lineWidth = 7;
      ctx.lineCap = ctx.lineJoin = "round";
      for (const stroke of strokes) {
        ctx.beginPath();
        stroke.forEach(([x, y], i) =>
          i ? ctx.lineTo(12 + x * s, 12 + y * s) : ctx.moveTo(12 + x * s, 12 + y * s),
        );
        if (stroke.length === 1) ctx.lineTo(12 + stroke[0][0] * s + 0.1, 12 + stroke[0][1] * s);
        ctx.stroke();
      }
    } else {
      ctx.fillStyle = "rgba(170,190,240,0.5)";
      ctx.font = "600 20px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("wartet auf Eingabe …", S / 2, S / 2 + 7);
    }
    this.cardTexture.needsUpdate = true;
  }

  /**
   * Draw the sample on the card, then send one pulse per inked pixel from
   * the card into its input neuron
   */
  showInput(strokes: Stroke[], extent: number, pixels: Float32Array, duration: number): void {
    this.clearActivity();
    this.drawCard(strokes, extent);
    this.pass = null;
    this.inputPixels = pixels;
    const travel = duration * 0.45;
    const local = new THREE.Vector3();
    this.card.updateMatrixWorld();
    for (let r = 0; r < GRID; r++) {
      const delay = (r / GRID) * duration * 0.4;
      for (let c = 0; c < GRID; c++) {
        const i = r * GRID + c;
        if (pixels[i] < 0.1) continue;
        local.set(
          ((c + 0.5) / GRID - 0.5) * CARD_SIZE * 0.85,
          (0.5 - (r + 0.5) / GRID) * CARD_SIZE * 0.85,
          0.02,
        );
        this.card.localToWorld(local);
        this.group.worldToLocal(local);
        this.spawnPulse(local, this.neuronPos[i], {
          delay,
          dur: travel,
          color: COLORS.layers[0],
          strength: 0.4 + 0.6 * pixels[i],
        });
      }
      this.schedule(delay + travel, () => {
        for (let c = 0; c < GRID; c++) this.nTarget[r * GRID + c] = pixels[r * GRID + c];
      });
    }
  }

  /**
   * Forward step through weight layer l (l → l+1). Pulses travel along the connections
   * that contribute most (|w · a|); when they arrive, layer l+1 lights up
   */
  forward(l: number, pass: ForwardPass, duration: number): void {
    this.pass = pass;
    const a = pass.acts[l];
    const contrib: [number, number][] = [];
    this.conns.forEach((c, k) => {
      if (c.l !== l) return;
      const v = c.i < 0 ? Math.abs(this.wTarget[k]) : Math.abs(this.wTarget[k] * a[c.i]);
      if (v > 1e-4) contrib.push([k, v]);
    });
    contrib.sort((x, y) => y[1] - x[1]);
    const top = contrib.slice(0, FORWARD_PULSES[l]);
    const max = top[0]?.[1] || 1;
    for (const [k, v] of top) {
      const c = this.conns[k];
      this.spawnPulse(c.a, c.b, {
        delay: Math.random() * duration * 0.25,
        dur: duration * 0.55,
        color: COLORS.forward,
        strength: 0.35 + 0.65 * (v / max),
        conn: k,
      });
    }
    this.schedule(duration * 0.75, () => this.setLayerActivations(l + 1, pass.acts[l + 1]));
  }

  /** Normalise a layer's activations to 0..1 for display */
  private setLayerActivations(layer: number, acts: Float32Array): void {
    const max = layer === 3 ? 1 : Math.max(1e-6, ...acts);
    for (let i = 0; i < acts.length; i++) this.nTarget[this.offsets[layer] + i] = acts[i] / max;
  }

  setRings({ target = -1, winner = -1 } = {}): void {
    this.targetRing.index = target;
    this.winnerRing.index = winner;
  }

  /**
   * Backprop step: the error starts at the outputs and travels back one layer at a time.
   * Neurons glow violet by |δ| (their share of the blame). Pulses follow the weights
   * with the biggest batch gradients |∂L/∂w|
   */
  backward(deltas: Float32Array[], grads: Gradients, duration: number): void {
    const wave = duration / 3;
    for (let step = 0; step < 3; step++) {
      const l = 2 - step;
      this.schedule(step * wave, () => {
        const d = deltas[l];
        const dMax = Math.max(1e-6, ...d.map(Math.abs));
        for (let j = 0; j < d.length; j++) this.eTarget[this.offsets[l + 1] + j] = Math.abs(d[j]) / dMax;
        const nIn = this.sizes[l];
        const list: [number, number][] = [];
        this.conns.forEach((c, k) => {
          if (c.l !== l) return;
          const g = Math.abs(c.i < 0 ? grads.gb[l][c.j] : grads.gW[l][c.j * nIn + c.i]);
          if (g > 1e-7) list.push([k, g]);
        });
        list.sort((x, y) => y[1] - x[1]);
        const top = list.slice(0, BACKWARD_PULSES[l]);
        const max = top[0]?.[1] || 1;
        for (const [k, g] of top) {
          const c = this.conns[k];
          this.spawnPulse(c.b, c.a, {
            delay: Math.random() * wave * 0.25,
            dur: wave * 0.85,
            color: COLORS.backward,
            strength: 0.35 + 0.65 * (g / max),
            conn: k,
            back: true,
          });
        }
      });
    }
  }

  /** Weights morph to their new values, changed lines flash, blame fades */
  applyUpdate(model: NeuralNetworkModel): void {
    this.setWeights(model, true);
    this.eTarget.fill(0);
  }

  private spawnPulse(
    from: THREE.Vector3,
    to: THREE.Vector3,
    { delay = 0, dur = 0.5, color, strength = 1, conn = -1, back = false }: PulseOptions,
  ): void {
    const p = this.pulse;
    const k = this.pulseNext;
    this.pulseNext = (k + 1) % this.pulseCount;
    p.from.set([from.x, from.y, from.z], k * 3);
    p.to.set([to.x, to.y, to.z], k * 3);
    p.rgb.set([color.r, color.g, color.b], k * 3);
    p.t[k] = -delay;
    p.dur[k] = Math.max(dur, 1e-3);
    p.strength[k] = strength;
    p.conn[k] = conn;
    p.back[k] = back ? 1 : 0;
    p.active[k] = 1;
  }

  // ───────────────────────────── per frame ─────────────────────────────

  update(dt: number, simDt: number): void {
    this.time += dt;
    this.simClock += simDt;
    if (this.timers.length) {
      const due = this.timers.filter((t) => t.at <= this.simClock).sort((a, b) => a.at - b.at);
      if (due.length) {
        this.timers = this.timers.filter((t) => t.at > this.simClock);
        due.forEach((t) => t.fn());
      }
    }

    // idle motion
    this.group.position.y = Math.sin(this.time * 0.5) * 0.08;
    this.group.rotation.y = Math.sin(this.time * 0.13) * 0.035;
    this.stars.rotation.y += dt * 0.004;

    this.updatePulses(simDt);
    this.updateNeurons(simDt);
    this.writeConnections(simDt);
    this.updateRings(dt);
    this.updateOutputLabels();
    this.card.lookAt(this.camera.position);
    this.updateCamera(dt);
    if (this.pointerDirty) {
      this.pointerDirty = false;
      this.pick();
    }

    this.controls.update();
    this.composer.render();
    this.labelRenderer.render(this.scene, this.camera);
  }

  private updatePulses(simDt: number): void {
    const p = this.pulse;
    const pos = this.pulsePoints.geometry.attributes.position;
    const col = this.pulsePoints.geometry.attributes.color;
    for (let k = 0; k < this.pulseCount; k++) {
      const o = k * 3;
      if (!p.active[k]) {
        col.array[o] = col.array[o + 1] = col.array[o + 2] = 0;
        continue;
      }
      p.t[k] += simDt;
      const u = p.t[k] / p.dur[k];
      if (u > 1) {
        p.active[k] = 0;
        col.array[o] = col.array[o + 1] = col.array[o + 2] = 0;
        continue;
      }
      if (u < 0) {
        col.array[o] = col.array[o + 1] = col.array[o + 2] = 0;
        continue;
      }
      const eased = 0.5 - 0.5 * Math.cos(Math.PI * u); // ease in-out
      for (let a = 0; a < 3; a++) pos.array[o + a] = p.from[o + a] + (p.to[o + a] - p.from[o + a]) * eased;
      const glow = Math.sin(Math.PI * u) ** 0.5 * p.strength[k] * 1.1;
      for (let a = 0; a < 3; a++) col.array[o + a] = p.rgb[o + a] * glow;
      const c = p.conn[k];
      if (c >= 0) {
        const arr = p.back[k] ? this.hlB : this.hlF;
        arr[c] = Math.max(arr[c], p.strength[k] * 0.9);
      }
    }
    pos.needsUpdate = col.needsUpdate = true;
  }

  private updateNeurons(simDt: number): void {
    const kN = 1 - Math.exp(-simDt * 7);
    const arr = this.neurons.instanceMatrix.array;
    const colArr = this.neurons.instanceColor!.array;
    const tmp = new THREE.Color();
    const hoverN = this.hover?.kind === "neuron" ? this.hover.k : -1;
    for (let k = 0; k < this.neuronCount; k++) {
      this.nDisp[k] += (this.nTarget[k] - this.nDisp[k]) * kN;
      this.eDisp[k] += (this.eTarget[k] - this.eDisp[k]) * kN;
      const l = this.neuronLayer[k];
      const v = this.nDisp[k];
      const e = this.eDisp[k];
      tmp.copy(COLORS.layers[l]).multiplyScalar(0.07 + 0.95 * v);
      if (e > 0.01) tmp.lerp(COLORS.backward.clone().multiplyScalar(0.5 + 1.2 * e), Math.min(1, e * 0.9));
      if (k === hoverN) tmp.addScalar(0.3);
      colArr.set([tmp.r, tmp.g, tmp.b], k * 3);
      const s =
        NEURON_RADIUS[l] *
        (0.85 + 0.35 * v + 0.2 * e) *
        (k === hoverN ? 1.35 : 1) *
        (1 + 0.035 * Math.sin(this.time * 2.1 + k * 1.37));
      const p = this.neuronPos[k];
      const o = k * 16;
      arr[o] = s;
      arr[o + 5] = s;
      arr[o + 10] = s;
      arr[o + 12] = p.x;
      arr[o + 13] = p.y;
      arr[o + 14] = p.z;
    }
    this.neurons.instanceMatrix.needsUpdate = true;
    this.neurons.instanceColor!.needsUpdate = true;

    const hoverB = this.hover?.kind === "bias" ? this.hover.k : -1;
    for (let l = 0; l < 3; l++) {
      const s = 0.22 * (l === hoverB ? 1.35 : 1) * (1 + 0.05 * Math.sin(this.time * 1.7 + l));
      const m = new THREE.Matrix4()
        .makeRotationY(this.time * 0.6)
        .scale(new THREE.Vector3(s, s, s))
        .setPosition(this.biasPos[l]);
      this.biasNodes.setMatrixAt(l, m);
      this.biasNodes.setColorAt(l, tmp.copy(COLORS.bias).multiplyScalar(l === hoverB ? 1.6 : 1.05));
    }
    this.biasNodes.instanceMatrix.needsUpdate = true;
    this.biasNodes.instanceColor!.needsUpdate = true;
  }

  /** Weight -> cylinder radius + colour, plus forward/backward/update highlights */
  private writeConnections(simDt: number): void {
    const kW = 1 - Math.exp(-simDt * 4);
    const decay = Math.exp(-simDt * 2.5);
    const decayFlash = Math.exp(-simDt * 1.6);
    const matrix = this.connMesh.instanceMatrix.array;
    const color = this.connMesh.instanceColor!.array;
    const basis = this.connBasis;
    const hoverSet = this.hover?.conns;
    const { positive, negative, forward, backward } = COLORS;
    for (let k = 0; k < this.conns.length; k++) {
      this.wDisp[k] += (this.wTarget[k] - this.wDisp[k]) * kW;
      this.hlF[k] *= decay;
      this.hlB[k] *= decay;
      this.flash[k] *= decayFlash;
      const b = this.conns[k].bundle;
      const w = this.wDisp[k];
      const mag = Math.min(1.5, Math.abs(w) / this.bundleScale[b]);
      const hs = BUNDLE_HIGHLIGHT[b];
      const hl = (this.hlF[k] + this.hlB[k]) * hs;
      const hov = hoverSet?.has(k) ? 1 : 0;
      const r = BUNDLE_RADIUS[b] * (0.3 + 0.85 * mag) * (1 + 0.8 * hl + 0.8 * this.flash[k] + 1.5 * hov);
      const o = k * 16;
      const q = k * 13;
      const len = basis[q + 12];
      matrix[o] = basis[q] * r;
      matrix[o + 1] = basis[q + 1] * r;
      matrix[o + 2] = basis[q + 2] * r;
      matrix[o + 3] = 0;
      matrix[o + 4] = basis[q + 3] * len;
      matrix[o + 5] = basis[q + 4] * len;
      matrix[o + 6] = basis[q + 5] * len;
      matrix[o + 7] = 0;
      matrix[o + 8] = basis[q + 6] * r;
      matrix[o + 9] = basis[q + 7] * r;
      matrix[o + 10] = basis[q + 8] * r;
      matrix[o + 11] = 0;
      matrix[o + 12] = basis[q + 9];
      matrix[o + 13] = basis[q + 10];
      matrix[o + 14] = basis[q + 11];
      matrix[o + 15] = 1;
      const base = w >= 0 ? positive : negative;
      const bright = BUNDLE_BRIGHT[b] * Math.min(1.2, 0.15 + mag) * (hoverSet && !hov ? 0.35 : 1) + hov * 0.9;
      const fl = this.flash[k] * 0.5 * hs;
      const c3 = k * 3;
      const f = this.hlF[k] * hs;
      const bk = this.hlB[k] * hs;
      color[c3] = base.r * bright + forward.r * f + backward.r * bk + fl;
      color[c3 + 1] = base.g * bright + forward.g * f + backward.g * bk + fl;
      color[c3 + 2] = base.b * bright + forward.b * f + backward.b * bk + fl;
    }
    this.connMesh.instanceMatrix.needsUpdate = true;
    this.connMesh.instanceColor!.needsUpdate = true;
  }

  private updateRings(dt: number): void {
    for (const ring of [this.targetRing, this.winnerRing]) {
      const want = ring.index >= 0 ? 1 : 0;
      ring.opacity += (want - ring.opacity) * (1 - Math.exp(-dt * 6));
      ring.mesh.material.opacity =
        ring.opacity * (ring === this.winnerRing ? 0.75 + 0.25 * Math.sin(this.time * 4) : 0.9);
      if (ring.index >= 0) ring.mesh.position.copy(this.neuronPos[this.offsets[3] + ring.index]);
      ring.mesh.scale.setScalar(ring === this.targetRing ? 1.18 : 1);
      ring.mesh.visible = ring.opacity > 0.01;
      ring.mesh.lookAt(this.camera.position);
    }
  }

  private updateOutputLabels(): void {
    this.outputLabels.forEach((o, i) => {
      const pct = Math.round(this.nDisp[this.offsets[3] + i] * 100);
      if (pct !== o.last) {
        o.pct.textContent = `${pct} %`;
        o.last = pct;
      }
      o.obj.element.classList.toggle("win", this.winnerRing.index === i);
    });
  }

  private updateCamera(dt: number): void {
    const tw = this.cameraTween;
    if (!tw) return;
    tw.t = Math.min(1, tw.t + dt / 1.6);
    const e = tw.t < 0.5 ? 4 * tw.t ** 3 : 1 - (-2 * tw.t + 2) ** 3 / 2;
    this.camera.position.lerpVectors(tw.fromPos, tw.toPos, e);
    this.controls.target.lerpVectors(tw.fromTarget, tw.toTarget, e);
    if (tw.t >= 1) this.cameraTween = null;
  }

  // ───────────────────────────── hover ─────────────────────────────

  private pick(): void {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = (mesh: THREE.Object3D) => this.raycaster.intersectObject(mesh, false)[0];
    let found = hit(this.neurons);
    if (found) return this.setHover({ kind: "neuron", k: found.instanceId! });
    found = hit(this.biasNodes);
    if (found) return this.setHover({ kind: "bias", k: found.instanceId! });
    found = hit(this.connMesh);
    if (found) return this.setHover({ kind: "conn", k: found.instanceId! });
    this.setHover(null);
  }

  private setHover(target: HoverTarget | null): void {
    if (!target) this.hover = null;
    else if (target.kind !== this.hover?.kind || target.k !== this.hover.k)
      this.hover = { ...target, conns: this.connectionsOf(target) };
    const el = this.tooltipEl;
    if (!target) {
      el.hidden = true;
      this.renderer.domElement.style.cursor = "";
      return;
    }
    el.innerHTML = this.tooltipHtml(this.hover!);
    el.hidden = false;
    const [x, y] = this.pointerClient;
    el.style.left = `${Math.min(x + 16, window.innerWidth - el.offsetWidth - 12)}px`;
    el.style.top = `${Math.min(y + 16, window.innerHeight - el.offsetHeight - 12)}px`;
    this.renderer.domElement.style.cursor = "help";
  }

  /** Connections to highlight while hovering: all lines touching a neuron / bias node, or the line itself */
  private connectionsOf(target: HoverTarget): Set<number> {
    const set = new Set<number>();
    if (target.kind === "conn") return set.add(target.k);
    const l = target.kind === "bias" ? target.k : this.neuronLayer[target.k];
    const i = target.kind === "bias" ? -1 : this.neuronIndex[target.k];
    this.conns.forEach((c, k) => {
      if (
        target.kind === "bias"
          ? c.i < 0 && c.l === l
          : (c.l === l && c.i === i) || (c.l === l - 1 && c.j === i)
      )
        set.add(k);
    });
    return set;
  }

  private neuronName(l: number, i: number): string {
    if (l === 0) return `Pixel (Zeile ${Math.floor(i / GRID) + 1}, Spalte ${(i % GRID) + 1})`;
    if (l === 3) return CLASSES[i];
    return `${LAYER_NAMES[l]} #${i + 1}`;
  }

  private tooltipHtml(hover: Hover): string {
    const p = this.pass;
    if (hover.kind === "neuron") {
      const l = this.neuronLayer[hover.k];
      const i = this.neuronIndex[hover.k];
      if (l === 0) {
        const x = this.inputPixels?.[i];
        return `<h4 class="t-input">Eingabeneuron</h4><p>${this.neuronName(0, i)}</p>
          <p class="mono">x = ${x == null ? "—" : num(x)}</p>
          <p class="muted">Wie viel Tinte in diesem Pixel liegt (0 = leer, 1 = voll). Der Wert geht an alle ${this.sizes[1]} Neuronen der verborgenen Schicht 1.</p>`;
      }
      const z = p?.zs[l - 1][i];
      const a = p?.acts[l][i];
      const b = this.model.b[l - 1][i];
      if (l === 3) {
        return `<h4 class="t-output">Ausgabeneuron · ${CLASS_ICONS[i]} ${CLASSES[i]}</h4>
          <p class="mono">z = ${formatSigned(z)} → Softmax → ${a == null ? "—" : pct(a, 1)}</p>
          <p class="mono">Bias b = ${formatSigned(b)}</p>
          <p class="muted">Sein Wert z wird zu einer Wahrscheinlichkeit. Das hellste Ausgabeneuron ist die Vorhersage.</p>`;
      }
      return `<h4 class="t-hidden">Neuron · ${LAYER_NAMES[l]} #${i + 1}</h4>
        <p class="mono">z = w·x + b = ${formatSigned(z)}</p>
        <p class="mono">a = ReLU(z) = ${a == null ? "—" : num(a)}</p>
        <p class="mono">Bias b = ${formatSigned(b)}</p>
        <p class="muted">Addiert seine ${this.sizes[l - 1]} gewichteten Eingaben plus Bias. ReLU lässt positive Werte durch und macht negative zu 0 – das Neuron ist dann „aus“.</p>`;
    }
    if (hover.kind === "bias") {
      return `<h4 class="t-bias">Bias-Knoten (konstant +1)</h4>
        <p class="muted">Seine Verbindungsstärken sind die Biases b der Schicht ${LAYER_NAMES[hover.k + 1]}. Ein Bias verschiebt, wie leicht ein Neuron anspringt – unabhängig von der Eingabe.</p>`;
    }
    const c = this.conns[hover.k];
    const w = this.wTarget[hover.k];
    const from = c.i < 0 ? "Bias (+1)" : this.neuronName(c.l, c.i);
    return `<h4 class="${w >= 0 ? "t-pos" : "t-neg"}">${c.i < 0 ? "Bias-Gewicht" : "Gewicht"}</h4>
      <p>${from} → ${this.neuronName(c.l + 1, c.j)}</p>
      <p class="mono">${c.i < 0 ? "b" : "w"} = ${formatSigned(w)}</p>
      <p class="muted">${w >= 0 ? "Positiv (cyan): regt" : "Negativ (orange): hemmt"} das empfangende Neuron${w >= 0 ? " an" : ""}. Dicker & heller = stärker. Das Training verändert diese Zahl bei jedem Schritt ein kleines bisschen.</p>`;
  }
}
