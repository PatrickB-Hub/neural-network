# Neural Network

An interactive 3D demo of a small neural network learning to recognise hand-drawn
**triangles, rectangles and circles**. Watch it train (forward pass, loss, backpropagation,
weight updates), then draw your own shape and watch the network predict it.

```bash
npm install
npm run dev     # http://localhost:5173
npm run typecheck
npm run format  # Prettier for src/ and test/
npm run build   # type-check + static build in dist/
```

## How it works

Written in TypeScript (strict) with Sass for styling; bundled by Vite.

- **Network:** 100 inputs (a 10×10 pixel image) → 16 ReLU → 10 ReLU → 3 softmax outputs,
  trained with cross-entropy loss, backpropagation and Adam. Plain JS, no ML library.
- **Data:** 270 synthetic "hand-drawn" shapes generated on startup (wobbly strokes,
  random rotation, aspect ratio and unclosed ends), plus 90 held-out ones for accuracy.
- **Features:** each drawing is cropped to its bounding box, stretched to fill the grid
  and rasterised to 10×10. Training data and your drawing go through the same function.
- **Visualisation:** each training cycle is split into phases on a pausable simulation
  clock: _input → forward ×3 → loss → backprop → update_. One sample is shown in detail,
  while the math runs on a mini-batch of 8.
