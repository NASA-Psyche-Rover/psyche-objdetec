# psyche-objdetec

Perception and navigation-decision system for an autonomous rover designed to traverse
the surface of asteroid Psyche. The project's focus is **mobility**: seeing what's in
front of the rover (obstacles) and what the ground looks like (slope, roughness,
drop-offs), and turning that into a real-time PROCEED / CAUTION / STOP signal for
navigation — all under the compute and power constraints of flight-candidate hardware.

The system combines:
- **Object detection** (YOLOv8n, Ultralytics) — detects obstacles/rocks and feeds two
  signals into the navigation decision: how much of the frame they cover (cluster
  density) and, via the depth map, how *close* the nearest one actually is.
- **Terrain risk estimation** (MiDaS-small monocular depth, ONNX Runtime) — estimates
  relative depth from a single camera frame and derives slope, roughness, and drop-off
  (ledge/crater) risk from it.
- **Decision layer** — combines terrain risk, nearest-object proximity, and obstacle
  cluster density into a single PROCEED / CAUTION / STOP signal, shown live on a HUD.

## Repo layout

```text
main.py                   Live pipeline: camera -> detection + terrain risk -> decision -> HUD
src/                       Importable modules used by main.py
  camera_stream.py           Webcam capture (+ OAK-D Lite scaffold, see below)
  detect.py                  Detector: YOLOv8 wrapper, falls back to pretrained weights
  terrain_risk.py             TerrainAnalyzer: MiDaS depth -> slope/roughness/drop risk
  decision.py                  should_proceed(): terrain risk + object proximity + density -> decision
  utils.py                      HUD drawing, sample-image loading, cluster density, object proximity
scripts/                   Standalone tools, not imported by main.py
  terrain_demo.py             Depth-only live demo + latency/FPS benchmark (webcam)
  quantize_profile.py         Benchmarks MiDaS across PyTorch / ONNX / OpenVINO backends
legacy/                    Retired prototypes, kept for reference only
  terrain_risk_cubli.py       Depth-Anything-V2 (transformers) terrain risk, built for
                                the "cubli" test platform — not used by the rover pipeline
models/                    yolov8n.pt, best.pt (custom-trained, currently empty until
                            trained), midas_small.onnx(+.data), midas_small_openvino/
data/sample_images/       Test images (Psyche/Ryugu/Mars-analog) for running without a camera
notebooks/train_yolov8.ipynb  Fine-tuning YOLOv8n on the asteroid dataset -> models/best.pt
```

## Architecture

```text
                         Camera (webcam today, OAK-D Lite planned)
                                       │
                        ┌──────────────┴──────────────┐
                        V                              V
              Detector (YOLOv8n)             TerrainAnalyzer (MiDaS/ONNX)
              boxes, labels                   depth map, slope, roughness,
                        │                       drop ratio -> risk_score
                        │                              │
                        ├──────────────┬───────────────┤
                        V              V               │
          compute_cluster_density()   estimate_object_proximity()
          (2D frame coverage)          (depth sampled at each box -> how
                        │              close the nearest object actually is)
                        │                              │
                        └──────────────┬───────────────┘
                                       V
          should_proceed(risk_score, object_proximity, cluster_density)
                                       │
                                       V
                        PROCEED / CAUTION / STOP  ->  HUD
```

`main.py` runs detection and terrain analysis on independent cadences
(`DETECT_EVERY`, `TERRAIN_EVERY` — every N frames, on a downscaled input) so a slow
model pass never blocks the display loop, and downscales further for detection to keep
things responsive on CPU-only hardware.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is split into: **core** (needed to run `main.py`), **benchmarking /
model export** (needed for `scripts/quantize_profile.py` and the legacy Depth-Anything
prototype, not for the live pipeline), and a commented-out **future** section for
`depthai` once the OAK-D Lite is set up.

## Usage

**Run the live pipeline:**
```bash
python main.py
```
- Opens a webcam feed (falls back to cycling through `data/sample_images/` if no webcam
  is found).
- HUD shows: navigation decision, terrain status + risk score, FPS, detected object
  labels, and a depth-map inset in the corner.
- Controls: `q` quits; `t` (or the on-screen **Test** button) opens a sample-image
  viewer with detection boxes drawn — useful for demoing detection without live
  hardware; `space` cycles sample images when running without a webcam.
- `HIGH_QUALITY` in `main()` toggles between a higher-resolution/higher-cadence profile
  and a lighter one tuned for constrained hardware (e.g. Raspberry Pi).

**Terrain-only demo/benchmark** (isolates depth inference from the rest of the stack):
```bash
python scripts/terrain_demo.py
```
Prints frame-drop rate and inference latency (mean/P95) on exit.

**Benchmark MiDaS across backends** (PyTorch vs. ONNX Runtime vs. OpenVINO):
```bash
python scripts/quantize_profile.py
```
Writes `quantize_results.json` to the project root.

**Train a custom detector** on the asteroid dataset:
```bash
jupyter notebook notebooks/train_yolov8.ipynb
```
Produces `models/best.pt`; `Detector` picks it up automatically and only falls back to
the pretrained `yolov8n.pt` if `best.pt` is missing or empty (it currently is, pending a
training run — see [Current Status](#current-status)).

## Design decisions

- **YOLOv8n**: balance of accuracy and inference speed, small enough for CPU-only
  edge hardware (Raspberry Pi-class), and trivially fine-tunable on a custom asteroid
  dataset.
- **MiDaS-small over metric depth**: gives *relative* depth from a single RGB frame with
  no calibrated stereo rig required — enough to detect slope, roughness, and drop-offs
  without needing true distances. (This constraint goes away once OAK-D Lite's stereo
  depth is integrated — see below.)
- **ONNX Runtime over PyTorch/transformers**: the original terrain-risk prototype ran
  MiDaS through a `transformers` pipeline (GPU-oriented, ~465 ms/inference on CPU). The
  ONNX Runtime path cuts that to ~45 ms — the difference between the rover being blind
  for tens of centimeters of travel per depth update vs. a few. See
  `scripts/quantize_profile.py` and `quantize_results.json`.
- **Unified risk score**: slope, roughness, and drop-ratio are three signals on
  unrelated scales. `TerrainAnalyzer` normalizes each to "fraction of its own
  threshold" and takes the max, so a single downstream cutoff (`risk_score >= 1.0`)
  works regardless of which hazard actually tripped — no per-signal special-casing
  needed in the decision layer.
- **Distance-aware object proximity, not just box coverage**: `compute_cluster_density`
  (fraction of the frame covered by boxes) can't distinguish a small rock right in
  front of the rover from a large boulder far away — both can cover the same fraction
  of the frame. `estimate_object_proximity` (in `src/utils.py`) samples the terrain
  depth map at each detected box's ground-contact point instead, so the STOP trigger
  reflects actual distance to the nearest object rather than its apparent size.
  `cluster_density` still feeds a lower-priority CAUTION signal for generally cluttered
  (but not necessarily close) terrain.

### Performance (dev environment, ARM64 Apple Silicon — not flight-candidate hardware)
- Depth inference latency: 465 ms → 45 ms (PyTorch/transformers → ONNX Runtime)
- Perception throughput: 2 FPS → 22 FPS
- Frame-drop rate: 92.6% → 25.7%

These numbers predate the OAK-D Lite; expect a much bigger jump once depth moves
on-camera (below).

## Current Status
- YOLOv8n object detection pipeline: implemented, running on the pretrained COCO
  weights (custom asteroid-obstacle model not yet trained — `models/best.pt` is an
  empty placeholder).
- MiDaS-small monocular depth + terrain risk: implemented, consolidated into one
  module (`src/terrain_risk.py`), ONNX Runtime backend.
- Navigation decision: combines terrain risk and obstacle cluster density into
  PROCEED / CAUTION / STOP.
- Cubli-platform terrain prototype (Depth-Anything-V2/transformers): retired to
  `legacy/`, superseded on the rover path by `src/terrain_risk.py`.
- Raspberry Pi / flight-candidate hardware benchmarking: not yet done.
- OAK-D Lite integration: not yet done — `src/camera_stream.py` has an untested
  scaffold (`OakDLiteCamera`) to build on once the camera is in hand.

## Future Implementation

**OAK-D Lite integration (near-term, camera-driven)**
- Validate `OakDLiteCamera` in `src/camera_stream.py` against real hardware (board
  socket assignments, mono resolution/FPS, RGB-depth alignment) — it's unverified
  boilerplate right now.
- Replace `TerrainAnalyzer._estimate_depth()`'s MiDaS pass with the camera's native
  stereo depth stream. This is a bigger deal than a backend swap: MiDaS gives
  *relative* depth (no units), the OAK-D Lite gives *metric* depth in millimeters —
  the slope/roughness/drop-ratio thresholds in `TerrainAnalyzer` were tuned against
  relative depth and will need to be re-derived against real-world distances (which
  also means the risk model becomes physically interpretable — e.g. "stop if the drop
  ahead is >15 cm" instead of a unitless ratio).
- Retiring host-side monocular depth also frees the CPU entirely for detection and
  planning, which matters a lot on power/compute-constrained flight hardware.

**YOLOv6 / on-device detection**
- OAK-D Lite's Myriad X VPU can run detection on-camera via DepthAI's
  `YoloDetectionNetwork` node, off-loading YOLO entirely from the host CPU alongside
  the depth move above.
- This needs the trained model exported to a `.blob` (via `blobconverter` or the
  OpenVINO toolchain) with anchor/mask config matching the export exactly — code for
  this isn't included yet since it requires a real trained model and the physical
  camera to validate against; get `models/best.pt` trained first (see below), then
  export and wire it into a `YoloDetectionNetwork` node alongside `OakDLiteCamera`.
- Once on-device detection lands, `src/detect.py` should grow a second backend behind
  the same `Detector` interface so `main.py` doesn't need to change.

**Detection quality**
- Train `models/best.pt` on a real asteroid/regolith obstacle dataset (notebook is
  ready in `notebooks/train_yolov8.ipynb`) and evaluate mAP/precision/recall — right
  now the pipeline silently runs on generic COCO classes.
- Consider a rock/regolith texture segmentation model alongside detection, for
  terrain classification finer-grained than depth-only heuristics can give.

**Terrain mapping (the other core project goal)**
- Right now terrain risk is a single reactive per-frame signal. Building an actual
  *map* — an occupancy grid or elevation map accumulated from OAK-D depth over time —
  is what "mapping the terrain" in the project goal actually calls for, and would let
  navigation move from single-frame reactive STOP/PROCEED to path planning around a
  costmap.
- Fuse an IMU-based tilt estimate with the camera-based slope estimate for redundancy
  (there's prior slope-sensing work in `legacy/terrain_risk_cubli.py`'s platform that's
  worth revisiting for this, even though that module itself is retired).

**Hardware validation**
- Benchmark the full pipeline (detection + terrain + decision) on Raspberry Pi 5 or
  actual flight-candidate hardware — all performance numbers so far are from a dev
  laptop.
- Extend `scripts/quantize_profile.py`-style profiling to power and memory usage, not
  just latency.

**Systems integration**
- ROS2 is a natural fit given the current module boundaries: a camera driver node,
  detection node, terrain-risk node, and decision node map directly onto
  `src/camera_stream.py`, `src/detect.py`, `src/terrain_risk.py`, and `src/decision.py`.
- Replace heuristic risk thresholds with a learned navigation policy once enough
  labeled traversal data exists.
- There's an earlier Flask MJPEG-streaming prototype at `main.py.save` (not wired into
  the current pipeline) that could be revived for a browser-based telemetry dashboard
  instead of the local OpenCV window.
