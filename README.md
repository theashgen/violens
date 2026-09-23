# Violence Detection in Surveillance Videos

A research prototype supporting uploaded-video classification, single-camera testing, and continuous multi-camera CCTV monitoring with live previews and **NORMAL / VIOLENCE** detection states. Training is an offline PyTorch workflow. The web application never trains models, generates demo predictions, or substitutes random weights for a missing checkpoint.

## Architecture

```text
RTSP / HTTP camera / paced local test video
  → per-camera PyAV capture → bounded timestamped rolling buffer
  → latest pending temporal window → shared MC3-18 checkpoint
  → violence score → per-camera hysteresis → camera state / active event
                                              ↓ WebSocket updates
Next.js /monitor ← sampled MJPEG previews ← FastAPI camera manager

Next.js /       → multipart upload → shared MC3-18 → whole-video prediction
Next.js /live   → single-camera/webcam diagnostic mode
Offline PyTorch training → artifacts/checkpoints/best.pt
```

| Location | Responsibility |
| --- | --- |
| `apps/web` | Upload workspace, single-camera diagnostics, multi-camera dashboard |
| `apps/api/violence_api/app.py` | Upload and camera REST APIs, WebSocket state updates, MJPEG previews |
| `apps/api/violence_api/surveillance.py` | Continuous capture, latest-window inference, per-camera states and hysteresis |
| `apps/api/violence_api/stream.py` | Shared frame preprocessing and legacy single-camera sessions |
| `packages/ml/violence_detection/video.py` | Shared decoding, sampling, pretrained preprocessing |
| `packages/ml/violence_detection/dataset.py` | Inspection, duplicate checks, split manifests, dataset loading |
| `packages/ml/violence_detection/model.py` | MC3-18 construction and device selection |
| `packages/ml/violence_detection/train.py` | Explicit supervised loop and best-validation checkpoint |
| `packages/ml/violence_detection/evaluate.py` | Held-out test evaluation and metrics |
| `packages/ml/violence_detection/inference.py` | Checkpoint loading and independent CLI inference |
| `configs` | Baseline settings and reviewed dataset exclusions |
| `scripts/download_dataset.sh` | Resumable Kaggle download and extraction |
| `tests` | ML and API behavior tests |
| `data`, `artifacts` | Locally generated inputs/results; ignored by Git |

No database, global frontend state library, job queue, or experiment tracking service is required. The web routes are `/` (uploads), `/live` (single-camera diagnostics), and `/monitor` (continuous monitoring). Its layout stacks at narrower widths, follows the system light/dark preference, supports keyboard upload and drag/drop, and distinguishes actual upload progress from waiting for inference.

## Setup

Run commands from the repository root. Use Python **3.12+**, Node **22+**, and Bun for the root workspace scripts. The checked-in locks capture the versions verified locally (Python 3.14, Node 26 on Apple Silicon). GPU availability and wheel compatibility depend on the host. PyAV wheels include FFmpeg libraries; a separate FFmpeg executable is not required by the application.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install --no-build-isolation --no-deps -e packages/ml -e apps/api
bun install --frozen-lockfile
```

`requirements.lock.txt` pins the Python environment, including development tools. Package manifests specify runtime dependencies; `requirements-dev.txt` lists direct test/lint dependencies. `bun.lock` pins the Bun workspace; `package-lock.json` is retained for npm users. To use npm without Bun, run commands directly with `npm --prefix apps/web run <command>` after `npm ci`. For CUDA, install the matching PyTorch/torchvision builds from the [official installer](https://pytorch.org/get-started/locally/) and record the resulting environment.

## Dataset: download, inspect, then split

The requested [Kaggle dataset](https://www.kaggle.com/datasets/magicearth25/video-violence-detection-dataset) combines **RLVS** and **RWF-2000**. The archive is approximately 13.2 GiB; allow at least 30 GiB for the archive and extracted videos. The download is public at the time of verification. If Kaggle later requires authentication, download it through your Kaggle account and extract it into `data/raw`.

```bash
bash scripts/download_dataset.sh
```

The baseline uses the RLVS collection independently:

```text
data/raw/Violence Fight Detection dataset/
  RLVS/
    train/{Fight,NonFight}/
    val/{Fight,NonFight}/
  ...other supplied collections...
```

Original videos are never rewritten. Inspection fully decodes each supported video and records frame count, approximate duration, FPS, resolution, SHA-256, class, supplied split, and failures. Labels are derived from recognized **class folder names**, never substrings in filenames: `0 = Non-Violence`, `1 = Violence`. Unknown or ambiguous layouts fail visibly. RWF-2000 can be inspected separately using its actual extracted root.

```bash
python -m violence_detection.dataset inspect \
  --root 'data/raw/Violence Fight Detection dataset/RLVS' \
  --output artifacts/metrics/rlvs-inspection.json

python -m violence_detection.dataset split \
  --report artifacts/metrics/rlvs-inspection.json \
  --exclude configs/rlvs-exclusions.csv \
  --acknowledge-unknown-groups \
  --output data/splits.csv
```

Observed RLVS inspection: 2,000 videos, 1,000 per original class. Of these, 1,996 meet decoding/length limits: 998 per class. Three videos exceed 120 seconds; one contains a corrupted frame. There are 14 extra exact-duplicate copies, including one pair with contradictory labels and three pairs crossing supplied train/validation boundaries.

The reviewed exclusion file removes **both** contradictory-label copies and the three training copies duplicated in validation. Same-split duplicates retain one representative. After these exclusions/deduplication:

| Split | Non-Violence | Violence | Total |
| --- | ---: | ---: | ---: |
| Train | 674 | 672 | 1,346 |
| Validation | 199 | 198 | 397 |
| Test | 118 | 120 | 238 |

The supplied `val` membership is preserved for retained clips. Since RLVS has no supplied test partition, 15% of the remaining training groups are reserved as a local test set using seed 42. For completely unsplit datasets, the command produces approximately 70/15/15 splits. Existing complete train/val/test assignments are retained. Every partition must contain both classes.

**Related-source leakage remains unverified.** Byte hashing detects identical files, not re-encodes, overlapping clips, common actors, or camera scenes. This mirror supplies generic filenames rather than trusted source IDs. `--acknowledge-unknown-groups` makes that limitation explicit; results from this split are provisional research measurements, not a source-independent benchmark. After source review, supply a complete CSV instead:

```csv
path,group
train/Fight/file_002001.mp4,source-video-001
...
```

Use `--groups path/to/groups.csv` in place of `--acknowledge-unknown-groups`. Groups cannot cross partitions. Duplicate labels and split conflicts fail rather than silently moving videos. A manifest contains relative paths, labels, split, SHA-256, and group. Training checks train/validation file hashes; evaluation checks test hashes and requires the exact manifest used during training.

## First milestone: one video → tensor → logits

```bash
python -m violence_detection.video \
  --video 'data/raw/Violence Fight Detection dataset/RLVS/train/Fight/file_002001.mp4'
```

Expected shape: **`[3, 16, 112, 112]`**, dtype **`torch.float32`**. Uploaded-video inference and the training dataset use this exact function. Continuous monitoring reuses its spatial transform but samples a short timestamped window instead of the entire video.

1. PyAV decodes sequentially. A bounded count pass detects corrupt frames and avoids unreliable container frame counts.
2. Uniform frame indices span the whole video, including its first and last frames. Short clips repeat indices.
3. A second decode converts and retains only the selected frames. Interframe codecs still require decoding intervening frames; memory does not grow with clip length.
4. `MC3_18_Weights.KINETICS400_V1.transforms()` performs bilinear resize to 128×171, 112×112 center crop, scaling, normalization, and channel arrangement. No custom normalization constants are used.
5. Batched input is **`[B, C, T, H, W]`**. Model output is **`[B, 2]`** logits. Softmax produces the predicted class and its score.

Limits: 120 seconds, 12,000 decoded frames, and 3840×2160 pixel area per frame. HTTP uploads additionally have a 100 MiB file limit. These limits apply to uploads, not continuous capture. Use monitoring mode for ongoing streams; uploads are not silently truncated.

## Model choice

All candidates have Kinetics-400 weights and mature torchvision implementations (torchvision's video API is still marked beta, so versions are pinned):

| Architecture | Pretrained parameters | Published compute | Decision |
| --- | ---: | ---: | --- |
| MC3-18 | 11.7M | 43.34 GFLOPs | Selected: smaller model, practical frozen-feature baseline |
| R3D-18 | 33.4M | 40.70 GFLOPs | Larger fully 3D backbone |
| R(2+1)D-18 | 31.5M | 40.52 GFLOPs | Larger factorized spatiotemporal backbone |

Compute figures are torchvision reference figures, **not local latency measurements**. MC3-18 reduces parameters, not necessarily FLOPs. Sources: [MC3-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.mc3_18.html), [R3D-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html), [R(2+1)D-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html).

The classifier is replaced with two outputs. By default, only this new head is trained; pretrained batch-normalization statistics remain frozen. Set `freeze_backbone: false` to fine-tune the backbone, usually with a lower learning rate and more compute. Pretrained weights are always used in training. Random models exist only in tests.

## Offline training and evaluation

```bash
export TORCH_HOME="$PWD/artifacts/torch"
python -m violence_detection.train --config configs/baseline.yaml
```

The first training run downloads official pretrained weights (~45 MiB). Configuration controls epochs, batch size, learning rate, sampled frames, seed, device, frozen backbone, workers, and output paths. Defaults are 20 epochs, batch size 4, learning rate 0.001, and 16 frames. Run from the root because configuration paths are root-relative.

`device: auto` selects CUDA, then Apple MPS, then CPU. Explicitly select CPU if an accelerator lacks a required operation; there is no silent device fallback after an operation fails. Fixed seeds and deterministic-algorithm warnings improve reproducibility, but bitwise identity across devices/library versions is not guaranteed.

Each epoch runs training followed by validation. The lowest validation loss selects `artifacts/checkpoints/best.pt`. History goes to `artifacts/metrics/training.json`. The test partition is not loaded during fitting or model selection. This is a new training run, not a resumable optimizer session; use separate output paths to preserve previous experiments.

```bash
python -m violence_detection.evaluate \
  --root 'data/raw/Violence Fight Detection dataset/RLVS' \
  --manifest data/splits.csv \
  --checkpoint artifacts/checkpoints/best.pt \
  --output artifacts/metrics/test_metrics.json
```

The JSON includes accuracy, binary precision/recall/F1 with **Violence as positive**, confusion matrix, sample count, loss, checkpoint hash, and manifest hash. Confusion matrix rows are actual labels and columns are predicted labels, ordered `[Non-Violence, Violence]`. Undefined precision is reported as zero. Do not repeatedly tune using the test metrics.

Checkpoints include model weights, architecture, ordered classes, preprocessing identifier, frame count, pretrained weight identifier, training config, selected epoch, validation metrics, manifest hash, and PyTorch/torchvision versions. Inference refuses incompatible metadata and never downloads weights.

## CLI inference

```bash
python -m violence_detection.inference \
  --video path/to/clip.mp4 \
  --checkpoint artifacts/checkpoints/best.pt
```

This prints the video path and JSON with the prediction and confidence. Use the CLI first when diagnosing whether a problem belongs to decoding/model inference, HTTP, or the browser.

## API

```bash
source .venv/bin/activate
export CHECKPOINT_PATH="$PWD/artifacts/checkpoints/best.pt"
uvicorn violence_api.app:app --host 127.0.0.1 --port 8000
```

Start one worker for this local baseline. Model inference runs outside the async event loop; concurrent inference attempts return 429 rather than running multiple memory-heavy forwards. Uploaded files use temporary storage and are removed after inference. Only the current active camera event is retained in memory; closed or interrupted events are not archived.

```bash
curl http://localhost:8000/health
curl -F 'video=@path/to/clip.mp4' http://localhost:8000/api/v1/predict
```

Successful prediction contract (illustrative values only):

```json
{"prediction": "violence", "confidence": 0.923}
```

`prediction` is `violence` or `non-violence`; `confidence` is a finite number in `[0,1]`. It is the softmax score of the predicted class, **not a calibrated probability of harm**.

| HTTP status | Meaning |
| --- | --- |
| 413 | Upload too large |
| 415 | Unsupported filename extension |
| 422 | Empty, undecodable, or over-limit video; missing video form field |
| 429 | Another inference is running |
| 503 | No trained checkpoint loaded |
| 500 | Model execution failed; inspect server logs |

`GET /health` returns 200 for a running process and separately reports `model_ready`. A missing checkpoint allows the UI to show an unavailable state; an existing but incompatible checkpoint fails API startup. Restart the API after training or changing checkpoints. `/docs` contains FastAPI's generated contract. No training endpoints exist.

Optional environment settings are listed in `apps/api/.env.example`. Export them in your shell; no dotenv loader is installed. `WEB_ORIGINS` defaults to `http://localhost:3000`; use a comma-separated list to allow another exact origin (including `http://127.0.0.1:3000` if you browse with that hostname).

## Web

```bash
# Optional; defaults to http://localhost:8000
cp apps/web/.env.example apps/web/.env.local
bun run dev
```

Open **http://localhost:3000**. Select/drop a supported video, preview it, and run analysis. Browser preview depends on codec support; an AVI/MKV that cannot preview can still be analyzed by PyAV. Browser metadata is informational; the server validates actual contents. The model must be ready before analysis is enabled. Upload percentage reflects transferred bytes; the subsequent waiting state has no invented inference percentage.

Production:

```bash
bun run build
bun run start
```

`NEXT_PUBLIC_API_URL` is embedded at build time, so rebuild after changing it. This project uses Next.js's supported webpack mode because Turbopack's CSS worker could not bind its internal port in the development sandbox. There are no remote font downloads. Light/dark tokens follow the OS preference without a theme package.

## Verification

```bash
source .venv/bin/activate
pytest -q
ruff check packages apps/api tests
ruff format --check packages apps/api tests
mypy
python -c 'import violence_detection.video, violence_detection.train, violence_api.app'
bun run lint
bun run typecheck
bun run build
npx playwright install chromium
bun run test
```

Python tests exercise real decoding, short-clip sampling, dtype/layout, shared preprocessing, two-class model output, checkpoint compatibility, labels, deterministic grouped splits, leakage rejection, API prediction shape, unavailable/busy models, corrupted uploads, and streaming size limits. Random weights in tests verify plumbing only and never supply research metrics or application results.

Browser tests exercise empty/selected/uploading states, both prediction labels, errors/retry, malformed responses, keyboard file selection, missing checkpoints, and mobile overflow. Their HTTP fixtures test UI behavior. Real trained-checkpoint requests must also be verified independently; mocks are not evidence of model quality.

Formatting: `npx prettier --write 'apps/web/src/**/*.{ts,tsx,css}'`; Python: `ruff format packages apps/api tests`.

## Continuous multi-camera monitoring (research prototype)

### Connection troubleshooting

The browser connects to the **backend WebSocket**, not directly to RTSP cameras.
The backend continuously captures camera frames and publishes status updates.
Install `requirements.lock.txt` (including `websockets`) and restart the API after
upgrading; bare Uvicorn without a WebSocket transport cannot upgrade connections.

```bash
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m uvicorn violence_api.app:app --host 0.0.0.0 --port 8000
```

Run one API worker: camera state and the shared model are process-local.
Set `NEXT_PUBLIC_API_URL` to the API address reachable from the browser and restart
Next.js (rebuild production). On another machine, `localhost` refers to that
machine, not the API host. HTTPS pages require an HTTPS API/WSS endpoint; reverse
proxies must forward WebSocket Upgrade headers. The dashboard reconnects with
backoff and marks displayed statuses as potentially stale when disconnected.

Each monitor tile includes a live sampled MJPEG preview from
`GET /api/v1/cameras/{camera_id}/preview`. All viewers share the camera's existing
capture worker; opening a preview does not create another RTSP connection. Preview
rate follows sampling (default up to 4 FPS), not full native camera FPS. Use a
buffering-disabled proxy and HTTP/2 for larger grids; HTTP/1.1 browsers limit
simultaneous connections per host. Previews carry camera imagery and require the
same private-network protections as the API.

`ONLINE` describes the camera connection; `BUFFERING` means frames are arriving
but the first detection window is still filling. Low-FPS streams repeat nearest
timestamped samples to create the model's 16-frame input rather than waiting for
16 distinct frames. Extremely sparse footage has less motion information and
needs separate validation.

The uploaded-video `/api/v1/predict` route remains available. Continuous monitoring is separately available at **http://localhost:3000/monitor**. The API retains MC3-18 and the trained checkpoint, and maintains one bounded timestamped frame deque per camera. It samples at the configured rate, makes overlapping windows, and serializes inference through the existing shared model lock; stale windows are skipped rather than queued. RTSP feeds reconnect with exponential backoff. Local test videos can be configured as `file:///absolute/path/to/video.mp4` and loop at EOF.

Configure cameras at API startup using `CAMERAS_JSON` (or add/remove them in the monitor dashboard):

```bash
export CAMERAS_JSON='[{"camera_id":"CCTV-01","name":"Entrance","url":"rtsp://user:pass@192.168.1.20/stream","window_seconds":4,"sample_fps":4,"stride_seconds":1,"positive_threshold":0.65,"negative_threshold":0.4,"positive_windows":3,"negative_windows":4}]'
```

Defaults are 4-second windows, 4 sampled frames/sec (16 frames), 1-second stride, score thresholds 0.65/0.40, and 3 positive / 4 negative windows. These are starter settings, not validated operating points. The camera API is `GET/POST /api/v1/cameras`, `DELETE /api/v1/cameras/{camera_id}`, and real-time WebSocket `/api/v1/cameras/ws`; uploaded inference and the previous single-stream live test remain. `OFFLINE` and `ERROR` are distinct from `NORMAL`. Event records are in memory and close when hysteresis returns the camera to normal. Camera URLs, which may include credentials, are not returned in status payloads.

Run a local looping test source by setting `url` to a `file://` URI. This exercises continuous decode and reconnect behavior, not camera realism. For RTSP, supply an accessible URL and ensure network/firewall access. The dashboard reports score, inference latency, dropped windows, and connection state. Inference latency measures an individual forward path. The current `processed_fps` field is inverse inference duration, not measured capture FPS or sustained throughput. GPU/CPU utilization and comprehensive queue telemetry are not currently measured.

### Multiple cameras and local testing

Append additional objects to `CAMERAS_JSON`, each with a unique `camera_id`:

```bash
export CAMERAS_JSON='[{"camera_id":"CCTV-01","url":"rtsp://192.168.1.20/stream"},{"camera_id":"CCTV-02","url":"rtsp://192.168.1.21/stream"}]'
python -m uvicorn violence_api.app:app --host 127.0.0.1 --port 8000
```

For hardware-free testing, replace a URL with `file:///absolute/path/to/test.mp4`.
The adapter paces by video timestamps, reconnects at EOF, and resets the detector
between loops. Use a video longer than the configured window. Camera registrations
added through the UI disappear on restart; `CAMERAS_JSON` restores configured ones.
URLs are server-accessible camera/file addresses, not browser WebSocket addresses.

A single model is shared across upload, diagnostic, and monitoring modes. Each
camera has at most one pending window; replacements increment dropped-window
counts. This bounds application backlog but does not establish a latency SLA:
network/decoder buffering, camera count, and hardware still need measurement.
The model lock does not guarantee fairness across cameras.

### Checkpoint compatibility

No migration or retraining is required to load existing compatible MC3-18
checkpoints. Monitoring uses the checkpoint frame count (normally 16) and existing
class order `0 = non-violence`, `1 = violence`. Functional compatibility is not
evidence that whole-video-trained weights are validated for temporal CCTV events.

### Training and limitations

Training remains unchanged: each source video currently produces one uniformly sampled 16-frame clip. Live inference samples frames from a short fixed-duration window. This temporal-distribution mismatch needs evaluation. RLVS video-level labels do **not** imply every short sub-window is positive; do not naively label every crop from a violence-labeled video as a positive training example. Window-aware training requires event annotations, carefully reviewed temporal localization, or a documented multiple-instance/weak-supervision strategy. The new stream feature does not itself train a CCTV-adapted checkpoint. Fine-tuning on representative, consented CCTV-style data and testing by independent camera/source are recommended; preserve source-disjoint splits and assess event-level recall/false alerts before tuning thresholds.

Scores are uncalibrated model outputs, not probabilities of harm. RLVS does not represent arbitrary real CCTV, low-light/noisy footage, viewpoints, compression, or demographics. Dataset accuracy is not evidence of reliable real-world detection. Evaluate domain shift, bias, calibration, event-level performance, and latency; keep a human in the loop. The current API is a local research prototype, without authentication, TLS termination, durable incidents, or production hardening; do not expose it to the public internet.
