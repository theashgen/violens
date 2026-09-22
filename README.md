# Violence Detection in Surveillance Videos

A Phase 1 research baseline: upload a video, run a trained video classifier, and review its **Violence / Non-Violence** prediction. Training is an offline PyTorch workflow. The web application never trains models, generates demo predictions, or substitutes random weights for a missing checkpoint.

## Architecture

```text
                         MONOREPO
 ┌──────────────────────────────┐
 │ apps/web                     │
 │ Next.js · Tailwind · shadcn/ui│
 └──────────────┬───────────────┘
                │ multipart HTTP
                ▼
 ┌──────────────────────────────┐
 │ apps/api                     │
 │ FastAPI · upload validation  │
 └──────────────┬───────────────┘
                │ Python function call
                ▼
 ┌──────────────────────────────┐
 │ packages/ml                  │◀── Offline inspection/training/evaluation CLIs
 │ PyAV → sampling → torchvision│
 │ MC3-18 → binary prediction   │
 └──────────────┬───────────────┘
                ▼
     artifacts/checkpoints/best.pt
```

| Location | Responsibility |
| --- | --- |
| `apps/web` | Server-rendered page shell and an interactive upload/analysis workspace |
| `apps/api/violence_api/app.py` | Health, bounded multipart uploads, one loaded model, inference responses |
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

No database, global frontend state library, job queue, or experiment tracking service is required. The web application has one page. Its layout stacks at narrower widths, follows the system light/dark preference, supports keyboard upload and drag/drop, and distinguishes actual upload progress from waiting for inference.

## Setup

Run commands from the repository root. Use Python **3.12+** and Node **22+**. The checked-in locks capture the versions verified locally (Python 3.14, Node 26 on Apple Silicon). GPU availability and wheel compatibility depend on the host. PyAV wheels include FFmpeg libraries; a separate FFmpeg executable is not required by the application.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install --no-build-isolation --no-deps -e packages/ml -e apps/api
npm ci
```

`requirements.lock.txt` pins the Python environment, including development tools. Package manifests specify runtime dependencies; `requirements-dev.txt` lists direct test/lint dependencies. `package-lock.json` pins the npm workspace. For CUDA, install the matching PyTorch/torchvision builds from the [official installer](https://pytorch.org/get-started/locally/) and record the resulting environment.

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

Expected shape: **`[3, 16, 112, 112]`**, dtype **`torch.float32`**. The API and training dataset use this exact function.

1. PyAV decodes sequentially. A bounded count pass detects corrupt frames and avoids unreliable container frame counts.
2. Uniform frame indices span the whole video, including its first and last frames. Short clips repeat indices.
3. A second decode converts and retains only the selected frames. Interframe codecs still require decoding intervening frames; memory does not grow with clip length.
4. `MC3_18_Weights.KINETICS400_V1.transforms()` performs bilinear resize to 128×171, 112×112 center crop, scaling, normalization, and channel arrangement. No custom normalization constants are used.
5. Batched input is **`[B, C, T, H, W]`**. Model output is **`[B, 2]`** logits. Softmax produces the predicted class and its score.

Limits: 120 seconds, 12,000 decoded frames, and 3840×2160 pixel area per frame. HTTP uploads additionally have a 100 MiB file limit. Long videos need a deliberate windowing approach in a later phase; silently truncating them would change the prediction semantics.

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

Start one worker for this local baseline. Model inference runs outside the async event loop; concurrent inference attempts return 429 rather than running multiple memory-heavy forwards. Uploaded files use temporary storage and are removed after inference. No incident history is retained.

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
npm run dev
```

Open **http://localhost:3000**. Select/drop a supported video, preview it, and run analysis. Browser preview depends on codec support; an AVI/MKV that cannot preview can still be analyzed by PyAV. Browser metadata is informational; the server validates actual contents. The model must be ready before analysis is enabled. Upload percentage reflects transferred bytes; the subsequent waiting state has no invented inference percentage.

Production:

```bash
npm run build
npm run start
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
npm run lint
npm run typecheck
npm run build
npx playwright install chromium
npm test
```

Python tests exercise real decoding, short-clip sampling, dtype/layout, shared preprocessing, two-class model output, checkpoint compatibility, labels, deterministic grouped splits, leakage rejection, API prediction shape, unavailable/busy models, corrupted uploads, and streaming size limits. Random weights in tests verify plumbing only and never supply research metrics or application results.

Browser tests exercise empty/selected/uploading states, both prediction labels, errors/retry, malformed responses, keyboard file selection, missing checkpoints, and mobile overflow. Their HTTP fixtures test UI behavior. Real trained-checkpoint requests must also be verified independently; mocks are not evidence of model quality.

Formatting: `npx prettier --write 'apps/web/src/**/*.{ts,tsx,css}'`; Python: `ruff format packages apps/api tests`.

## Scope and limitations

This is a **video-only, whole-clip research classifier**. A single uniformly sampled clip can miss a brief incident, and center cropping can discard relevant context. Sampling whole-video indices differs from the pretrained evaluation's fixed-rate, multi-clip protocol. Spatial preprocessing follows the weights, while temporal sampling is a documented baseline approximation. Longer or variable-frame-rate clips can have uneven temporal coverage.

Confidence is uncalibrated, labels can be noisy, and RLVS contains scenes beyond fixed surveillance cameras. Performance on this dataset does not establish performance on live CCTV. Source-group independence, domain shift, bias, calibration, and event-level performance require further research. Keep human review in the loop.

Not implemented: audio, motion/pose branches, fusion, RTSP/CCTV streams, sliding windows, event localization, alerts, incident storage, authentication, cloud deployment, or an operational monitoring dashboard. The API is for local research, not an internet-facing upload service.
