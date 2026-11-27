# Violence Detection in Surveillance Videos
## Model Technologies and Viva Preparation Guide

**Ashwin S  |  Prudhviraj K B  |  Pranav R**  
Guide: Enjula Uchoi  -  Lovely Professional University (LPU)
_November 2026 - prepared for presentation and viva preparation_

---

## 1. Project at a Glance

This project is a **violence-detection research prototype** for surveillance video. A deep learning model watches short clips of video and outputs one of two labels: **non-violence (0)** or **violence (1)**. The same trained model powers three modes: analysis of uploaded videos through a web app, a single-camera live diagnostic page, and continuous multi-camera CCTV monitoring with NORMAL/VIOLENCE states.

Everything in this guide is grounded in the actual repository code and artifacts. The core model is an **MC3-18 residual 3D convolutional network**, pretrained on the **Kinetics-400** action dataset, with its final classification layer replaced and trained on the RLVS fight dataset. The backbone stays frozen; only the small new head is trained.

| Fact | Value |
|---|---|
| Core model | MC3-18 (mixed 3D/2D residual CNN), torchvision implementation |
| Pretraining | Kinetics-400 official weights (MC3_18_Weights.KINETICS400_V1) |
| Input per clip | 16 frames, tensor [B, 3, 16, 112, 112] |
| Head | Linear 512 -> 2 logits, softmax; only trainable part (frozen backbone) |
| Dataset | RLVS (Real Life Violence Situations), 2,000 inspected videos |
| Trainable weights | 1,026 (512 x 2 + 2); backbone's 11.49 M weights frozen |
| Test results | 238 clips: 92.86% accuracy, 88.15% precision, 99.17% recall, 93.33% F1 |
| Confusion matrix | [[102, 16], [1, 119]] - rows actual, columns predicted, order [NV, V] |

> **How to read this guide.** Sections 2 to 7 explain the actual technology and why each choice was made. Section 8 states limitations honestly. Sections 9 and 10 contain the speaking scripts and 50 likely viva questions with answers. Sections 11 and 12 are a last-minute revision sheet and glossary.


### 1.1 System architecture in one picture

```text
RTSP / HTTP camera / paced local test video
  |  per-camera PyAV capture (bounded timestamped rolling buffer)
  v
latest 16-frame temporal window  ----->  shared MC3-18 checkpoint
  |                                       (frozen backbone + trained head)
  v                                       |
violence score [0..1]  <------------------+
  |
  +--> upload mode:   whole-video softmax -> prediction + confidence (JSON)
  +--> monitor mode:  per-camera hysteresis (thresholds + window counters)
                      -> camera state NORMAL / VIOLENCE -> active event
                      -> WebSocket updates to the Next.js /monitor dashboard
  +--> live mode:     single-camera diagnostic page for testing
```

Key architectural property: **one model instance is shared** by upload, live and monitoring modes, and all inference is serialized through a single lock. Training is a fully separate offline PyTorch workflow; the web application never trains models, fakes predictions, or substitutes random weights when a checkpoint is missing.


## 2. The Core Model: MC3-18 in Depth


### 2.1 The problem: recognising events, not pixels

Violence in a video is a **temporal event**: a punch or a scuffle is defined by how pixels move over time, not by any single still image. A model that looks at one frame can confuse a raised arm in a fight with a raised arm in a dance. That is why the project uses a **video model**: it consumes a stack of 16 frames at once and can learn motion patterns directly.


### 2.2 From 2D to 3D convolutions

A 2D convolution slides a small weight grid over height and width. A **3D convolution** extends the kernel with a third axis, time, so a kernel of size 3x3x3 slides over height, width **and** frames. Because one filter sees several frames together, it can respond to motion (an arm swinging) the way a 2D filter responds to an edge. The cost is parameters and compute: the same kernel now covers a 3D volume.

The 2019 paper *A Closer Look at Spatiotemporal Convolutions for Action Recognition* (Tran et al., CVPR 2018) studied three designs that torchvision implements as R3D-18, R(2+1)D-18 and MC3-18:

- **R3D-18** uses full 3D 3x3x3 kernels everywhere. Most expressive, most parameters (33.4 M).
- **R(2+1)D-18** factorizes each 3D kernel into a 1x3x3 spatial conv followed by a 3x1x1 temporal conv. Fewer parameters than R3D for the same depth and often better accuracy, but 31.5 M params.
- **MC3-18 (Mixed Convolution)** uses full 3D kernels only in the earliest layers and cheap spatial 2D kernels (1x3x3) afterwards: the intuition is that early layers detect low-level motion edges, while later layers analyse appearance which changes less over time. Only 11.7 M params.

In the torchvision implementation used here (verified directly against the installed weights), MC3-18's **stem and first stage use true 3D kernels** (temporal extent 3), while **stages 2 to 4 use 1x3x3 kernels** (temporal extent 1, i.e. 2D). This is exactly the 'mixed' idea.


### 2.3 Why MC3-18 was selected

All three candidates ship official Kinetics-400 weights in torchvision:

| Architecture | Pretrained params | Kinetics-400 acc@1 | Compute (GFLOPs) | Decision |
|---|---|---|---|---|
| MC3-18 | 11.7 M | 63.96% | 43.34 | Selected: smallest backbone, practical frozen-feature baseline |
| R3D-18 | 33.4 M | 63.20% | 40.70 | Rejected: 2.9x parameters for lower accuracy |
| R(2+1)D-18 | 31.5 M | 67.46% | 40.52 | Rejected: more accurate but 2.7x parameters |

Reasons for the decision:

- **Smallest backbone** (11.7 M pretrained parameters) with a mature, tested torchvision implementation and official pretrained weights - important for CPU/MPS training and serving.
- As a **frozen-feature baseline**, the backbone's Kinetics motion features are reused as-is; the published Kinetics accuracy gap between candidates matters less when only a linear head adapts.
- R(2+1)D-18 and R3D-18 carry 2.7x-2.9x the parameters for similar or modestly better published accuracy - a poor trade for a prototype on limited hardware.

> The parameter and GFLOP figures are torchvision's published reference figures, not local latency measurements. MC3-18 reduces parameters, not necessarily FLOPs.


### 2.4 Transfer learning from Kinetics-400

**Transfer learning** means starting from weights that already learned a related task. Kinetics-400 (Kay et al., 2017) is 400 human action classes over about 240,000 training clips - running, punching, hugging, diving and so on. A network trained on it already knows low-level motion edges, body-part layouts and short action dynamics.

This matters because the project's training set is tiny: **1,346 training clips**. Training a video backbone from scratch on 1,346 clips would memorize them (overfitting) within an epoch. Instead, the frozen Kinetics features act as a fixed, high-quality 'motion vocabulary', and the small head learns only how to read violence out of that vocabulary.

The pretrained MC3-18 checkpoint file is about 45 MB and reaches 63.96% top-1 accuracy on Kinetics-400 (torchvision published figures). Pretrained weights are **always** used in real training runs; randomly initialized models exist only inside the test suite to verify plumbing, never to produce results.


### 2.5 Frozen backbone and the trained head

```text
model = mc3_18(weights=MC3_18_Weights.KINETICS400_V1)   # pretrained backbone
model.fc = nn.Linear(model.fc.in_features, 2)            # 400-class head -> 2-class head

# freeze everything except the new head
for name, parameter in model.named_parameters():
    parameter.requires_grad = name.startswith("fc.")
```

Only the linear head (1,026 weights: 512 x 2 plus 2 biases) receives gradients. The optimizer is built over `parameters()` filtered by `requires_grad`, so frozen weights are never updated.

**A subtle but viva-worthy detail:** during training the code calls `model.train()` and, when the backbone is frozen, immediately `model.eval()`. Why two modes? BatchNorm layers behave differently: in train mode they would update their running statistics from each batch, silently corrupting the frozen backbone's normalizations. Keeping `eval()` for the whole frozen model keeps the pretrained BatchNorm statistics untouched while the head still trains. This matches the 'frozen pretrained BN' behaviour stated in the README.

**Why freeze at all?** Three reasons: (a) 1,346 clips cannot reliably tune 11.5 M weights; (b) training is dramatically faster and cheaper; (c) results are more stable and reproducible. The tradeoff is ceiling: the backbone cannot specialize to fight footage. The config exposes `freeze_backbone: false` to fine-tune the whole network, usually with a lower learning rate and more compute, as documented future work.


### 2.6 Verified tensor walkthrough

The shapes below were traced through the actual model with a batch of one clip. The temporal dimension 16 survives the whole network; only spatial resolution shrinks.

| Stage | Operation | Output shape | Notes |
|---|---|---|---|
| Input | 16 uniformly sampled frames after preprocessing | [B, 3, 16, 112, 112] | Channels, Time, Height, Width |
| Stem | Conv3d 3x7x7 (temporal 3), stride (1,2,2), + BatchNorm3d + ReLU | [B, 64, 16, 56, 56] | Spatial halved; temporal depth kept |
| layer1 | 2 BasicBlocks, full 3x3x3 convolutions | [B, 64, 16, 56, 56] | Only stage with full 3D kernels |
| layer2 | 2 BasicBlocks, 1x3x3 (spatial) convolutions | [B, 128, 16, 28, 28] | 2D convolutions from here on |
| layer3 | 2 BasicBlocks, 1x3x3 convolutions | [B, 256, 16, 14, 14] |  |
| layer4 | 2 BasicBlocks, 1x3x3 convolutions | [B, 512, 16, 7, 7] |  |
| avgpool | AdaptiveAvgPool3d(1), then flatten | [B, 512] | One 512-d feature vector per clip |
| fc (replaced) | Linear 512 -> 2 (the only trained part) | [B, 2] | Logits for the two classes |
| softmax | Exponentiate and normalize | [B, 2] | Probabilities summing to 1 |

Two shapes to memorize: input **[B, 3, 16, 112, 112]** and output **[B, 2]** logits. For a single video, B = 1. The two logits are for classes ordered ['non-violence', 'violence'].


### 2.7 From logits to a prediction

Softmax turns the two logits into probabilities: `exp(logit) / sum(exp(logits))`. The predicted class is the larger one (`argmax`), and the **confidence** is the winning softmax score, a number in [0, 1]. Example contract of the API: `{"prediction": "violence", "confidence": 0.923}`.

> **Confidence is not a calibrated probability.** 0.9 does not mean '90% chance of real harm'; it is the softmax score of the predicted class, and it has not been calibrated. The code itself raises if it is not finite and the API contract only guarantees a finite number in [0, 1]. Say exactly this in the viva - examiners respect the honesty.


## 3. The Data Pipeline


### 3.1 Decoding with PyAV

**PyAV** is Python bindings for the FFmpeg libraries. It decodes MP4, AVI, MOV, MKV and WebM without needing a separate FFmpeg executable, because the wheels bundle the libraries. The project uses it in two places: `packages/ml/violence_detection/video.py` for offline loading and the API for uploads; plus per-camera capture threads for monitoring.

Why not OpenCV? OpenCV's frame seeking is unreliable for some containers and variable-frame-rate files, and its decode path offers weaker guarantees about codec handling. PyAV gives frame-accurate sequential decoding and direct access to per-frame properties such as `frame.is_corrupt`.


### 3.2 Two-pass, uniform temporal sampling

Given a video, `load_video()` produces exactly 16 preprocessed frames:

1. **Pass 1 (count).** The video is fully decoded without retaining frames. This verifies limits (120 s, 12,000 frames, 4K area), detects corrupt frames everywhere (not just the first frame), and counts frames. The code deliberately avoids trusting container metadata, which is unreliable for many files.
2. **Index selection.** `np.linspace(0, count - 1, 16)` picks 16 evenly spaced indices spanning the whole video, always including the first and last frames. A clip shorter than 16 frames repeats indices, so the shape stays [3, 16, 112, 112].
3. **Pass 2 (sample).** The video is decoded again; only the selected indices are converted to RGB and kept, each immediately transformed to 3x112x112. Inter-frame codecs still require decoding intervening frames (keyframe dependency), but memory never grows with clip length because only 16 frames are retained.
4. **Consistency check.** If the second pass sees a different frame count, loading fails loudly instead of producing misaligned samples.

A 'frame-index uniform' clip means every part of the video contributes roughly equally, which matches the classification task: a verdict about the whole clip, not one moment in it. The preprocessing identifier stored everywhere is `MC3_18_Weights.KINETICS400_V1.transforms;uniform-index-v1`.


### 3.3 The official Kinetics preprocessing

Each sampled frame goes through `MC3_18_Weights.KINETICS400_V1.transforms()`, torchvision's official transform for these weights (verified against the installed source):

- bilinear **resize to 128 x 171** (with antialias off, as torchvision hard-codes for video presets),
- **center crop 112 x 112**,
- convert to float and **scale to [0, 1]**,
- **normalize** with the Kinetics statistics: mean (0.43216, 0.394666, 0.37645), std (0.22803, 0.22145, 0.216989),
- stack the 16 frames into [C, T, H, W].

Using the official transform is not cosmetic: the pretrained weights were optimized for inputs with exactly these statistics. Any custom normalization would silently shift the input distribution away from what the network expects. No custom constants are used anywhere.

Safety limits enforced before and during decoding:

| Limit | Value | Applies to |
|---|---|---|
| Duration | 120 seconds | Uploads and offline dataset loading (not continuous capture) |
| Decoded frames | 12,000 | Any single decoded video |
| Frame size | 3840 x 2160 pixels (4K area) | Every decoded frame |
| HTTP upload size | 100 MiB | The /api/v1/predict route |
| Sampled frames | 2 to 64 (default 16) | load_video() |


### 3.4 The dataset: RLVS

The project's baseline uses the **RLVS (Real Life Violence Situations)** collection: 2,000 real-world YouTube videos, 1,000 labelled Fight and 1,000 NonFight, distributed through Kaggle (the configured Kaggle archive combines RLVS and RWF-2000; only RLVS is used for the baseline). Labels are derived **only from class folder names** (Fight -> 1, NonFight -> 0), never from filename substrings, and ambiguous layouts fail visibly.

Inspection fully decodes every video and records frame count, duration, FPS, resolution, SHA-256 hash, class and supplied split. The observed results: 2,000 videos, of which 1,996 pass decoding and length checks (998 per class). Three videos exceed 120 seconds and one contains a corrupted frame. There are **14 extra exact-duplicate copies**, including one pair with contradictory labels and three pairs whose duplicates cross the supplied train/val boundary. A reviewed exclusion list (`configs/rlvs-exclusions.csv`) removes the contradictory copies and the cross-split duplicates; same-split duplicates keep one representative.

Final split manifest (after exclusions and deduplication):

| Split | Non-violence | Violence | Total |
|---|---|---|---|
| Train | 674 | 672 | 1,346 |
| Validation | 199 | 198 | 397 |
| Test | 118 | 120 | 238 |


### 3.5 Splitting: groups, hashes and leakage

Splits are produced by `python -m violence_detection.dataset split` with **seed 42** using scikit-learn's `GroupShuffleSplit`. Since RLVS has no supplied test partition, 15% of the remaining training pool is reserved as a local test set, then 15% of the original pool becomes validation; supplied train/val membership is preserved for retained clips. Every partition must contain both classes or the command fails.

**Groups and hashes are the leak protection.** By default each video's group is its SHA-256 hash, so identical files always share a group and can never land in different splits. The manifest stores path, label, split, hash and group; `validate_records` rejects any group or hash that crosses splits, and rejects identical videos carrying conflicting labels. Training re-verifies the SHA-256 of every train/val file before fitting, and evaluation re-verifies test files.

> **Honest limit (say this proactively):** byte hashing catches identical files only. Re-encodes, overlapping clips, common actors or shared camera scenes are **not** detected, because the mirror provides generic filenames instead of trusted source IDs. The command therefore requires `--acknowledge-unknown-groups`, and the README states the resulting numbers are **provisional research measurements, not a source-independent benchmark**. A source-disjoint claim would need a curated groups CSV mapping clips to true sources.


## 4. Training


### 4.1 Configuration

| Setting | configs/baseline.yaml default | Saved checkpoint run |
|---|---|---|
| epochs | 20 | 1 |
| batch_size | 4 | 4 |
| learning_rate | 0.001 | 0.001 |
| num_frames | 16 | 16 |
| seed | 42 | 42 |
| freeze_backbone | true | true |
| device | auto (CUDA, then MPS, then CPU) | auto |
| num_workers | 0 | 0 |
| cpu_threads | 4 | 4 |

`device: auto` picks CUDA, then Apple MPS, then CPU. There is no silent device fallback after an operation fails - the run stops visibly instead. Reproducibility comes from fixed seeds (random, numpy, torch), `torch.use_deterministic_algorithms(True, warn_only=True)` and `cudnn.benchmark = False`; bitwise identity across devices and library versions is still not guaranteed, which is normal.

**About the saved run:** `configs/baseline.yaml` defaults to 20 epochs, but the delivered checkpoint was produced by a run configured with **epochs: 1** (recorded inside the checkpoint and visible in `artifacts/metrics/training.json`, which contains exactly one epoch entry). This guide reports only what that run actually achieved and does not invent multi-epoch curves or ablation studies.


### 4.2 Loss and optimizer

**Loss: cross-entropy.** The model outputs two logits; softmax converts them to a probability distribution; cross-entropy penalizes negative log probability of the true class: `loss = -log p(true class)`. It is the standard choice for classification, well behaved with nearly balanced classes (674/672 in training). A non-finite loss aborts the run before any checkpoint is saved.

**Optimizer: AdamW with learning rate 0.001.** Adam is a gradient-descent variant that keeps per-parameter adaptive learning rates from running estimates of the gradient mean and variance; AdamW (Loshchilov and Hutter, ICLR 2019) decouples weight decay from the gradient step, which regularizes without distorting the adaptive rates. For a small head on a frozen backbone it converges quickly and is robust to the learning-rate choice - a good fit for a one-epoch-friendly, low-tuning baseline. SGD with momentum would need schedules and tuning to beat it here.

Only parameters with `requires_grad=True` (the head) are passed to the optimizer, and `optimizer.zero_grad(set_to_none=True)` clears stale gradients each step.


### 4.3 One epoch of training, step by step

1. Read `configs/baseline.yaml`, seed everything, select device.
2. Read the manifest and **verify the SHA-256 of every train/val video**; any changed file stops the run.
3. Build DataLoaders: shuffle for train, fixed-seed generator, batch size 4; each item decodes and preprocesses one clip to [3, 16, 112, 112] on the fly.
4. Build the pretrained MC3-18, replace the head, freeze the backbone (train mode + eval mode to freeze BatchNorm statistics).
5. For each batch: forward pass -> [B, 2] logits -> cross-entropy -> backward -> AdamW step.
6. After the epoch, evaluate on validation (397 clips) with the same metrics code used for test.
7. If **validation loss is the lowest so far**, save `artifacts/checkpoints/best.pt`; history is appended to `artifacts/metrics/training.json` after every epoch.

The **test split is never loaded during fitting or model selection**. This is what makes the test numbers meaningful.


### 4.4 What the saved run actually shows

Recorded epoch (artifacts/metrics/training.json):

| Epoch | Train loss | Val accuracy | Val precision | Val recall | Val F1 | Val loss |
|---|---|---|---|---|---|---|
| 1 | 0.3285 | 89.17% | 82.70% | 98.99% | 90.11% | 0.2456 |

Validation confusion matrix at that epoch: [[158, 41], [2, 196]] (rows actual, columns predicted, order [non-violence, violence]). Validation loss 0.2456 was the lowest available, so epoch 1 was selected. **Why is one epoch already good?** Because the pretrained Kinetics backbone already extracts strong motion features; a fresh linear head on top of good features can be trained in a single pass over 1,346 clips. A longer run might improve further or begin to overfit - that is future work, not a claim made here.


### 4.5 The checkpoint and its metadata

`best.pt` stores far more than weights, and inference refuses to load an incompatible file:

| Field | Value |
|---|---|
| format_version | 1 |
| architecture | mc3_18 |
| class_names | ['non-violence', 'violence'] |
| preprocessing | MC3_18_Weights.KINETICS400_V1.transforms;uniform-index-v1 |
| num_frames | 16 |
| pretrained_weights | MC3_18_Weights.KINETICS400_V1 |
| epoch | 1 |
| validation (at selection) | accuracy 89.17%, loss 0.2456, CM [[158, 41], [2, 196]] |
| manifest_sha256 | c4afc39bfe575e12361ede16e5d2b1f65d3fa6a06cb3f15b0441dbe070313952 |
| torch / torchvision | 2.14.0 / 0.29.0 |
| File size | about 46 MB (weights + metadata) |

Because metadata travels with the weights, the system can fail fast and visibly when, say, the preprocessing identifier or class order does not match - instead of serving garbage predictions. Inference also **never downloads weights**.


## 5. Evaluation


### 5.1 Metrics and how they are defined

Evaluation uses scikit-learn with **violence as the positive class** (`average='binary', pos_label=1`), and undefined precision is reported as zero. Confusion matrix rows are actual labels and columns are predictions, ordered [non-violence, violence].

Held-out test results (artifacts/metrics/test_metrics.json):

| Metric | Value | How it is computed |
|---|---|---|
| Accuracy | 92.86% | (102 + 119) / 238 = 221 / 238 |
| Precision (violence as positive) | 88.15% | 119 / (119 + 16): of every clip called violence, how many were right |
| Recall (violence as positive) | 99.17% | 119 / (119 + 1): of real violent clips, how many were caught |
| F1 score | 93.33% | Harmonic mean of precision and recall |
| Cross-entropy loss | 0.2092 | Mean softmax cross-entropy over the 238 test clips |
| Samples | 238 | 118 non-violence, 120 violence |

The evaluation also re-verifies every test file's SHA-256 and **refuses to run if the manifest hash differs from the one stored in the checkpoint** - guaranteeing the test set is exactly the one isolated during training. Checkpoint and manifest hashes are recorded in the output JSON for auditability.


### 5.2 Reading the confusion matrix

|  | Predicted: Non-violence | Predicted: Violence |
|---|---|---|
| Actual: Non-violence (118) | 102 (true negatives) | 16 (false alarms) |
| Actual: Violence (120) | 1 (missed violence) | 119 (true positives) |

- **119 of 120 violent clips were caught** (recall 99.17%): only one miss. For a safety-oriented task, missing real violence is the costliest error, so a high recall is exactly what you want.
- **16 non-violent clips were flagged as violence** (precision 88.15%). Most are likely clips with fight-like motion (crowds, fast movement, camera shake) sharing features with real fights.
- The imbalance in error types is a **property of the operating point**: this checkpoint predicts the positive class generously. Deployment thresholds (0.65/0.40 in monitoring) and hysteresis shape the real trade-off between missed events and false alarms.

> **Statistical honesty:** with 238 test clips, a 92.86% accuracy carries real uncertainty; a 95% Wilson interval computed from the observed counts spans roughly 89% to 95%. Small test sets give wide intervals - quoting the interval shows understanding, not weakness.


### 5.3 Why the test set stays untouched

- Test metrics are only trustworthy if test data never influenced fitting **or model selection**. Here selection uses validation loss; test is loaded once, by `evaluate`, with hash verification.
- The README's rule: do not repeatedly tune against test metrics - that turns test into a second validation set and quietly overfits to it.
- Evaluation requires the exact manifest used at training (hash comparison), so the reported numbers cannot silently describe a different split.


## 6. Inference and Deployment


### 6.1 Checkpoint loading gate

`load_checkpoint()` uses `torch.load(..., weights_only=True)` (safe unpickling), then verifies architecture name, class names, preprocessing identifier, format version and a sane frame count before building `mc3_18(weights=None)` and loading the state dict. The model is moved to the selected device and set to `eval()`. All later inference runs under `torch.inference_mode()`, which disables autograd tracking for speed and memory.


### 6.2 Upload API flow (FastAPI)

1. Startup: the API loads `CHECKPOINT_PATH` (default `artifacts/checkpoints/best.pt`). A **missing** checkpoint lets the UI show an unavailable state; an **existing but incompatible** one fails startup loudly. A single Uvicorn worker is required because model and camera state are process-local.
2. POST `/api/v1/predict` with multipart video: extension check, streaming size limit (a middleware caps bodies at 100 MiB + 1 MiB slack), then a **non-blocking acquire of the inference lock** - if another inference is running, the client gets 429 immediately.
3. The upload is written to a temp file, decoded and sampled by the **same `load_video()` used in training**, forwarded through the model, and softmaxed. The temp file is removed afterwards.
4. Response: `{"prediction": "violence" | "non-violence", "confidence": <0..1>}`; errors map to the codes below. `/health` reports `model_ready` separately from process liveness.

| HTTP status | Meaning |
|---|---|
| 413 | Upload larger than 100 MiB |
| 415 | File extension not supported (MP4, AVI, MOV, MKV, WebM) |
| 422 | Empty, undecodable or over-limit video; missing form field |
| 429 | Another inference is running (single-model lock) |
| 503 | No trained checkpoint loaded |
| 500 | Model execution failed; check server logs |

CORS origins come from `WEB_ORIGINS` (default `http://localhost:3000`). The FastAPI contract is browsable at `/docs`. There are **no training endpoints** - by design.


### 6.3 Continuous monitoring

The monitor reuses the same shared model and adds per-camera streaming machinery:

- **One capture thread per camera** decodes RTSP (over TCP, 5 s timeout) or a paced local `file://` test video that loops at EOF, reconnecting with exponential backoff (1 s doubling to a cap).
- Sampled frames (default 4 fps) go into a **bounded timestamped deque per camera**; older frames are dropped once the 4-second window is covered, so memory never grows.
- Every `stride_seconds` (1 s) the camera builds a window: 16 target timestamps are spread across the window and the **nearest timestamped frames are chosen** (low-FPS streams repeat nearest samples). The window replaces any pending one - the **dropped-windows counter** shows backlog instead of queueing stale clips.
- A consumer thread takes the newest pending window and runs inference under the **shared model lock**; stale windows older than 2 strides are skipped rather than executed.
- The score is the softmax violence column of [1, 2] logits, then **hysteresis** turns scores into a stable state machine (below).

Monitoring defaults (starter settings, not validated operating points):

| Setting | Default | Meaning |
|---|---|---|
| window_seconds | 4 | Length of the temporal window per inference |
| sample_fps | 4 | Frames sampled per second: 4 s x 4 fps = 16 frames |
| stride_seconds | 1 | How often a fresh window is classified |
| positive_threshold | 0.65 | Violence score at or above this counts as a positive window |
| negative_threshold | 0.40 | Score at or below this counts as a negative window |
| positive_windows | 3 | Consecutive positive windows needed to raise VIOLENCE |
| negative_windows | 4 | Consecutive negative windows needed to return to NORMAL |


### 6.4 Hysteresis: why two thresholds and window counters

Borderline scores flicker. If a camera flipped NORMAL/VIOLENCE on every single window, the UI would strobe and alerts would be useless. The state machine demands **3 consecutive windows at or above 0.65** to raise VIOLENCE and **4 consecutive windows at or below 0.40** to return to NORMAL; scores between the thresholds reset both counters. This adds a small detection delay (a few seconds) in exchange for a stable, explainable state - the classic engineering trade-off between responsiveness and stability. States OFFLINE and ERROR are distinct from NORMAL; only the current active event is kept in memory.


### 6.5 Upload evaluation vs monitoring: a documented mismatch

Training samples 16 frames **across the whole video**; monitoring samples 16 frames from a **4-second window**. The checkpoint was therefore trained on whole-video statistics and is applied to short windows at inference - a temporal-distribution mismatch the README flags explicitly. Also, RLVS video-level labels do **not** mean every 4-second sub-window of a violent video shows violence; naive window-level training on video labels would be wrong. Fixing this properly needs window-aware training: event annotations, reviewed temporal localization, or a documented weak-supervision strategy. The monitoring feature does not itself produce a CCTV-adapted checkpoint.


## 7. Why Each Choice: Alternatives and Trade-offs

Viva examiners love 'why did you do X instead of Y?'. This table covers the main decisions; each row is a ready answer.

| Decision | Why it was made | Alternative and its tradeoff |
|---|---|---|
| 3D CNN over the whole clip | Learns motion directly from a stack of frames; a single model handles space and time | 2D CNN + LSTM: more flexible temporal modelling but needs a second stage, more engineering and usually more data |
| MC3-18 (mixed 3D/2D) | Smallest mature torchvision video model with Kinetics-400 weights (11.7 M params) | R(2+1)D-18 scores higher on Kinetics (67.5%) but has 2.7x parameters; R3D-18 is larger still |
| Transfer learning from Kinetics-400 | Training a video backbone from scratch on 1,346 clips carries substantial overfitting risk | Training from scratch: needs orders of magnitude more labelled video and compute |
| Frozen backbone, trainable head | Only 1,026 trainable weights: fast, stable, low risk on a small dataset | Full fine-tuning (freeze_backbone: false): could adapt more, but needs a lower LR, more epochs and can overfit or drift |
| PyAV decoding | Bundled FFmpeg libraries, frame-accurate sequential decode, no external executable needed | OpenCV: frame seeking is unreliable on some containers/codecs; separate FFmpeg adds setup burden |
| Uniform index sampling of the whole video | Matches how the baseline is defined and evaluated; covers the entire clip | Fixed-window sampling: faster for long videos but trains and evaluates on a narrower view |
| Official Kinetics transforms | Guarantees the exact resize, crop, scaling and normalization statistics the pretrained weights expect | Custom preprocessing: any drift in statistics silently degrades a pretrained model |
| Cross-entropy loss | Standard for softmax classification; well behaved with 2 logits and balanced classes | Focal loss or class weights: useful under imbalance; here classes are nearly balanced (1,346 train: 674/672) |
| AdamW, lr 0.001 | Robust to learning-rate choice for a small head; decoupled weight decay is standard practice | SGD + momentum: often better final accuracy at scale, but needs schedules and tuning |
| Best-validation-loss checkpoint | Keeps the epoch that generalizes best on held-out data, guarding against late overfitting | Last-epoch checkpoint: simpler, but silently keeps an overfit model |
| Hash-based duplicate and group handling | Exact duplicate files cannot leak across splits; conflicting labels fail loudly | Plain random split: silently inflates metrics when copies of one clip appear in train and test |
| Serialized inference (one model lock) | Bounds memory on a small host; concurrent requests get a clear 429 instead of OOM | Parallel inference: higher throughput but risks memory pressure and device contention |
| Hysteresis for camera states | Smooths flickering NORMAL/VIOLENCE toggles caused by borderline scores | Raw thresholding: instant reaction but noisy state changes and alert flapping |
| Softmax confidence reported as-is | Simple, finite, bounded in [0,1]; clearly documented as an uncalibrated score | Platt/isotonic calibration: gives true probabilities but needs a held-out calibration set and ongoing validation |


## 8. Limitations and Honest Caveats

1. **One recorded epoch.** The delivered checkpoint is from a run with epochs: 1. Validation loss was 0.2456; more training might improve it. No multi-epoch tuning or ablation studies exist, and none should be claimed.
2. **Small test set.** 238 clips give wide confidence intervals (roughly 89-95% for the 92.86% accuracy at the 95% level).
3. **Source leakage unverified.** Hashing catches byte-identical files only; related-source splits need a curated groups CSV. Numbers are provisional research measurements.
4. **Temporal mismatch.** Whole-video-trained weights are applied to 4-second monitoring windows; this needs evaluation before monitoring results are trusted.
5. **Video-level labels.** A violence-labelled video does not make every sub-window positive; window-aware training needs proper annotations.
6. **Uncalibrated scores.** The confidence is a softmax score, not a validated probability of harm; no calibration curve exists.
7. **Dataset domain shift.** RLVS comes from YouTube-style footage; real CCTV differs in lighting, viewpoint, compression and demographics. Dataset accuracy is not evidence of reliable real-world detection. Bias and domain-shift evaluation are future work.
8. **Prototype security.** The API is a local research prototype without authentication, TLS termination or durable incident storage; camera events live in memory only. It must not be exposed to the public internet, and a human should stay in the loop for any action taken on alerts.
9. **Compute figures are reference figures.** The GFLOP numbers are torchvision's published values, not local latency benchmarks; the same applies to published Kinetics accuracies.


## 9. Presentation Speaking Notes


### 9.1 Two-minute script (about 260 words)

"Our project detects violence in video streams. The core is a deep learning model that takes a short clip - 16 frames - and outputs one of two labels: violence or non-violence, with a confidence score.

"We chose an MC3-18 network, a residual 3D convolutional network from torchvision that is pretrained on the Kinetics-400 action dataset. 3D convolutions slide over frames as well as pixels, so the network sees motion directly - exactly what a fight is. MC3 is 'mixed': only the earliest layers use full 3D kernels; the rest use cheaper 2D ones, which keeps the model small at 11.7 million parameters. Because our dataset is small - about 1,350 training clips - we freeze the pretrained backbone and train only a new two-output classification head, about a thousand weights, with cross-entropy loss and the AdamW optimizer.

"Videos are decoded with PyAV, 16 frames are picked uniformly across the whole clip, and preprocessed exactly as Kinetics expects - resize to 128 by 171, crop to 112 by 112, normalize with Kinetics statistics. We audited the dataset for duplicates and split it by hash groups with a fixed seed, so identical files cannot leak between train and test.

"On 238 held-out clips the model reaches 92.9 percent accuracy, 99.2 percent recall - it caught 119 of 120 violent clips - with 88 percent precision. The same checkpoint powers a FastAPI backend: upload analysis, a live camera page, and a multi-camera monitor where a hysteresis state machine turns frame scores into stable NORMAL/VIOLENCE camera states. Scores are uncalibrated, the split is not source-disjoint, and the monitoring windows differ from training - we document all of this honestly in the report."


### 9.2 Five-minute script (add these blocks to the 2-minute script)

**Block 1 - data pipeline (about 1 minute).** "Let me walk through the pipeline. A video enters through PyAV, which decodes frame by frame with FFmpeg libraries bundled in the wheel. We do two passes: a counting pass that validates limits and finds corrupt frames anywhere, then a sampling pass that keeps only 16 evenly spaced frames - first and last included; short clips repeat indices so the tensor shape never changes. Every frame is resized to 128 by 171, center-cropped to 112 by 112, scaled to [0, 1] and normalized with the exact Kinetics means and standard deviations - no custom constants, because pretrained weights assume their training statistics."

**Block 2 - training mechanics (about 45 seconds).** "Training verifies every train and validation file's SHA-256 before starting, then runs cross-entropy on batches of four with AdamW at learning rate 0.001. After each epoch we evaluate on validation and keep the checkpoint with the lowest validation loss; the checkpoint carries architecture, class order, preprocessing identifier and hashes, and inference refuses an incompatible file. Our saved run is one epoch - validation accuracy 89.2 percent - which already works because the frozen Kinetics features are strong and only the linear head trains."

**Block 3 - results (about 45 seconds).** "On the untouched test set of 238 clips: accuracy 92.86 percent, precision 88.15, recall 99.17, F1 93.33. The confusion matrix tells the story: 102 true negatives, 16 false alarms, 1 miss, 119 true positives. Missing violence is the expensive error for a safety task, so we operate at high recall; the 16 false alarms are fight-like motion and can be filtered by the monitoring thresholds. With only 238 clips the 95 percent interval on accuracy is wide - roughly 89 to 95 percent - and we say so."

**Block 4 - system and limits (about 45 seconds).** "The deployment keeps one shared model behind a lock: a second concurrent request gets HTTP 429 instead of running a second memory-heavy forward pass. Cameras each hold a bounded buffer; every second the newest 16-frame window is scored, and hysteresis - three positive windows above 0.65 to raise an alert, four negative windows below 0.4 to clear - keeps states stable. Honest limits: one recorded epoch, related-source leakage unverified, uncalibrated scores, YouTube-sourced data rather than real CCTV, and a documented mismatch between whole-video training and 4-second monitoring windows. Next steps: fine-tuning on consented CCTV-style data, window-aware training with event annotations, and probability calibration."


## 10. Likely Viva Questions and Answers

50 questions, ordered from basic to challenging. Group A: basics. Group B: model. Group C: data. Group D: training. Group E: evaluation. Group F: system and critical thinking.


#### Group A. Basics

**Q1. What does your project do, in one sentence?**

It classifies short video clips as violence or non-violence using a pretrained 3D CNN, and serves the model through a web app for uploads, a live single-camera page and a multi-camera CCTV monitor.

**Q2. Why is this problem hard for classical computer vision?**

Violence is a temporal event defined by motion between frames; single-frame appearance cues (raised arm, crowd) are ambiguous. It also needs robustness to lighting, viewpoint and camera quality, which rule out simple heuristics.

**Q3. Walk me through the whole system.**

Video arrives by upload or camera capture; PyAV decodes it; 16 frames are sampled and preprocessed exactly like Kinetics; the frozen MC3-18 backbone + trained linear head produce 2 logits; softmax gives a prediction and score; the upload route returns JSON, while monitoring feeds scores into a per-camera hysteresis state machine shown on a Next.js dashboard via WebSocket.

**Q4. Which dataset did you use and what is in it?**

RLVS - Real Life Violence Situations - 2,000 real YouTube videos, 1,000 Fight and 1,000 NonFight, obtained via a Kaggle archive. Labels come from class folder names only.

**Q5. What exactly is the input tensor to the model?**

A batch of shape [B, 3, 16, 112, 112]: batch, RGB channels, 16 frames, 112x112 pixels, normalized with Kinetics statistics. A single video is B=1.

**Q6. What is the model's output and how do you read it?**

[B, 2] logits for classes ['non-violence', 'violence']; softmax converts to probabilities; the larger probability is the prediction and its value is the reported confidence.

**Q7. What are your headline results?**

On 238 held-out test clips: accuracy 92.86%, precision 88.15%, recall 99.17%, F1 93.33%; confusion matrix [[102, 16], [1, 119]] with violence as the positive class.


#### Group B. Model and architecture

**Q8. Why a 3D CNN instead of a 2D CNN?**

Because violence is motion. 3D convolutions have a temporal extent, so filters respond to how pixels change across frames - directly learning motion patterns. A 2D CNN sees only stills and would need a separate temporal model (LSTM/TCN) bolted on.

**Q9. What does MC3-18 mean?**

Mixed Convolution ResNet-18 for video (Tran et al., CVPR 2018): full 3D kernels only at the earliest layers, spatial 1x3x3 (2D) kernels afterwards. In torchvision's implementation the stem and first stage use 3x3x3 convs and stages 2-4 use 1x3x3 convs - verified against the weights.

**Q10. Compare R3D-18, R(2+1)D-18 and MC3-18.**

R3D: full 3x3x3 everywhere, 33.4 M params, Kinetics acc@1 63.20%. R(2+1)D: factorized spatial + temporal kernels, 31.5 M params, 67.46%. MC3: 3D early, 2D late, 11.7 M params, 63.96%. We chose MC3 for the frozen-feature baseline because it is 2.7-2.9x smaller with comparable published accuracy.

**Q11. Why not the more accurate R(2+1)D-18?**

Its 67.5% Kinetics advantage applies to full fine-tuning at scale; with a frozen backbone on 1.3k clips the extra 20 M parameters mainly add memory, compute and risk. For a prototype that must train and serve on modest hardware, MC3-18 is the better engineering choice.

**Q12. What is transfer learning and why Kinetics-400?**

Starting from weights trained on a related large task. Kinetics-400 has about 240k clips of 400 human actions, so the backbone already encodes human motion primitives. Our 1,346 training clips are far too few to learn those from scratch without severe overfitting.

**Q13. What exactly did you train?**

Only the new classification head: a Linear layer from 512 features to 2 logits, i.e. 1,026 weights. All 11.49 M backbone weights are frozen (requires_grad=False).

**Q14. Why freeze the backbone? What is the trade-off?**

Small data, fast and stable training, reproducible results, low compute. The trade-off is a lower ceiling: the backbone cannot specialize to fight-specific features. Fine-tuning is supported via freeze_backbone: false with a lower learning rate.

**Q15. What happens to BatchNorm layers when the backbone is frozen?**

The whole model is put in eval() mode during training so BatchNorm uses its pretrained running statistics and does not update them from our small batches - otherwise the frozen backbone's input normalization would silently drift.

**Q16. Trace the tensor through the network.**

[B,3,16,112,112] -> stem 3x7x7 conv -> [B,64,16,56,56] -> layer1 (3D) -> same shape -> layer2 [B,128,16,28,28] -> layer3 [B,256,16,14,14] -> layer4 [B,512,16,7,7] -> global average pool + flatten -> [B,512] -> fc -> [B,2] -> softmax. Temporal depth 16 is preserved throughout; only spatial size shrinks.

**Q17. Why is there a residual (skip) connection in each block?**

The skip adds a block's input to its output, so each block only needs to learn a small residual correction. This eases gradient flow in deep networks (He et al., 2016) and lets an 18-layer video network train stably.

**Q18. What does the confidence number actually mean?**

The softmax score of the winning class: a relative model output in [0,1], not a calibrated probability of real harm. Calibration (Platt/isotonic on a held-out set) is documented future work.


#### Group C. Data pipeline

**Q19. Why PyAV rather than OpenCV or a shell ffmpeg?**

PyAV binds the FFmpeg libraries directly (they ship inside the wheel), decodes frame-accurately and sequentially, exposes per-frame properties like is_corrupt, and avoids relying on OpenCV's less reliable frame seeking or on an external executable.

**Q20. How are the 16 frames chosen from a video?**

Uniformly: np.linspace over the decoded frame count, always including the first and last frames. A first decode pass counts frames and validates limits (the container's own frame count is not trusted); a second pass retains only the selected frames, transformed to 3x112x112 each.

**Q21. What if a video has fewer than 16 frames?**

Indices repeat: a 5-frame video samples [0,1,2,3,4,4,4,...]. The shape stays [3,16,112,112]. Repeating frames carry less temporal information, which the model handles but which is naturally a weaker signal.

**Q22. Why decode the video twice?**

Pass 1 fully validates the file (limits, corrupt frames anywhere, true frame count) without storing anything; pass 2 keeps only 16 frames in memory. Inter-frame codecs force decoding intervening frames either way, so the two passes cost decode time but bound memory.

**Q23. What preprocessing is applied and why those exact constants?**

Per frame: bilinear resize to 128x171, center crop 112x112, scale to [0,1], normalize with the Kinetics-400 mean/std (0.43216, 0.394666, 0.37645 / 0.22803, 0.22145, 0.216989). These are the official transforms of the pretrained weights - using anything else would shift the input distribution away from what the network was trained on.

**Q24. Why 16 frames and 112x112 pixels?**

They are the MC3-18/Kinetics architecture defaults: Kinetics models were trained at 112x112 with 8-16 frame clips, and torchvision's weights and transforms are built for that regime. 16 frames gives a wider temporal view than 8 while staying affordable.

**Q25. What limits protect the pipeline?**

120 s duration, 12,000 decoded frames, 3840x2160 pixel area, 100 MiB uploads, and a sampled-frame range of 2-64. Uploads are rejected with clear errors rather than silently truncated.

**Q26. How did you split train/val/test, and why that way?**

GroupShuffleSplit with seed 42, keeping RLVS's supplied train/val for retained clips and reserving 15% of the training pool as a local test set. Group = SHA-256 hash by default, so exact duplicates can never straddle splits; conflicting labels or split-crossing hashes fail loudly. Final: 1,346 / 397 / 238.

**Q27. How do you detect duplicates and leakage?**

Inspection hashes every video; duplicate groups are reported and deduplicated with reviewed exclusions (14 extra copies found, including one contradictory-label pair and three pairs crossing train/val). Manifest validation rejects any group/hash crossing splits, and training and evaluation re-verify file hashes.

**Q28. Can you claim source-disjoint results?**

No. Hashing cannot detect re-encodes, overlapping footage, shared actors or scenes; the mirror has generic filenames rather than trusted source IDs. The pipeline requires explicitly acknowledging unknown groups, and results are presented as provisional research measurements.


#### Group D. Training

**Q29. What loss function did you use and why?**

Softmax cross-entropy: with 2 logits it penalizes negative log probability of the true class, is the standard objective for classification, and suits the nearly balanced classes (674/672). A non-finite loss aborts the run before saving.

**Q30. Why AdamW at learning rate 0.001?**

AdamW keeps Adam's adaptive per-parameter rates and decouples weight decay, giving fast, robust convergence for the small head with minimal tuning - appropriate for a one-epoch-friendly frozen-backbone baseline. SGD+momentum could match or beat it at scale but needs schedules and tuning.

**Q31. How long did you train, honestly?**

The delivered checkpoint comes from a run configured with epochs: 1 (the config default is 20). training.json contains exactly one epoch: train loss 0.3285, validation accuracy 89.17%, validation loss 0.2456. One epoch suffices to reach 92.86% test accuracy because the frozen Kinetics features are strong and only the linear head adapts. We claim nothing beyond this.

**Q32. How is the best checkpoint selected?**

After every epoch the model is evaluated on validation; the epoch with the **lowest validation loss** overwrites best.pt. Loss is used instead of accuracy because it is a finer-grained signal (confidence-aware, differentiable) and is the same quantity the optimizer reduces. Test data is never used for selection.

**Q33. How do you make training reproducible?**

Fixed seeds for random/numpy/torch (seed 42), a seeded DataLoader generator, use_deterministic_algorithms(True, warn_only=True), cudnn.benchmark=False, and pinned torch/torchvision versions recorded in the checkpoint. Bitwise identity across devices is still not guaranteed - that is stated, not hidden.

**Q34. What does the checkpoint contain besides weights?**

format_version, architecture, class order, preprocessing identifier, frame count, pretrained-weight identifier, full training config, selected epoch, validation metrics, manifest SHA-256, and torch/torchvision versions - enough for inference to reject any incompatible model.


#### Group E. Evaluation and results

**Q35. Define precision, recall and F1 for this task.**

With violence as positive: precision = TP / (TP + FP) - of clips we called violent, the fraction that truly were (88.15%). Recall = TP / (TP + FN) - of truly violent clips, the fraction we caught (99.17%). F1 is their harmonic mean (93.33%), balancing the two.

**Q36. Interpret your confusion matrix.**

Rows are actual, columns predicted, order [non-violence, violence]: [[102, 16], [1, 119]]. 102 true negatives, 16 false alarms (non-violent called violent), 1 missed violent clip, 119 true positives. The model rarely misses violence but over-predicts it somewhat.

**Q37. Why is recall so high and precision lower? Is that good?**

The checkpoint's operating point predicts the positive class generously: 119/120 violent clips caught at the cost of 16 false alarms. For a safety prototype, missing real violence is the more dangerous error, so this bias is defensible; the monitoring thresholds and hysteresis then shape the actual alert trade-off. In deployment the thresholds are the tuning surface.

**Q38. Why keep a test set the training never sees?**

Because metrics only measure generalization if the data influenced neither fitting nor model selection. Selection uses validation loss; test is loaded once by the evaluation command, which also verifies test file hashes and that the manifest matches training's manifest hash.

**Q39. How confident are you in 92.86% accuracy?**

Moderately: 238 clips give a wide 95% Wilson interval, roughly 89-95%. Also the split is not source-disjoint, and the dataset is YouTube footage rather than real CCTV - so the number measures this benchmark, not real-world deployment performance.


#### Group F. System, deployment and critical thinking

**Q40. How does continuous monitoring work end to end?**

Each camera runs a capture thread decoding RTSP (TCP, 5 s timeout) or a looping local file, sampling at 4 fps into a bounded timestamped buffer. Every second the newest 16-frame window is built by nearest-timestamp matching and queued as the camera's single pending window; a consumer thread runs it through the shared model under a global lock; the softmax violence score feeds hysteresis; states and events stream to the dashboard over WebSocket; MJPEG previews reuse the same capture.

**Q41. What is hysteresis and why do you need it?**

A two-threshold state machine: 3 consecutive windows at or above 0.65 raise VIOLENCE; 4 consecutive windows at or below 0.40 return to NORMAL; in-between scores reset the counters. It prevents state flapping from borderline scores at the cost of a few seconds of detection delay.

**Q42. What happens if two users upload videos at the same time?**

Inference is serialized: the second request fails the non-blocking lock acquire and gets HTTP 429 with a retry hint, instead of running a second memory-heavy forward pass. One worker process is required because model and camera state are process-local.

**Q43. What is the biggest limitation of the model?**

Honest answer: domain transfer. It was trained on YouTube-style RLVS footage with video-level labels, one recorded epoch, and a split that is not source-disjoint; monitoring applies it to 4-second windows it was never trained on. Dataset accuracy is not evidence of reliable real-world CCTV detection.

**Q44. How would you improve the project next?**

Fine-tune (unfrozen) on representative, consented CCTV-style data; train window-aware with event annotations or weak supervision; build source-disjoint splits from a curated groups CSV; calibrate scores on held-out data; measure latency/throughput and event-level recall/false-alert rate on cameras; keep a human in the loop.

**Q45. Is this system production-ready?**

No - and the README says so. It is a local research prototype: no authentication, no TLS, events only in memory, one worker, starter thresholds, uncalibrated scores. It must not face the public internet. The engineering (hash audits, checkpoint gating, honest errors) is solid for research, not for deployment.

**Q46. Your checkpoint says epoch 1 but the config says 20 epochs. Explain.**

configs/baseline.yaml defaults to 20 epochs, but the run that produced best.pt was configured with epochs: 1; the checkpoint records its own config and epoch, and training.json has exactly one entry. Best-validation-loss selection then kept that epoch. We report what exists - one epoch - and do not claim additional training.

**Q47. What safeguards exist against loading a wrong or corrupted checkpoint?**

torch.load with weights_only=True; verification of architecture, class names, preprocessing identifier, format version and frame-count range before building the model; a mismatched manifest hash aborts evaluation; startup fails loudly on an incompatible checkpoint instead of serving random weights.

**Q48. Why softmax over two logits instead of a single sigmoid?**

Two logits with softmax is the standard multi-class formulation that generalizes beyond two classes, keeps the code identical if more classes arrive, and pairs directly with cross-entropy and argmax. A single sigmoid output is equivalent for two classes but would tie the code to binary-specific plumbing.

**Q49. What do the tests cover?**

Python tests exercise real decoding, short-clip sampling, dtype/layout, shared preprocessing, two-class outputs, checkpoint compatibility, labels, deterministic grouped splits, leakage rejection, API prediction shapes, busy/unavailable models, corrupted uploads and stream size limits. Browser tests cover UI states and error handling with HTTP fixtures - mocks verify UI behavior, not model quality.

**Q50. Explain 'temporal-distribution mismatch' in one breath.**

Training saw 16 frames spread across whole videos; monitoring feeds 16 frames from 4-second windows, and video-level labels never guaranteed those windows are violent - so live scores operate off-distribution and need dedicated evaluation before being trusted.


## 11. Quick Revision Sheet

One-page cram list. If you can answer these bullets from memory, you are ready.

Numbers to know cold:

| Item | Value |
|---|---|
| Model / backbone | MC3-18, torchvision, Kinetics-400 pretrained (11.7 M params), frozen |
| Trainable head | Linear 512 -> 2 = 1,026 weights; AdamW lr 0.001; cross-entropy |
| Input / output | [B, 3, 16, 112, 112] -> [B, 2] logits -> softmax |
| Preprocessing | resize 128x171 -> crop 112x112 -> scale [0,1] -> Kinetics normalize |
| Sampling | 16 uniform frame indices, whole video; PyAV two-pass decode |
| Dataset | RLVS 2,000 inspected; 1,996 usable; splits 1,346 / 397 / 238; seed 42 |
| Saved run | epochs: 1; train loss 0.3285; val acc 89.17%, val loss 0.2456 |
| Test results | acc 92.86% \| precision 88.15% \| recall 99.17% \| F1 93.33% \| loss 0.2092 |
| Confusion matrix | [[102, 16], [1, 119]]; rows actual, cols predicted, [NV, V] |
| Monitoring defaults | 4 s window, 4 fps, 1 s stride, thresholds 0.65/0.40, 3+/4- windows |
| Upload limits | 100 MiB, 120 s, 12,000 frames, 4K area; errors 413/415/422/429/500/503 |
| Checkpoint hash | 698e6d253c38f66abdcebd6e35f47c81222dccbb33ec3c10381ebd17ee483f3d |

- MC3 = mixed 3D/2D: 3D early (stem + layer1), 2D (1x3x3) later.
- Frozen backbone keeps pretrained BatchNorm statistics: train() then eval().
- Kinetics-400: ~240k clips, 400 actions; pretrained top-1 63.96%.
- Transfer learning reason: pretrained features reduce data requirements and overfitting risk.
- Uniform sampling spans the whole clip; short clips repeat indices.
- Group = SHA-256: exact duplicates cannot cross splits; related-source leakage unverified.
- Checkpoint selection: lowest validation loss; test untouched until final evaluation.
- Recall 99% because only 1 of 120 violent clips missed; 16 false alarms lower precision.
- Confidence = softmax score, NOT calibrated probability.
- Monitoring mismatch: whole-video training vs 4-second windows - documented, unvalidated.
- Hysteresis: 3 consecutive positives >= 0.65 to alert; 4 negatives <= 0.40 to clear.
- One shared model + inference lock -> HTTP 429 when busy; temp uploads deleted.
- Checkpoint carries metadata; incompatible checkpoint = loud failure, never random weights.


## 12. Glossary

| Term | Meaning |
|---|---|
| 3D convolution | A convolution whose kernel also spans time; learns motion directly from frame stacks |
| MC3 | Mixed Convolution: full 3D kernels early, spatial-only 1x3x3 kernels later |
| Residual connection | Skip path adding a block's input to its output; stabilizes deep network training |
| Backbone | The feature-extracting body of the network (everything except the final classifier) |
| Head | The final classifier; here a Linear 512 -> 2 layer, the only trained part |
| Transfer learning | Reusing weights pretrained on a large task (Kinetics-400) for a smaller one |
| Frozen weights | Parameters excluded from updates (requires_grad=False) |
| BatchNorm | Layer normalizing activations; keeps running statistics that must stay frozen when backbone is frozen |
| Logits | Raw model scores before softmax; here [B, 2] |
| Softmax | Converts logits to probabilities that sum to 1 |
| Cross-entropy | Classification loss: negative log probability of the true class |
| AdamW | Adam optimizer with decoupled weight decay (Loshchilov & Hutter, 2019) |
| Precision / Recall / F1 | Positive-class error split; F1 is their harmonic mean |
| Confusion matrix | Table of actual vs predicted counts; rows actual here |
| PyAV | Python bindings to FFmpeg libraries for decoding media |
| Uniform sampling | Choosing 16 evenly spaced frame indices across the whole video |
| GroupShuffleSplit | sklearn splitter that keeps whole groups on one side of a split |
| Data leakage | Test information influencing training; here guarded by hashes and groups |
| SHA-256 | Cryptographic file hash used to detect duplicates and verify integrity |
| Hysteresis | Two-threshold state machine that prevents rapid state flapping |
| Calibration | Making scores trustworthy as probabilities; not done here, documented as future work |
| MJPEG | Motion JPEG: an HTTP stream of JPEG frames used for camera previews |
| WebSocket | Push channel used to stream camera states to the dashboard |


## 13. References

1. D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, M. Paluri. A Closer Look at Spatiotemporal Convolutions for Action Recognition (R3D, R(2+1)D, MC3). CVPR 2018. arXiv:1711.11248.
2. W. Kay et al. The Kinetics Human Action Video Dataset. arXiv:1705.06950, 2017.
3. M. Cheng, K. Cai, M. Li. RWF-2000: An Open Large Scale Video Database for Violence Detection. arXiv:1911.05913 (ICPR 2021). The Kaggle archive used here combines RLVS and RWF-2000; the baseline uses the RLVS collection.
4. Kaggle: Video Violence Detection Dataset (RLVS + RWF-2000 mirror), kaggle.com/datasets/magicearth25/video-violence-detection-dataset.
5. K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR 2016. arXiv:1512.03385.
6. I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization (AdamW). ICLR 2019. arXiv:1711.05101.
7. S. Ioffe, C. Szegedy. Batch Normalization: Accelerating Deep Network Training. ICML 2015. arXiv:1502.03167.
8. torchvision video models and transforms (official docs): docs.pytorch.org/vision/stable/models.html - video classification section.
9. PyAV documentation: pyav.org/docs/stable/.
10. Project sources: README.md and packages/ml/violence_detection/, apps/api/violence_api/, configs/, artifacts/metrics/ in this repository.

