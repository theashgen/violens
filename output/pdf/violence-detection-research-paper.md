# Violence Detection in Surveillance Videos Using Transfer Learning and Continuous Multi-Camera Monitoring

Ashwin Sathiya Moorthy | Registration No. 12308018 | Roll No. 46 | theashgen06@gmail.com

Prudhviraj K B | Registration No. 12307060 | Roll No. 17 | prudhvirajkb3@gmail.com

Pranav R | Registration No. 12306134 | Roll No. 36 | pranavsandle@gmail.com
B.Tech in Artificial Intelligence and Machine Learning
Lovely Professional University, Punjab, India

## Abstract

We evaluated transfer-learned video violence classification on RLVS, three independent external clip benchmarks, and unedited continuous multi-camera surveillance recordings. Within RLVS, SHA-256 and frame-based perceptual hashing formed visually grouped partitions before splitting; because RLVS lacks source identifiers, we further evaluated the exact fixed RLVS checkpoints on 2,150 external clips with verified zero overlap: RWF-2000 validation (400 CCTV clips), Bus Violence (1,400 multi-camera transit clips), and AIRTLab (350 dual-camera clips including non-violent lookalike actions). Across three 20-epoch head-training seeds at threshold 0.50, frozen MC3-18 averaged 96.73% accuracy on 397 grouped RLVS test clips (ResNet-18: 95.97%) but fell to 71.92% on RWF-2000 (AUC 0.808 vs. 0.699), 56.02% on Bus Violence (AUC 0.710 vs. 0.562), and 56.29% on AIRTLab (AUC 0.551 vs. 0.442), exposing substantial domain shift and a 70.8% false-positive rate on close-contact non-violent lookalikes. On 34 unedited official UCF-Crime test recordings (38.0 min, 0.465 camera-hours negative exposure, 11 human-annotated events), default monitor hysteresis (0.65 for 3 windows) matched 5/11 events (4/5 Fighting; median delay 11.5 s) at 32.27 false alerts per camera-hour, whereas a separate continuous validation-calibrated policy (0.85 for 5 windows) cut false alerts fivefold to 6.45 per camera-hour while lowering recall to 3/11. Live multi-camera trials on 1 to 16 unedited UCF-Crime streams scaled linearly through 8 streams (6.83 windows/s, 0.0% replaced) before saturating the serialized lock at 12 and 16 streams (10.19 windows/s, 24.0% replaced). Independent human inter-annotator agreement on RLVS remains open, and the system must be used only as an unverified human-in-the-loop triage prototype.

**Keywords:** violence detection; video classification; MC3-18; transfer learning; RLVS; RWF-2000; Bus Violence; AIRTLab; UCF-Crime; multi-camera monitoring.

## I Introduction

A person watching several surveillance feeds can miss short incidents, especially when multiple feeds compete for attention. An automated classifier can triage footage for human review, but ordinary activity frequently resembles physical aggression. A celebratory jump, a hug, a crowded scene, or someone helping another person stand up can trigger spurious scores depending on camera angle, occlusion, and compression.

We set out to measure how well a pretrained spatiotemporal video model classifies RLVS clips, how reliably its fixed checkpoints generalize to provenance-verified external surveillance datasets, and how the same checkpoint behaves inside a continuous multi-camera application. The upload view returns one prediction per video, a diagnostic view accepts a browser webcam or network camera, and a continuous monitor tracks multiple cameras concurrently. All three load the same compatible checkpoint contract.

To address dataset leakage, training rigor, continuous monitoring validation, non-violent lookalike error modes, and false-alert governance, we combine five reproducible protocols: perceptual-hash grouped RLVS partitioning, multi-seed 20-epoch training comparing MC3-18 and frame-averaged ResNet-18, zero-shot evaluation on three independent external benchmarks (RWF-2000, Bus Violence, and AIRTLab), continuous event detection and validation-only alarm calibration on unedited UCF-Crime CCTV recordings, and live multi-camera concurrency scaling through lock saturation.

<!-- page -->
## II Related work and research positioning

### A. Residual and spatiotemporal representations

He et al. introduced shortcut connections to ease optimization in deep visual networks [1]. Video models must also capture motion across frames. Tran et al. compared full 3D convolutions, mixed 3D and 2D convolutions, and factorized spatiotemporal convolutions in residual networks [2]. We adopted torchvision's mixed-convolution MC3-18 initialized with Kinetics-400 action-recognition weights [3,4] and trained only a two-output linear classifier head.

### B. Violence benchmarks and cross-domain shift

Public violence benchmarks differ sharply in capture provenance and scene realism. RLVS [6] aggregates short internet clips (1.0 to 11.3 s) without source camera identifiers, whereas RWF-2000 [5] compiles 5-second real CCTV surveillance clips. Bus Violence [9] provides 1,400 clips from three fixed cameras inside a moving public bus under CC-BY-4.0 (DOI 10.5281/zenodo.7044203), explicitly designed as an external domain-shift benchmark. AIRTLab [10] records 350 Full HD dual-camera clips in a controlled university lab whose non-violent class explicitly targets false-positive lookalikes such as hugs, high-fives, jumping, and hand gestures.

Short isolated clips also omit the temporal reality of surveillance feeds, where rare violent events occur inside long uneventful streams. UCF-Crime [11] provides untrimmed real-world CCTV recordings with independent human frame-level temporal boundaries for test events, and UCA [12] extends UCF-Crime with independent event timestamps on disjoint splits. Evaluating on both provenance-verified external clip sets and unedited continuous CCTV directly tests whether high within-dataset clip accuracy transfers to surveillance monitoring.

### C. Optimization and interpretation of confidence

We train linear heads with AdamW [7] on frozen backbone embeddings, keeping trainable parameters small (1,026 weights). As Guo et al. showed [8], neural softmax outputs are generally uncalibrated. A softmax score of 0.90 reflects relative ranking confidence rather than a 90% probability of real violence, especially under domain shift and low real-world event prevalence.

### D. Position of the present study

Rather than relying solely on one visually grouped RLVS split or short edited replays, we evaluate fixed RLVS-trained checkpoints across three zero-overlap external clip benchmarks and unedited continuous UCF-Crime CCTV streams. We separate RLVS clip threshold selection from continuous camera alarm hysteresis calibration using disjoint validation streams, and we stress-test the actual multi-camera worker past hardware saturation to observe real window replacement behavior.

<!-- page -->
## III Dataset audit, external sets, and label analysis

### A. RLVS acquisition and perceptual grouping

The RLVS subset of the local Kaggle archive [6] contained 2,000 clips (1,000 per class folder). Three clips exceeded the 120-second decoding limit and one had a corrupted frame, leaving 1,996 decodable files. All filenames are generic sequential identifiers (file_NNNNNN) without camera, actor, or recording-session metadata.

To reduce near-duplicate leakage within RLVS, we combined exact SHA-256 hashes with 12-frame perceptual DCT fingerprints (32 by 32 grayscale, low-frequency 8 by 8 block minus DC, 63-bit median threshold). Two clips link if at least four distinct frames match within an 8-bit Hamming distance along a consistent temporal offset. Connected components identified 285 byte-distinct visual links and 96 multi-file components before exclusions (1 conflicting-label exact-hash pair, 12 exact-duplicate groups, and 83 visually linked multi-clip groups).

We excluded the 2 conflicting-label files and 13 extra exact duplicates, retaining 1,981 clips in 1,803 visual groups. A stratified five-fold group split (seed 42) assigned one fold to validation, one to test, and three to training with zero group-ID or SHA-256 overlap.

[[TABLE_SPLITS]]

Because RLVS lacks source metadata, this grouping is a visual proxy rather than verified source independence; substantial crops or repeated scenes can still cross RLVS folds.

### B. Provenance-verified external test benchmarks

To establish genuine source-independent evaluation without retraining or test tuning, we assembled three external benchmarks totaling 2,150 clips and verified every clip against all 1,996 usable RLVS files using the exact same SHA-256 and 12-frame perceptual-hash screen:

1) **RWF-2000 official validation [5]**: 400 real CCTV clips (200 Fight, 200 NonFight, 5.0 s each) from the local archive; 0 exact and 0 perceptual overlaps with RLVS.

2) **Bus Violence [9]**: 1,400 balanced clips (700 Violence, 700 NoViolence, 16 to 48 frames at 25 fps) recorded on three fixed transit cameras (two 960 by 540 corner views and one 1280 by 960 fisheye) under CC-BY-4.0 (DOI 10.5281/zenodo.7044203); 0 exact and 0 perceptual overlaps with RLVS.

3) **AIRTLab [10]**: 350 Full HD dual-camera clips (230 violent, 120 non-violent; cam1 and cam2) recorded at Università Politecnica delle Marche with per-clip action-class CSVs; 0 exact and 0 perceptual overlaps with RLVS.

### C. RLVS label screen and non-violent lookalikes

On RLVS, folder names supply the only labels. We drew 24 test clips (12 per class from distinct groups, seed 932) without consulting model scores. An AI-assisted five-frame contact-sheet screen recorded 0 clear folder-label disagreements and 9 ambiguous clips (2 non-violence, 7 violence) caused by occlusion, blur, dark interiors, fast camera motion, or distant contact. Original folder labels were retained unmodified in all RLVS metrics.

Crucially, this five-frame AI screen is exploratory and does not substitute for independent human annotation; **multi-annotator human relabeling and inter-annotator agreement (e.g., Cohen's kappa) on RLVS remain open** because independent human raters were not available in this study. To provide concrete empirical evidence on non-violent false-positive drivers instead of speculation, Section IX quantifies false-positive rates across AIRTLab's independently human-annotated non-violent close-contact and celebratory action categories.

<!-- page -->
## IV Proposed system and video representation

### A. Components and data flow

The ML package decodes videos, audits datasets, trains models, and evaluates checkpoints. At startup, the FastAPI service loads the offline checkpoint, validates uploads, and manages continuous camera sessions. Next.js provides views for clip uploads, single-camera diagnostics, and multi-camera monitoring.

[[FIG_ARCH]]

An upload returns one prediction for the full clip, whereas continuous monitoring scores overlapping four-second windows cut from timestamped camera samples. Both paths share identical spatial preprocessing and class ordering.

### B. Bounded decoding and temporal sampling

Upload inference decodes once to check frame count, timestamps, corruption, and resolution, then decodes a second time retaining only selected frames. Limits are 120 seconds, 12,000 decoded frames, 3840 by 2160 maximum area, and 100 MiB HTTP upload size. For N decoded frames and T = 16 requested samples, selected indices follow i(k) = floor(k(N-1)/(T-1)) for k = 0 to 15. On long uploaded clips, spacing widens so brief events can fall between samples.

### C. Spatial preprocessing and checkpoint contract

Frames use the resize, center crop, scaling, and normalization tied to the pretrained weights [4], yielding float32 tensors of shape [B,3,16,112,112]. Center cropping can trim peripheral interactions, and the RGB baseline operates without person detection, tracking, or pose estimation.

Each checkpoint records architecture, class order, frame count, preprocessing ID, base weights, selected epoch, validation metrics, training settings, package versions, and manifest hash. Startup verifies contract compatibility; an incompatible checkpoint aborts startup, while an absent checkpoint leaves prediction explicitly unavailable.

<!-- page -->
## V Model adaptation and training protocol

### A. MC3-18 and the ResNet-18 comparison backbone

The primary model uses torchvision MC3-18 with Kinetics-400 weights and a two-output linear classifier (1,026 trainable parameters) on its 512-dimensional pooled representation. For comparison, we trained an identical linear head on frame-averaged 512-dimensional features from frozen ImageNet-pretrained ResNet-18 [1] using the same 16 frame indices. Because the two pipelines differ in both temporal convolutions and pretraining corpus, the comparison evaluates the full 3D spatiotemporal pipeline against a 2D static-appearance baseline.

### B. Optimization and multi-seed model selection

Frozen backbone features were extracted once in evaluation mode (fixing batch-normalization statistics) on Apple Silicon MPS. Linear heads trained on CPU with AdamW, batch size 32, learning rate 0.001, and two-class cross-entropy for 20 epochs across three random seeds (41, 42, and 43) on the identical grouped RLVS split. Each run saved the checkpoint with minimum RLVS validation loss. Neither RLVS test clips nor any external or continuous test videos influenced epoch selection or head weights.

[[TABLE_TRAINING]]

### C. Separate clip and continuous alarm validation protocols

We keep two distinct threshold protocols strictly separated from test evaluation:

1) **Clip-level threshold selection (RLVS validation only)**: For each seed and backbone, we swept thresholds from 0.05 to 0.95 (step 0.05) on the 396 RLVS validation clips to maximize violence F1 (ties broken by higher precision, then higher threshold), locking the cutoff before scoring RLVS test and external clip benchmarks.

2) **Continuous event-alarm calibration (UCF-Crime continuous validation only)**: Single-clip F1 tuning does not calibrate operational hysteresis alerts. We therefore evaluated candidate hysteresis rules on a strictly disjoint set of 24 unedited UCF-Crime validation streams (Section VII-A), locked the winning alarm rule, and evaluated both the fixed default rule and the validation-calibrated rule once on the untouched official UCF-Crime continuous test corpus.

<!-- page -->
## VI Continuous monitoring methodology

### A. Capture and recent-window construction

Each camera runs a capture thread, a bounded deque of timestamped preprocessed frames, a JPEG preview buffer, and an inference consumer. By default, capture samples up to 4 frames per second and schedules a 4.0-second sliding window every 1.0 second. Local test files are paced by presentation timestamps (or nominal frame rate when absent) so decoding runs at real-time playback speed; stream failures transition the camera OFFLINE and trigger exponential reconnect backoff.

Once the four-second startup buffer fills, 16 target timestamps are spaced uniformly across [t - 4.0, t] and matched to the nearest buffered sample. Retaining one sample prior to the window boundary avoids empty spans at low source frame rates.

### B. Latest-window scheduling and serialized inference

Each camera stores at most one pending clip. If a newer scheduled window arrives before the consumer acquires the shared inference lock, the newer clip replaces the pending slot and increments `dropped_windows`. After acquiring the shared lock (serialized across cameras, uploads, and diagnostics), the consumer discards any pending window older than twice the stride (2.0 s). This keeps memory strictly bounded regardless of camera count, trading window coverage for bounded latency once aggregate demand exceeds hardware throughput.

### C. Hysteresis and event lifetime

By default, three consecutive windows scoring at or above 0.65 transition a camera from NORMAL to VIOLENCE, and four consecutive windows at or below 0.40 clear the alert. Mid-range scores reset both counters while preserving an active state.

[[FIG_STATES]]

An active event tracks start time, latest update, and peak score in memory. Disconnections clear the buffer, pending clip, counters, and event while incrementing a generation counter so in-flight inferences from prior connections cannot overwrite new state.

<!-- page -->
## VII Unedited continuous monitoring benchmark

### A. Unedited UCF-Crime corpora and disjoint split

To evaluate continuous monitoring on real unedited surveillance footage with independent human event boundaries rather than edited clips, we used UCF-Crime [11] (CC0-1.0) and UCA [12], verifying zero exact or perceptual overlap with RLVS across all 58 continuous videos:

1) **Official continuous test set (34 unedited CCTV videos, 38.0 min)**: All 10 official physical-violence test recordings from Sultani et al.'s `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` [11] (5 Fighting, 3 Assault, 2 Abuse; 11 human-annotated violent intervals) plus a deterministic seeded sample (seed 42, 1 to 15 MiB) of 24 official `Testing_Normal_Videos`. Excluding startup and all annotated events plus their grace periods, this provides **1,673.3 s (0.465 camera-hours)** of unedited negative exposure—12 times longer than the earlier 140-second edited RLVS replay check.

2) **Disjoint continuous validation set (24 unedited CCTV videos, 25.9 min)**: 10 non-test Fighting and Assault recordings (14 violent intervals annotated independently in UCA [12]) plus 14 seeded `Training-Normal-Videos` (seed 101), providing **1,101.2 s (0.306 camera-hours)** of disjoint negative exposure for validation-only alarm calibration with zero video overlap against the test set.

### B. Event detection and validation-only alarm calibration

We scored every unedited video with the fixed RLVS-trained `mc3-head-42` using exact 4.0-second windows, 4 fps sampling, and 1.0-second stride. The first alert in an annotated event or its 4-second grace period counts as a match. Later alerts in that interval count as duplicates and are excluded from event precision. Alerts outside these intervals count as false alerts. Delay is measured from annotated onset. Sweeping `positive_threshold` (0.50 to 0.95) and `positive_windows` (2 to 5) strictly on the 24 continuous validation streams maximized validation event F1 at **0.85 for 5 consecutive windows** (`negative_threshold` = 0.40, `negative_windows` = 4), reducing validation false alerts from 5 (16.35/cam-h) to 2 (6.54/cam-h) while holding recall at 7/14 (50.0%).

[[TABLE_CONTINUOUS]]

On the untouched 34-video official UCF-Crime test set (window ROC-AUC **0.7767**), the uncalibrated default policy (0.65 / 3w) matched **5/11 events (45.5%)**—specifically **4/5 Fighting (80.0%)**, 1/4 Assault, and 0/2 Abuse—with median delay **11.5 s** (mean 12.8 s), but triggered **15 false alerts (32.27 per camera-hour; 10 across the 24 normal feeds)**. Locking the validation-calibrated policy (0.85 / 5w) cut test false alerts **fivefold to 3 (6.45 per camera-hour, only 1 across 24 normal feeds)** and doubled event precision (25.0% to 50.0%, F1 35.3%), at the explicit cost of lower recall (**3/11 = 27.3%**) and longer delay (median 31.5 s). By contrast, our earlier 4-stream edited RLVS replay (46 s per stream, 140 s negative exposure) had shown 6/8 matches and 0.0 false alerts/cam-h, showing that the edited replay result did not carry over to this continuous benchmark.

<!-- page -->
### C. Live multi-camera concurrency and lock saturation

We ran the unmodified `SurveillanceManager` threads, capture buffers, serialized `infer_lock`, and default hysteresis (0.65 / 3w) in real time on Apple Silicon MPS across **1, 2, 4, 8, 12, and 16 simultaneous distinct unedited UCF-Crime test streams** (each natively >= 46.3 s, benchmarked for 42.0 s with zero looping and zero offline failures).

[[TABLE_MONITOR]]

[[FIG_SCALING]]

Throughput scaled linearly from 0.86 to 6.83 windows/s across 1 to 8 streams with **0.0% replaced windows** and stable p50/p95 inference latency (84.4/101.7 ms at 8 streams). Lock saturation began at **12 streams** (9.69 windows/s, 11/420 replaced = **2.6%**) and plateaued at **16 streams** (**10.19 windows/s**, 138/576 replaced = **24.0%**, p50/p95 latency flat at 83.1/95.5 ms). Under simultaneous activity at 8, 12, and 16 streams, the live worker matched 3/4, 3/5, and 3/5 active early events (mean delays 8.00 to 8.26 s) with 1, 3, and 3 false alerts. This empirically verifies bounded-memory backpressure under saturation while defining the single-host hardware ceiling (~10 windows/s on MPS).

<!-- page -->
## VIII Held-out and external classification results

### A. RLVS grouped test results across seeds

On the 397 visually grouped RLVS test clips at a fixed 0.50 threshold, MC3-18 averaged **96.73 +/- 0.00% accuracy** and **96.75 +/- 0.01% F1** across seeds 41, 42, and 43, compared with **95.97 +/- 0.44% accuracy** and **95.95 +/- 0.44% F1** for frame-averaged ResNet-18. These multi-seed 20-epoch runs replace the original single-epoch checkpoint.

[[TABLE_RESULTS]]

The predesignated seed-42 MC3-18 run produced 193 true positives, 191 true negatives, 8 false positives, and 5 false negatives on RLVS test (seed 41 was identical; seed 43 reached 384/397 with 194 TP, 190 TN, 9 FP, and 4 FN).

[[FIG_CONFUSION]]

### B. Training and validation loss curves

Mean training cross-entropy fell smoothly across 20 epochs for both backbones (MC3-18: 0.5267 to 0.0717; ResNet-18: 0.5302 to 0.1019), while mean validation loss dropped from 0.3824 to 0.1298 (epoch 19) for MC3-18 and from 0.3802 to 0.1333 for ResNet-18. Lowest validation loss selected epoch 19 for all three MC3-18 runs and epochs 19, 20, and 20 for ResNet-18.

[[FIG_METRICS]]

### C. Zero-shot generalization to external benchmarks

Evaluating the exact same six fixed RLVS checkpoints on the three zero-overlap external benchmarks reveals a large domain-shift gap hidden by within-RLVS evaluation.

[[TABLE_EXTERNAL]]

At threshold 0.50, MC3-18 accuracy fell from 96.73% on RLVS to **71.92 +/- 0.88%** on RWF-2000 val (F1 71.28%, **AUC 0.808**), **56.02 +/- 1.30%** on Bus Violence (precision 81.70%, recall 15.57% due to conservative scores on short transit fisheye clips, **AUC 0.710**; official `test.txt` AUC 0.734), and **56.29 +/- 0.57%** on AIRTLab (F1 67.40%, **AUC 0.551**). Applying the RLVS validation-selected clip thresholds changed accuracy only modestly (RWF-2000: 72.67%; Bus Violence: 58.71%; AIRTLab: 58.48%).

Crucially, **MC3-18 outperformed frame-averaged ResNet-18 in ROC-AUC on every external dataset** (+0.109 on RWF-2000, +0.148 on Bus Violence, and +0.109 on AIRTLab where ResNet-18 fell to 0.442, below chance). Whereas static 2D appearance cues nearly matched MC3-18 inside RLVS (95.97% vs. 96.73%), spatiotemporal convolutions transferred substantially better across real camera domains—yet still proved insufficient for uncalibrated zero-shot deployment.

<!-- page -->
## IX Threshold selection, lookalikes, and responsible deployment

### A. Clip-level validation threshold vs. alarm hysteresis

On RLVS, validation-only F1 maximization selected threshold 0.35 for MC3-18 seeds 41 and 42 and 0.50 for seed 43 (mean test F1 96.68 +/- 0.14% vs. 96.75 +/- 0.01% at 0.50). For seed 42, lowering the clip cutoff from 0.50 to 0.35 recovered one positive clip (194 vs. 193) at the cost of two additional false positives (10 vs. 8).

[[TABLE_THRESHOLDS]]

As demonstrated in Section VII-B, clip-level cutoffs (0.35 to 0.50) are unsuitable for continuous streams. Operational alarm thresholds must be calibrated on separate continuous validation recordings; raising hysteresis from 0.65 / 3w to the continuous-validation-selected 0.85 / 5w cut UCF-Crime test false alerts from 32.27 to 6.45 per camera-hour while exposing a stark recall-versus-alert-burden tradeoff (45.5% to 27.3% recall).

### B. Empirical false-positive audit on non-violent lookalikes

To quantify how benign close-contact or vigorous non-violent movements trigger false positives, we broke down `mc3-head-42` predictions at 0.50 across AIRTLab's 120 independently annotated non-violent clips (dual cameras). Overall non-violent false-positive rate reached **70.8%** (specificity 29.2%).

[[TABLE_LOOKALIKES]]

Playful contact (friendly punch: 100.0%, hug: 78.1%, highfive: 75.0%), exuberant motion (jump: 88.9%), and emphatic upper-body motion (greet: 69.7%, handgestures: 63.3%) routinely scored above 0.50. Together with organized sports and crowd scenes observed in the RLVS 24-clip screen, these measurements confirm that an RGB action classifier without contextual reasoning confuses rapid benign human interaction with physical aggression.

### C. Low-prevalence risk and human-in-the-loop workflow

Even transferring the optimistic RLVS seed-42 sensitivity (97.47%) and specificity (95.98%) to a 1% real-world violence prevalence yields a positive predictive value of only 19.67% (16.45% at cutoff 0.35). Under real continuous CCTV shift conditions (Section VII-B), 6.45 to 32.27 false alerts per camera-hour across a multi-camera facility would overwhelm operators without disciplined triage.

Accordingly, **model scores and camera alerts must never be treated as confirmed violent incidents**, and **this prototype is not suitable for autonomous surveillance, security dispatch, or enforcement decisions**. In a human-in-the-loop workflow, the dashboard routes flagged intervals to trained operators who review pre- and post-alert context, check camera angle and visibility, and log a mandatory disposition (confirmed, false alert, or insufficient evidence) plus scene category (sports, play, crowding, occlusion). Any threshold revision must be locked on separate site validation footage rather than adjusted ad hoc on test clips.

<!-- page -->
## X Limitations and follow-up validation

### A. Label reliability and RLVS provenance

Although external evaluation on Bus Violence, AIRTLab, and RWF-2000 resolved the lack of source-independent test evidence, **independent human relabeling and inter-annotator agreement (Cohen's kappa) on RLVS remain open**. Our 24-clip five-frame RLVS check was AI-assisted and exploratory, not a human reliability study. A complete RLVS label audit requires at least two independent human annotators reviewing full-motion clips on a larger stratified sample under a written rubric, reporting raw agreement, Cohen's kappa, and adjudication outcomes before retraining.

### B. Domain shift, subtle violence, and monitoring scope

Zero-shot evaluation showed sharp accuracy drops on transit fisheye footage (Bus Violence), hard non-violent lookalikes (AIRTLab), and unedited CCTV (UCF-Crime), where subtle abuse (0/2) and distant assaults (1/4) were frequently missed even when overt fights reached 4/5 recall under default settings. Improving operational utility requires fine-tuning beyond a frozen linear head, probability calibration on site-representative data, and hard-negative training on sports, hugs, play, and crowded scenes.

Our continuous UCF-Crime tests used local file pacing rather than live RTSP/ONVIF transport over congested networks, and total negative test exposure (0.465 camera-hours across 34 streams) still falls short of multi-day field shifts. Live network jitter, packet loss, lighting cycles, and multi-worker durable event persistence (replacing the current single-process in-memory registry) remain required before field deployment, alongside privacy, consent, retention, and audit-logging controls.

<!-- page -->
## XI Reproducibility and conclusion

### A. Experiment records and software provenance

All experiments run under PyTorch 2.14.0 and torchvision 0.29.0 on Apple Silicon MPS (feature extraction, continuous scoring, and live multi-camera trials) and CPU (head training). `scripts/review_features.py` extracts RLVS features and hashes; `scripts/review_experiments.py` builds the grouped RLVS split (manifest SHA-256 `9df60ccff3cdba9550586f1cbf7e1781ae20714ed9c83dd10fec7f795a8195f3`) and trains the six 20-epoch heads. `scripts/eval_external_clips.py` audits RLVS overlap and evaluates fixed heads on RWF-2000, Bus Violence, and AIRTLab. `scripts/eval_ucf_continuous.py` scores unedited UCF-Crime validation and test streams, performs validation-only alarm calibration, and executes live 1 to 16 stream `SurveillanceManager` trials. `scripts/build_review_results.py` compiles `output/pdf/review-results.json` for report generation.

[[TABLE_PROVENANCE]]

### B. Conclusion

Across three 20-epoch seeds on 397 visually grouped RLVS test clips, frozen MC3-18 averaged 96.73% accuracy and 96.75% F1 (ResNet-18: 95.97% accuracy). However, provenance-verified zero-shot evaluation on 2,150 external clips with zero RLVS overlap (RWF-2000 val, Bus Violence, and AIRTLab) exposed substantial cross-domain drops alongside a consistent ROC-AUC advantage for spatiotemporal MC3-18 (+0.109 to +0.148 over ResNet-18) and a 70.8% false-positive rate on close-contact non-violent lookalikes.

On 34 unedited official UCF-Crime CCTV recordings (38.0 min, 0.465 camera-hours negative exposure), default hysteresis matched 5/11 events (4/5 Fighting) at 32.27 false alerts/cam-h, whereas separate continuous validation calibration (0.85 / 5w) cut false alerts fivefold to 6.45/cam-h with reduced recall (3/11). Live `SurveillanceManager` trials across 1 to 16 unedited streams demonstrated zero window drops through 8 streams and bounded-memory backpressure at lock saturation (10.19 windows/s, 24.0% replaced at 16 streams). Independent human inter-annotator RLVS verification, domain fine-tuning, and multi-hour live RTSP field validation remain necessary before operational deployment.

## References

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR, pp. 770-778, 2016. https://arxiv.org/abs/1512.03385.

[2] D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, "A Closer Look at Spatiotemporal Convolutions for Action Recognition," CVPR, pp. 6450-6459, 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html.

[3] W. Kay et al., "The Kinetics Human Action Video Dataset," arXiv:1705.06950, 2017. https://arxiv.org/abs/1705.06950.

[4] PyTorch contributors, "MC3-18 model and pretrained weights," Torchvision documentation. https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.mc3_18.html.

[5] M. Cheng, K. Cai, and M. Li, "RWF-2000: An Open Large Scale Video Database for Violence Detection," ICPR, 2021. https://arxiv.org/abs/1911.05913.

[6] magicearth25, "Video violence detection dataset (RLVS and RWF-2000 archive)," Kaggle. https://www.kaggle.com/datasets/magicearth25/video-violence-detection-dataset.

[7] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR, 2019. https://arxiv.org/abs/1711.05101.

[8] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," ICML, PMLR 70, pp. 1321-1330, 2017. https://arxiv.org/abs/1706.04599.

[9] L. Ciampi et al., "Bus Violence: An Open Benchmark for Video Violence Detection on Public Transport," Sensors, vol. 22, no. 21, p. 8345, 2022; dataset Zenodo DOI: 10.5281/zenodo.7044203 (CC-BY-4.0). https://doi.org/10.5281/zenodo.7044203.

[10] M. Bianculli et al., "A dataset for automatic violence detection in videos," Data in Brief, vol. 33, p. 106587, 2020. https://doi.org/10.1016/j.dib.2020.106587.

[11] W. Sultani, C. Chen, and M. Shah, "Real-World Anomaly Detection in Surveillance Videos," CVPR, pp. 6479-6488, 2018. https://arxiv.org/abs/1801.04264.

[12] T. Yuan et al., "Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges," CVPR, 2024. https://github.com/xuange923/Surveillance-Video-Understanding.
