# Violence Detection in Surveillance Videos Using Transfer Learning and Continuous Multi-Camera Monitoring

Ashwin S | Registration No. 12308018 | theashgen06@gmail.com

Prudhviraj K B | Registration No. 12307060 | prudhvirajkb3@gmail.com

Pranav R | Registration No. 12306134 | pranavsandle@gmail.com
B.Tech in Artificial Intelligence and Machine Learning
Lovely Professional University, Punjab, India

## Abstract

Automated video analysis can help operators review large volumes of surveillance footage, but a video classifier alone does not constitute a dependable monitoring system. This paper presents an implemented research prototype combining a transfer-learned mixed-convolution network, an audited dataset pipeline, uploaded-video inference, and continuous multi-camera monitoring. The classifier adapts Kinetics-400-pretrained MC3-18 features to two classes while freezing the backbone. A public RLVS collection containing 2,000 videos is inspected for decoding failures, excessive duration, exact duplicates, and label conflicts. Reviewed exclusions and deduplication yield 1,346 training, 397 validation, and 238 test clips. The available checkpoint records one training epoch and achieves 92.86% accuracy, 88.15% precision, 99.17% recall, and 93.33% F1 on the held-out test clips. Continuous monitoring uses bounded timestamped buffers, overlapping temporal windows, serialized inference, and per-camera hysteresis, with a Next.js interface backed by FastAPI. These engineering capabilities are distinguished from the measured video-level experiment: neither event-level surveillance performance nor camera throughput has been established. Related-source leakage also remains unverified because the supplied filenames lack trusted source identities. The principal contribution is an inspectable end-to-end baseline with traceable results and explicit operational limits, rather than a novel architecture or a deployment-ready safety system.

**Keywords:** violence detection; video classification; MC3-18; transfer learning; RLVS; temporal windows; surveillance monitoring.

## I Introduction

Surveillance applications produce more footage than a person can consistently examine in real time. A screening model can prioritize clips for review, provided its output is treated as uncertain evidence about visual content. Violence recognition is particularly difficult because similar motion may occur in sports, play, crowd activity, or physical assistance. Camera viewpoint, occlusion, scene context, and video compression can change the appearance of the same action. A useful research system must therefore make its evaluation scope and failure behavior as visible as its predictions.

This project investigates a practical baseline: can a pretrained video representation support binary violence classification while remaining accessible through a working local application? The implemented system accepts uploaded videos, supports a browser webcam or network camera in a diagnostic mode, and provides a separate continuous multi-camera dashboard. All application predictions depend on a compatible trained checkpoint. There is no random-weight fallback and no training endpoint exposed through the interface.

The study addresses three questions. First, what video-level classification performance is supported by the available saved experiment? Second, how can training and application inference share a well-defined preprocessing contract? Third, what additional mechanisms are needed to turn intermittent model predictions into bounded, inspectable camera states? These questions connect model evaluation with the engineering constraints of streaming inputs without equating the two.

The contributions are an audited manifest-based data workflow; a frozen-backbone MC3-18 baseline with checkpoint provenance; a shared decoding and preprocessing path; an API and three application views; and a monitoring design using recent windows and hysteresis. The architecture combines established methods. The reported classification result is local and provisional, and no state-of-the-art comparison is claimed.

The practical motivation is not limited to model accuracy. A browser may send a valid container with a codec it cannot preview, a camera may disconnect during a prediction, or concurrent requests may compete for the same accelerator. These cases can produce misleading displays unless the service explicitly tracks availability, source state, and the meaning of a score. The project treats these as first-class requirements while keeping the statistical evaluation restricted to labeled clips.

The remainder of the paper reviews related methods, documents the dataset audit, explains the representation and training protocol, and describes the monitoring and application layers. It then reports the saved experiment, interprets uncertainty and error asymmetry, and identifies the evidence required for stronger surveillance claims.

<!-- page -->
## II Related work and research positioning

### A. Residual and spatiotemporal representations

Residual learning introduces shortcut connections that support the optimization of deep visual networks [1]. For video, spatial appearance and temporal evolution must both be represented. Tran et al. study full three-dimensional, mixed, and factorized spatiotemporal convolutions within residual architectures [2]. Their mixed-convolution formulation motivates the MC3-18 backbone selected here. The present work uses an existing implementation rather than designing new residual blocks or a new temporal operator.

The model begins with Kinetics-400 weights [3,4]. Such pretraining provides a general action representation, but does not by itself define a violence detector. The original classification head is replaced and the new two-class head is fitted to the local dataset. The distinction matters: the provenance of pretrained features and the evidence for the target classification task are separate components of the experiment.

### B. Violence datasets and protocol differences

Cheng et al. introduce RWF-2000 as a surveillance-oriented violence dataset and describe a Flow Gated Network [5]. Their work illustrates why camera-specific material is relevant to this application. However, this paper evaluates the RLVS collection extracted from a public combined archive [6]; it does not evaluate RWF-2000. Published scores obtained on another dataset, split, or sampling protocol are not used as numerical competitors to the local result.

A dataset name alone is insufficient to establish comparability. A mirror may contain several collections, retain nonstandard folder assignments, or include byte-identical copies under different names. Here, the archive contents are inspected before split creation. The report distinguishes the original discovered count, decodable records, reviewed exclusions, and final manifest membership. This makes the evaluated population narrower and more precise than an unqualified statement that the system was tested on all downloaded footage.

### C. Optimization and interpretation of confidence

The classifier head is optimized using AdamW, following the decoupled weight-decay formulation of Loshchilov and Hutter [7]. The code uses the framework optimizer directly; it does not implement a new optimization method. Freezing the backbone limits the number of trainable parameters and makes the experiment a feature-transfer baseline rather than full video-network fine-tuning.

Guo et al. distinguish classification confidence from calibrated correctness likelihood [8]. This distinction is relevant when a model score appears in a monitoring interface. The project displays the softmax score but performs no probability calibration study. Consequently, a score of 0.9 must not be interpreted as a verified 90% chance that a real-world harmful event is occurring.

### D. Position of the present study

The study is best understood as a systems-oriented baseline with a small empirical evaluation. It contributes reproducible data handling and an application that uses the trained model, while retaining the limits of a single checkpoint and a single local split. There are no repeated-seed results, trained architecture comparisons, optical-flow experiments, or event-localization annotations in the available evidence.

The engineering design emphasizes observable failure: missing checkpoints disable prediction, invalid uploads produce explicit errors, camera disconnects become states, and competing inference requests are controlled. These properties support development and review. They do not establish that an alert policy is accurate, fair, or acceptable in an operational setting. The following sections therefore report video classification and monitoring behavior as related but distinct aspects of the prototype.

A useful comparison must hold more than the class names constant. Spatial resolution, number of frames, choice of best epoch, treatment of duplicates, and independence of the test population can all change a reported score. For this reason, the literature is used to explain design choices rather than construct a ranking across incompatible results. A controlled local comparison remains a future experiment.

The same principle applies to the phrase real time. An interactive interface and continuous capture are observable software capabilities. Real-time performance additionally requires a specified input rate, latency budget, hardware environment, and overload policy. This paper documents the overload policy but reports no measured real-time performance guarantee.

<!-- page -->
## III Dataset audit and experimental partitions

### A. Acquisition and label definition

The project uses the RLVS portion of the public Video violence detection dataset archive [6]. Inspection discovers 2,000 videos, evenly divided between the two folder-defined classes. There are 1,951 MP4 files and 49 AVI files. Labels are derived from recognized class directories such as Fight and NonFight rather than from filename substrings. The ordered labels are 0 for non-violence and 1 for violence, and the same ordering is stored in each checkpoint.

The inspection pass decodes each supported video sequentially and records relative path, dimensions, frame count, approximate duration, frame rate, supplied split, class, and SHA-256. Four videos fail the baseline limits: three exceed 120 seconds and one contains a corrupted frame. The remaining 1,996 records contain 998 clips per class. Their recorded mean duration is approximately 4.91 seconds, with a minimum of 1.00 second and a maximum of 11.32 seconds. These statistics describe the usable records before final deduplication.

### B. Duplicate and conflict handling

The audit identifies 14 extra exact-duplicate copies among usable records. At least one duplicate pair has contradictory class labels, and three pairs cross the supplied training/validation boundary. The reviewed exclusion CSV removes both members of the contradictory pair and the training member of each of the three cross-boundary pairs. Validation membership is therefore preserved for those reviewed duplicate pairs. Same-split duplicates retain one representative.

After reviewed exclusions and deduplication, 1,981 distinct retained clips remain. A SHA-256 match detects byte-identical files; it does not detect re-encoding, temporal overlap, a common original video, or repeated actors and locations. The use of hashes is a reproducibility safeguard with a specific scope, not proof of source independence.

### C. Partition construction

The supplied validation assignment is retained for the surviving clips. As the inspected RLVS collection has no supplied test partition, a seeded group shuffle reserves approximately 15% of the remaining training pool for a local test set. Seed 42 fixes the assignment. The final manifest contains path, label, split, SHA-256, and group. Every split must contain both classes, and path, hash, and group assignments are checked for split conflicts.

[[TABLE_SPLITS]]

The main training routine verifies the hashes of training and validation files. Evaluation requires the exact manifest fingerprint recorded during training and verifies the test file hashes. This reduces accidental data changes after inspection. The test split is not loaded by the optimization loop and does not select the saved model.

### D. Remaining leakage uncertainty

Trusted source identifiers are unavailable in this mirror. The manifest therefore uses exact-file hashes as groups after an explicit acknowledgement that related-source leakage is unknown. The experiment cannot be described as camera-disjoint, actor-disjoint, or original-video-disjoint. A new review supplying complete source groups would strengthen the protocol and could change the resulting scores. The present partitions are a documented local research split, not an official RLVS benchmark protocol.

### E. Audit accounting and reproducibility

The final class totals are 991 non-violence and 990 violence clips. Their small difference follows the reviewed conflict handling and deduplication rather than deliberate class rebalancing. The final training set remains nearly balanced, so the reported run uses ordinary unweighted cross-entropy. The exclusion list contains explicit reasons, allowing a reviewer to distinguish a decoding failure from a label conflict or a cross-partition duplicate.

Inspection and split generation are separate operations. This avoids silently rewriting the source archive or changing partitions whenever a model is trained. A corrected source-group mapping would be a new experimental protocol and should receive a new manifest and checkpoint fingerprint; it should not overwrite the identity of the existing result.

<!-- page -->
## IV Proposed system and video representation

### A. Components and data flow

The implementation separates offline model development from online prediction. The ML package owns video decoding, dataset inspection, model construction, training, checkpoint loading, and evaluation. The FastAPI service owns upload validation and camera sessions. The Next.js application provides an upload workspace, a single-camera diagnostic view, and a multi-camera monitor. Training produces the artifact consumed by the service; browsing the application never modifies model parameters.

[[FIG_ARCH]]

The uploaded-video path returns a single prediction for the submitted clip. The continuous path repeatedly forms short windows from timestamped camera samples. Both use the same spatial transform and model class ordering, but their temporal sampling policies differ. This separation is intentional and is central to interpreting the results.

### B. Bounded decoding and temporal sampling

For uploaded videos, a first decode counts frames and checks corruption, duration, frame count, and resolution. A second decode retains only the uniformly selected frames. Interframe compression still requires decoding intervening frames, so the two-pass method limits retained memory without making decoding constant-time. Video limits are 120 seconds, 12,000 decoded frames, and a maximum frame area of 3840 by 2160 pixels. HTTP uploads also have a 100 MiB file limit.

For N decoded frames and T requested samples, the selected index is i(k) = floor(k(N-1)/(T-1)), for k from 0 to T-1. The recorded experiment uses T = 16. The first and last frames are included, and short videos repeat indices. Uniform coverage provides a simple whole-clip representation, although a brief incident can occur between selected frames and be missed.

### C. Spatial preprocessing and tensor layout

The project applies the transform associated with the pretrained weight specification [4]: resize, center crop, rescaling, and normalization. The same implementation is shared by the offline dataset and uploaded-video inference. The resulting clip has shape [3,16,112,112] in float32 format; batching produces [B,3,16,112,112]. Continuous capture applies the spatial transform to incoming sampled frames before constructing its temporal tensor.

Center cropping is deterministic and easy to reproduce, but it can remove relevant action near a frame boundary. Fixed-size resizing can also reduce detail for small or distant people. No person detector, tracker, pose estimator, region proposal, or camera-specific crop is implemented in the evaluated baseline.

### D. Checkpoint contract

A checkpoint records architecture, ordered classes, frame count, preprocessing identifier, weight provenance, selected epoch, validation metrics, training configuration, software versions, and manifest hash. Inference checks compatibility and constructs the model without downloading pretrained weights. An incompatible existing checkpoint fails startup; a missing file leaves the process available while reporting that the model is not ready. This makes model availability an explicit application state rather than an implicit assumption.

Two types of temporal coverage should be distinguished. Whole-video sampling distributes a fixed number of observations across the submitted duration, so its effective temporal spacing changes with clip length. A four-second monitoring window instead targets a fixed temporal extent. Reusing spatial preprocessing prevents a normalization mismatch, but does not eliminate this difference in motion scale or the possibility that relevant actions fall between samples.

<!-- page -->
## V Model adaptation and training protocol

### A. Binary classifier

The model is torchvision MC3-18 with a two-output linear classifier replacing the original head. Let f(X) denote the frozen representation of an input video tensor. The adapted logits are z = Wf(X) + b. For the model's 512-dimensional pooled representation, the trainable head contains 512 times 2 weights and 2 biases, or 1,026 parameters. The rest of the network supplies fixed pretrained features during the recorded run.

The predicted class is the index of the larger softmax value. For the violence class, s = exp(z1)/(exp(z0)+exp(z1)). The uploaded-video API returns the winning class and its score, so its confidence field is not always the violence score. The monitoring path explicitly reads the class-1 score before applying its thresholds. Keeping these semantics separate prevents confusion when comparing an upload result with a camera tile.

### B. Training objective and optimization

Training minimizes mean two-class cross-entropy. Only parameters whose names begin with the classifier prefix are passed to AdamW. The saved run uses batch size 4, learning rate 0.001, seed 42, zero data-loader workers, and four CPU threads. Weight decay is not explicitly overridden in the code; the optimizer therefore uses its library default. The experiment does not include class weighting, label smoothing, a learning-rate scheduler, or gradient accumulation.

The code sets Python, NumPy, and PyTorch seeds, disables cuDNN benchmarking, and requests deterministic algorithms with warnings. This supports repeatability without guaranteeing identical results across device types and software versions. Automatic device selection prefers CUDA, then Apple MPS, then CPU. The checkpoint records the requested automatic setting, but does not retain an explicit resolved training-device field. No exact training-hardware or runtime claim is made here.

### C. Frozen backbone behavior

All backbone parameters have gradients disabled. The model is placed in evaluation mode during the frozen-feature training loop, which also prevents batch-normalization running statistics from changing. The linear classifier remains trainable in evaluation mode because its parameter gradients are still enabled. This detail is necessary to distinguish freezing weights alone from fully freezing the feature extractor's behavior.

Each epoch is followed by validation over 397 clips. The checkpoint with the lowest validation loss is saved. The test partition is excluded from fitting and model selection. The current baseline configuration permits 20 epochs, but the checkpoint configuration and saved history both record only one completed epoch. The paper reports the one-epoch artifact and does not imply that a 20-epoch training schedule was completed.

[[TABLE_TRAINING]]

### D. Scope of the adaptation

This experiment measures the usefulness of a pretrained representation after fitting a small classification head. It does not isolate the effect of pretraining against random initialization, compare frozen and fine-tuned backbones, or establish convergence. Such comparisons require additional predeclared runs on the same reviewed data protocol. The available result is a baseline checkpoint, not a completed hyperparameter study.

The recorded run processes all 1,346 training examples, including the smaller final mini-batch, before validation. Loss is accumulated with sample-count weighting so a short last batch does not receive the same influence as a complete batch in the epoch summary. Evaluation disables gradient tracking and uses inference-time batch-normalization behavior. Validation and test reports include loss alongside thresholded classification metrics, but the available aggregate reports do not preserve enough score-level detail to reconstruct ROC or precision-recall curves.

No data augmentation is applied by the dataset loader beyond the deterministic pretrained transform. The experiment therefore avoids an unreported source of sampling variability, while also leaving the potential benefit of augmentation untested. Future augmentation studies should distinguish appearance changes from transformations that alter the action semantics.

<!-- page -->
## VI Continuous monitoring methodology

### A. Capture and recent-window construction

Each configured camera has a capture worker, bounded timestamped frame storage, a recent preview, and an inference consumer. The monitoring defaults are a four-second window, sampling at up to four frames per second, and a one-second stride. The capture path supports validated network stream schemes and local video files for paced testing. Network failures cause a transition to an offline state and reconnect attempts with backoff.

The buffer retains recent samples plus one sample preceding the current window boundary. Once the window has filled, 16 target timestamps are spaced across it. Each target selects the nearest available sample, repeating samples when a source is sparse. This avoids requiring 16 distinct decoded frames before inference on a low-frame-rate source. It cannot reconstruct motion that the source never supplied.

### B. Latest-window scheduling

Only the latest pending clip is retained for each camera. A new window replaces an older pending window and increments a dropped-window counter. A shared lock serializes model execution across cameras, uploads, and diagnostic streams. A consumer acquires the lock before taking the latest window and discards windows that are older than twice that camera's stride. This bounds pending inference work and prioritizes freshness over exhaustive analysis.

The design limits queue growth but does not guarantee fairness between cameras. Consumers compete for the shared lock, and model throughput has not been benchmarked for increasing camera counts. If the model is consistently slower than the requested aggregate window rate, some windows will be replaced or discarded. Consequently, adding cameras can reduce temporal coverage even while memory remains bounded.

### C. Hysteresis and event state

A camera becomes VIOLENCE after three consecutive scores at or above 0.65. Once in that state, four consecutive scores at or below 0.40 return it to NORMAL. Scores between the thresholds reset the counters while preserving an existing violence state. A positive window alone does not immediately open an event. Before the confirmation count is reached, the camera can be reported as NORMAL, which must not be interpreted as a guarantee of absence.

[[FIG_STATES]]

With a one-second stride, three positive windows can confirm a state approximately two seconds after the first qualifying prediction, after the initial buffer has filled. This is an ideal scheduling interpretation, not a measured detection-delay result. Capture delay, inference contention, sampling phase, and the trained model's response can add further delay.

### D. Event lifetime and disconnection

The active event records its start, latest update, and highest score. Events exist only in memory; closed events are removed rather than archived. A disconnect clears buffered frames, pending work, scores, counters, and the active event. A generation counter prevents a completed successful inference from an earlier connection from overwriting the newer state. The implementation does not provide a durable incident history, guaranteed alert delivery, or an evidence-retention service. Those capabilities would require a separate design and evaluation.

The monitor reports connection status and detection state separately. An ONLINE connection can still be BUFFERING while the first temporal window is collected. Conversely, a retained prediction is not evidence that fresh frames are arriving. Public camera snapshots include time since the last sampled frame, processed rate, inference latency, and dropped windows to support diagnosis. These values describe the running instance; no aggregate latency or capacity measurements were saved for the experiment.

Local test files are paced using presentation timestamps, with a frame-rate fallback when timestamps are absent. This prevents an offline decoder from treating an entire video as an instantaneous live burst.

<!-- page -->
## VII Application implementation and verification

### A. Uploaded-video workspace

The upload page supports file selection, drag and drop, local preview, and explicit submission. It checks filename extension, file size, and available browser metadata before enabling analysis. Browser codec support is independent of server decodability; an AVI or MKV file may lack a preview while remaining suitable for PyAV decoding. The API remains responsible for validating actual video contents and decoded limits.

The interface separates byte-transfer progress from inference waiting. A percentage is displayed only while upload events provide measurable transfer progress. Once transfer completes, the application waits for the server without inventing a model-completion percentage. Prediction payloads are checked for allowed class strings and finite scores in the interval from zero to one.

### B. API behavior and resource control

The service loads one shared checkpoint during startup. Model forwards run outside the asynchronous event loop, and a nonblocking inference lock rejects competing upload or diagnostic inference with HTTP 429. Uploaded files use temporary storage that is cleaned after processing. A request-body limit protects the prediction endpoint before the multipart parser writes an arbitrarily large upload to disk; application-level checks also enforce the file-size limit.

[[TABLE_API]]

The health endpoint distinguishes process availability from model readiness. It can return HTTP 200 while reporting that no trained checkpoint is loaded. This allows the application to explain an unavailable model instead of treating every running backend as prediction-ready. Existing incompatible checkpoint metadata causes startup to fail visibly.

### C. Diagnostic camera and monitor views

The single-camera view receives browser webcam keyframes or requests server-side decoding of a network camera. It polls status and requests analysis periodically. This is a diagnostic mode with a consuming frame queue, distinct from the timestamped overlapping windows used by the continuous monitor. It is not the experimental protocol underlying the video-level test metrics.

The multi-camera dashboard receives state snapshots over a WebSocket, reconnects with backoff, and marks displayed statuses as potentially stale while disconnected. Native MJPEG image elements show sampled camera previews. Viewers share existing capture workers, so opening a preview does not create another camera connection. Preview rate follows sampled capture and is not a claim of full native-frame-rate playback.

### D. Verification evidence

During preparation of this paper, the Python suite passed 27 tests. These exercise decoding and tensor layout, model/checkpoint compatibility, split constraints, API error behavior, webcam lifecycle, hysteresis, stale prediction rejection, and paced local monitoring. Ruff lint and mypy checks also passed. A formatting check flagged one auxiliary RLVS helper; formatting is not included as a completed clean check.

The running upload, diagnostic, and monitor routes each returned HTTP 200, and the API health response reported a loaded MC3-18 model. These checks establish startup and selected functional behavior. They do not establish browser usability under every device, successful connection to an external CCTV installation, production security, or statistically reliable multi-camera capacity. Random-weight models in tests verify software plumbing and do not contribute to the research accuracy reported below.

The application is intentionally separated from the training process so a user cannot accidentally present an untrained model as a result. A model-ready response reflects a checkpoint that passed loading checks, not certification of its predictive quality. Similarly, returning a correctly shaped prediction establishes the HTTP contract, while accuracy requires independent labeled evaluation. Keeping these meanings separate helps prevent development tests, demonstration requests, and experimental measurements from being combined into a single unsupported claim.

<!-- page -->
## VIII Experimental results

### A. Artifact consistency and evaluation population

The primary result comes from the saved test evaluation over 238 clips. Its checkpoint fingerprint matches the current best.pt file, and its manifest fingerprint matches the current splits.csv file. The checkpoint contains epoch 1 and the same validation statistics as the one-entry training history. These consistency checks link the reported result to an identifiable model and partition rather than to an illustrative interface response.

There are 118 non-violence and 120 violence clips in the test partition. Evaluation uses the whole-video uniform sampling policy, the checkpoint's 16-frame setting, and the shared spatial transform. The winning logit selects the class. The positive class for precision, recall, and F1 is violence. No camera hysteresis is applied to these test predictions.

[[TABLE_RESULTS]]

### B. Confusion matrix and error asymmetry

The classifier correctly identifies 102 of 118 negative clips and 119 of 120 positive clips. It therefore makes 16 false-positive and one false-negative predictions, with 221 correct classifications overall. The high positive recall is accompanied by lower positive precision. Approximately 11.85% of the 135 positive predictions are incorrect under the available labels.

[[FIG_CONFUSION]]

Specificity is 102/118 = 86.44%, and the false-positive rate is 16/118 = 13.56%. The false-negative rate is 1/120 = 0.83%. Balanced accuracy, the mean of sensitivity and specificity, is 92.80%. These quantities are derived directly from the stored confusion matrix; they do not require access to individual confidence values. The error counts reveal behavior that overall accuracy alone would obscure.

### C. Validation and training evidence

The one saved epoch has training loss 0.3285 and validation loss 0.2456. Validation accuracy is 89.17%, precision 82.70%, recall 98.99%, and F1 90.11%. The validation confusion matrix contains 158 true negatives, 41 false positives, two false negatives, and 196 true positives. Thus the same recall-heavy error pattern appears in both available partitions.

A single epoch does not support a learning-curve plot or a convergence claim. The lower test loss of 0.2092 and higher test accuracy relative to validation are observations of these fixed partitions, not evidence that test data is intrinsically easier or that further training would improve results. No causal explanation of individual errors is available without a separate visual error review.

For clarity, accuracy is (TP+TN)/n, precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is 2TP/(2TP+FP+FN). Substituting TP=119, TN=102, FP=16, and FN=1 reproduces the reported values. Precision concerns the reliability of positive predictions, whereas recall concerns the fraction of labeled positives recovered. Neither metric describes how early within a video an incident is recognized, because each test clip receives only one classification.

<!-- page -->
## IX Statistical and operational interpretation

### A. Sampling uncertainty

A score computed from 238 clips is an estimate for the evaluated population, not a universal property of the model. A two-sided 95% Wilson interval for the observed accuracy of 221/238 is approximately 88.86% to 95.49%. Corresponding intervals are approximately 81.61% to 92.57% for precision, 95.43% to 99.85% for recall, and 79.10% to 91.48% for specificity. These intervals are newly derived from the saved counts using z = 1.96.

The Wilson calculation assumes independent Bernoulli observations for the relevant proportion. Related clips or repeated source scenes can violate that assumption. The intervals should therefore be read as descriptive, conditional summaries, not as a correction for the unresolved source-group problem. There is no repeated-seed or repeated-split distribution in the available evidence.

[[FIG_METRICS]]

### B. Prevalence and alert burden

The test partition is nearly class balanced, while many real camera streams may contain long periods without violent events. Precision depends on prevalence as well as sensitivity and specificity. As an illustrative calculation only, if sensitivity and specificity remained 99.17% and 86.44% at 1% positive prevalence, the implied positive predictive value would be about 6.88%. This is a hypothetical population calculation, not a measured deployment result.

Using r for sensitivity, q for specificity, and p for prevalence, the calculation is PPV = (r p)/((r p) + (1-q)(1-p)). The assumption that clip-level rates transfer to another population is strong and untested. The example explains why high recall on balanced clips cannot establish a manageable alert workload on continuous CCTV footage.

### C. What hysteresis does and does not establish

Multiple qualifying windows can reduce rapid state changes, but adjacent windows overlap and their errors are dependent. It would be incorrect to cube the clip false-positive rate to estimate the probability of a three-window false alarm. Hysteresis thresholds and consecutive-window counts require their own event-level validation on continuous recordings with timestamps and negative durations.

The stored evaluation also does not establish detection delay, false alerts per camera-hour, event recall, or localization precision. Those quantities need annotations describing when incidents begin and end, as well as a definition of what counts as a matched alert. A video-level positive label alone cannot supply this information for every four-second sub-window.

### D. Other saved outputs

The repository contains a separate 20-clip output with perfect classification. Its small sample and absent checkpoint/manifest fingerprint fields make it unsuitable as the primary evaluation result. It is excluded from the main results table rather than promoted as 100% accuracy. The 238-clip fingerprinted artifact is the basis for the paper's headline measurements. This distinction prevents a convenience check from replacing the more complete available evaluation.

The plotted validation and test summaries come from one checkpoint and different fixed partitions. Their difference should not be interpreted as a statistically tested improvement from validation to test. Both reflect finite samples, and neither provides an estimate of training-seed variance. A paired comparison of two future models would require predictions on the same clips; aggregate metrics alone are insufficient to recover which clips one model corrects and another loses.

Calibration would require score-level outputs and an independent calibration protocol. The current confusion matrix supports thresholded performance summaries, but cannot establish expected calibration error, a reliability curve, or a useful abstention threshold.

<!-- page -->
## X Limitations, responsible use, and future experiments

### A. Validity of the empirical claim

The principal limitation is unverified related-source leakage. Exact hashes prevent identical retained files from crossing partitions, but do not establish independent camera scenes or original videos. The labels are inherited from folders, and no independent relabeling agreement study is available. Only one completed training epoch and one seed are recorded. These constraints limit both generalization claims and any comparison with external literature.

The model receives a small center crop of 16 uniformly spaced frames. It may ignore brief actions, peripheral interactions, or contextual information outside the crop. It processes RGB appearance without audio, pose, optical flow, or person identity. The stored aggregate metrics do not identify which visual factors caused the 17 test errors. Explanations involving occlusion, sports, lighting, or camera shake remain hypotheses until the individual errors are reviewed.

### B. Temporal and operational limitations

Training labels describe complete clips, whereas continuous monitoring applies the model to short overlapping windows. A positive clip need not contain violence in every sub-window. Naively assigning its label to every temporal crop would introduce additional label noise. A stronger window-level study requires localized annotations, careful review, or a documented weak-supervision strategy.

The service is designed as a local single-worker prototype. Camera state and the shared model live in process memory. Restarting loses active sessions and events; multiple independent API workers would not share the same camera registry. The current system has no durable event database, role-based access control, or validated external alerting integration. Remote access therefore requires an appropriate surrounding deployment design, especially because previews can contain sensitive camera imagery.

### C. Human review and privacy

Predictions should prioritize review rather than determine guilt, intent, or a person's identity. The binary label describes learned visual categories under a dataset protocol. It does not infer the legal or social meaning of an interaction. Softmax confidence is uncalibrated, and the study contains no demographic fairness assessment or consent audit of the public mirror.

Camera credentials should remain outside committed configuration files, and access to streams should be restricted. The monitoring design avoids archiving footage or closed events by default, but short-lived previews still expose visual information to connected viewers. Any operational study should define authorized access, retention, consent, reviewer responsibilities, and procedures for false alerts before collecting new footage.

### D. Prioritized future work

The first experimental priority is a reviewed source-group manifest followed by a fresh source-disjoint split. A predeclared comparison should then evaluate frozen MC3-18 against selective fine-tuning and alternative backbones, with repeated seeds and unchanged test data. This would separate representation choice, optimization variation, and training duration instead of attributing all effects to a single architectural label.

A second study should annotate continuous CCTV recordings and report event recall, false alerts per hour, delay distributions, and camera-specific performance. Thresholds must be selected using validation recordings before final testing. Calibration and uncertainty-based review policies can be evaluated separately. Engineering work should benchmark increasing camera counts, fairness under lock contention, reconnect recovery, shutdown behavior, and preview bandwidth. None of these proposed experiments is presented as already completed.

A useful error-review extension would record each test path, true label, predicted label, and both class scores, then review mistakes with a fixed taxonomy. Categories might include missing temporal coverage, action outside the crop, ambiguous labels, and visually similar nonviolent movement. Review should include correct predictions as controls and avoid changing the test set merely to improve a score. The present paper does not assign those categories retrospectively without seeing the corresponding clips.

Operational evaluation should also separate unavailable input from confident negative classification. A dropped window or disconnected camera is an absence of evidence, and should be counted in coverage reporting rather than silently removed from the denominator.

<!-- page -->
## XI Reproducibility and conclusion

### A. Reproducing the available result

The primary evidence consists of the inspected dataset report, reviewed exclusion CSV, retained split manifest, checkpoint, training history, and fingerprinted test report. A reproducer should acquire the same dataset contents, regenerate or verify the manifest, and confirm that the checkpoint and manifest fingerprints match before comparing metrics. A new download can differ from the archived mirror even when its dataset title is unchanged.

The evaluation entry point is violence_detection.evaluate, with dataset root, manifest, checkpoint, and output arguments. It validates checkpoint metadata and the manifest fingerprint, verifies test file hashes, and uses the same frame loader as training. The training entry point is violence_detection.train. To reproduce the available run, its configuration must specify one epoch rather than assume the current 20-epoch baseline configuration produced the recorded checkpoint.

[[TABLE_PROVENANCE]]

For the recorded checkpoint, the manifest SHA-256 is c4afc39bfe575e12361ede16e5d2b1f65d3fa6a06cb3f15b0441dbe070313952. The checkpoint SHA-256 is 698e6d253c38f66abdcebd6e35f47c81222dccbb33ec3c10381ebd17ee483f3d. These identify the evaluated artifacts; they are not substitutes for publishing the permitted source data and complete protocol. The code and local generated artifacts remain subject to their respective distribution rights.

### B. Conclusion

This project implements an end-to-end violence-classification research prototype with a pretrained video backbone, audited data handling, checkpoint validation, an inference API, and upload and camera interfaces. The saved one-epoch frozen-backbone MC3-18 checkpoint achieves 92.86% accuracy and 93.33% positive-class F1 on 238 locally held-out RLVS clips. Its errors are asymmetric: one positive clip is missed, while 16 negative clips are classified as violence.

The continuous monitoring implementation adds recent-window sampling, bounded pending work, reconnect handling, previews, and per-camera hysteresis. These are functioning engineering mechanisms, but the available experiment does not validate event detection or operational alert quality. Source-independent testing and continuous-video annotations remain necessary before stronger claims can be made. The defensible outcome is a traceable baseline and an inspectable application that can support those next experiments.

## References

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR, pp. 770-778, 2016. https://arxiv.org/abs/1512.03385.

[2] D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, "A Closer Look at Spatiotemporal Convolutions for Action Recognition," CVPR, pp. 6450-6459, 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html.

[3] W. Kay et al., "The Kinetics Human Action Video Dataset," arXiv:1705.06950, 2017. https://arxiv.org/abs/1705.06950.

[4] PyTorch contributors, "MC3-18 model and pretrained weights," Torchvision documentation. https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.mc3_18.html. Accessed November 15, 2026.

[5] M. Cheng, K. Cai, and M. Li, "RWF-2000: An Open Large Scale Video Database for Violence Detection," arXiv:1911.05913, 2019. https://arxiv.org/abs/1911.05913.

[6] magicearth25, "Video violence detection dataset," Kaggle public archive. https://www.kaggle.com/datasets/magicearth25/video-violence-detection-dataset. RLVS collection audited locally; original data rights retained by their owners.

[7] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR, 2019. https://arxiv.org/abs/1711.05101.

[8] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," ICML, PMLR 70, pp. 1321-1330, 2017. https://arxiv.org/abs/1706.04599.
