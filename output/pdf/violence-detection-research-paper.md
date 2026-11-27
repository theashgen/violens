# Violence Detection in Surveillance Videos Using Transfer Learning and Continuous Multi-Camera Monitoring

Ashwin Sathiya Moorthy | Registration No. 12308018 | Roll No. 46 | theashgen06@gmail.com

Prudhviraj K B | Registration No. 12307060 | Roll No. 17 | prudhvirajkb3@gmail.com

Pranav R | Registration No. 12306134 | Roll No. 36 | pranavsandle@gmail.com
B.Tech in Artificial Intelligence and Machine Learning
Lovely Professional University, Punjab, India

## Abstract

We evaluated violence classification on the RLVS video collection and measured the prototype's continuous monitoring workers in a paced replay benchmark. A frame-based perceptual hash screen and file hashes grouped visually related footage before splitting. We trained linear heads on frozen MC3-18 and frame-averaged ResNet-18 features for 20 epochs with three seeds. On 397 grouped test clips, mean accuracy at a fixed 0.50 threshold was 96.73% for MC3-18 and 95.97% for ResNet-18. We chose a second clip threshold on validation clips and reported its test results separately. A review of sampled frames from 24 clips identified 9 ambiguous cases and 0 clear folder-label disagreements. The four-stream replay completed 130 scheduled windows and matched six of eight inserted events. The archive lacks original-source identifiers, and the edited replays offer limited negative exposure. The monitoring results describe a local prototype under controlled conditions.

**Keywords:** violence detection; video classification; MC3-18; transfer learning; RLVS; temporal windows; surveillance monitoring.

## I Introduction

A person watching several surveillance feeds can miss short incidents, especially when the feeds compete for attention. A classifier can flag footage for review. The difficult cases are often ones in which ordinary activity resembles violence. A collision during a game, a crowded scene, or someone helping another person stand up can look different depending on the camera angle, what is occluded, and how much detail survives compression.

We set out to measure how well a pretrained video model classifies complete RLVS clips and to see what it takes to use that same model in a working local application. The upload view returns one prediction per video. A diagnostic view accepts a browser webcam or network camera. A third view monitors multiple cameras continuously. All three use a compatible trained checkpoint. If the checkpoint is missing, prediction is unavailable.

The offline workflow records which files were excluded, how clips were split, and which checkpoint produced the reported scores. Training and upload inference use the same decoder and preprocessing. For the camera monitor, we added a windowing policy and per-camera state rules so the dashboard can keep updating even when several feeds share the model.

The application also has to handle cases that clip accuracy cannot describe. A browser may be unable to preview a clip the server can decode. A camera can disconnect while an inference request is running. Several feeds may ask for predictions faster than the model can serve them. The implementation gives each of these cases a visible outcome. We measured classification on grouped clips and measured worker performance and alerts on paced replays.

<!-- page -->
## II Related work and research positioning

### A. Residual and spatiotemporal representations

He et al. introduced shortcut connections to ease optimization in deep visual networks [1]. Video models also need to account for how the picture changes between frames. Tran et al. compared full 3D convolutions, mixed 3D and 2D convolutions, and factorized spatiotemporal convolutions in residual networks [2]. We used their mixed-convolution design as implemented by torchvision's MC3-18.

The starting weights were trained on Kinetics-400 for human action recognition [3,4]. We reused that network's representation and replaced its original classifier with two outputs. The new head learned the difference between the violence and non-violence labels in our data. The RLVS evaluation therefore concerns the adapted model on our partitions.

### B. Violence datasets and protocol differences

Cheng et al. introduced RWF-2000 as a surveillance video dataset and proposed a Flow Gated Network [5]. Our archive contains more than one collection, and our experiment used its RLVS portion [6]. The two datasets have different footage and evaluation protocols. We use earlier work to explain model choices and report our results for the audited RLVS subset.

Copies of the same video can appear in separate folders or under different filenames. If one copy enters training and another appears in validation or test, the result can reflect familiarity with that footage. We inspected the archive before assigning the test set and recorded the difference between the discovered files, decodable records, and retained clips.

### C. Optimization and interpretation of confidence

The training code uses AdamW, which applies weight decay separately from Adam's gradient update [7]. We optimize the classifier head while leaving the feature extractor fixed. This keeps the number of trainable parameters small.

Guo et al. discuss how a classifier's reported confidence can differ from how often it is correct [8]. The application displays the model's softmax score, which we did not calibrate. A softmax score of 0.9 gives the model's relative confidence. To interpret it as a 90% chance of a harmful event, we would need calibration data from the setting where the system is used.

### D. Position of the present study

We compare two frozen backbones on the same grouped split using three training seeds. The 2D baseline averages frame embeddings and the 3D model integrates frame order within convolutional filters. We also run the monitoring workers at a fixed replay rate to observe throughput and alert behavior, leaving camera-network conditions and unedited event recordings for a field study.

<!-- page -->
## III Dataset audit and grouped partitions

### A. Acquisition and label definition

The RLVS subset of the public archive [6] contained 2,000 clips, 1,000 under each class folder. Three clips exceeded the 120-second limit and one had a corrupted frame. The audit retained 1,996 decodable files for fingerprinting. The folder label is class 0 for non-violence and class 1 for violence.

### B. Perceptual duplicate detection

We verified file hashes against the inspection records and sampled 12 evenly spaced frames from every usable clip. Each frame was resized to 32 by 32 grayscale pixels. We took the low-frequency 8 by 8 block of a two-dimensional discrete cosine transform, removed the DC coefficient, and compared the other 63 coefficients with their median to produce a binary fingerprint.

We compared all clip pairs. A pair links when at least four distinct samples in each clip have near matches and at least four matches line up under a single temporal offset. Each matching pair of frames differs by no more than eight bits. The offset allows some tolerance for trimming. We also joined exact SHA-256 matches and took connected components so linked clips always travel together. The screen found 285 links between byte-distinct files and 96 components with more than one file before exclusions. These are candidate visual relationships. A spot check of eight linked pairs found visually related scenes, including segments of the same sports broadcast and successive shots of the same setting. This check did not establish original recording identity.

### C. Final partitions

We excluded 2 clips in groups whose folder labels disagreed. Within remaining groups, we kept one representative of each exact hash and retained other linked clips in the same group. We assigned groups with a stratified five-fold shuffle seeded at 42. One fold supplies validation, a second supplies the test set, and the other three supply training. We checked that every group belongs to only one partition.

[[TABLE_SPLITS]]

The new split has 1,981 clips in 1,803 groups. The original archive supplies no trusted camera, actor, or recording IDs. Visual groups are proxies for shared footage. The test split is independent of the training and validation splits by the recorded group IDs. Substantial crops, large time shifts, or repeated actors in visually different footage can still escape the screen.

### D. Label review

We drew 12 test clips per folder class from distinct groups, using seed 932 and without consulting model scores. An AI assistant visually screened five chronological frames per clip and recorded a clear violence label, clear non-violence label, or an ambiguous result. The review found 0 clear folder-label disagreements and 9 ambiguous clips out of 24. The test metrics retain the supplied labels, so the review does not selectively remove difficult test clips. No second annotator was available to estimate inter-annotator agreement.

The ambiguous set included occluded interactions, fast camera motion, and scenes whose context was unclear from five frames. Examples of clear negatives included field sports and a staged performance. These observations illustrate where folder labels can be difficult to verify, but give no estimate of annotator agreement.

<!-- page -->
## IV Proposed system and video representation

### A. Components and data flow

The ML package decodes videos, inspects the dataset, trains the model, and evaluates checkpoints. At startup, the FastAPI service loads the checkpoint produced offline. It validates uploads and manages camera sessions. Next.js provides the views for video uploads, camera diagnostics, and continuous monitoring.

[[FIG_ARCH]]

An upload produces one prediction for the entire clip. Continuous monitoring scores short windows cut from timestamped camera samples. Both paths use the same spatial transform and class order.

### B. Bounded decoding and temporal sampling

The upload path first decodes the video to count frames and check timestamps, corruption, and resolution. It then decodes the file again and retains only the selected frames. Interframe compression still requires the decoder to read the intervening frames. The two passes therefore reduce the amount of video held in memory. The limits are 120 seconds, 12,000 decoded frames, and a maximum frame area of 3840 by 2160 pixels. The HTTP upload limit is 100 MiB.

For N decoded frames and T requested samples, the selected index is i(k) = floor(k(N-1)/(T-1)), for k from 0 to T-1. We used T = 16. This includes the first and last frames and repeats indices when a clip is too short. The spacing grows with clip length, so a brief incident can fall between two samples.

### C. Spatial preprocessing and tensor layout

We use the resize, center crop, rescaling, and normalization associated with the pretrained weights [4]. Training and uploaded-video inference use the same implementation. Each clip is a float32 tensor with shape [3,16,112,112]. A batch has shape [B,3,16,112,112]. Continuous capture applies the same spatial transform to incoming sampled frames before stacking them into a clip.

Center cropping can cut out an interaction near the edge of a frame. Resizing can also hide details of people far from the camera. The baseline uses a fixed crop and operates directly on RGB, without a person detector, tracker, pose estimator, or region proposal.

### D. Checkpoint contract

The checkpoint records the architecture, class order, frame count, preprocessing identifier, original weights, selected epoch, validation metrics, training settings, software versions, and manifest hash. Inference verifies compatibility before restoring the model. The server needs no additional pretrained weight download. An incompatible checkpoint prevents startup. When the file is absent, the server starts and reports that the model is unavailable.

The upload path spreads its samples across the length of the file. A monitoring window covers four seconds by default. This changes the amount of motion the same 16 frames can represent, even though each frame receives the same spatial transform.

<!-- page -->
## V Model adaptation and training protocol

### A. MC3-18 and the comparison backbone

The primary model is torchvision MC3-18 with Kinetics-400 weights and a new two-output linear classifier. Its pooled 512-dimensional vector becomes two logits through the 1,026-parameter head. Mixed spatiotemporal convolutions let MC3-18 use changes across frames before pooling.

For comparison, we added a frozen ImageNet-pretrained ResNet-18 [1]. It uses the same 16 frame indices, applies its pretrained image transform, and averages the 512-dimensional per-frame embeddings. We fit the same linear classifier on those averages. The comparison changes both temporal modeling and the pretraining dataset, so it measures the two complete pipelines. It cannot isolate a causal contribution from either difference.

### B. Optimization and model selection

We computed each frozen embedding once and cached it. Both backbones stayed in evaluation mode, which fixes their batch-normalization statistics. Each head trained with AdamW and mean two-class cross-entropy, using batches of 32 and a learning rate of 0.001. Seeds 41, 42, and 43 changed the initial head weights and training order. All runs used the same grouped partitions.

We set a 20-epoch budget for each run. For a small linear head on fixed features, this gives the optimizer repeated passes with modest cost and provides enough observations to see how validation loss changes. We saved the epoch with the lowest validation loss. The test set played no role in the choice of epoch, hyperparameters, or threshold.

[[TABLE_TRAINING]]

### C. Validation-only thresholds

For each seed and backbone we scored validation clips with the selected checkpoint. We tested thresholds from 0.05 to 0.95 in steps of 0.05 and maximized violence-class F1. Ties were broken by higher precision, then higher threshold. We locked the threshold before computing test metrics. We also report results at the fixed 0.50 threshold so both backbones can be compared under the same decision rule.

The upload API exposes the predicted class and its softmax score. The monitor uses the violence score with separate state-transition thresholds. Clip-level F1 tuning does not calibrate scores or select an event-level alarm policy.

<!-- page -->
## VI Continuous monitoring methodology

### A. Capture and recent-window construction

Each camera has a capture worker, a bounded buffer of timestamped samples, a recent preview, and a consumer for inference. By default, the capture worker samples up to four frames per second, and the monitor scores a four-second window with a one-second stride. The capture code accepts validated network stream URLs and local test files. A network failure moves the camera offline and triggers reconnect attempts with backoff.

The buffer keeps recent samples plus one sample just before the window boundary. After the window fills, the monitor spaces 16 target timestamps across it and uses the closest available sample for each timestamp. A slow source can contribute the same frame to several positions. Local test files are paced using presentation timestamps, or the frame rate when timestamps are missing, so decoding runs at playback speed.

### B. Latest-window scheduling

Each camera can have one pending clip. If a newer window arrives first, it replaces the pending clip and increments the dropped-window counter. A shared lock allows one model call at a time across the monitor, uploads, and the diagnostic view. After acquiring the lock, a camera consumer takes its latest window and discards it if it is older than twice the camera's stride.

Consumers compete for the same lock. When the total arrival rate exceeds the model's capacity, clips may be replaced or discarded. Adding cameras can therefore reduce how much footage gets scored even though the pending queue stays bounded. We did not measure how throughput changes as cameras are added.

### C. Hysteresis and event state

Three consecutive scores at or above 0.65 change a camera's state to VIOLENCE. Four consecutive scores at or below 0.40 return it to NORMAL. Scores between the thresholds reset the counters while preserving an active violence state. A camera can still read NORMAL after the first or second qualifying score.

[[FIG_STATES]]

At a one-second stride, the third qualifying prediction could arrive about two seconds after the first, once the buffer has filled. This is the timing implied by the settings. Capture delays, sampling phase, inference contention, and the model's response to the footage can increase it.

### D. Event lifetime and disconnection

An active event records its start, last update, and highest score. The monitor stores it in memory and removes it when it closes. A disconnect clears the buffer, pending window, score, counters, and event. A generation counter prevents a completed inference from the old connection from replacing the state of a new connection. The available event information is limited to the current process.

The dashboard shows connection and detection states separately. A camera can be ONLINE while its first window is still BUFFERING. The per-camera snapshot includes the age of the last frame, processed rate, inference latency, and the count of dropped windows. These values describe the running instance. The experiment did not save aggregate latency or capacity measurements.

<!-- page -->
## VII Continuous replay benchmark

### A. Replay construction and labels

We assembled four 46-second files from reviewed, held-out RLVS clips with clear labels. Each replay contains non-violent footage and two five-second violence inserts, beginning at seconds 10 and 25. We resampled the clips to 384 by 216 at 12 frames per second. Short segments loop within their allotted time. The inserts and cuts are controlled test conditions, and each event's onset is the edit boundary. A delay calculated from that boundary is a replay delay.

### B. Concurrent stream trials

We ran the project's capture and inference workers on the same Apple Silicon host with PyTorch MPS. Each trial replayed the same selected inputs at their presentation timestamps, with one, two, or four simultaneous streams. We scored 42 seconds per stream, including the four-second startup period, and kept the default four-second window, one-second stride, and hysteresis settings. The MC3-18 head from seed 42 was designated for the replay before seeing the test results.

We counted scheduled windows and completed predictions in the observation interval. The worker's dropped-window counter records replacement of a pending clip by a newer one. A gap between scheduled, completed, and replaced windows may include work pending at cutoff or a window that expired. Latency covers the time from the worker's inference call through score computation. Throughput counts completed windows per elapsed wall-clock second across all cameras.

[[TABLE_MONITOR]]

### C. Event matching

The first alert starting within an insert or in the four seconds after it counts as a match. Later alerts in the same interval are tracked separately. An insert with no matched alert is a miss, even when a prior alert was already active. We report delay only for matched events. False alert rate divides unmatched alerts by the negative exposure in camera-hours, excluding startup and each matching interval. Each stream contributes 20 seconds of such exposure per trial, so the rate is sensitive to a single alert.

[[TABLE_EVENTS]]

Event recall was 50% with one stream, 50% with two, and 75% with four. The four-stream trial completed all 130 scheduled windows at 3.10 windows per second, with a p95 inference latency of 114.51 ms. The 75% figure comes from a wider mix of replay inserts. It does not establish that higher concurrency improves detection. Across the three trials, the worker recorded zero replacements and zero false alerts in 140 seconds of scored negative exposure.

The counts describe this short replay benchmark. Some clips contribute to more than one camera or trial, and the edits can produce motion changes that an unedited feed would not have. An annotated collection of long recordings from unrelated cameras is needed for a field estimate.

<!-- page -->
## VIII Held-out classification results

### A. Results across seeds

The new grouped test set contains 397 clips. Each seed produces one checkpoint per backbone. The table reports the mean and sample standard deviation across the three seeds, in percentage points. All seeds share one test set. The standard deviations measure variation from head initialization and training order on that split.

[[TABLE_RESULTS]]

The main comparison uses a fixed decision threshold of 0.50. MC3-18 reached 96.73% mean accuracy and 96.75% mean F1. The ResNet-18 frame-average model reached 95.97% mean accuracy and 95.95% mean F1. These values replace the original one-epoch experiment as the main results.

### B. Errors in the fixed-threshold run

The seed-42 MC3-18 run recorded 193 true positives, 191 true negatives, 8 false positives, and 5 false negatives at 0.50. We use this predesignated seed in the confusion matrix and replay. At the same cutoff, the other seeds had the same accuracy but sometimes disagreed on individual clips.

[[FIG_CONFUSION]]

Precision gives the fraction of predicted positives that match the folder label. Recall gives the fraction of folder-labeled positives found. F1 is their harmonic mean. Accuracy counts correct labels over all clips. We report these clip metrics independently of the camera state machine.

### C. Loss across epochs

The figure plots the mean training and validation cross-entropy at every epoch for each backbone. Individual seed histories and the selected epochs are included in the experiment results.

[[FIG_METRICS]]

Across the three MC3-18 runs, mean training loss fell from 0.5267 at epoch 1 to 0.0717 at epoch 20. Mean validation loss fell from 0.3824 to 0.1341. Validation loss selected epoch 19 for all three MC3-18 heads and epochs 19, 20, and 20 for the ResNet-18 heads.

<!-- page -->
## IX Threshold selection and false alerts

### A. Clip classification thresholds

The table compares the 0.50 rule with the threshold selected from validation for the seed-42 MC3-18 model. We use the same model and the same split in both rows of each partition. Thresholds for all runs and the complete validation sweeps are recorded with the experiment results.

[[TABLE_THRESHOLDS]]

Threshold changes move the trade-off between missed clips and flags for review. Tuning for clip-level F1 weights precision and recall symmetrically in the validation split. A camera alarm would need a separate event-based comparison of thresholds and hysteresis counts on long validation recordings before use.

### B. Interpreting scores at a low event rate

The new test split is much closer to balanced than many operational camera feeds would be. At a hypothetical 1% prevalence, transferring the seed-42 fixed-threshold test sensitivity and specificity without change would imply 19.67% positive predictive value. This calculation illustrates how changing prevalence can affect review workload. The test protocol cannot establish that its rates will transfer.

Sports, play, physical assistance, and crowd activity can involve pushes, rapid movement, or close contact. These visual similarities can cause false positives. Camera shake and occlusion can make interactions harder to interpret. The error mechanism for a particular clip is unknown without reviewing the footage.

### C. Human review workflow

The dashboard should route high scores to a human reviewer as unverified flags. The reviewer should inspect the surrounding footage, check whether the camera view provides enough context, and record a disposition of confirmed, false alert, or insufficient evidence. False alerts and uncertain cases can be logged with their time and scene category for later analysis. Threshold changes should be evaluated using validation data drawn from those reviewed cases.

The model's output describes a clip under a binary visual label. It cannot establish identity, intent, or responsibility. Any follow-up action should depend on the review and the applicable site procedures. The monitoring prototype has a current-event view in memory. A durable review log and access controls would require separate implementation.

For the predesignated MC3-18 run, validation selected 0.35. On the grouped test set, lowering the cutoff from 0.50 to 0.35 recovered one additional positive clip and flagged two additional negative clips. This increase in recall came with lower test precision.

<!-- page -->
## X Limitations and follow-up validation

### A. Related footage and label reliability

The group split prevents exact matches and detected visual links from crossing training, validation, and test. The screen can miss genuinely related clips and link visually similar unrelated scenes. Trusted recording, camera, actor, and location identifiers would allow a stronger source-disjoint split. The archive also mixes staged footage, sports broadcasts, and other sources with camera recordings. The three training seeds capture variation on one partition and do not estimate sensitivity to how the groups were split.

A single AI-assisted visual screen covered a small stratified sample using five frames per clip. The review assessed what was visible in the sampled frames. A full-video study would have two independent annotators use a shared labeling rubric, report raw agreement and Cohen's kappa, and adjudicate disagreements before training or evaluation.

### B. Monitoring and deployment

A positive clip label applies to the whole clip. Some windows in it may contain mostly ordinary activity. Our replay onset is the start of the inserted segment, and the matching rule allows a four-second grace interval. Overlapping windows share frames. Their scores can remain correlated, so alert-level error rates must be measured on continuous footage.

The short replay load trials use local files with one capture resolution and no network transport. They do not test RTSP jitter, packet loss, network reconnection, or sustained performance under longer runs. The observed false alert rate derives from seconds of negative exposure per camera and should not be extrapolated to an operational shift.

The prototype stores camera state and active events in a single worker's memory. It loses them on restart and gives separate workers separate registries. An operational study should secure preview access, keep credentials out of committed configuration, define consent and retention before recording, and record how reviewers handled flags.

### C. Independent field evaluation

A field evaluation needs unedited recordings from independent sites and an annotation protocol with event boundaries. Validation recordings should determine the operating thresholds and counts. A separate field test should then measure event recall and delay, false alerts over substantially longer negative exposure, per-camera coverage, and variation across lighting and views. A review of individual errors should include correct cases for comparison and record whether an error involved clipping, occlusion, fast motion, or an ambiguous label.

<!-- page -->
## XI Reproducibility and conclusion

### A. Experiment records

The feature extraction script verifies source hashes, decodes clips, and saves pooled features in chunks. Run scripts/review_features.py for MC3-18 and add --architecture resnet for the comparison. The script uses PyTorch 2.14.0 and torchvision 0.29.0 on Apple Silicon MPS. The head training runs on CPU. scripts/review_experiments.py generates the split and fits the six heads. The grouped manifest SHA-256 is 9df60ccff3cdba9550586f1cbf7e1781ae20714ed9c83dd10fec7f795a8195f3.

The review sample and its labels are included in review-results.json. For a rerun, the sample labels can be restored from that file by mapping each sample ID to its review and note fields in review-labels.json. scripts/review_monitor.py provides the sample, replays, and benchmark stages. Replay assembly uses FFmpeg, and the benchmark loads the MC3-18 head from seed 42. scripts/build_review_results.py assembles the data used by the report generator. The deployed checkpoint from the original demonstration is separate from these experiment heads.

[[TABLE_PROVENANCE]]

### B. Conclusion

On 397 clips in the visually grouped test split, frozen MC3-18 averaged 96.73% accuracy and 96.75% positive-class F1 over three 20-epoch head-training runs. The ResNet-18 comparison averaged 95.97% accuracy and 95.95% F1. The report also separates validation-only threshold selection from the fixed-threshold comparison.

The monitoring workers processed up to four paced replays on the local host. The benchmark measures alerts and delays under that specific setup. Longer unedited recordings, independent event annotation, and known original sources are still needed before using the prototype to make operational decisions.

## References

[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR, pp. 770-778, 2016. https://arxiv.org/abs/1512.03385.

[2] D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, "A Closer Look at Spatiotemporal Convolutions for Action Recognition," CVPR, pp. 6450-6459, 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html.

[3] W. Kay et al., "The Kinetics Human Action Video Dataset," arXiv:1705.06950, 2017. https://arxiv.org/abs/1705.06950.

[4] PyTorch contributors, "MC3-18 model and pretrained weights," Torchvision documentation. https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.mc3_18.html. Accessed November 15, 2026.

[5] M. Cheng, K. Cai, and M. Li, "RWF-2000: An Open Large Scale Video Database for Violence Detection," arXiv:1911.05913, 2019. https://arxiv.org/abs/1911.05913.

[6] magicearth25, "Video violence detection dataset," Kaggle public archive. https://www.kaggle.com/datasets/magicearth25/video-violence-detection-dataset. RLVS collection audited locally; original data rights retained by their owners.

[7] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR, 2019. https://arxiv.org/abs/1711.05101.

[8] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," ICML, PMLR 70, pp. 1321-1330, 2017. https://arxiv.org/abs/1706.04599.
