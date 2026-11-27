"""Continuous unedited CCTV evaluation (UCF-Crime) and multi-camera SurveillanceManager scaling."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path
from statistics import mean, median

import av
import numpy as np
import torch
from scipy.fft import dctn
from sklearn.metrics import roc_auc_score
from violence_api.stream import tensor_from_frame
from violence_api.surveillance import SurveillanceManager
from violence_detection.model import build_model, select_device

REVIEW_DIR = Path("artifacts/review")
UCF_DIR = Path("data/external/ucf-crime")
WINDOW_SECONDS = 4.0
SAMPLE_FPS = 4.0
STRIDE_SECONDS = 1.0
GRACE_SECONDS = 4.0


def load_rlvs_hashes():
    audit = json.loads(Path("artifacts/metrics/rlvs-inspection.json").read_text())
    records = audit["videos"]
    rlvs_hashes = np.concatenate(
        [np.load(REVIEW_DIR / f"features-{i:04}.npz")["hashes"] for i in range(0, len(records), 24)]
    )
    packed = np.packbits(rlvs_hashes, axis=-1).view(np.uint64).squeeze(-1)
    sha_set = {r["sha256"] for r in records}
    return records, packed, sha_set


def load_mc3_seed42():
    torch.set_num_threads(2)
    torch.hub.set_dir(str(Path("artifacts/torch/hub").resolve()))
    device = select_device()
    model = build_model().eval().to(device)
    model.fc.load_state_dict(torch.load(REVIEW_DIR / "mc3-head-42.pt", weights_only=True))
    with torch.inference_mode():
        model(torch.zeros((1, 3, 16, 112, 112), device=device)).cpu()
    return model, device


def score_continuous_video(video_path: Path, model, device, rlvs_records, rlvs_packed, rlvs_sha):
    with video_path.open("rb") as f:
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()

    sampled_times = []
    sampled_tensors = []
    sampled_hashes = []
    fps = 30.0
    total_frames = 0
    duration = 0.0

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.average_rate:
            fps = float(stream.average_rate)
        last_t = -1e9
        for idx, frame in enumerate(container.decode(stream)):
            total_frames = idx + 1
            t = float(frame.time) if frame.time is not None else (idx / fps)
            duration = max(duration, t)
            if t - last_t < (1.0 / SAMPLE_FPS) - 1e-6:
                continue
            last_t = t
            rgb = frame.to_ndarray(format="rgb24")
            sampled_times.append(t)
            sampled_tensors.append(tensor_from_frame(rgb))
            if len(sampled_hashes) < 120 and (len(sampled_times) % 4 == 1):
                grey = frame.reformat(width=32, height=32, format="gray").to_ndarray()
                low = dctn(grey.astype(float), norm="ortho")[:8, :8].ravel()[1:]
                sampled_hashes.append(low > np.median(low))

    # Perceptual hash overlap check against RLVS using 12 evenly spaced hashes
    overlap_type, overlap_path = None, None
    if sha256 in rlvs_sha:
        overlap_type = "exact"
    elif len(sampled_hashes) >= 12:
        h_idx = np.linspace(0, len(sampled_hashes) - 1, 12).astype(int)
        phash = np.stack([sampled_hashes[i] for i in h_idx])
        packed = np.packbits(phash, axis=-1).view(np.uint64).squeeze(-1)
        distances = np.bitwise_count(packed[:, None] ^ rlvs_packed[:, None, :])
        matches = distances <= 8
        cands = np.flatnonzero((matches.any(1).sum(1) >= 4) & (matches.any(2).sum(1) >= 4))
        for idx in cands:
            if any(np.diagonal(matches[idx], offset).sum() >= 4 for offset in range(-8, 9)):
                overlap_type = "visual"
                overlap_path = rlvs_records[int(idx)]["path"]
                break

    ts_arr = np.array(sampled_times, dtype=np.float64)
    if len(ts_arr) < 2 or ts_arr[-1] - ts_arr[0] < WINDOW_SECONDS:
        return {
            "sha256": sha256,
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration,
            "rlvs_overlap_type": overlap_type,
            "rlvs_overlap_path": overlap_path,
            "windows": [],
        }

    # Build sliding 4.0s windows at 1.0s stride matching SurveillanceManager logic
    window_ends = []
    clips = []
    t0 = ts_arr[0]
    next_due = t0 + WINDOW_SECONDS
    while next_due <= ts_arr[-1] + 1e-6:
        cutoff = next_due - WINDOW_SECONDS
        targets = np.linspace(cutoff, next_due, 16)
        indices = np.abs(ts_arr[:, None] - targets).argmin(axis=0)
        clip = torch.stack([sampled_tensors[int(i)] for i in indices], dim=1)  # (3, 16, 112, 112)
        window_ends.append(float(next_due))
        clips.append(clip)
        next_due += STRIDE_SECONDS

    scores = []
    with torch.inference_mode():
        for b0 in range(0, len(clips), 16):
            batch = torch.stack(clips[b0 : b0 + 16]).to(device)
            probs = model(batch).softmax(dim=1)[:, 1].cpu().numpy().tolist()
            scores.extend(probs)

    return {
        "sha256": sha256,
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": duration,
        "rlvs_overlap_type": overlap_type,
        "rlvs_overlap_path": overlap_path,
        "windows": [{"t": round(t, 2), "score": round(float(s), 6)} for t, s in zip(window_ends, scores)],
    }


def score_split(split: str):
    manifest_path = UCF_DIR / f"{split}_manifest.json"
    out_path = REVIEW_DIR / f"ucf-{split}-scores.json"
    manifest = json.loads(manifest_path.read_text())
    existing = {}
    if out_path.exists():
        old = json.loads(out_path.read_text())
        existing = {v["name"]: v for v in old.get("videos", [])}

    rlvs_records, rlvs_packed, rlvs_sha = load_rlvs_hashes()
    model, device = load_mc3_seed42()
    all_entries = manifest["positives"] + manifest["normals"]
    results = []
    for idx, item in enumerate(all_entries, 1):
        name = item["name"]
        if name in existing:
            results.append(existing[name])
            continue
        vpath = Path(item["file"])
        t_start = time.perf_counter()
        scored = score_continuous_video(vpath, model, device, rlvs_records, rlvs_packed, rlvs_sha)
        if "intervals_frames" in item:
            intervals_sec = [
                [round(a / scored["fps"], 2), round(b / scored["fps"], 2)]
                for a, b in item["intervals_frames"]
            ]
        else:
            intervals_sec = item.get("intervals_sec", [])
        entry = {
            "name": name,
            "file": item["file"],
            "category": item["category"],
            "intervals_sec": intervals_sec,
            **scored,
        }
        results.append(entry)
        out_path.write_text(json.dumps({"split": split, "videos": results}, indent=2))
        print(
            f"[{split}] {idx}/{len(all_entries)} {name}: dur={scored['duration_sec']:.1f}s "
            f"fps={scored['fps']:.1f} wins={len(scored['windows'])} "
            f"ov={scored['rlvs_overlap_type']} ({time.perf_counter()-t_start:.1f}s)",
            flush=True,
        )


def simulate_hysteresis(windows, pos_th, neg_th, pos_win, neg_win):
    state = "NORMAL"
    pos_c = 0
    neg_c = 0
    alert_starts = []
    for w in windows:
        s = w["score"]
        if s >= pos_th:
            pos_c += 1
            neg_c = 0
        elif s <= neg_th:
            neg_c += 1
            pos_c = 0
        else:
            pos_c = neg_c = 0
        if state != "VIOLENCE" and pos_c >= pos_win:
            state = "VIOLENCE"
            alert_starts.append(w["t"])
        elif state == "VIOLENCE" and neg_c >= neg_win:
            state = "NORMAL"
    return alert_starts


def evaluate_policy(videos, pos_th, neg_th, pos_win, neg_win):
    total_events = 0
    matched_events = 0
    false_alerts = 0
    duplicate_alerts = 0
    delays = []
    negative_seconds = 0.0
    total_duration = 0.0
    by_cat = {}
    win_labels, win_scores = [], []

    for v in videos:
        wins = v["windows"]
        if not wins:
            continue
        dur = v["duration_sec"]
        total_duration += dur
        intervals = v["intervals_sec"]
        cat = v["category"]
        by_cat.setdefault(cat, {"videos": 0, "events": 0, "matched": 0, "false_alerts": 0, "delays": []})
        by_cat[cat]["videos"] += 1

        # Window-level AUC labels (positive if window [t-4, t] overlaps any ground-truth event)
        for w in wins:
            w_start, w_end = w["t"] - WINDOW_SECONDS, w["t"]
            is_pos = any(not (w_end < a or w_start > b) for a, b in intervals)
            win_labels.append(1 if is_pos else 0)
            win_scores.append(w["score"])

        # Evaluable negative exposure: duration minus initial 4s startup minus matched event+grace intervals
        # Merge overlapping [a, b + GRACE_SECONDS] intervals clipped to [4.0, dur]
        pos_spans = []
        for a, b in sorted(intervals):
            s0 = max(WINDOW_SECONDS, a)
            s1 = min(dur, b + GRACE_SECONDS)
            if s1 > s0:
                if pos_spans and s0 <= pos_spans[-1][1]:
                    pos_spans[-1][1] = max(pos_spans[-1][1], s1)
                else:
                    pos_spans.append([s0, s1])
        evaluable_total = max(0.0, dur - WINDOW_SECONDS)
        pos_exposure = sum(b - a for a, b in pos_spans)
        neg_sec = max(0.0, evaluable_total - pos_exposure)
        negative_seconds += neg_sec

        alerts = simulate_hysteresis(wins, pos_th, neg_th, pos_win, neg_win)
        total_events += len(intervals)
        by_cat[cat]["events"] += len(intervals)

        matched_local = set()
        for a_t in alerts:
            ev_idx = next(
                (i for i, (s, e) in enumerate(intervals) if s <= a_t <= e + GRACE_SECONDS),
                None,
            )
            if ev_idx is None:
                false_alerts += 1
                by_cat[cat]["false_alerts"] += 1
            elif ev_idx in matched_local:
                duplicate_alerts += 1
            else:
                matched_local.add(ev_idx)
                delay = a_t - intervals[ev_idx][0]
                delays.append(delay)
                by_cat[cat]["delays"].append(delay)
        matched_events += len(matched_local)
        by_cat[cat]["matched"] += len(matched_local)

    recall = matched_events / total_events if total_events else 0.0
    precision = matched_events / (matched_events + false_alerts) if (matched_events + false_alerts) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fa_per_hr = (false_alerts * 3600.0 / negative_seconds) if negative_seconds > 0 else 0.0
    auc = float(roc_auc_score(win_labels, win_scores)) if len(set(win_labels)) > 1 else None

    return {
        "policy": {
            "positive_threshold": pos_th,
            "negative_threshold": neg_th,
            "positive_windows": pos_win,
            "negative_windows": neg_win,
        },
        "videos": len(videos),
        "total_duration_sec": round(total_duration, 1),
        "negative_seconds": round(negative_seconds, 1),
        "negative_camera_hours": round(negative_seconds / 3600.0, 4),
        "events": total_events,
        "events_matched": matched_events,
        "event_recall": recall,
        "event_precision": precision,
        "event_f1": f1,
        "false_alerts": false_alerts,
        "duplicate_alerts": duplicate_alerts,
        "false_alerts_per_camera_hour": round(fa_per_hr, 2),
        "delay_mean_sec": round(mean(delays), 2) if delays else None,
        "delay_median_sec": round(median(delays), 2) if delays else None,
        "delays_sec": [round(d, 2) for d in delays],
        "window_roc_auc": round(auc, 4) if auc is not None else None,
        "by_category": {
            k: {
                "videos": v["videos"],
                "events": v["events"],
                "matched": v["matched"],
                "false_alerts": v["false_alerts"],
                "delay_mean_sec": round(mean(v["delays"]), 2) if v["delays"] else None,
            }
            for k, v in sorted(by_cat.items())
        },
    }


def calibrate_and_eval():
    val_videos = json.loads((REVIEW_DIR / "ucf-val-scores.json").read_text())["videos"]
    test_videos = json.loads((REVIEW_DIR / "ucf-test-scores.json").read_text())["videos"]

    # Confirm zero filename/SHA overlap between val and test and zero RLVS overlap
    assert not ({v["name"] for v in val_videos} & {v["name"] for v in test_videos})

    val_sweep = []
    for pos_th in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        for pos_win in (2, 3, 4, 5):
            neg_th = round(min(0.40, pos_th - 0.15), 2)
            res = evaluate_policy(val_videos, pos_th, neg_th, pos_win, 4)
            val_sweep.append(res)

    # Select strictly on validation: maximize event_f1, break ties by lower false_alerts_per_camera_hour, then higher threshold
    best_val = max(
        val_sweep,
        key=lambda r: (
            round(r["event_f1"], 6),
            -r["false_alerts_per_camera_hour"],
            r["policy"]["positive_threshold"],
            r["policy"]["positive_windows"],
        ),
    )
    val_default = evaluate_policy(val_videos, 0.65, 0.40, 3, 4)
    test_default = evaluate_policy(test_videos, 0.65, 0.40, 3, 4)
    bp = best_val["policy"]
    test_calibrated = evaluate_policy(
        test_videos,
        bp["positive_threshold"],
        bp["negative_threshold"],
        bp["positive_windows"],
        bp["negative_windows"],
    )

    out = {
        "val_rlvs_overlaps": sum(1 for v in val_videos if v["rlvs_overlap_type"] is not None),
        "test_rlvs_overlaps": sum(1 for v in test_videos if v["rlvs_overlap_type"] is not None),
        "val_default": val_default,
        "val_selected": best_val,
        "test_default": test_default,
        "test_calibrated": test_calibrated,
        "val_sweep_summary": [
            {
                **r["policy"],
                "event_recall": round(r["event_recall"], 4),
                "event_precision": round(r["event_precision"], 4),
                "event_f1": round(r["event_f1"], 4),
                "false_alerts": r["false_alerts"],
                "false_alerts_per_camera_hour": r["false_alerts_per_camera_hour"],
                "delay_mean_sec": r["delay_mean_sec"],
            }
            for r in val_sweep
        ],
    }
    (REVIEW_DIR / "ucf-continuous-results.json").write_text(json.dumps(out, indent=2))
    print("=== CONTINUOUS CALIBRATION & TEST SUMMARY ===", flush=True)
    for label, block in (
        ("VAL  default   (0.65/3w)", val_default),
        (f"VAL  selected  ({bp['positive_threshold']}/{bp['positive_windows']}w)", best_val),
        ("TEST default   (0.65/3w)", test_default),
        (f"TEST calibrated({bp['positive_threshold']}/{bp['positive_windows']}w)", test_calibrated),
    ):
        print(
            f"{label}: matched={block['events_matched']}/{block['events']} "
            f"rec={block['event_recall']*100:.1f}% prec={block['event_precision']*100:.1f}% "
            f"f1={block['event_f1']*100:.1f}% FA={block['false_alerts']} "
            f"({block['false_alerts_per_camera_hour']:.2f}/cam-h over {block['negative_camera_hours']:.2f} cam-h) "
            f"delay_mean={block['delay_mean_sec']}s delay_med={block['delay_median_sec']}s "
            f"win_auc={block['window_roc_auc']}",
            flush=True,
        )


class TimedManager(SurveillanceManager):
    def __init__(self, model):
        super().__init__(model, 16, threading.Lock())
        self.observations = []
        self.observer_lock = threading.Lock()

    def _infer(self, camera, clip, *, locked=False, generation=None):
        previous = camera.state
        super()._infer(camera, clip, locked=locked, generation=generation)
        elapsed = camera.inference_latency_ms
        if elapsed is None or camera.state == "ERROR":
            return
        with self.observer_lock:
            self.observations.append(
                {
                    "camera": camera.camera_id,
                    "when": time.monotonic(),
                    "latency_ms": elapsed,
                    "score": camera.score,
                    "starts_alert": previous != "VIOLENCE" and camera.state == "VIOLENCE",
                }
            )


def run_live_multicam_trial(model, stream_specs, duration_sec: float, policy: dict):
    manager = TimedManager(model)
    first_frame = {}
    processed_schedules = {spec["camera_id"]: set() for spec in stream_specs}
    try:
        for spec in stream_specs:
            manager.add(
                {
                    "camera_id": spec["camera_id"],
                    "url": Path(spec["file"]).resolve().as_uri(),
                    "positive_threshold": policy["positive_threshold"],
                    "negative_threshold": policy["negative_threshold"],
                    "positive_windows": policy["positive_windows"],
                    "negative_windows": policy["negative_windows"],
                }
            )
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            for cid, cam in manager.cameras.items():
                with cam.lock:
                    if cam.first_frame_at is not None:
                        first_frame.setdefault(cid, cam.first_frame_at)
                    if cam.last_inference:
                        processed_schedules[cid].add(cam.last_inference)
            if len(first_frame) == len(stream_specs):
                break
            time.sleep(0.02)
        assert len(first_frame) == len(stream_specs), f"Only {len(first_frame)}/{len(stream_specs)} cameras started"
        end_at = max(first_frame.values()) + duration_sec
        while time.monotonic() < end_at:
            for cid, cam in manager.cameras.items():
                with cam.lock:
                    if cam.last_inference and cam.last_inference - first_frame[cid] <= duration_sec:
                        processed_schedules[cid].add(cam.last_inference)
                    if cam.state == "ERROR":
                        raise RuntimeError(f"Camera {cid} ERROR: {cam.error}")
            time.sleep(0.025)
        stats = manager.snapshot()
        with manager.observer_lock:
            rows = [
                dict(row, seconds=row["when"] - first_frame[row["camera"]])
                for row in manager.observations
                if row["when"] - first_frame[row["camera"]] <= duration_sec
            ]
    finally:
        manager.stop_all()

    latencies = [r["latency_ms"] for r in rows]
    scheduled = sum(len(s) for s in processed_schedules.values())
    replaced = sum(int(c["dropped_windows"]) for c in stats)
    offline_count = sum(1 for c in stats if c["connection_status"] != "ONLINE")

    # Event matching for events whose onset falls inside [0, duration_sec - 4]
    total_events, matched, false_alerts, delays, neg_sec = 0, 0, 0, [], 0.0
    for spec in stream_specs:
        cid = spec["camera_id"]
        intervals = [
            (a, min(b, duration_sec))
            for a, b in spec["intervals_sec"]
            if a < duration_sec - WINDOW_SECONDS
        ]
        total_events += len(intervals)
        pos_spans = []
        for a, b in sorted(intervals):
            s0, s1 = max(WINDOW_SECONDS, a), min(duration_sec, b + GRACE_SECONDS)
            if s1 > s0:
                if pos_spans and s0 <= pos_spans[-1][1]:
                    pos_spans[-1][1] = max(pos_spans[-1][1], s1)
                else:
                    pos_spans.append([s0, s1])
        neg_sec += max(0.0, (duration_sec - WINDOW_SECONDS) - sum(b - a for a, b in pos_spans))
        alerts = [r["seconds"] for r in rows if r["camera"] == cid and r["starts_alert"]]
        local_matched = set()
        for a_t in alerts:
            ev = next((i for i, (s, e) in enumerate(intervals) if s <= a_t <= e + GRACE_SECONDS), None)
            if ev is None:
                false_alerts += 1
            elif ev not in local_matched:
                local_matched.add(ev)
                delays.append(a_t - intervals[ev][0])
        matched += len(local_matched)

    return {
        "cameras": len(stream_specs),
        "duration_seconds_per_camera": duration_sec,
        "scheduled": scheduled,
        "completed": len(rows),
        "replaced_windows": replaced,
        "replaced_rate": round(replaced / scheduled, 4) if scheduled else 0.0,
        "completion_rate": round(len(rows) / scheduled, 4) if scheduled else 0.0,
        "throughput_per_second": round(len(rows) / duration_sec, 2),
        "latency_p50_ms": round(float(np.median(latencies)), 1) if latencies else None,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
        "events": total_events,
        "events_matched": matched,
        "false_alerts": false_alerts,
        "negative_seconds": round(neg_sec, 1),
        "false_alerts_per_camera_hour": round(false_alerts * 3600.0 / neg_sec, 2) if neg_sec > 0 else 0.0,
        "delay_mean_seconds": round(mean(delays), 2) if delays else None,
        "delays_seconds": [round(d, 2) for d in delays],
        "offline_at_end": offline_count,
    }


def live_multicam():
    test_videos = json.loads((REVIEW_DIR / "ucf-test-scores.json").read_text())["videos"]
    cont_results = json.loads((REVIEW_DIR / "ucf-continuous-results.json").read_text())
    policy = {
        "positive_threshold": 0.65,
        "negative_threshold": 0.40,
        "positive_windows": 3,
        "negative_windows": 4,
    }
    duration_sec = 42.0

    # Choose unedited continuous UCF-Crime test videos with native duration >= 46s (zero looping!)
    long_pos = sorted(
        [v for v in test_videos if v["intervals_sec"] and v["duration_sec"] >= duration_sec + 4.0],
        key=lambda v: v["intervals_sec"][0][0],
    )
    long_norm = sorted(
        [v for v in test_videos if not v["intervals_sec"] and v["duration_sec"] >= duration_sec + 4.0],
        key=lambda v: v["name"],
    )
    print(f"Eligible unedited >=46s streams: {len(long_pos)} positive, {len(long_norm)} normal", flush=True)

    # Interleave positives and normals so even small camera counts include live events & real normal feeds
    pool = []
    # Put positives whose first event begins inside [4, 42] first!
    early_pos = [v for v in long_pos if v["intervals_sec"][0][0] <= duration_sec - 8]
    late_pos = [v for v in long_pos if v["intervals_sec"][0][0] > duration_sec - 8]
    ordered_pos = early_pos + late_pos
    pi, ni = 0, 0
    while len(pool) < 16:
        if pi < len(ordered_pos) and (len(pool) % 2 == 0 or ni >= len(long_norm)):
            pool.append(ordered_pos[pi])
            pi += 1
        elif ni < len(long_norm):
            pool.append(long_norm[ni])
            ni += 1
        else:
            break

    model, device = load_mc3_seed42()
    results = {
        "device": str(device),
        "policy": policy,
        "duration_seconds_per_camera": duration_sec,
        "stream_pool": [{"name": v["name"], "category": v["category"], "duration_sec": round(v["duration_sec"], 1), "intervals_sec": v["intervals_sec"]} for v in pool],
        "runs": [],
    }
    for count in (1, 2, 4, 8, 12, 16):
        if count > len(pool):
            print(f"Skipping {count} streams (only {len(pool)} >=55s unedited videos available)", flush=True)
            continue
        specs = [
            {
                "camera_id": f"cam-{i}",
                "name": pool[i]["name"],
                "file": pool[i]["file"],
                "intervals_sec": pool[i]["intervals_sec"],
            }
            for i in range(count)
        ]
        run = run_live_multicam_trial(model, specs, duration_sec, policy)
        results["runs"].append(run)
        (REVIEW_DIR / "ucf-multicam-results.json").write_text(json.dumps(results, indent=2))
        print(
            f"Streams={count:2d} | done={run['completed']}/{run['scheduled']} "
            f"| win/s={run['throughput_per_second']:.2f} | p50/p95={run['latency_p50_ms']}/{run['latency_p95_ms']}ms "
            f"| replaced={run['replaced_windows']} ({run['replaced_rate']*100:.1f}%) "
            f"| matched={run['events_matched']}/{run['events']} | FA={run['false_alerts']}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("score-val", "score-test", "calibrate-and-eval", "live-multicam"))
    args = parser.parse_args()
    if args.stage == "score-val":
        score_split("val")
    elif args.stage == "score-test":
        score_split("test")
    elif args.stage == "calibrate-and-eval":
        calibrate_and_eval()
    elif args.stage == "live-multicam":
        live_multicam()


if __name__ == "__main__":
    main()
