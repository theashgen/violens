"""Measure the real surveillance worker on paced replays of reviewed test clips."""

import argparse
import csv
import json
import random
import subprocess
import threading
import time
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw
from violence_api.surveillance import SurveillanceManager
from violence_detection.model import build_model, select_device

OUTPUT = Path("artifacts/review")
ROOT = Path("data/raw/Violence Fight Detection dataset/RLVS").resolve()
BENCHMARK_SECONDS = 42
GRACE_SECONDS = 4


def select_review():
    """Select distinct test groups within each class without looking at model scores."""
    with (OUTPUT / "grouped-splits.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(932)
    selected = []
    for label in ("0", "1"):
        pool = [row for row in rows if row["split"] == "test" and row["label"] == label]
        rng.shuffle(pool)
        seen = set()
        for row in pool:
            if row["group"] not in seen:
                selected.append(row)
                seen.add(row["group"])
            if len(seen) == 12:
                break
    for index, row in enumerate(selected, 1):
        row["id"] = f"R{index:02}"
    (OUTPUT / "review-sample.json").write_text(json.dumps(selected, indent=2))
    for start in range(0, len(selected), 6):
        sheet = Image.new("RGB", (1000, 6 * 145), "#f2f4f7")
        draw = ImageDraw.Draw(sheet)
        for k, row in enumerate(selected[start : start + 6]):
            with av.open(str(ROOT / row["path"])) as container:
                frames = [
                    Image.fromarray(
                        frame.reformat(width=192, height=108, format="rgb24").to_ndarray()
                    )
                    for frame in container.decode(video=0)
                ]
            y = k * 145
            draw.text((10, y + 5), row["id"], fill="#16283b")
            for j, idx in enumerate(np.linspace(0, len(frames) - 1, 5).astype(int)):
                sheet.paste(frames[idx], (10 + j * 196, y + 25))
        sheet.save(OUTPUT / f"review-sheet-{start // 6 + 1}.png")


def make_replays():
    sample = json.loads((OUTPUT / "review-sample.json").read_text())
    annotations = json.loads((OUTPUT / "review-labels.json").read_text())
    assert {r["id"] for r in sample} == set(annotations)
    reviewed = [dict(row, **annotations[row["id"]]) for row in sample]
    negatives = [r for r in reviewed if r["review"] == "non-violence" and r["label"] == "0"]
    positives = [r for r in reviewed if r["review"] == "violence" and r["label"] == "1"]
    assert len(negatives) >= 4 and len(positives) >= 4
    output = OUTPUT / "replays"
    output.mkdir(exist_ok=True)
    timelines = {}
    for cam_id in range(4):
        timeline = []
        cursor = 0
        # Vary the footage between cameras and keep time for recovery between events.
        for step, seconds in enumerate((10, 5, 10, 5, 16)):
            is_event = step % 2 == 1
            row = (positives if is_event else negatives)[
                (cam_id + step // 2) % (len(positives) if is_event else len(negatives))
            ]
            segment = output / f"cam-{cam_id}-{step}.mp4"
            # Extend short segments by looping. Window timing refers to the edited replay.
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-stream_loop",
                    "-1",
                    "-i",
                    str(ROOT / row["path"]),
                    "-t",
                    str(seconds),
                    "-an",
                    "-vf",
                    "fps=12,scale=384:216:force_original_aspect_ratio=decrease,"
                    "pad=384:216:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    str(segment),
                ],
                check=True,
            )
            if is_event:
                timeline.append({"source": row["id"], "start": cursor, "end": cursor + seconds})
            cursor += seconds
        concat = output / f"cam-{cam_id}.txt"
        concat.write_text("".join(f"file cam-{cam_id}-{i}.mp4\n" for i in range(5)))
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat),
                "-c",
                "copy",
                str(output / f"cam-{cam_id}.mp4"),
            ],
            check=True,
        )
        timelines[str(cam_id)] = timeline
    (OUTPUT / "replay-events.json").write_text(json.dumps(timelines, indent=2))
    mismatches = [
        r["id"]
        for r in reviewed
        if r["review"] not in ("ambiguous", "violence" if r["label"] == "1" else "non-violence")
    ]
    audit = {
        "size": len(reviewed),
        "mismatches": mismatches,
        "ambiguous": [r["id"] for r in reviewed if r["review"] == "ambiguous"],
    }
    (OUTPUT / "label-audit.json").write_text(json.dumps(audit, indent=2))


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


def run_trial(model, count, events):
    manager = TimedManager(model)
    first_frame = {}
    processed_schedules = {str(i): set() for i in range(count)}
    try:
        for i in range(count):
            uri = (OUTPUT / f"replays/cam-{i}.mp4").resolve().as_uri()
            manager.add({"camera_id": str(i), "url": uri})
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            for cid, cam in manager.cameras.items():
                with cam.lock:
                    if cam.first_frame_at is not None:
                        first_frame.setdefault(cid, cam.first_frame_at)
                    if cam.last_inference:
                        processed_schedules[cid].add(cam.last_inference)
            if len(first_frame) == count:
                break
            time.sleep(0.02)
        assert len(first_frame) == count, manager.snapshot()
        deadline = max(first_frame.values()) + BENCHMARK_SECONDS
        while time.monotonic() < deadline:
            for cid, cam in manager.cameras.items():
                with cam.lock:
                    if (
                        cam.last_inference
                        and cam.last_inference - first_frame[cid] <= BENCHMARK_SECONDS
                    ):
                        processed_schedules[cid].add(cam.last_inference)
                    if cam.state in {"OFFLINE", "ERROR"}:
                        raise RuntimeError(f"Unexpected camera failure: {cid} {cam.error}")
            time.sleep(0.025)
        stats = manager.snapshot()
        with manager.observer_lock:
            rows = [
                dict(row, seconds=row["when"] - first_frame[row["camera"]])
                for row in manager.observations
                if row["when"] - first_frame[row["camera"]] <= BENCHMARK_SECONDS
            ]
    finally:
        manager.stop_all()
    latencies = [r["latency_ms"] for r in rows]
    matched, false_alerts, duplicates, delays, negative_seconds = 0, 0, 0, [], 0
    per_camera = {}
    for cid in first_frame:
        intervals = events[cid]
        negative_seconds += (
            BENCHMARK_SECONDS - sum(e["end"] - e["start"] + GRACE_SECONDS for e in intervals) - 4
        )  # initial buffering is outside evaluable exposure
        alerts = [r["seconds"] for r in rows if r["camera"] == cid and r["starts_alert"]]
        local_matched = set()
        for alert in alerts:
            event = next(
                (
                    i
                    for i, e in enumerate(intervals)
                    if e["start"] <= alert <= e["end"] + GRACE_SECONDS
                ),
                None,
            )
            if event is None:
                false_alerts += 1
            elif event in local_matched:
                duplicates += 1
            else:
                local_matched.add(event)
                delays.append(alert - intervals[event]["start"])
        matched += len(local_matched)
        completed = sum(r["camera"] == cid for r in rows)
        scheduled = len(processed_schedules[cid])
        per_camera[cid] = {"scheduled": scheduled, "completed": completed, "alerts": alerts}
    scheduled = sum(len(s) for s in processed_schedules.values())
    replaced = sum(int(camera["dropped_windows"]) for camera in stats)
    return {
        "cameras": count,
        "duration_seconds_per_camera": BENCHMARK_SECONDS,
        "events": sum(len(events[cid]) for cid in first_frame),
        "events_matched": matched,
        "false_alerts": false_alerts,
        "duplicate_alerts": duplicates,
        "negative_seconds": negative_seconds,
        "false_alerts_per_camera_hour": false_alerts * 3600 / negative_seconds,
        "delays_seconds": delays,
        "scheduled": scheduled,
        "completed": len(rows),
        "replaced_windows": replaced,
        "throughput_per_second": len(rows) / BENCHMARK_SECONDS,
        "latency_p50_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "per_camera": per_camera,
        "observations": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sample", "replays", "benchmark"))
    args = parser.parse_args()
    if args.stage == "sample":
        select_review()
    elif args.stage == "replays":
        make_replays()
    else:
        torch.set_num_threads(2)
        torch.hub.set_dir(str(Path("artifacts/torch/hub").resolve()))
        device = select_device()
        model = build_model().eval().to(device)
        model.fc.load_state_dict(torch.load(OUTPUT / "mc3-head-42.pt", weights_only=True))
        with torch.inference_mode():
            model(torch.zeros((1, 3, 16, 112, 112), device=device)).cpu()
        results = {"device": str(device), "runs": []}
        events = json.loads((OUTPUT / "replay-events.json").read_text())
        for count in (1, 2, 4):
            run = run_trial(model, count, events)
            results["runs"].append(run)
            (OUTPUT / "monitor-results.json").write_text(json.dumps(results, indent=2))
            print(
                {k: v for k, v in run.items() if k not in ("observations", "per_camera")},
                flush=True,
            )


if __name__ == "__main__":
    main()
