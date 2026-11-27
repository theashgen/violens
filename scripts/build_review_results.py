"""Collect the measured reviewer-study results used to rebuild the report."""

import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path("artifacts/review")
DEST = Path("output/pdf/review-results.json")


def load(name):
    return json.loads((ROOT / name).read_text())


def main():
    experiments = load("experiments.json")
    audit = experiments["partition"]
    trial_data = load("monitor-results.json")
    sample = load("review-sample.json")
    labels = load("review-labels.json")
    manifest = ROOT / "grouped-splits.csv"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == audit["manifest_sha256"]
    assert len(trial_data["runs"]) == 3
    models = {}
    for name in ("mc3", "resnet"):
        model = experiments[name]
        assert [run["seed"] for run in model["runs"]] == [41, 42, 43]
        rows = [
            {key: value for key, value in run.items() if key != "test_scores"}
            for run in model["runs"]
        ]
        models[name] = {"runs": rows, "summary": model["summary"]}
    monitor = {"device": trial_data["device"], "runs": []}
    for row in trial_data["runs"]:
        summary = {key: value for key, value in row.items() if key != "observations"}
        delays = row["delays_seconds"]
        summary["delay_mean_seconds"] = mean(delays) if delays else None
        monitor["runs"].append(summary)
    evidence = {
        "audit": {
            k: v
            for k, v in audit.items()
            if k not in ("multi_clip_groups", "nonidentical_visual_edges")
        },
        "visual_edge_count": len(audit["nonidentical_visual_edges"]),
        "multi_clip_group_count": len(audit["multi_clip_groups"]),
        "label_review": load("label-audit.json"),
        "sample": [dict(row, **labels[row["id"]]) for row in sample],
        "models": models,
        "monitor": monitor,
        "replay_events": load("replay-events.json"),
    }
    DEST.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
