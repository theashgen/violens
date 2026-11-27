"""Group visually related RLVS clips and compare two frozen backbones."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

OUTPUT = Path("artifacts/review")
AUDIT = Path("artifacts/metrics/rlvs-inspection.json")
SEEDS = (41, 42, 43)
EPOCHS = 20
BACKBONES = ("mc3", "resnet")
THRESHOLDS = np.linspace(0.05, 0.95, 19).round(2)


def load_features(prefix, size, key):
    return np.concatenate(
        [np.load(OUTPUT / f"{prefix}-{i:04}.npz")[key] for i in range(0, size, 24)]
    )


def grouped_partition(records, fingerprints):
    # Pack each 63-bit frame signature into one integer to compare Hamming distance.
    hashes = np.packbits(fingerprints, axis=-1).view(np.uint64).squeeze(-1)
    parents = list(range(len(records)))

    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i

    def join(i, j):
        before, after = sorted([root(i), root(j)])
        parents[after] = before

    exact = {}
    groups = {}
    edges = []
    for i, record in enumerate(records):
        key = record["sha256"]
        if key in exact:
            join(i, exact[key])
        exact[key] = i
        if i == len(records) - 1:
            continue
        distances = np.bitwise_count(hashes[i, :, None] ^ hashes[i + 1 :, None, :])
        matches = distances <= 8
        # Require four distinct samples on each side plus four aligned samples
        # along one temporal offset. Offsets can pick up partially trimmed copies.
        candidates = np.flatnonzero((matches.any(1).sum(1) >= 4) & (matches.any(2).sum(1) >= 4))
        for k in candidates:
            if any(np.diagonal(matches[k], offset).sum() >= 4 for offset in range(-8, 9)):
                j = i + 1 + int(k)
                if records[j]["sha256"] != key:
                    edges.append([records[i]["path"], records[j]["path"]])
                join(i, j)

    for i in range(len(records)):
        groups.setdefault(root(i), []).append(i)
    excluded = set()
    kept = []
    for members in groups.values():
        if len({records[i]["label"] for i in members}) > 1:
            excluded.update(members)
            continue
        retained_hashes = set()
        for i in members:
            digest = records[i]["sha256"]
            if digest not in retained_hashes:
                kept.append(i)
                retained_hashes.add(digest)
    kept = np.array(sorted(kept))
    group_ids = np.array([root(i) for i in kept])
    labels = np.array([records[i]["label"] for i in kept])
    folds = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(kept, labels, group_ids)
    )
    masks = {"val": kept[folds[0][1]], "test": kept[folds[1][1]]}
    masks["train"] = np.setdiff1d(kept, np.concatenate(list(masks.values())))
    assert not (set(masks["val"]) & set(masks["test"]))
    seen = {}
    manifest = OUTPUT / "grouped-splits.csv"
    with manifest.open("w", newline="") as handle:
        fields = ("path", "label", "split", "sha256", "group")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, indices in masks.items():
            for i in indices:
                group = f"visual-{root(i):04}"
                assert seen.setdefault(group, split) == split
                writer.writerow(
                    dict(
                        **{key: records[i][key] for key in ("path", "label", "sha256")},
                        split=split,
                        group=group,
                    )
                )
    audit = {
        "usable": len(records),
        "retained": len(kept),
        "distinct_groups": len(seen),
        "nonidentical_visual_edges": edges,
        "multi_clip_groups": [
            [records[i]["path"] for i in m] for m in groups.values() if len(m) > 1
        ],
        "excluded_conflicting_groups": sorted(records[i]["path"] for i in excluded),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "partitions": {
            split: {
                "clips": len(ids),
                "groups": len({root(i) for i in ids}),
                "class_counts": np.bincount(
                    [records[i]["label"] for i in ids], minlength=2
                ).tolist(),
            }
            for split, ids in masks.items()
        },
    }
    (OUTPUT / "group-audit.json").write_text(json.dumps(audit, indent=2))
    return masks, audit


def metrics(labels, scores, threshold):
    pred = np.asarray(scores) >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(np.mean(pred == labels)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
    }


def train_head(features, labels, masks, seed):
    torch.manual_seed(seed)
    head = nn.Linear(features.shape[1], 2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    rng = torch.Generator().manual_seed(seed)
    train, val = (torch.as_tensor(masks[k]) for k in ("train", "val"))
    loss_fn = nn.CrossEntropyLoss()
    history, best = [], float("inf")
    for epoch in range(1, EPOCHS + 1):
        tally = 0
        for indices in train[torch.randperm(len(train), generator=rng)].split(32):
            optimizer.zero_grad()
            loss = loss_fn(head(features[indices]), labels[indices])
            loss.backward()
            optimizer.step()
            tally += loss.item() * len(indices)
        with torch.no_grad():
            val_loss = loss_fn(head(features[val]), labels[val]).item()
        history.append({"epoch": epoch, "train_loss": tally / len(train), "val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            selected = epoch
            weights = {key: value.detach().clone() for key, value in head.state_dict().items()}
    head.load_state_dict(weights)
    with torch.no_grad():
        val_scores = head(features[val]).softmax(dim=1)[:, 1].numpy()
    sweep = [
        dict(threshold=float(t), **metrics(labels[val].numpy(), val_scores, t)) for t in THRESHOLDS
    ]
    # Choose for clip-level F1 using validation only. Prefer higher precision on ties.
    winner = max(sweep, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    test = masks["test"]
    with torch.no_grad():
        test_scores = head(features[test]).softmax(dim=1)[:, 1].numpy()
    return dict(
        seed=seed,
        selected_epoch=selected,
        history=history,
        validation_sweep=sweep,
        validation_selected=winner,
        test_default=metrics(labels[test].numpy(), test_scores, 0.5),
        test_selected=metrics(labels[test].numpy(), test_scores, winner["threshold"]),
        test_scores=test_scores.tolist(),
    ), weights


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-only", action="store_true")
    args = parser.parse_args()
    records = json.loads(AUDIT.read_text())["videos"]
    masks, audit = grouped_partition(records, load_features("features", len(records), "hashes"))
    print(audit["partitions"], flush=True)
    if args.split_only:
        return
    torch.set_num_threads(2)
    labels = torch.tensor([r["label"] for r in records])
    results = {"partition": audit, "test_paths": [records[i]["path"] for i in masks["test"]]}
    for architecture in BACKBONES:
        prefix = "features" if architecture == "mc3" else "resnet"
        features = torch.from_numpy(load_features(prefix, len(records), "features"))
        runs = []
        for seed in SEEDS:
            record, weights = train_head(features, labels, masks, seed)
            torch.save(weights, OUTPUT / f"{architecture}-head-{seed}.pt")
            runs.append(record)
        results[architecture] = {"runs": runs, "summary": {}}
        for policy in ("test_default", "test_selected"):
            results[architecture]["summary"][policy] = {}
            for key in ("accuracy", "precision", "recall", "f1"):
                values = [row[policy][key] for row in runs]
                results[architecture]["summary"][policy][key] = {
                    "mean": np.mean(values),
                    "sd": np.std(values, ddof=1),
                }
        print(architecture, results[architecture]["summary"], flush=True)
        (OUTPUT / "experiments.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
