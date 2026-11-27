"""Evaluate fixed RLVS-trained heads on external clip benchmarks with perceptual-hash leakage checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import torch
from scipy.fft import dctn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import MC3_18_Weights
from violence_detection.model import build_model, select_device

REVIEW_DIR = Path("artifacts/review")
SEEDS = (41, 42, 43)
SAMPLES = 12


def load_rlvs_reference():
    audit = json.loads(Path("artifacts/metrics/rlvs-inspection.json").read_text())
    records = audit["videos"]
    hashes_list = [
        np.load(REVIEW_DIR / f"features-{i:04}.npz")["hashes"]
        for i in range(0, len(records), 24)
    ]
    rlvs_hashes = np.concatenate(hashes_list)
    packed = np.packbits(rlvs_hashes, axis=-1).view(np.uint64).squeeze(-1)
    sha_to_idx = {r["sha256"]: idx for idx, r in enumerate(records)}
    return records, packed, sha_to_idx


def decode_clip(path: Path):
    with path.open("rb") as handle:
        sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame)
    n = len(frames)
    if n == 0:
        raise ValueError(f"No video frames in {path}")
    img_idx = np.linspace(0, n - 1, 16).astype(int)
    hash_idx = np.linspace(0, n - 1, SAMPLES).astype(int)
    wanted = set(img_idx.tolist()) | set(hash_idx.tolist())
    rgb_map, hash_map = {}, {}
    for idx in sorted(wanted):
        frame = frames[idx]
        if idx in hash_idx:
            grey = frame.reformat(width=32, height=32, format="gray").to_ndarray()
            low = dctn(grey.astype(float), norm="ortho")[:8, :8].ravel()[1:]
            hash_map[idx] = low > np.median(low)
        if idx in img_idx:
            rgb_map[idx] = frame.to_ndarray(format="rgb24")
    raw = torch.from_numpy(np.stack([rgb_map[j] for j in img_idx])).permute(0, 3, 1, 2)
    mc3_t = MC3_18_Weights.KINETICS400_V1.transforms()(raw)
    res_t = ResNet18_Weights.IMAGENET1K_V1.transforms()(raw)
    phash = np.stack([hash_map[j] for j in hash_idx])
    return sha256, n, mc3_t, res_t, phash


def check_rlvs_overlap(sha256: str, phash: np.ndarray, rlvs_records, rlvs_packed, rlvs_sha):
    if sha256 in rlvs_sha:
        idx = rlvs_sha[sha256]
        return "exact", rlvs_records[idx]["path"]
    packed = np.packbits(phash, axis=-1).view(np.uint64).squeeze(-1)
    # Compare (12,) against (N, 12) using exact same rule as review_experiments.py
    distances = np.bitwise_count(packed[:, None] ^ rlvs_packed[:, None, :])  # (N, 12, 12)
    matches = distances <= 8
    candidates = np.flatnonzero((matches.any(1).sum(1) >= 4) & (matches.any(2).sum(1) >= 4))
    for idx in candidates:
        if any(np.diagonal(matches[idx], offset).sum() >= 4 for offset in range(-8, 9)):
            return "visual", rlvs_records[int(idx)]["path"]
    return None, None


def compute_metrics(labels, scores, threshold):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    pred = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) else 0.0
    auc = float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else None
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(pred == labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": specificity,
        "fpr": fpr,
        "roc_auc": auc,
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def collect_rwf2000_val():
    root = Path("data/raw/Violence Fight Detection dataset/RWF-2000/val")
    items = []
    for folder, label in (("NonFight", 0), ("Fight", 1)):
        for p in sorted((root / folder).iterdir()):
            if p.is_file() and not p.name.startswith("."):
                items.append({"path": str(p), "rel": f"val/{folder}/{p.name}", "label": label, "meta": {}})
    return items


def collect_bus_violence():
    root = Path("data/external/bus-violence/extracted")
    v_dir = root / "Violence"
    nv_dir = root / "NoViolence"
    if not v_dir.exists() or not nv_dir.exists():
        raise FileNotFoundError(f"Could not locate Violence/NoViolence under {root}")
    test_names = {
        line.strip() for line in (root / "test.txt").read_text().splitlines() if line.strip()
    } if (root / "test.txt").exists() else set()
    items = []
    for folder, label in ((nv_dir, 0), (v_dir, 1)):
        for p in sorted(folder.rglob("*.mp4")):
            if p.name.startswith("."):
                continue
            items.append({
                "path": str(p),
                "rel": str(p.relative_to(root)),
                "label": label,
                "meta": {"official_split": "test" if p.name in test_names else "train"},
            })
    return items


def collect_airtlab():
    root = Path("data/external/airtlab/violence-detection-dataset")

    def read_action_csv(csv_path: Path):
        mapping = {}
        if csv_path.exists():
            with csv_path.open(newline="") as f:
                reader = csv.reader(f, delimiter=";")
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        clip_name = row[0].strip()
                        actions = [a.strip().lower() for a in row[1].split(",") if a.strip()]
                        mapping[clip_name] = actions
        return mapping

    nv_actions = read_action_csv(root / "nonviolent-action-classes.csv")
    v_actions = read_action_csv(root / "violent-action-classes.csv")

    items = []
    for folder, label in (("non-violent", 0), ("violent", 1)):
        for cam in ("cam1", "cam2"):
            for p in sorted((root / folder / cam).glob("*.mp4")):
                actions = (nv_actions if label == 0 else v_actions).get(p.name, [])
                items.append({
                    "path": str(p),
                    "rel": f"{folder}/{cam}/{p.name}",
                    "label": label,
                    "meta": {"camera": cam, "clip_num": p.stem, "actions": actions},
                })
    return items


def evaluate_dataset(name: str, items: list[dict]):
    out_npz = REVIEW_DIR / f"ext-{name}-features.npz"
    out_meta = REVIEW_DIR / f"ext-{name}-meta.json"
    out_json = REVIEW_DIR / f"ext-{name}-results.json"

    rlvs_records, rlvs_packed, rlvs_sha = load_rlvs_reference()

    if out_npz.exists() and out_meta.exists():
        print(f"[{name}] Loading cached features from {out_npz}", flush=True)
        cached = np.load(out_npz)
        mc3_feats = cached["mc3"]
        res_feats = cached["resnet"]
        meta_records = json.loads(out_meta.read_text())
    else:
        torch.set_num_threads(2)
        torch.hub.set_dir(str(Path("artifacts/torch/hub").resolve()))
        device = select_device()
        mc3 = build_model().to(device).eval()
        mc3.fc = torch.nn.Identity()
        res = resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()
        res.fc = torch.nn.Identity()

        print(f"[{name}] Extracting features for {len(items)} clips on {device}", flush=True)
        mc3_list, res_list, meta_records = [], [], []
        with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
            for start in range(0, len(items), 24):
                batch_items = items[start : start + 24]
                decoded = list(pool.map(lambda it: decode_clip(Path(it["path"])), batch_items))
                for b0 in range(0, len(decoded), 4):
                    chunk = decoded[b0 : b0 + 4]
                    mc3_in = torch.stack([c[2] for c in chunk]).to(device)
                    res_in = torch.stack([c[3] for c in chunk]).to(device)
                    mc3_out = mc3(mc3_in).cpu().numpy()
                    res_out = res(res_in.flatten(0, 1)).reshape(len(chunk), 16, 512).mean(dim=1).cpu().numpy()
                    mc3_list.append(mc3_out)
                    res_list.append(res_out)
                for it, (sha256, n_frames, _, _, phash) in zip(batch_items, decoded):
                    ov_type, ov_path = check_rlvs_overlap(sha256, phash, rlvs_records, rlvs_packed, rlvs_sha)
                    meta_records.append({
                        "rel": it["rel"],
                        "label": it["label"],
                        "sha256": sha256,
                        "frames": n_frames,
                        "rlvs_overlap_type": ov_type,
                        "rlvs_overlap_path": ov_path,
                        "meta": it["meta"],
                    })
                if start % 120 == 0 or start + 24 >= len(items):
                    print(f"[{name}] {min(start + 24, len(items))}/{len(items)}", flush=True)
        mc3_feats = np.concatenate(mc3_list, axis=0)
        res_feats = np.concatenate(res_list, axis=0)
        np.savez(out_npz, mc3=mc3_feats, resnet=res_feats)
        out_meta.write_text(json.dumps(meta_records, indent=2))

    # Load RLVS validation-selected thresholds from experiments.json (never tuned on external test!)
    experiments = json.loads((REVIEW_DIR / "experiments.json").read_text())
    val_thresholds = {
        arch: {r["seed"]: r["validation_selected"]["threshold"] for r in experiments[arch]["runs"]}
        for arch in ("mc3", "resnet")
    }

    labels_all = np.array([r["label"] for r in meta_records])
    clean_mask = np.array([r["rlvs_overlap_type"] is None for r in meta_records])
    exact_ov = [r for r in meta_records if r["rlvs_overlap_type"] == "exact"]
    visual_ov = [r for r in meta_records if r["rlvs_overlap_type"] == "visual"]

    result = {
        "dataset": name,
        "total_clips": len(meta_records),
        "class_counts_all": [int((labels_all == 0).sum()), int((labels_all == 1).sum())],
        "exact_rlvs_overlaps": len(exact_ov),
        "visual_rlvs_overlaps": len(visual_ov),
        "overlap_examples": [
            {"rel": r["rel"], "type": r["rlvs_overlap_type"], "rlvs_path": r["rlvs_overlap_path"]}
            for r in (exact_ov + visual_ov)[:20]
        ],
        "clean_clips": int(clean_mask.sum()),
        "class_counts_clean": [
            int(((labels_all == 0) & clean_mask).sum()),
            int(((labels_all == 1) & clean_mask).sum()),
        ],
        "models": {},
    }

    for arch, feats_np in (("mc3", mc3_feats), ("resnet", res_feats)):
        feats = torch.from_numpy(feats_np)
        runs = []
        for seed in SEEDS:
            head = torch.nn.Linear(512, 2)
            head.load_state_dict(torch.load(REVIEW_DIR / f"{arch}-head-{seed}.pt", weights_only=True))
            head.eval()
            with torch.inference_mode():
                scores = head(feats).softmax(dim=1)[:, 1].numpy()
            t_val = val_thresholds[arch][seed]
            run_entry = {
                "seed": seed,
                "rlvs_val_threshold": t_val,
                "all_default": compute_metrics(labels_all, scores, 0.50),
                "all_val_selected": compute_metrics(labels_all, scores, t_val),
                "clean_default": compute_metrics(labels_all[clean_mask], scores[clean_mask], 0.50),
                "clean_val_selected": compute_metrics(labels_all[clean_mask], scores[clean_mask], t_val),
            }
            if name == "airtlab" and arch == "mc3" and seed == 42:
                # Per-action false-positive rate breakdown for non-violent lookalikes
                action_stats = {}
                for idx, rec in enumerate(meta_records):
                    if rec["label"] == 0:
                        for act in rec["meta"].get("actions", []):
                            st = action_stats.setdefault(act, {"clips": 0, "fp_050": 0, "scores": []})
                            st["clips"] += 1
                            st["fp_050"] += int(scores[idx] >= 0.50)
                            st["scores"].append(float(scores[idx]))
                run_entry["nonviolent_action_fp"] = {
                    act: {
                        "clips": st["clips"],
                        "fp_050": st["fp_050"],
                        "fpr_050": st["fp_050"] / st["clips"],
                        "mean_score": float(np.mean(st["scores"])),
                    }
                    for act, st in sorted(action_stats.items())
                }
            if name == "bus_violence" and arch == "mc3" and seed == 42:
                split_stats = {}
                for idx, rec in enumerate(meta_records):
                    sp = rec["meta"].get("official_split", "unknown")
                    split_stats.setdefault(sp, []).append((rec["label"], float(scores[idx])))
                run_entry["by_official_split_050"] = {
                    sp: compute_metrics([x[0] for x in vals], [x[1] for x in vals], 0.50)
                    for sp, vals in sorted(split_stats.items())
                    if len(vals) >= 10
                }
            runs.append(run_entry)
        summary = {}
        for policy in ("all_default", "all_val_selected", "clean_default", "clean_val_selected"):
            summary[policy] = {}
            for key in ("accuracy", "precision", "recall", "f1", "specificity", "fpr", "roc_auc"):
                vals = [r[policy][key] for r in runs if r[policy][key] is not None]
                if vals:
                    summary[policy][key] = {
                        "mean": float(np.mean(vals)),
                        "sd": float(np.std(vals, ddof=1)),
                    }
        result["models"][arch] = {"runs": runs, "summary": summary}

    out_json.write_text(json.dumps(result, indent=2))
    print(f"\n=== {name.upper()} SUMMARY ===", flush=True)
    print(f"Total: {result['total_clips']} | Clean: {result['clean_clips']} (exact_ov={result['exact_rlvs_overlaps']}, visual_ov={result['visual_rlvs_overlaps']})", flush=True)
    for arch in ("mc3", "resnet"):
        s = result["models"][arch]["summary"]["clean_default"]
        print(
            f"  {arch:6s} @0.50 (clean): acc={s['accuracy']['mean']*100:.2f}+/-{s['accuracy']['sd']*100:.2f}% "
            f"prec={s['precision']['mean']*100:.2f}% rec={s['recall']['mean']*100:.2f}% "
            f"f1={s['f1']['mean']*100:.2f}+/-{s['f1']['sd']*100:.2f}% auc={s['roc_auc']['mean']:.4f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("rwf2000_val", "bus_violence", "airtlab"))
    args = parser.parse_args()
    collectors = {
        "rwf2000_val": collect_rwf2000_val,
        "bus_violence": collect_bus_violence,
        "airtlab": collect_airtlab,
    }
    items = collectors[args.dataset]()
    evaluate_dataset(args.dataset, items)


if __name__ == "__main__":
    main()
