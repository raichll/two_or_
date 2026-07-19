import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from env_inspection_pipeline import label_to_int, parse_float, parse_dt, read_json


@dataclass
class RiskExample:
    event_id: str
    event_date: str
    text: str
    label: int


class RiskDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        encoded = self.tokenizer(
            item.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(item.label, dtype=torch.long)
        encoded["event_id"] = item.event_id
        encoded["event_date"] = item.event_date
        return encoded


def ensure_dir(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_name(day_text):
    dt = parse_dt(day_text)
    if not dt:
        return None
    d = dt.date()
    if date(2024, 1, 1) <= d <= date(2024, 8, 31):
        return "train"
    if date(2024, 9, 1) <= d <= date(2024, 10, 31):
        return "calibration"
    if date(2024, 11, 1) <= d <= date(2024, 12, 31):
        return "test"
    return None


def clean_text(value):
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    return text.replace("\n", " ").replace("\r", " ")


def leakage_flag(row, cfg):
    keywords = cfg.get("leakage_keywords", [])
    for col in cfg.get("leakage_sensitive_text_columns", []):
        text = clean_text(row.get(col))
        if any(keyword and keyword in text for keyword in keywords):
            return True
    return False


def coordinate_tokens(row, cfg):
    lon = parse_float(row.get(cfg["longitude_column"]))
    lat = parse_float(row.get(cfg["latitude_column"]))
    tokens = []
    if lon is not None:
        tokens.append(f"longitude_bin_{int(lon * 10)}")
    if lat is not None:
        tokens.append(f"latitude_bin_{int(lat * 10)}")
    return " ".join(tokens)


def compose_text(row, cfg):
    parts = []
    for col in cfg["pre_event_text_columns"]:
        value = clean_text(row.get(col))
        if value:
            parts.append(f"{col}: {value}")
    event_desc = clean_text(row.get("事件描述"))
    if event_desc and not leakage_flag(row, cfg):
        parts.append(f"事件描述: {event_desc}")
    coord = coordinate_tokens(row, cfg)
    if coord:
        parts.append(coord)
    return " [SEP] ".join(parts)


def load_examples(config_path, data_path, max_train=0, seed=2026):
    cfg = read_json(config_path)
    df = pd.read_csv(data_path or cfg["data_file"], encoding="utf-8-sig", dtype=str).fillna("")
    examples = []
    split_counts = {"train": 0, "calibration": 0, "test": 0}
    label_counts = {s: {0: 0, 1: 0} for s in split_counts}
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        label = label_to_int(row_dict.get(cfg["label_column"]))
        if label is None:
            continue
        day_text = clean_text(row_dict.get(cfg["date_column"]))
        split = split_name(day_text)
        if split is None:
            continue
        text = compose_text(row_dict, cfg)
        if not text:
            continue
        event_id = clean_text(row_dict.get(cfg["id_column"]))
        if not event_id:
            continue
        examples.append((split, RiskExample(event_id=event_id, event_date=day_text, text=text, label=label)))
        split_counts[split] += 1
        label_counts[split][label] += 1

    if max_train and max_train > 0:
        rng = random.Random(seed)
        train = [item for item in examples if item[0] == "train"]
        rest = [item for item in examples if item[0] != "train"]
        positives = [item for item in train if item[1].label == 1]
        negatives = [item for item in train if item[1].label == 0]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        positive_target = min(len(positives), max(1, int(max_train * 0.25)))
        negative_target = min(len(negatives), max_train - positive_target)
        train = positives[:positive_target] + negatives[:negative_target]
        rng.shuffle(train)
        examples = train + rest

    by_split = {"train": [], "calibration": [], "test": []}
    for split, example in examples:
        by_split[split].append(example)
    metadata = {
        "split_counts": {k: len(v) for k, v in by_split.items()},
        "label_counts": {
            split: {
                "negative": sum(1 for item in items if item.label == 0),
                "positive": sum(1 for item in items if item.label == 1),
            }
            for split, items in by_split.items()
        },
        "max_train_used": max_train,
    }
    return cfg, by_split, metadata


def train_epoch(model, loader, optimizer, scheduler, device, pos_weight=None):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="training", leave=False):
        batch.pop("event_id")
        batch.pop("event_date")
        labels = batch["labels"].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**inputs).logits
        if pos_weight is None:
            loss = F.cross_entropy(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels, weight=pos_weight)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else None


def predict_logits(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predicting", leave=False):
            event_ids = list(batch.pop("event_id"))
            event_dates = list(batch.pop("event_date"))
            labels = batch.pop("labels").numpy().tolist()
            inputs = {k: v.to(device) for k, v in batch.items()}
            logits = model(**inputs).logits.detach().cpu().numpy()
            for event_id, event_date, label, logit in zip(event_ids, event_dates, labels, logits):
                rows.append(
                    {
                        "event_id": event_id,
                        "date": event_date,
                        "label": int(label),
                        "logit0": float(logit[0]),
                        "logit1": float(logit[1]),
                        "margin": float(logit[1] - logit[0]),
                    }
                )
    return rows


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def add_probabilities(rows, temperature, bias):
    for row in rows:
        raw = sigmoid(row["margin"])
        calibrated = sigmoid((row["margin"] + bias) / temperature)
        row["risk_raw"] = raw
        row["risk"] = calibrated
    return rows


def conformal_quantile(residuals, nominal_coverage):
    clean = sorted(float(v) for v in residuals if v is not None)
    if not clean:
        return 1.0
    index = int(math.ceil((len(clean) + 1) * nominal_coverage)) - 1
    index = max(0, min(index, len(clean) - 1))
    return float(clean[index])


def risk_bin_edges(rows, bin_count):
    probs = np.array([row["risk"] for row in rows], dtype=float)
    if bin_count <= 1 or len(probs) < bin_count * 20:
        return [0.0, 1.0]
    raw = np.quantile(probs, np.linspace(0.0, 1.0, bin_count + 1)).tolist()
    edges = [0.0]
    for value in raw[1:-1]:
        value = float(value)
        if value > edges[-1] + 1e-8:
            edges.append(value)
    edges.append(1.0)
    return edges


def bin_index(probability, edges):
    for idx in range(len(edges) - 1):
        left, right = edges[idx], edges[idx + 1]
        if idx == len(edges) - 2:
            if left <= probability <= right:
                return idx
        elif left <= probability < right:
            return idx
    return max(0, len(edges) - 2)


def fit_conformal_model(calibration_rows, nominal_coverage=0.90, bin_count=5):
    edges = risk_bin_edges(calibration_rows, bin_count)
    global_residuals = [abs(row["label"] - row["risk"]) for row in calibration_rows]
    global_radius = conformal_quantile(global_residuals, nominal_coverage)
    radii = {}
    counts = {}
    for idx in range(len(edges) - 1):
        residuals = [
            abs(row["label"] - row["risk"])
            for row in calibration_rows
            if bin_index(row["risk"], edges) == idx
        ]
        counts[idx] = len(residuals)
        radii[idx] = conformal_quantile(residuals, nominal_coverage) if residuals else global_radius
    return {
        "nominal_coverage": nominal_coverage,
        "bin_edges": edges,
        "global_radius": global_radius,
        "bin_radii": radii,
        "bin_counts": counts,
    }


def apply_conformal_model(rows, model):
    edges = model["bin_edges"]
    radii = model["bin_radii"]
    for row in rows:
        p = max(0.0, min(1.0, float(row["risk"])))
        b = bin_index(p, edges)
        radius = max(0.0, min(1.0, float(radii.get(b, model["global_radius"]))))
        lower = max(0.0, p - radius)
        upper = min(1.0, p + radius)
        row["conformal_bin"] = b
        row["conformal_radius"] = radius
        row["conformal_lower"] = lower
        row["conformal_upper"] = upper
        # The robust TOP uses downside reward deviation; rule-priority is deterministic.
        row["uncertainty_radius"] = p - lower
    return rows


def conformal_coverage(rows, model):
    covered = []
    widths = []
    downside = []
    for row in rows:
        p = max(0.0, min(1.0, float(row["risk"])))
        b = bin_index(p, model["bin_edges"])
        radius = max(0.0, min(1.0, float(model["bin_radii"].get(b, model["global_radius"]))))
        lower = max(0.0, p - radius)
        upper = min(1.0, p + radius)
        y = float(row["label"])
        covered.append(1.0 if lower <= y <= upper else 0.0)
        widths.append(upper - lower)
        downside.append(p - lower)
    return {
        "n": len(rows),
        "positives": int(sum(row["label"] for row in rows)),
        "nominal_coverage": model["nominal_coverage"],
        "empirical_coverage": float(np.mean(covered)) if covered else None,
        "mean_interval_width": float(np.mean(widths)) if widths else None,
        "mean_downside_radius": float(np.mean(downside)) if downside else None,
    }


def fit_temperature_bias(rows):
    labels = np.array([row["label"] for row in rows], dtype=float)
    margins = np.array([row["margin"] for row in rows], dtype=float)

    def nll(params):
        log_temperature, bias = params
        temperature = max(float(np.exp(log_temperature)), 1e-3)
        probs = 1.0 / (1.0 + np.exp(-(margins + bias) / temperature))
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        return -float(np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))

    res = minimize(nll, x0=np.array([0.0, 0.0]), method="Nelder-Mead")
    return float(np.exp(res.x[0])), float(res.x[1])


def expected_calibration_error(labels, probs, bins=10):
    labels = np.array(labels, dtype=float)
    probs = np.array(probs, dtype=float)
    ece = 0.0
    bin_rows = []
    for b in range(bins):
        left = b / bins
        right = (b + 1) / bins
        if b == bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        count = int(mask.sum())
        if count == 0:
            bin_rows.append({"bin": b, "left": left, "right": right, "count": 0, "confidence": None, "accuracy": None})
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += count / len(labels) * abs(confidence - accuracy)
        bin_rows.append(
            {
                "bin": b,
                "left": left,
                "right": right,
                "count": count,
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    return float(ece), bin_rows


def topk_recall(labels, probs, k):
    order = np.argsort(-np.array(probs, dtype=float))[: min(k, len(probs))]
    positives = max(1, int(np.sum(labels)))
    return float(np.sum(np.array(labels)[order]) / positives)


def best_f1_threshold(rows, probability_key="risk"):
    labels = np.array([row["label"] for row in rows])
    probs = np.array([row[probability_key] for row in rows])
    best = (0.0, 0.5)
    for threshold in np.linspace(0.02, 0.80, 157):
        preds = (probs >= threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1]


def metrics_for(rows, probability_key="risk", threshold=0.5):
    labels = [row["label"] for row in rows]
    probs = [row[probability_key] for row in rows]
    preds = [1 if p >= threshold else 0 for p in probs]
    ece, bins = expected_calibration_error(labels, probs, bins=10)
    return {
        "n": len(rows),
        "positives": int(sum(labels)),
        "threshold": threshold,
        "auc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else None,
        "pr_auc": average_precision_score(labels, probs) if len(set(labels)) > 1 else None,
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "brier": brier_score_loss(labels, probs),
        "ece": ece,
        "top20_recall": topk_recall(labels, probs, 20),
        "top50_recall": topk_recall(labels, probs, 50),
        "top100_recall": topk_recall(labels, probs, 100),
        "calibration_bins": bins,
    }


def write_predictions(path, rows, id_column):
    ensure_dir(path)
    fieldnames = [
        id_column,
        "date",
        "split",
        "label",
        "risk_raw",
        "risk",
        "conformal_bin",
        "conformal_radius",
        "conformal_lower",
        "conformal_upper",
        "uncertainty_radius",
        "logit0",
        "logit1",
        "margin",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    id_column: row["event_id"],
                    "date": row["date"],
                    "split": row["split"],
                    "label": row["label"],
                    "risk_raw": round(row["risk_raw"], 8),
                    "risk": round(row["risk"], 8),
                    "conformal_bin": row.get("conformal_bin"),
                    "conformal_radius": round(row.get("conformal_radius", 0.0), 8),
                    "conformal_lower": round(row.get("conformal_lower", 0.0), 8),
                    "conformal_upper": round(row.get("conformal_upper", 1.0), 8),
                    "uncertainty_radius": round(row["uncertainty_radius"], 8),
                    "logit0": round(row["logit0"], 8),
                    "logit1": round(row["logit1"], 8),
                    "margin": round(row["margin"], 8),
                }
            )


def write_metric_csv(path, metrics):
    ensure_dir(path)
    rows = []
    for split, data in metrics["splits"].items():
        for prob_key, label in (("raw", "uncalibrated"), ("calibrated", "temperature_calibrated")):
            item = data[prob_key]
            rows.append(
                {
                    "split": split,
                    "probability": label,
                    "n": item["n"],
                    "positives": item["positives"],
                    "threshold": item["threshold"],
                    "auc": item["auc"],
                    "pr_auc": item["pr_auc"],
                    "precision": item["precision"],
                    "recall": item["recall"],
                    "f1": item["f1"],
                    "brier": item["brier"],
                    "ece": item["ece"],
                    "top20_recall": item["top20_recall"],
                    "top50_recall": item["top50_recall"],
                    "top100_recall": item["top100_recall"],
                }
            )
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_conformal_csv(path, metrics):
    ensure_dir(path)
    rows = []
    for level_key, data in metrics["conformal"].items():
        for split, item in data["coverage"].items():
            rows.append(
                {
                    "nominal_coverage": item["nominal_coverage"],
                    "split": split,
                    "n": item["n"],
                    "positives": item["positives"],
                    "empirical_coverage": item["empirical_coverage"],
                    "mean_interval_width": item["mean_interval_width"],
                    "mean_downside_radius": item["mean_downside_radius"],
                    "global_radius": data["model"]["global_radius"],
                    "bin_edges": "|".join(f"{edge:.6f}" for edge in data["model"]["bin_edges"]),
                    "bin_radii": "|".join(
                        f"{idx}:{radius:.6f}" for idx, radius in sorted(data["model"]["bin_radii"].items())
                    ),
                }
            )
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Train and calibrate a chronological BERT risk model for inspection dispatch.")
    parser.add_argument("--config", default="config/defaults.json")
    parser.add_argument("--data")
    parser.add_argument("--model-name", default="uer/chinese_roberta_L-2_H-128")
    parser.add_argument("--output-dir", default="outputs/bert_calibrated_risk_model")
    parser.add_argument(
        "--predictions",
        default="outputs/risk_predictions_bert_chrono_public.csv",
    )
    parser.add_argument("--metrics-json", default="outputs/risk_model_bert_chrono_metrics.json")
    parser.add_argument("--metrics-csv", default="outputs/risk_model_bert_chrono_metrics.csv")
    parser.add_argument("--conformal-csv", default="outputs/risk_model_conformal_coverage.csv")
    parser.add_argument("--conformal-levels", default="0.80,0.90,0.95")
    parser.add_argument("--conformal-bins", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=0, help="0 uses all chronological training examples.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-save", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg, by_split, metadata = load_examples(args.config, args.data, args.max_train, args.seed)
    if not by_split["train"] or not by_split["calibration"] or not by_split["test"]:
        raise RuntimeError(f"Chronological split is empty: {metadata}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loaders = {
        split: DataLoader(RiskDataset(items, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=(split == "train"))
        for split, items in by_split.items()
    }
    train_labels = [ex.label for ex in by_split["train"]]
    neg = max(1, sum(1 for label in train_labels if label == 0))
    pos = max(1, sum(1 for label in train_labels if label == 1))
    pos_weight = torch.tensor([1.0, neg / pos], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, len(loaders["train"]) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    losses = []
    for epoch in range(args.epochs):
        loss = train_epoch(model, loaders["train"], optimizer, scheduler, device, pos_weight=pos_weight)
        losses.append(loss)
        print(json.dumps({"epoch": epoch + 1, "train_loss": loss}, ensure_ascii=False))

    if not args.skip_save:
        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer.save_pretrained(args.output_dir)
        model.save_pretrained(args.output_dir, safe_serialization=False)

    all_rows = []
    split_rows = {}
    for split, loader in loaders.items():
        rows = predict_logits(model, loader, device)
        for row in rows:
            row["split"] = split
        split_rows[split] = rows
        all_rows.extend(rows)

    temperature, calibration_bias = fit_temperature_bias(split_rows["calibration"])
    for rows in split_rows.values():
        add_probabilities(rows, temperature, calibration_bias)
    add_probabilities(all_rows, temperature, calibration_bias)

    conformal_levels = [float(x.strip()) for x in args.conformal_levels.split(",") if x.strip()]
    conformal_models = {
        level: fit_conformal_model(split_rows["calibration"], nominal_coverage=level, bin_count=args.conformal_bins)
        for level in conformal_levels
    }
    primary_level = min(conformal_levels, key=lambda level: abs(level - 0.90)) if conformal_levels else 0.90
    primary_model = conformal_models[primary_level]
    for rows in split_rows.values():
        apply_conformal_model(rows, primary_model)
    apply_conformal_model(all_rows, primary_model)

    metrics = {
        "model_name": args.model_name,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "temperature": temperature,
        "calibration_bias": calibration_bias,
        "metadata": metadata,
        "train_losses": losses,
        "splits": {},
        "conformal_primary_level": primary_level,
        "conformal": {},
    }
    raw_threshold = best_f1_threshold(split_rows["calibration"], probability_key="risk_raw")
    calibrated_threshold = best_f1_threshold(split_rows["calibration"], probability_key="risk")
    metrics["raw_threshold"] = raw_threshold
    metrics["calibrated_threshold"] = calibrated_threshold
    for split, rows in split_rows.items():
        metrics["splits"][split] = {
            "raw": metrics_for(rows, probability_key="risk_raw", threshold=raw_threshold),
            "calibrated": metrics_for(rows, probability_key="risk", threshold=calibrated_threshold),
        }
    for level, model_data in conformal_models.items():
        metrics["conformal"][f"{level:.2f}"] = {
            "model": model_data,
            "coverage": {split: conformal_coverage(rows, model_data) for split, rows in split_rows.items()},
        }

    ensure_dir(args.metrics_json)
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_metric_csv(args.metrics_csv, metrics)
    write_conformal_csv(args.conformal_csv, metrics)
    write_predictions(args.predictions, all_rows, cfg["id_column"])
    print(
        json.dumps(
            {
                "metrics": args.metrics_json,
                "predictions": args.predictions,
                "temperature": temperature,
                "calibration_bias": calibration_bias,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
