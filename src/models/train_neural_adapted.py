"""Adapted neural baseline: same RoBERTa recipe, multi-source training (T2).

Trains RoBERTa-base on HaluEval train PLUS the RAGTruth non-QA train rows
(summarization + data-to-text, 10,056 rows), mirroring the EC-XGB m3 recipe.
Validation stays HaluEval val only, so Platt calibration and early stopping
remain source-only and comparable with the source-only neural baseline.
RAGTruth QA rows never enter training. Three seeds, mixed precision.

Run (repo root, .venv):
  python src/models/train_neural_adapted.py --smoke
  python src/models/train_neural_adapted.py --seeds 42,123,456 --epochs 3
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.models.config import DATA_PROCESSED, RESULTS_DIR
from src.models.train_pipeline import ece

ROOT = Path(__file__).resolve().parents[2]
QA = DATA_PROCESSED / "qa_clean.parquet"
UNIFIED = DATA_PROCESSED / "unified_records.parquet"
EXTERNAL = DATA_PROCESSED / "external_features.parquet"
OUT = RESULTS_DIR / "review" / "neural_adapted"
MAX_LEN = 256
BATCH = 8
GRAD_ACCUM = 2
LR = 2e-5
EPOCHS = 3
PATIENCE = 1


class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": torch.tensor(int(self.labels[i]), dtype=torch.long),
        }


def encode(df, tokenizer):
    texts = [
        f"Question: {q}\nContext: {c if isinstance(c, str) else ''}\nAnswer: {a}"
        for q, c, a in zip(df["question"].fillna(""), df["context"].fillna(""), df["answer"].fillna(""))
    ]
    enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    p_clip = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "ece": float(ece(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "nll": float(-(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)).mean()),
        "predicted_positive_rate": float(pred.mean()),
    }


def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels.extend(batch.pop("labels").cpu().numpy().tolist())
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(**batch).logits
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy().tolist())
    return np.asarray(probs), np.asarray(labels)


def run_seed(seed, tokenizer, data, ext_data, device, epochs, smoke):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = max(1, len(data["train_ids"]) // (BATCH * GRAD_ACCUM))
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")

    train_ds = TextDataset(data["train_ids"], data["train_mask"], data["train_y"])
    val_ds = TextDataset(data["val_ids"], data["val_mask"], data["val_y"])
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    best_f1, best_state, bad = -1.0, None, 0
    history = []
    t0 = time.time()
    epoch_bar = tqdm(range(epochs), desc=f"seed {seed}", unit="epoch", dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        batch_bar = tqdm(train_loader, desc=f"  epoch {epoch + 1}/{epochs}", unit="batch",
                         leave=False, dynamic_ncols=True)
        for step, batch in enumerate(batch_bar, 1):
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = model(**batch, labels=labels).loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            running += loss.item() * GRAD_ACCUM
            if step % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 25 == 0:
                batch_bar.set_postfix(loss=f"{running / step:.4f}")
        p_val, y_val = predict(model, val_loader, device)
        m = metrics(y_val, p_val)
        history.append({"epoch": epoch + 1, "train_loss": running / max(1, len(train_loader)), "val_f1": m["f1"], "val_auroc": m["auroc"]})
        elapsed = time.time() - t0
        print(f"seed {seed} epoch {epoch + 1}/{epochs} ({100 * (epoch + 1) / epochs:.0f}%) | "
              f"val F1 {m['f1']:.4f} AUROC {m['auroc']:.4f} | {elapsed:.0f}s", flush=True)
        epoch_bar.set_postfix(val_f1=f"{m['f1']:.4f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.to(device)

    def scores(df):
        ids, mask = encode(df, tokenizer)
        ds = TextDataset(ids, mask, df["label"].to_numpy())
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
        return predict(model, loader, device)

    p_val, y_val = scores(data["val_df"])
    p_test, y_test = scores(data["test_df"])

    platt = LogisticRegression(max_iter=2000).fit(p_val.reshape(-1, 1), y_val)
    p_test_cal = platt.predict_proba(p_test.reshape(-1, 1))[:, 1]

    result = {
        "seed": seed,
        "epochs_run": len(history),
        "train_seconds": time.time() - t0,
        "history": history,
        "in_domain_raw": metrics(y_test, p_test),
        "in_domain_calibrated": metrics(y_test, p_test_cal),
        "external": {},
    }
    pred_rows = []
    for name, df in ext_data.items():
        p, y = scores(df)
        p_cal = platt.predict_proba(p.reshape(-1, 1))[:, 1]
        result["external"][name] = {"raw": metrics(y, p), "source_calibrated": metrics(y, p_cal)}
        for sid, lab, raw, cal in zip(df["sample_id"], y, p, p_cal):
            pred_rows.append({
                "seed": seed, "dataset": name, "sample_id": sid, "label": int(lab),
                "score_raw": float(raw), "score_calibrated": float(cal),
            })
    for sid, lab, raw, cal in zip(data["test_df"]["sample_id"], y_test, p_test, p_test_cal):
        pred_rows.append({
            "seed": seed, "dataset": "halueval_test", "sample_id": sid, "label": int(lab),
            "score_raw": float(raw), "score_calibrated": float(cal),
        })
    result["predictions"] = pred_rows
    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    smoke = args.smoke
    epochs = 1 if smoke else args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    qa = pd.read_parquet(QA)
    train_df = qa[qa["split"] == "train"].reset_index(drop=True)
    val_df = qa[qa["split"] == "val"].reset_index(drop=True)
    test_df = qa[qa["split"] == "test"].reset_index(drop=True)
    if smoke:
        train_df = train_df.sample(400, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(200, random_state=42).reset_index(drop=True)
        test_df = test_df.sample(200, random_state=42).reset_index(drop=True)

    unified = pd.read_parquet(UNIFIED, columns=["sample_id", "question", "context", "answer"])
    ext = pd.read_parquet(EXTERNAL, columns=["sample_id", "source_dataset", "task", "split", "label"])
    ext = ext.merge(unified, on="sample_id", how="left", validate="one_to_one")
    adapt_train = ext[
        (ext.source_dataset == "ragtruth")
        & (ext.task.isin(["summarization", "data_to_text"]))
        & (ext.split == "train")
    ][["sample_id", "question", "context", "answer", "label"]].reset_index(drop=True)
    assert (ext[(ext.source_dataset == "ragtruth") & (ext.task == "qa") & (ext.split == "train")].shape[0] == 0), \
        "RAGTruth QA must have no train split"
    train_df = pd.concat([train_df[["sample_id", "question", "context", "answer", "label"]], adapt_train],
                         ignore_index=True)
    print(f"adapted train rows: {len(train_df)} (halueval {len(train_df) - len(adapt_train)} + ragtruth non-qa {len(adapt_train)})")
    ext_data = {
        "ragtruth_qa_test": ext[(ext.source_dataset == "ragtruth") & (ext.task == "qa") & (ext.split == "test")].reset_index(drop=True),
        "ragtruth_all_test": ext[(ext.source_dataset == "ragtruth") & (ext.split == "test")].reset_index(drop=True),
        "faithbench": ext[ext.source_dataset == "faithbench"].reset_index(drop=True),
    }
    if smoke:
        ext_data = {k: v.sample(min(100, len(v)), random_state=42).reset_index(drop=True) for k, v in ext_data.items()}

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    data = {
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "train_y": train_df["label"].to_numpy(),
        "val_y": val_df["label"].to_numpy(),
    }
    data["train_ids"], data["train_mask"] = encode(train_df, tokenizer)
    data["val_ids"], data["val_mask"] = encode(val_df, tokenizer)

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        results.append(run_seed(seed, tokenizer, data, ext_data, device, epochs, smoke))

    all_preds = pd.DataFrame([r for res in results for r in res.pop("predictions")])
    all_preds.to_parquet(OUT / "neural_adapted_predictions.parquet", index=False)
    (OUT / "neural_adapted_per_seed.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = []
    for res in results:
        rows.append({"seed": res["seed"], "epochs_run": res["epochs_run"], "train_seconds": res["train_seconds"],
                     **{f"raw_{k}": v for k, v in res["in_domain_raw"].items()},
                     **{f"cal_{k}": v for k, v in res["in_domain_calibrated"].items()}})
    pd.DataFrame(rows).to_csv(OUT / "neural_adapted_per_seed.csv", index=False)

    summary = {"seeds": seeds, "model": "roberta-base", "max_len": MAX_LEN, "batch": BATCH,
               "grad_accum": GRAD_ACCUM, "lr": LR, "epochs": epochs, "smoke": smoke,
               "in_domain_raw": {}, "in_domain_calibrated": {}, "external": {}}
    for key in ("in_domain_raw", "in_domain_calibrated"):
        for metric in ("f1", "auroc", "pr_auc", "mcc", "ece", "brier", "nll"):
            vals = [res[key][metric] for res in results]
            summary[key][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    for name in ext_data:
        summary["external"][name] = {}
        for variant in ("raw", "source_calibrated"):
            for metric in ("f1", "auroc", "predicted_positive_rate", "ece", "brier"):
                vals = [res["external"][name][variant][metric] for res in results]
                summary["external"][name][f"{variant}_{metric}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    (OUT / "neural_adapted_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "in_domain_raw": {k: round(v["mean"], 4) for k, v in summary["in_domain_raw"].items()},
        "in_domain_calibrated": {k: round(v["mean"], 4) for k, v in summary["in_domain_calibrated"].items()},
        "ragtruth_all_raw": {k: round(v["mean"], 4) for k, v in summary["external"]["ragtruth_all_test"].items() if k.startswith("raw_")},
        "saved": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
