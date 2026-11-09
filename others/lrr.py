"""Repository LRR scoring and one-vs-rest evaluation."""

if __name__ == "__main__":
    from baseline_cli import run_single

    run_single("lrr")
    raise SystemExit(0)
import json
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")


class UserConfig:
    MODEL_PATH = "models/llama-7b"
    DATASETS = [
        {"name": "TT_xlnet_large", "test_path": "data/test.json", "max_samples": None}
    ]
    OUTPUT_DIR = "outputs"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_TOKEN_LENGTH = 512


class LRRFeatureExtractor:

    def __init__(self):
        self.device = UserConfig.DEVICE
        print(f"[Init] Loading Model from: {UserConfig.MODEL_PATH} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            UserConfig.MODEL_PATH, use_fast=False, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            UserConfig.MODEL_PATH,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def _compute_lrr_score(self, logits, input_ids):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        sorted_indices = shift_logits.argsort(dim=-1, descending=True)
        matches = sorted_indices == shift_labels.unsqueeze(-1)
        ranks = matches.float().argmax(dim=-1) + 1
        log_ranks = torch.log(ranks.float())
        mask = shift_labels != self.tokenizer.pad_token_id
        if mask.sum().item() > 0:
            avg_log_prob = token_log_probs[mask].mean().item()
            avg_log_rank = log_ranks[mask].mean().item()
            nll = -avg_log_prob
            return nll / (avg_log_rank + 1e-08)
        return 0.0

    def extract_batch(self, text_list, desc="Extracting"):
        records = []
        for text in tqdm(text_list, desc=desc):
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=UserConfig.MAX_TOKEN_LENGTH,
            ).to(self.device)
            if inputs.input_ids.size(1) < 10:
                continue
            with torch.no_grad():
                logits = self.model(**inputs).logits
                lrr_score = self._compute_lrr_score(logits, inputs.input_ids)
            records.append({"lrr_score": lrr_score})
        return records


def load_and_process_test_data(json_path, max_samples, extractor, cache_path):
    if os.path.exists(cache_path):
        print(f"[Data] Loading cached features from {cache_path}")
        return pd.read_csv(cache_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_records = []
    for label, texts in data.items():
        selected_texts = texts[:max_samples] if max_samples else texts
        print(f"Processing {label}: {len(selected_texts)} samples")
        features = extractor.extract_batch(selected_texts, desc=f"[{label}]")
        for feat in features:
            all_records.append({"label": label, "lrr_score": feat["lrr_score"]})
    df = pd.DataFrame(all_records)
    df.to_csv(cache_path, index=False)
    return df


def save_report_as_image(report_dict, dataset_name):
    """Render the one-vs-rest evaluation report as an image."""
    df_report = pd.DataFrame(
        {key: value for key, value in report_dict.items() if isinstance(value, dict)}
    ).transpose()
    fig, ax = plt.subplots(figsize=(8, len(df_report) * 0.6 + 2))
    ax.axis("off")
    tbl = ax.table(
        cellText=df_report.values.round(4),
        colLabels=df_report.columns,
        rowLabels=df_report.index,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.2)
    plt.title(f"Zero-Shot OvR Report: {dataset_name}", fontsize=14, pad=20)
    img_path = os.path.join(UserConfig.OUTPUT_DIR, f"{dataset_name}_ovr_report.png")
    plt.savefig(img_path, bbox_inches="tight", dpi=300)
    print(f"📊 Report image saved: {img_path}")
    plt.close()


def main():
    need_model = False
    for ds_cfg in UserConfig.DATASETS:
        cache_path = os.path.join(
            UserConfig.OUTPUT_DIR, f"{ds_cfg['name']}_all_samples_test_feat.csv"
        )
        if not os.path.exists(cache_path):
            need_model = True
            break
    extractor = LRRFeatureExtractor() if need_model else None
    for ds_cfg in UserConfig.DATASETS:
        name = ds_cfg["name"]
        print(f"\n{'#' * 40}\n# Task: {name} (Zero-Shot OvR)\n{'#' * 40}")
        cache_path = os.path.join(
            UserConfig.OUTPUT_DIR, f"{name}_all_samples_test_feat.csv"
        )
        df_test = load_and_process_test_data(
            ds_cfg["test_path"], ds_cfg["max_samples"], extractor, cache_path
        )
        classes = df_test["label"].unique()
        y_test = df_test["label"].astype(str).values
        scores = df_test["lrr_score"].values
        ovr_metrics = {}
        auc_list, acc_list, f1_list = ([], [], [])
        print(
            f"\n{'Class (Binary OvR)':<25} | {'AUC':<10} | {'Binary ACC':<10} | {'Binary F1':<10}"
        )
        print("-" * 70)
        for cls in classes:
            y_true_bin = (y_test == cls).astype(int)
            current_scores = -scores if "human" in str(cls).lower() else scores
            try:
                auc = roc_auc_score(y_true_bin, current_scores)
                thresh = np.median(current_scores)
                y_pred_bin = (current_scores >= thresh).astype(int)
                acc = accuracy_score(y_true_bin, y_pred_bin)
                f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            except Exception as e:
                print(f"[Warning] Failed for {cls}: {e}")
                auc, acc, f1 = (0.5, 0.0, 0.0)
            ovr_metrics[str(cls)] = {
                "AUC": round(auc, 4),
                "ACC": round(acc, 4),
                "F1": round(f1, 4),
            }
            auc_list.append(auc)
            acc_list.append(acc)
            f1_list.append(f1)
            print(f"{str(cls):<25} | {auc:<10.4f} | {acc:<10.4f} | {f1:<10.4f}")
        mean_auc = np.mean(auc_list)
        mean_acc = np.mean(acc_list)
        mean_f1 = np.mean(f1_list)
        ovr_metrics["Macro Average"] = {
            "AUC": round(mean_auc, 4),
            "ACC": round(mean_acc, 4),
            "F1": round(mean_f1, 4),
        }
        ovr_metrics["evaluation_protocol"] = (
            "source-wise One-vs-Rest; each class versus all others; "
            "test-distribution median threshold; arithmetic mean of binary F1"
        )
        ovr_metrics["mean_sourcewise_ovr_f1"] = round(float(mean_f1), 4)
        print("-" * 70)
        print(f"Paper primary mean source-wise OvR F1: {mean_f1:.4f}")
        json_path = os.path.join(UserConfig.OUTPUT_DIR, f"{name}_ovr_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ovr_metrics, f, indent=4)
        save_report_as_image(ovr_metrics, name)
