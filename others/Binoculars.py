"""Repository Binoculars score-ratio baseline and one-vs-rest evaluation."""

if __name__ == "__main__":
    from baseline_cli import run_single

    run_single("Binoculars")
    raise SystemExit(0)
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore")
GPU_ID = "1"


class UserConfig:
    OBSERVER_MODEL_PATH = "tiiuae/falcon-7b"
    PERFORMER_MODEL_PATH = "tiiuae/falcon-7b-instruct"
    DATASETS = [{"name": "arxiv", "test_path": "data/test.json", "max_samples": None}]
    OUTPUT_DIR = "outputs"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_TOKEN_LENGTH = 1024


class BinocularsFeatureExtractor:

    def __init__(self):
        self.device = UserConfig.DEVICE
        if self.device == "cuda":
            gpu_count = torch.cuda.device_count()
            print(f"[Init] Using {gpu_count} GPUs.")
        print(f"[Init] Loading Observer Model (LLaMA)...")
        self.tokenizer_obs = AutoTokenizer.from_pretrained(
            UserConfig.OBSERVER_MODEL_PATH,
            trust_remote_code=True,
            use_fast=False,
            padding_side="right",
        )
        if self.tokenizer_obs.pad_token is None:
            self.tokenizer_obs.pad_token = self.tokenizer_obs.eos_token
        self.observer = AutoModelForCausalLM.from_pretrained(
            UserConfig.OBSERVER_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        ).eval()
        print(f"[Init] Loading Performer Model (Qwen)...")
        self.tokenizer_perf = AutoTokenizer.from_pretrained(
            UserConfig.PERFORMER_MODEL_PATH,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer_perf.pad_token is None:
            self.tokenizer_perf.pad_token = self.tokenizer_perf.eos_token
        self.performer = AutoModelForCausalLM.from_pretrained(
            UserConfig.PERFORMER_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        ).eval()
        if (self.tokenizer_obs.get_vocab() != self.tokenizer_perf.get_vocab()
                or self.observer.config.vocab_size != self.performer.config.vocab_size):
            raise ValueError(
                "Binoculars cross-perplexity requires identical token ID mappings "
                "and vocabulary sizes. Use compatible observer/performer checkpoints "
                "(e.g. the original paired model family), not LLaMA plus Qwen."
            )

    def extract_batch(self, text_list, label_name):
        records = []
        for text in tqdm(text_list, desc=f"Feat Extract [{label_name}]"):
            inputs_obs = self.tokenizer_obs(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=UserConfig.MAX_TOKEN_LENGTH,
                padding=False,
            ).to(self.device)
            inputs_perf = self.tokenizer_perf(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=UserConfig.MAX_TOKEN_LENGTH,
                padding=False,
            ).to(self.device)
            input_ids_obs, input_ids_perf = (
                inputs_obs.input_ids,
                inputs_perf.input_ids,
            )
            if (input_ids_obs != self.tokenizer_obs.pad_token_id).sum().item() < 2:
                continue
            with torch.no_grad():
                shift_logits_obs = (
                    self.observer(**inputs_obs).logits[..., :-1, :].contiguous()
                )
                shift_labels_obs = input_ids_obs[..., 1:].contiguous()
                mask_obs = shift_labels_obs != self.tokenizer_obs.pad_token_id
                shift_logits_perf = (
                    self.performer(**inputs_perf).logits[..., :-1, :].contiguous()
                )
                shift_labels_perf = input_ids_perf[..., 1:].contiguous()
                mask_perf = shift_labels_perf != self.tokenizer_perf.pad_token_id
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss_obs = loss_fct(
                    shift_logits_obs.view(-1, shift_logits_obs.size(-1)),
                    shift_labels_obs.view(-1),
                ).view(shift_labels_obs.shape)
                log_ppl = (
                    loss_obs[mask_obs].mean().item()
                    if mask_obs.sum().item() > 0
                    else 0.0
                )
                # Cross-perplexity: entropy of the observer distribution under
                # the performer distribution, not the performer's text NLL.
                observer_probs = torch.softmax(shift_logits_obs.float(), dim=-1)
                performer_log_probs = torch.log_softmax(shift_logits_perf.float(), dim=-1)
                cross_entropy = -(observer_probs * performer_log_probs).sum(dim=-1)
                log_x_ppl = cross_entropy[mask_obs].mean().item()
                score = log_ppl / log_x_ppl if log_x_ppl != 0 else 0
            records.append({"label": label_name, "binoculars_score": score})
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
        features = extractor.extract_batch(selected_texts, label_name=label)
        all_records.extend(features)
    df = pd.DataFrame(all_records)
    df.to_csv(cache_path, index=False)
    return df


def save_report_as_image(report_dict, dataset_name):
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
    plt.title(f"Zero-Shot OvR Report (Binoculars): {dataset_name}", fontsize=14, pad=20)
    img_path = os.path.join(UserConfig.OUTPUT_DIR, f"{dataset_name}_ovr_report.png")
    plt.savefig(img_path, bbox_inches="tight", dpi=300)
    print(f"📊 Report image saved: {img_path}")
    plt.close()


def plot_binoculars_distribution(df, dataset_name):
    print("🎨 Generating Score Distribution Plot...")
    LABEL_MAPPING = {
        "human": "gpt-3.5-turbo",
        "text-davinci-003": "text-davinci-003",
        "bigscience/bloomz": "bloomz",
        "gpt-3.5-turbo": "human",
        "cohere": "cohere",
        "dolly-v2-12b": "dolly-v2-12b",
    }
    plot_df = df.copy()
    plot_df["plot_label"] = plot_df["label"].map(lambda x: LABEL_MAPPING.get(x, x))
    plt.rcParams["font.family"] = "serif"
    sns.set_theme(style="ticks", font="serif")
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = plot_df["plot_label"].unique()
    colors = sns.color_palette("tab10", n_colors=max(10, len(labels)))
    color_idx = 0
    sorted_labels = sorted(labels, key=lambda x: 0 if "human" in str(x).lower() else 1)
    for label in sorted_labels:
        subset = (
            plot_df[plot_df["plot_label"] == label]["binoculars_score"].dropna().values
        )
        weights = np.ones_like(subset) / len(subset) * 100
        if "human" in str(label).lower():
            c = "dimgray"
        else:
            c = colors[color_idx]
            color_idx += 1
        ax.hist(
            subset,
            bins=80,
            weights=weights,
            alpha=0.6,
            label=label,
            color=c,
            edgecolor="none",
        )
    ax.set_xlabel("Binoculars Score", fontsize=16, fontweight="bold")
    ax.set_ylabel("Frequency (%)", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12)
    sns.despine()
    ax.legend(
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
        fontsize=12,
        ncol=2,
        borderaxespad=0.0,
    )
    pdf_path = os.path.join(UserConfig.OUTPUT_DIR, f"{dataset_name}_score_dist.pdf")
    png_path = os.path.join(UserConfig.OUTPUT_DIR, f"{dataset_name}_score_dist.png")
    plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"✅ Distribution plot saved to:\n  - {pdf_path}")
    plt.close()


def main():
    print("=" * 60)
    print(f"🚀 Starting Binoculars (LLaMA+Qwen) Zero-Shot OvR Pipeline")
    print(f"📁 Output Directory: {UserConfig.OUTPUT_DIR}")
    print("=" * 60)
    need_model = False
    for ds_cfg in UserConfig.DATASETS:
        cache_path = os.path.join(
            UserConfig.OUTPUT_DIR, f"{ds_cfg['name']}_all_samples_test_feat.csv"
        )
        if not os.path.exists(cache_path):
            need_model = True
            break
    extractor = BinocularsFeatureExtractor() if need_model else None
    for ds_cfg in UserConfig.DATASETS:
        name = ds_cfg["name"]
        print(f"\n{'#' * 40}\n# Task: {name} (Zero-Shot OvR)\n{'#' * 40}")
        cache_path = os.path.join(
            UserConfig.OUTPUT_DIR, f"{name}_all_samples_test_feat.csv"
        )
        df_test = load_and_process_test_data(
            ds_cfg["test_path"], ds_cfg["max_samples"], extractor, cache_path
        )
        plot_binoculars_distribution(df_test, name)
        y_test = df_test["label"].astype(str).values
        scores = df_test["binoculars_score"].values
        classes = np.unique(y_test)
        ovr_metrics = {}
        auc_list, acc_list, f1_list = ([], [], [])
        print(
            f"\n{'Class (Binary OvR)':<25} | {'AUC':<10} | {'Binary ACC':<10} | {'Binary F1':<10}"
        )
        print("-" * 70)
        for cls in classes:
            y_true_bin = (y_test == cls).astype(int)
            current_scores = scores if "human" in str(cls).lower() else -scores
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
