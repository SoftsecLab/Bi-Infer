"""Repository GhostBuster feature extraction and supervised evaluation."""

if __name__ == "__main__":
    from baseline_cli import run_single

    run_single("GhostBuster")
    raise SystemExit(0)
import os
import json
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")


class Config:
    PROXY_MODELS = [
        {"type": "llama", "path": "models/llama-7b"},
        {"type": "qwen", "path": "models/qwen2.5-7b-instruct"},
    ]
    INPUT_BASE_DIR = "outputs"
    OUTPUT_BASE_DIR = "outputs"
    TASK_NAMES = ["dataset"]
    PER_CLASS_TRAIN_SAMPLES = -1
    PER_CLASS_TEST_SAMPLES = -1
    MAX_TOKEN_LENGTH = 1024
    GPU_ID = 1
    DEVICE = (
        f"cuda:{GPU_ID}" if torch.cuda.is_available() and GPU_ID in [0, 1] else "cpu"
    )


def convert_to_python_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_type(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, bool):
        return obj
    else:
        return obj


def sample_per_class(samples, labels, per_class_samples):
    class2samples = {}
    class2labels = {}
    for s, l in zip(samples, labels):
        if l not in class2samples:
            class2samples[l] = []
            class2labels[l] = []
        class2samples[l].append(s)
        class2labels[l].append(l)
    new_samples, new_labels = ([], [])
    for cls in class2samples:
        cls_samples = class2samples[cls]
        cls_labels = class2labels[cls]
        take = (
            len(cls_samples)
            if per_class_samples == -1 or len(cls_samples) <= per_class_samples
            else per_class_samples
        )
        new_samples.extend(cls_samples[:take])
        new_labels.extend(cls_labels[:take])
    return (new_samples, new_labels)


def load_raw_data_from_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class GhostBusterSingleExtractor:
    """Load one model at a time and extract five GhostBuster features."""

    def __init__(self, model_config):
        self.device = Config.DEVICE
        self.model_type = model_config["type"]
        self.model_path = model_config["path"]
        print(
            f"\n[{self.device.upper()}] Loading {self.model_type.upper()} from: {os.path.basename(self.model_path)} ..."
        )
        trust_remote = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=trust_remote, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map={"": self.device},
        ).eval()

    def unload(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_stats(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_TOKEN_LENGTH,
        ).to(self.device)
        if inputs.input_ids.size(1) < 3:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()
        log_probs_dist = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs_dist.gather(-1, shift_labels.unsqueeze(-1)).squeeze(
            -1
        )
        mask = shift_labels != self.tokenizer.pad_token_id
        if mask.sum().item() == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        valid_log_probs = token_log_probs[mask]
        mean_lp = valid_log_probs.mean().item()
        var_lp = valid_log_probs.var().item() if valid_log_probs.size(0) > 1 else 0.0
        min_lp = valid_log_probs.min().item()
        max_lp = valid_log_probs.max().item()
        median_lp = valid_log_probs.median().item()
        return [mean_lp, var_lp, min_lp, max_lp, median_lp]


def main():
    if torch.cuda.is_available() and Config.GPU_ID in [0, 1]:
        torch.cuda.set_device(Config.GPU_ID)
        print(
            f"\n[*] Device: cuda:{Config.GPU_ID} ({torch.cuda.get_device_name(Config.GPU_ID)})"
        )
    else:
        print(f"\n[*] Device: {Config.DEVICE}")
    print("=" * 75)
    print(f"🚀 GhostBuster Multi-Task Supervised OvR Detection Pipeline")
    print("=" * 75)
    print("\n>>> [Phase 1] Pre-loading all datasets...")
    all_tasks_meta = {}
    datasets_dict = {}
    for task_name in Config.TASK_NAMES:
        train_path = Config.TRAIN_DATA_PATH
        test_path = Config.TEST_DATA_PATH
        output_dir = os.path.join(Config.OUTPUT_BASE_DIR, f"ghostbuster_{task_name}")
        os.makedirs(output_dir, exist_ok=True)
        train_raw_data = load_raw_data_from_json(train_path)
        test_raw_data = load_raw_data_from_json(test_path)
        if not train_raw_data or not test_raw_data:
            print(f"[!] Skipping task {task_name}: training or test data not found.")
            continue

        def flatten_data(raw_dict):
            samples, labels = ([], [])
            for cls_name, texts in raw_dict.items():
                for t in texts:
                    if t.strip():
                        samples.append(t)
                        labels.append(cls_name)
            return (samples, labels)

        train_samples, train_labels = flatten_data(train_raw_data)
        test_samples, test_labels = flatten_data(test_raw_data)
        train_samples, train_labels = sample_per_class(
            train_samples, train_labels, Config.PER_CLASS_TRAIN_SAMPLES
        )
        test_samples, test_labels = sample_per_class(
            test_samples, test_labels, Config.PER_CLASS_TEST_SAMPLES
        )
        all_tasks_meta[task_name] = {
            "train_labels": train_labels,
            "test_labels": test_labels,
            "test_samples": test_samples,
            "output_dir": output_dir,
            "train_feats": [],
            "test_feats": [],
        }
        datasets_dict[f"{task_name}_train"] = train_samples
        datasets_dict[f"{task_name}_test"] = test_samples
    if not datasets_dict:
        print("\n[!] No valid tasks; exiting.")
        return
    print("\n>>> [Phase 2] Serial Feature Extraction & Checkpointing...")
    for model_idx, model_config in enumerate(Config.PROXY_MODELS):
        m_type = model_config["type"]
        datasets_to_process = {}
        for data_name, samples in datasets_dict.items():
            is_train = data_name.endswith("_train")
            task_name = data_name[:-6] if is_train else data_name[:-5]
            split = "train" if is_train else "test"
            out_dir = all_tasks_meta[task_name]["output_dir"]
            save_path = os.path.join(
                out_dir, f"gb_all_samples_feat_{m_type}_{split}.npy"
            )
            if os.path.exists(save_path):
                print(
                    f"[*] Cache found: loading [{task_name}-{split}] {m_type} features."
                )
                loaded_feats = np.load(save_path)
                if is_train:
                    all_tasks_meta[task_name]["train_feats"].append(loaded_feats)
                else:
                    all_tasks_meta[task_name]["test_feats"].append(loaded_feats)
            else:
                datasets_to_process[data_name] = samples
        if not datasets_to_process:
            print(
                f"[*] ✅ Model {m_type} features loaded for every task; skipping model loading."
            )
            continue
        extractor = GhostBusterSingleExtractor(model_config)
        for data_name, samples in datasets_to_process.items():
            is_train = data_name.endswith("_train")
            task_name = data_name[:-6] if is_train else data_name[:-5]
            split = "train" if is_train else "test"
            out_dir = all_tasks_meta[task_name]["output_dir"]
            save_path = os.path.join(
                out_dir, f"gb_all_samples_feat_{m_type}_{split}.npy"
            )
            extracted_feats = []
            for text in tqdm(
                samples, desc=f"   Inferencing {data_name} with {m_type}", leave=False
            ):
                try:
                    stats = extractor.get_stats(text)
                except Exception:
                    stats = [0.0, 0.0, 0.0, 0.0, 0.0]
                extracted_feats.append(stats)
            feat_array = np.array(extracted_feats, dtype=np.float32)
            np.save(save_path, feat_array)
            print(
                f"  Saved intermediate features: [{data_name}] {m_type} features saved to: {save_path}"
            )
            if is_train:
                all_tasks_meta[task_name]["train_feats"].append(feat_array)
            else:
                all_tasks_meta[task_name]["test_feats"].append(feat_array)
        extractor.unload()
    print("\n>>> [Phase 3] Training Classifiers & Evaluation...")
    for task_name, meta in all_tasks_meta.items():
        print(f"\n" + "=" * 60)
        print(f" Evaluation task: {task_name} ")
        print("=" * 60)
        X_train = np.concatenate(meta["train_feats"], axis=1)
        X_test = np.concatenate(meta["test_feats"], axis=1)
        y_train = np.array(meta["train_labels"])
        y_test = np.array(meta["test_labels"])
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
        # GhostBuster's central idea is to search combinations of weak-LM
        # probability features. Expand first/second-order interactions and
        # select discriminative features on training data only. This remains
        # a compact adaptation, not the official exhaustive feature search.
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)
        selector = SelectKBest(score_func=f_classif, k=min(64, X_train.shape[1]))
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Paper Section 4.2: fit each source-vs-rest classifier independently.
        # A document may be positive for several binary classifiers, or none.
        classes = np.unique(y_train)
        binary_predictions = {}
        positive_scores = {}
        for class_name in classes:
            train_binary = (y_train == class_name).astype(int)
            binary_clf = LogisticRegression(
                solver="liblinear", max_iter=1000, random_state=42
            )
            binary_clf.fit(X_train_scaled, train_binary)
            binary_predictions[class_name] = binary_clf.predict(X_test_scaled)
            positive_scores[class_name] = binary_clf.predict_proba(X_test_scaled)[:, 1]
        # Auxiliary forced-choice classification only; NOT used for OvR F1.
        y_prob = np.column_stack([positive_scores[c] for c in classes])
        y_pred = classes[np.argmax(y_prob, axis=1)]
        document_macro_f1 = f1_score(
            y_test, y_pred, labels=classes, average="macro", zero_division=0
        )
        individual_metrics = {}
        auc_list, acc_list, f1_list = ([], [], [])
        print(f"{'Class (Target vs Rest)':<25} | {'AUC':<8} | {'ACC':<8} | {'F1':<8}")
        print(f"{'-' * 60}")
        for i, class_name in enumerate(classes):
            y_test_bin = (y_test == class_name).astype(int)
            y_pred_bin = binary_predictions[class_name]
            pred_scores = y_prob[:, i]
            try:
                if np.sum(y_test_bin) > 0:
                    auc = roc_auc_score(y_test_bin, pred_scores)
                else:
                    auc = 0.0
                acc = accuracy_score(y_test_bin, y_pred_bin)
                f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
            except Exception:
                auc, acc, f1 = (0.0, 0.0, 0.0)
            auc_list.append(auc)
            acc_list.append(acc)
            f1_list.append(f1)
            individual_metrics[str(class_name)] = {
                "auc": round(auc, 4),
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
            }
            print(f"{str(class_name):<25} | {auc:<8.4f} | {acc:<8.4f} | {f1:<8.4f}")
        print(f"{'-' * 60}")
        mean_auc, mean_acc, mean_f1 = (
            np.mean(auc_list),
            np.mean(acc_list),
            np.mean(f1_list),
        )
        print(
            f"🎯 Macro Avg Metrics       | {mean_auc:<8.4f} | {mean_acc:<8.4f} | {mean_f1:<8.4f}"
        )
        print(f"Paper primary mean source-wise OvR F1: {mean_f1:.4f}")
        print(f"Auxiliary forced-choice document Macro-F1: {document_macro_f1:.4f}")
        print(f"{'=' * 60}")
        out_dir = meta["output_dir"]
        final_metrics = {
            "document_macro_f1": float(document_macro_f1),
            "macro_avg_auc": round(mean_auc, 4),
            "macro_avg_accuracy": round(mean_acc, 4),
            "macro_avg_f1": round(mean_f1, 4),
            "mean_sourcewise_ovr_f1": round(mean_f1, 4),
            "individual_metrics": individual_metrics,
        }
        metrics_path = os.path.join(out_dir, "metrics_result.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                convert_to_python_type(final_metrics), f, indent=4, ensure_ascii=False
            )
        pred_path = os.path.join(out_dir, "predictions.json")
        results = []
        for i in range(len(meta["test_samples"])):
            results.append(
                convert_to_python_type(
                    {
                        "sample_text": meta["test_samples"][i],
                        "true_label_name": y_test[i],
                        "pred_label_name": y_pred[i],
                        "is_correct": bool(y_test[i] == y_pred[i]),
                    }
                )
            )
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"📁 Task {task_name} metrics and predictions saved to: {out_dir}")
