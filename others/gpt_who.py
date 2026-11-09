"""Repository GPT-Who surprisal features and logistic regression."""

if __name__ == "__main__":
    from baseline_cli import run_single

    run_single("gpt_who")
    raise SystemExit(0)
import json
import torch
import numpy as np
import os
import sys
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
MODEL_PATHS = ["models/llama-7b", "models/qwen2.5-7b-instruct"]
INPUT_BASE_DIR = "outputs"
OUTPUT_BASE_DIR = "outputs"
TASK_NAMES = ["dataset"]
DATASETS_CONFIG = []
for task in TASK_NAMES:
    DATASETS_CONFIG.append(
        {
            "task_name": task,
            "train_file": os.path.join(INPUT_BASE_DIR, task, "train_data.json"),
            "test_file": os.path.join(INPUT_BASE_DIR, task, "test_data.json"),
            "output_dir": os.path.join(OUTPUT_BASE_DIR, f"llama_qwen_{task}"),
        }
    )
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAIN_SAMPLES_PER_CLASS = None
TEST_SAMPLES_PER_CLASS = None


class SurprisalFeatureExtractor:

    def __init__(self, model_path, device):
        print(f"\n[{device.upper()}] Loading model: {os.path.basename(model_path)} ...")
        self.device = device
        self.window_size = 20
        self.feature_dim = 4 + self.window_size * 2
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map={"": self.device},
                trust_remote_code=True,
            )
            self.model.eval()
            if self.tokenizer.pad_token == "[PAD]":
                self.model.resize_token_embeddings(len(self.tokenizer))
        except Exception as e:
            print(f"Model loading failed: {e}")
            sys.exit(1)

    def calculate_surprisals_gpu(self, text):
        """Compute surprisal on the GPU"""
        encodings = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        )
        input_ids = encodings.input_ids.to(self.device)
        if input_ids.size(1) < self.window_size + 1:
            return None
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        valid_logits = shift_logits.view(-1, shift_logits.size(-1)).to(torch.float32)
        surprisals_tensor = loss_fct(valid_logits, shift_labels.view(-1))
        return surprisals_tensor.cpu().numpy()

    def extract_uid_features(self, surprisals):
        """Extract 44-dimensional features"""
        if surprisals is None:
            return None
        mean_s = np.mean(surprisals)
        var_s = np.var(surprisals)
        diffs = np.diff(surprisals)
        mean_diff = np.mean(np.abs(diffs))
        mean_sq_diff = np.mean(diffs**2)
        num_tokens = len(surprisals)
        spans = [
            surprisals[i : i + self.window_size]
            for i in range(num_tokens - self.window_size + 1)
        ]
        span_vars = [np.var(s) for s in spans]
        if not span_vars:
            return None
        min_idx = np.argmin(span_vars)
        max_idx = np.argmax(span_vars)
        return np.concatenate(
            [[mean_s, var_s, mean_diff, mean_sq_diff], spans[min_idx], spans[max_idx]]
        )


def load_raw_dataset(
    file_path, label_map=None, is_train=True, max_samples_per_class=None
):
    if not os.path.exists(file_path):
        return ([], [], label_map)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if is_train:
        original_classes = sorted(list(data.keys()))
        filtered_classes = original_classes
        current_label_map = {cls: i for i, cls in enumerate(filtered_classes)}
    else:
        current_label_map = label_map
    texts_list, labels_list = ([], [])
    for label_name, texts in data.items():
        if label_name not in current_label_map:
            continue
        if max_samples_per_class is not None:
            texts = texts[:max_samples_per_class]
        label_id = current_label_map[label_name]
        for text in texts:
            if text.strip():
                texts_list.append(text)
                labels_list.append(label_id)
    return (texts_list, labels_list, current_label_map)


def main():
    print(f"=== GPT-Who feature extraction (device: {DEVICE}) ===")
    print("\n[Phase 1] Loading dataset texts...")
    tasks_data = {}
    for config in DATASETS_CONFIG:
        task_name = config["task_name"]
        print(f"  -> Loading task: {task_name}")
        if not os.path.exists(config["output_dir"]):
            os.makedirs(config["output_dir"], exist_ok=True)
        train_texts, train_labels, label_map = load_raw_dataset(
            config["train_file"],
            is_train=True,
            max_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        )
        test_texts, test_labels, _ = load_raw_dataset(
            config["test_file"],
            label_map=label_map,
            is_train=False,
            max_samples_per_class=TEST_SAMPLES_PER_CLASS,
        )
        if not train_texts or not test_texts:
            print(
                f"  Warning: task {task_name} has missing or empty JSON files; skipping."
            )
            continue
        tasks_data[task_name] = {
            "config": config,
            "train_texts": train_texts,
            "train_labels": train_labels,
            "test_texts": test_texts,
            "test_labels": test_labels,
            "label_map": label_map,
            "all_train_feats": [],
            "all_test_feats": [],
        }
    if not tasks_data:
        print("Error: no datasets could be loaded; exiting.")
        return
    print("\n[Phase 2] Extracting surrogate features...")
    for model_path in MODEL_PATHS:
        extractor = SurprisalFeatureExtractor(model_path, DEVICE)
        model_name = os.path.basename(model_path)
        for task_name, task in tasks_data.items():
            model_train_f = []
            for text in tqdm(
                task["train_texts"], desc=f"[{task_name}] Train - {model_name}"
            ):
                surprisals = extractor.calculate_surprisals_gpu(text)
                feats = extractor.extract_uid_features(surprisals)
                if feats is None:
                    feats = np.full(extractor.feature_dim, np.nan)
                model_train_f.append(feats)
            task["all_train_feats"].append(np.array(model_train_f))
            model_test_f = []
            for text in tqdm(
                task["test_texts"], desc=f"[{task_name}] Test - {model_name}"
            ):
                surprisals = extractor.calculate_surprisals_gpu(text)
                feats = extractor.extract_uid_features(surprisals)
                if feats is None:
                    feats = np.full(extractor.feature_dim, np.nan)
                model_test_f.append(feats)
            task["all_test_feats"].append(np.array(model_test_f))
        print(f"[*] Unloading model {model_name} and clearing GPU memory...")
        del extractor.model
        del extractor.tokenizer
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\n[Phase 3] Training classifiers and saving results...")
    for task_name, task in tasks_data.items():
        print(f"\n" + "=" * 50)
        print(f" Processing task: {task_name} ")
        print("=" * 50)
        config = task["config"]
        label_map = task["label_map"]
        X_train = np.concatenate(task["all_train_feats"], axis=1)
        X_test = np.concatenate(task["all_test_feats"], axis=1)
        y_train = np.array(task["train_labels"])
        y_test = np.array(task["test_labels"])
        print(f"[*] Imputing missing values...")
        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        print(
            f"[*] Training logistic regression; feature dimension={X_train.shape[1]})..."
        )
        clf = LogisticRegression(
            max_iter=10000, multi_class="multinomial", solver="lbfgs"
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        sorted_items = sorted(label_map.items(), key=lambda x: x[1])
        target_names = [item[0] for item in sorted_items]
        target_ids = [item[1] for item in sorted_items]
        num_classes = len(target_names)
        acc = accuracy_score(y_test, y_pred)
        report_str = classification_report(
            y_test, y_pred, target_names=target_names, digits=4, zero_division=0
        )
        txt_filename = os.path.join(config["output_dir"], "classification_report.txt")
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(report_str)
        print(f"✅ Classification report saved: {txt_filename}")
        print(report_str)
        y_test_bin = label_binarize(y_test, classes=target_ids)
        if num_classes == 2 and y_test_bin.shape[1] == 1:
            y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))
        individual_aucs = {}
        auc_scores = []
        for i, class_name in enumerate(target_names):
            try:
                if np.sum(y_test_bin[:, i]) > 0:
                    auc = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
                else:
                    auc = 0.0
            except Exception:
                auc = 0.0
            auc_scores.append(auc)
            individual_aucs[str(i)] = {"class_name": class_name, "auc": auc}
        json_output = {
            "accuracy": acc,
            "document_macro_f1": float(f1_score(y_test, y_pred, labels=target_ids, average="macro", zero_division=0)),
            "mean_auc": np.mean(auc_scores),
            "individual_aucs": individual_aucs,
        }
        json_filename = os.path.join(config["output_dir"], "test_metrics.json")
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=4)
        print(f"✅ Metrics JSON saved: {json_filename}")
