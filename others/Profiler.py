"""PROFILER context-loss features, random forest training and test evaluation."""

if __name__ == "__main__":
    from baseline_cli import run_single

    run_single("Profiler")
    raise SystemExit(0)
import json
import torch
import numpy as np
import os
import sys
import scipy.stats
import gc
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class Config:
    PROXY_MODEL_PATHS = ["models/llama-7b", "models/qwen2.5-7b-instruct"]
    TRAIN_DATA_PATH = "data/train.json"
    TEST_DATA_PATH = "data/test.json"
    OUTPUT_DIR = "outputs"
    CONTEXT_WINDOW_SIZE = 4
    MAX_SEQ_LEN = 512
    SAMPLES_PER_CLASS_TRAIN = None
    SAMPLES_PER_CLASS_TEST = None
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ProfilerFeatureExtractor:

    def __init__(self, model_path, device, window_size=4):
        print(f"\n[{device.upper()}] Loading surrogate model: {model_path} ...")
        self.device = device
        self.window_size = window_size
        self.half_window = window_size // 2
        self.expected_dim = (
            self.window_size * 18 + self.window_size * (self.window_size - 1) // 2
        )
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

    def _compute_context_losses(self, input_ids, logits):
        """Compute the context-loss matrix."""
        ids = input_ids[0]
        logs = logits[0][:-1]
        seq_len = logs.size(0)
        valid_start = self.half_window
        valid_end = seq_len - self.half_window
        if valid_start >= valid_end:
            return None
        context_losses = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        valid_logits = logs[valid_start:valid_end].to(torch.float32)
        for j in range(-self.half_window, self.half_window + 1):
            if j == 0 and self.window_size % 2 == 0:
                continue
            target_indices = torch.arange(valid_start, valid_end) + j + 1
            target_tokens = ids[target_indices.long()].to(self.device)
            loss = loss_fct(valid_logits, target_tokens)
            context_losses.append(loss.cpu().numpy())
            if len(context_losses) >= self.window_size:
                break
        if not context_losses:
            return None
        return np.stack(context_losses, axis=1)

    def _extract_independent_patterns(self, loss_matrix):
        """Extract independent pattern features."""
        features = []
        N, W = loss_matrix.shape
        for w in range(W):
            col = loss_matrix[:, w]
            stats = [
                np.mean(col),
                np.std(col),
                np.min(col),
                np.max(col),
                np.median(col),
                np.var(col),
            ]
            features.extend(stats)
            if len(col) > 1:
                diff1 = np.diff(col)
                stats_d1 = [
                    np.mean(diff1),
                    np.std(diff1),
                    np.min(diff1),
                    np.max(diff1),
                    np.median(diff1),
                    np.var(diff1),
                ]
                features.extend(stats_d1)
            else:
                features.extend([0] * 6)
            if len(col) > 2:
                diff2 = np.diff(col, n=2) / 2.0
                stats_d2 = [
                    np.mean(diff2),
                    np.std(diff2),
                    np.min(diff2),
                    np.max(diff2),
                    np.median(diff2),
                    np.var(diff2),
                ]
                features.extend(stats_d2)
            else:
                features.extend([0] * 6)
        return np.array(features)

    def _extract_correlated_patterns(self, loss_matrix):
        """Extract correlated pattern features."""
        features = []
        probs_matrix = scipy.special.softmax(loss_matrix, axis=0)
        for i, j in combinations(range(loss_matrix.shape[1]), 2):
            p = probs_matrix[:, i] + 1e-09
            q = probs_matrix[:, j] + 1e-09
            kl_sym = np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))
            features.append(kl_sym)
        return np.array(features)

    def get_features(self, text):
        if not text or not text.strip():
            return None
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_SEQ_LEN,
            padding=False,
        ).to(self.device)
        if inputs.input_ids.size(1) < self.window_size + 5:
            return None
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
            )
        loss_matrix = self._compute_context_losses(inputs.input_ids, outputs.logits)
        if loss_matrix is None:
            return None
        feat_ip = self._extract_independent_patterns(loss_matrix)
        feat_cp = self._extract_correlated_patterns(loss_matrix)
        final_feats = np.concatenate([feat_ip, feat_cp])
        final_feats = np.nan_to_num(final_feats, nan=0.0, posinf=0.0, neginf=0.0)
        return final_feats


class DataProcessor:

    def __init__(self, raw_data):
        all_classes = sorted(list(raw_data.keys()))
        print(f"[*] Input classes: {all_classes}")
        self.valid_classes = list(all_classes)
        print(f"[*] Classes: {self.valid_classes}")
        self.class2id = {c: i for i, c in enumerate(self.valid_classes)}
        self.id2class = {i: c for c, i in self.class2id.items()}

    def load_samples(self, file_path, max_per_class=None):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples, labels = ([], [])
        for cls_name, texts in data.items():
            norm_cls = cls_name.strip()
            if norm_cls not in self.class2id:
                continue
            label_idx = self.class2id[norm_cls]
            if max_per_class:
                texts = texts[:max_per_class]
            for t in texts:
                if t.strip():
                    samples.append(t)
                    labels.append(label_idx)
        return (samples, labels)


def save_evaluation_results(y_true, y_pred, y_prob, label_map, output_dir):
    """Save classification reports and AUC metrics."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    target_names = [label_map[i] for i in range(len(label_map))]
    report_str = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, zero_division=0
    )
    txt_path = os.path.join(output_dir, "classification_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"[*] Classification report saved: {txt_path}")
    print(report_str)
    acc = accuracy_score(y_true, y_pred)
    classes = list(range(len(label_map)))
    y_true_bin = label_binarize(y_true, classes=classes)
    if len(classes) == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
    individual_aucs = {}
    auc_scores = []
    for i, class_name in enumerate(target_names):
        try:
            if np.sum(y_true_bin[:, i]) > 0:
                auc_val = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            else:
                auc_val = 0.0
        except:
            auc_val = 0.0
        auc_scores.append(auc_val)
        individual_aucs[str(i)] = {"class_name": class_name, "auc": float(auc_val)}
    mean_auc = np.mean(auc_scores)
    json_output = {
        "accuracy": float(acc),
        "document_macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "mean_auc": float(mean_auc),
        "individual_aucs": individual_aucs,
    }
    json_path = os.path.join(output_dir, "test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=4)
    print(f"[*] Metrics JSON saved: {json_path}")


def main():
    print("=== PROFILER: multi-surrogate feature fusion ===")
    with open(Config.TRAIN_DATA_PATH, "r") as f:
        raw_train = json.load(f)
    processor = DataProcessor(raw_train)
    print("\n>>> Loading data...")
    train_texts, train_labels = processor.load_samples(
        Config.TRAIN_DATA_PATH, Config.SAMPLES_PER_CLASS_TRAIN
    )
    test_texts, test_labels = processor.load_samples(
        Config.TEST_DATA_PATH, Config.SAMPLES_PER_CLASS_TEST
    )
    if not train_texts or not test_texts:
        print("[!] Insufficient data; exiting")
        return
    all_train_feats = []
    all_test_feats = []
    for model_path in Config.PROXY_MODEL_PATHS:
        extractor = ProfilerFeatureExtractor(
            model_path, Config.DEVICE, Config.CONTEXT_WINDOW_SIZE
        )
        model_name = os.path.basename(model_path)
        model_train_f = []
        for t in tqdm(train_texts, desc=f"Extracting Train - {model_name}"):
            f = extractor.get_features(t)
            if f is None:
                f = np.full(extractor.expected_dim, np.nan)
            model_train_f.append(f)
        all_train_feats.append(np.array(model_train_f))
        model_test_f = []
        for t in tqdm(test_texts, desc=f"Extracting Test - {model_name}"):
            f = extractor.get_features(t)
            if f is None:
                f = np.full(extractor.expected_dim, np.nan)
            model_test_f.append(f)
        all_test_feats.append(np.array(model_test_f))
        print(f"\n[*] Unloading model {model_name} and clearing GPU memory...")
        del extractor.model
        del extractor.tokenizer
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\n>>> Concatenating surrogate features...")
    X_train = np.concatenate(all_train_feats, axis=1)
    X_test = np.concatenate(all_test_feats, axis=1)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    print(">>> Imputing missing values...")
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    print(f"\n>>> Training random forest; feature dimension={X_train.shape[1]})...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("\n>>> Generating predictions...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    save_evaluation_results(
        y_test, y_pred, y_prob, processor.id2class, Config.OUTPUT_DIR
    )
    print("\n=== Pipeline complete ===")
