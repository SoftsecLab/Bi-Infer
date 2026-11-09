"""Repository SeqXGPT sequence tagging and document evaluation."""

if __name__ == "__main__":
    from baseline_cli import run_single

    run_single("seqxgpt")
    raise SystemExit(0)
import os
import sys
import json
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss, TransformerEncoder, TransformerEncoderLayer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")


class Config:
    PROXY_MODELS = [
        {"type": "llama", "path": "models/llama-7b"},
        {"type": "qwen", "path": "models/qwen2.5-7b-instruct"},
    ]
    INPUT_BASE_DIR = "outputs"
    OUTPUT_BASE_DIR = "outputs"
    TASK_NAMES = ["dataset"]
    TASKS = []
    for task in TASK_NAMES:
        TASKS.append(
            {
                "task_name": task,
                "train": os.path.join(INPUT_BASE_DIR, task, "train_data.json"),
                "test": os.path.join(INPUT_BASE_DIR, task, "test_data.json"),
                "output_dir": os.path.join(OUTPUT_BASE_DIR, f"seqxgpt_{task}"),
            }
        )
    MODEL_SAVE_NAME = "seqxgpt_auto_label.pth"
    REPORT_SAVE_NAME = "classification_report.txt"
    METRICS_SAVE_NAME = "test_metrics.json"
    GPU_INDEX = 1
    DEVICE = f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu"
    MAX_SAMPLES_PER_CLASS_TRAIN = None
    MAX_SAMPLES_PER_CLASS_TEST = None
    BATCH_SIZE = 16
    SEQ_LEN = 1024
    EPOCHS = 10
    LR = 5e-05
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1


class LabelProcessor:

    def __init__(self, raw_data: Dict[str, List[str]]):
        raw_classes = [k.strip().lower() for k in raw_data.keys()]
        self.raw_classes = list(dict.fromkeys(raw_classes))
        self.valid_classes = self.raw_classes.copy()
        self.class2id = {cls: i for i, cls in enumerate(self.valid_classes)}
        self.id2class = {i: cls for cls, i in self.class2id.items()}
        self.tag2id = {}
        self.id2tag = {}
        counter = 0
        prefixes = ["B-", "M-", "E-", "S-"]
        for cls in self.valid_classes:
            for pre in prefixes:
                tag = f"{pre}{cls}"
                self.tag2id[tag] = counter
                self.id2tag[counter] = tag
                counter += 1
        self.num_tags = len(self.tag2id)

    def encode_sentence(self, word_list: List[str], class_name: str) -> List[int]:
        class_name = class_name.strip().lower()
        if class_name not in self.class2id:
            return []
        length = len(word_list)
        if length == 0:
            return []
        if length == 1:
            return [self.tag2id[f"S-{class_name}"]]
        ids = [0] * length
        ids[0] = self.tag2id[f"B-{class_name}"]
        ids[-1] = self.tag2id[f"E-{class_name}"]
        m_id = self.tag2id[f"M-{class_name}"]
        for i in range(1, length - 1):
            ids[i] = m_id
        return ids


def load_data(
    file_path: str, valid_classes: list = None, max_per_class: Optional[int] = None
):
    if not os.path.exists(file_path):
        return ([], [], {})
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if valid_classes is not None:
        valid_classes_norm = [cls.strip().lower() for cls in valid_classes]
        raw_data = {
            k: v for k, v in raw_data.items() if k.strip().lower() in valid_classes_norm
        }
    samples, classes = ([], [])
    for cls_name, texts in raw_data.items():
        current_texts = [t for t in texts if t.strip()]
        if max_per_class is not None and max_per_class > 0:
            current_texts = current_texts[:max_per_class]
        for text in current_texts:
            samples.append(text)
            classes.append(cls_name)
    return (samples, classes, raw_data)


class SeqXGPTFeatureExtractor:

    def __init__(self, model_config):
        self.device = Config.DEVICE
        self.m_type = model_config["type"]
        m_path = model_config["path"]
        print(f"\n[{self.device.upper()}] Loading surrogate model: {self.m_type}")
        try:
            trust_remote = True
            self.tokenizer = AutoTokenizer.from_pretrained(
                m_path, trust_remote_code=trust_remote
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                m_path,
                device_map={"": self.device},
                torch_dtype=torch.float16,
                trust_remote_code=trust_remote,
            )
            self.model.eval()
        except Exception as e:
            print(f"[!] Loading model {self.m_type} failed: {e}")
            sys.exit(1)

    def free_memory(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[*] Unloading model {self.m_type} and clearing GPU memory.")

    def extract_aligned_scores(self, text: str, target_words: List[str]):
        if not text.strip() or len(target_words) == 0:
            return []
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=Config.SEQ_LEN,
                padding=False,
            ).to(self.device)
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits[0, :-1, :]
            labels = input_ids[0, 1:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=1, index=labels.unsqueeze(1)
            ).squeeze(1)
            token_log_probs = torch.cat(
                [torch.tensor([0.0], device=self.device), token_log_probs]
            )
            offset_mapping = (
                self.tokenizer(
                    text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=Config.SEQ_LEN,
                    return_tensors="pt",
                )["offset_mapping"][0]
                .cpu()
                .numpy()
            )
            token_log_probs = token_log_probs.cpu().numpy()
            aligned_scores = []
            curr_pos = 0
            for word in target_words:
                word_start = text.find(word, curr_pos)
                if word_start == -1:
                    word_start = curr_pos
                word_end = word_start + len(word)
                curr_pos = word_end
                relevant_probs = []
                for token_idx, (t_start, t_end) in enumerate(offset_mapping):
                    if max(word_start, t_start) < min(word_end, t_end):
                        if token_idx < len(token_log_probs):
                            relevant_probs.append(token_log_probs[token_idx])
                score = np.mean(relevant_probs) if relevant_probs else 0.0
                aligned_scores.append(score)
            return aligned_scores
        except Exception:
            return []


class SeqXGPTModel(nn.Module):

    def __init__(self, num_tags, num_proxies, seq_len=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.cnn_out_dim = 64
        self.embed_dim = num_proxies * self.cnn_out_dim
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, self.embed_dim))
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=512,
            batch_first=True,
            activation="relu",
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_tags)
        # Original SeqXGPT describes a linear classification layer, not a CRF.
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)

    def forward(self, features, labels=None):
        batch_size, seq_len, num_models = features.shape
        cnn_outputs = []
        for model_idx in range(num_models):
            single_feature = features[:, :, model_idx : model_idx + 1].permute(0, 2, 1)
            cnn_out = self.conv(single_feature).transpose(1, 2)
            cnn_outputs.append(cnn_out)
        out = torch.cat(cnn_outputs, dim=-1)
        out = out + self.pos_enc[:, :seq_len, :]
        out = self.norm(out)
        if labels is not None:
            mask = labels.gt(-1)
            out = self.transformer(out, src_key_padding_mask=~mask)
        else:
            out = self.transformer(out)
        logits = self.classifier(out)
        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fct(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )
            output["loss"] = loss
            preds = logits.argmax(dim=-1).masked_fill(~mask, -1)
            output["preds"] = preds
        return output


class SeqXGPTDataset(Dataset):

    def __init__(self, features, words, class_names, label_processor: LabelProcessor):
        self.features, self.words, self.labels = ([], [], [])
        for i in range(len(features)):
            seq_ids = label_processor.encode_sentence(words[i], class_names[i])
            if (
                len(seq_ids) == 0
                or len(features[i]) == 0
                or len(seq_ids) != len(features[i])
            ):
                continue
            self.features.append(features[i])
            self.words.append(words[i])
            self.labels.append(seq_ids)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "text": self.words[idx],
        }


def collate_fn(batch):
    features = [item["features"] for item in batch]
    labels = [item["labels"] for item in batch]
    texts = [item["text"] for item in batch]
    features_padded = nn.utils.rnn.pad_sequence(
        features, batch_first=True, padding_value=0.0
    )
    labels_padded = nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-1
    )
    return {"features": features_padded, "labels": labels_padded, "text": texts}


class Trainer:

    def __init__(self, model, train_loader, test_loader, label_processor, output_dir):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.label_processor = label_processor
        self.output_dir = output_dir
        self.optimizer = AdamW(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )
        total_steps = len(train_loader) * Config.EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(Config.WARMUP_RATIO * total_steps),
            num_training_steps=total_steps,
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}", leave=False
        )
        for batch in pbar:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(Config.DEVICE)
            output = self.model(batch["features"], batch["labels"])
            loss = output["loss"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / len(self.train_loader)

    def evaluate(self, final_report=False):
        self.model.eval()
        all_preds, all_labels, all_probs = ([], [], [])
        document_true, document_pred = [], []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(Config.DEVICE)
                output = self.model(batch["features"], batch["labels"])
                logits = output["logits"]
                preds = output["preds"]
                labels = batch["labels"]
                probs = F.softmax(logits, dim=-1)
                mask = labels != -1
                all_preds.extend(preds[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
                all_probs.extend(probs[mask].cpu().numpy())
                # One classification decision per original document. Decode tag
                # classes before voting so B/M/E/S variants vote together.
                for item_idx in range(labels.size(0)):
                    valid = mask[item_idx]
                    if not valid.any():
                        continue
                    predicted_tags = preds[item_idx][valid].cpu().tolist()
                    predicted_classes = [
                        self.label_processor.class2id[
                            self.label_processor.id2tag[tag].split("-", 1)[1]
                        ] for tag in predicted_tags
                    ]
                    true_tag = labels[item_idx][valid][0].item()
                    true_class = self.label_processor.class2id[
                        self.label_processor.id2tag[true_tag].split("-", 1)[1]
                    ]
                    document_true.append(true_class)
                    document_pred.append(
                        np.bincount(predicted_classes,
                                    minlength=len(self.label_processor.class2id)).argmax()
                    )
        if not all_preds:
            return
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        if not final_report:
            print(f"   [Val] Token-ACC: {accuracy:.4f} | Token-F1: {f1:.4f}")
        else:
            os.makedirs(self.output_dir, exist_ok=True)

            def tag_id_to_class_idx(tag_id):
                if tag_id == -1:
                    return -1
                class_str = self.label_processor.id2tag[tag_id].split("-", 1)[1]
                return self.label_processor.class2id[class_str]

            cls_preds = [tag_id_to_class_idx(p) for p in all_preds]
            cls_labels = [tag_id_to_class_idx(l) for l in all_labels]
            target_ids = list(self.label_processor.class2id.values())
            target_names = list(self.label_processor.class2id.keys())
            report_str = classification_report(
                cls_labels,
                cls_preds,
                labels=target_ids,
                target_names=target_names,
                digits=4,
                zero_division=0,
            )
            txt_path = os.path.join(self.output_dir, Config.REPORT_SAVE_NAME)
            document_report = classification_report(
                document_true, document_pred,
                labels=target_ids, target_names=target_names,
                digits=4, zero_division=0,
            )
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("DOCUMENT-LEVEL SOURCE CLASSIFICATION (primary)\n")
                f.write(document_report)
                f.write("\nTOKEN-LEVEL TAG CLASSIFICATION (diagnostic)\n")
                f.write(report_str)
                f.write(
                    f"\n    accuracy                           {accuracy:.4f}       {len(cls_labels)}\n"
                )
            print(f"\n[*] Classification report saved: {txt_path}")
            num_classes = len(target_names)
            probs_np = np.array(all_probs)
            class_probs = np.zeros((len(all_labels), num_classes))
            for tag_id, tag_name in self.label_processor.id2tag.items():
                class_idx = self.label_processor.class2id[tag_name.split("-", 1)[1]]
                class_probs[:, class_idx] += probs_np[:, tag_id]
            class_probs = class_probs / (class_probs.sum(axis=1, keepdims=True) + 1e-09)
            labels_bin = label_binarize(cls_labels, classes=target_ids)
            if num_classes == 2 and labels_bin.shape[1] == 1:
                labels_bin = np.hstack((1 - labels_bin, labels_bin))
            individual_aucs = {}
            auc_scores = []
            for i, class_name in enumerate(target_names):
                try:
                    if np.sum(labels_bin[:, i]) > 0:
                        auc = roc_auc_score(labels_bin[:, i], class_probs[:, i])
                    else:
                        auc = 0.0
                except:
                    auc = 0.0
                auc_scores.append(auc)
                individual_aucs[str(i)] = {"class_name": class_name, "auc": auc}
            json_output = {
                "accuracy": accuracy,
                "document_accuracy": accuracy_score(document_true, document_pred),
                "document_macro_f1": f1_score(
                    document_true, document_pred, labels=target_ids,
                    average="macro", zero_division=0,
                ),
                "token_macro_f1": f1,
                "mean_auc": np.mean(auc_scores),
                "individual_aucs": individual_aucs,
            }
            json_path = os.path.join(self.output_dir, Config.METRICS_SAVE_NAME)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_output, f, indent=4)

    def train(self):
        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch + 1}: train loss={train_loss:.4f}")
        # Hold out test data for a single evaluation after training.
        self.evaluate(final_report=True)


def main():
    if torch.cuda.is_available() and Config.GPU_INDEX in [0, 1]:
        torch.cuda.set_device(Config.GPU_INDEX)
        print(
            f"\n[*] Device: cuda:{Config.GPU_INDEX} ({torch.cuda.get_device_name(Config.GPU_INDEX)})"
        )
    else:
        print(f"\n[*] Device: {Config.DEVICE}")
    print("=" * 60)
    print("=== SeqXGPT pipeline ===")
    print("=" * 60)
    task_vault = {}
    for task in Config.TASKS:
        t_name = task["task_name"]
        print(f"\n>>> Initializing task: {t_name}")
        _, _, raw_train = load_data(task["train"], max_per_class=None)
        if not raw_train:
            continue
        label_proc = LabelProcessor(raw_train)
        tr_txt, tr_cls, _ = load_data(
            task["train"], label_proc.valid_classes, Config.MAX_SAMPLES_PER_CLASS_TRAIN
        )
        te_txt, te_cls, _ = load_data(
            task["test"], label_proc.valid_classes, Config.MAX_SAMPLES_PER_CLASS_TEST
        )
        if not tr_txt or not te_txt:
            continue
        tr_words = [t.split()[: Config.SEQ_LEN] for t in tr_txt]
        te_words = [t.split()[: Config.SEQ_LEN] for t in te_txt]
        num_models = len(Config.PROXY_MODELS)
        tr_feats = [np.zeros((len(w), num_models), dtype=np.float32) for w in tr_words]
        te_feats = [np.zeros((len(w), num_models), dtype=np.float32) for w in te_words]
        os.makedirs(task["output_dir"], exist_ok=True)
        task_vault[t_name] = {
            "label_processor": label_proc,
            "train_classes": tr_cls,
            "test_classes": te_cls,
            "train_words": tr_words,
            "test_words": te_words,
            "train_texts": tr_txt,
            "test_texts": te_txt,
            "train_feats": tr_feats,
            "test_feats": te_feats,
            "output_dir": task["output_dir"],
        }
    for model_idx, model_config in enumerate(Config.PROXY_MODELS):
        m_type = model_config["type"]
        need_extract = False
        for t_name, data in task_vault.items():
            train_cache = os.path.join(
                data["output_dir"], f"seqxgpt_all_samples_feat_{m_type}_train.npy"
            )
            test_cache = os.path.join(
                data["output_dir"], f"seqxgpt_all_samples_feat_{m_type}_test.npy"
            )
            if os.path.exists(train_cache) and os.path.exists(test_cache):
                print(f"[*] Cache found: {t_name} loaded {m_type} features.")
                tr_loaded = np.load(train_cache, allow_pickle=True)
                te_loaded = np.load(test_cache, allow_pickle=True)
                for i in range(len(tr_loaded)):
                    v_len = min(len(tr_loaded[i]), data["train_feats"][i].shape[0])
                    data["train_feats"][i][:v_len, model_idx] = tr_loaded[i][:v_len]
                for i in range(len(te_loaded)):
                    v_len = min(len(te_loaded[i]), data["test_feats"][i].shape[0])
                    data["test_feats"][i][:v_len, model_idx] = te_loaded[i][:v_len]
                data[f"cached_{m_type}"] = True
            else:
                data[f"cached_{m_type}"] = False
                need_extract = True
        if not need_extract:
            print(
                f"✅ Model {m_type} features loaded for all tasks; skipping model loading."
            )
            continue
        extractor = SeqXGPTFeatureExtractor(model_config)
        for t_name, data in task_vault.items():
            if data.get(f"cached_{m_type}", False):
                continue
            print(f"\n>>> [{m_type}] Extracting task features: {t_name}")
            m_train_feats = []
            for i, text in enumerate(
                tqdm(data["train_texts"], desc="Train", leave=False)
            ):
                aligned = extractor.extract_aligned_scores(text, data["train_words"][i])
                v_len = min(len(aligned), data["train_feats"][i].shape[0])
                if v_len > 0:
                    data["train_feats"][i][:v_len, model_idx] = aligned[:v_len]
                m_train_feats.append(aligned)
            m_test_feats = []
            for i, text in enumerate(
                tqdm(data["test_texts"], desc="Test", leave=False)
            ):
                aligned = extractor.extract_aligned_scores(text, data["test_words"][i])
                v_len = min(len(aligned), data["test_feats"][i].shape[0])
                if v_len > 0:
                    data["test_feats"][i][:v_len, model_idx] = aligned[:v_len]
                m_test_feats.append(aligned)
            train_cache = os.path.join(
                data["output_dir"], f"seqxgpt_all_samples_feat_{m_type}_train.npy"
            )
            test_cache = os.path.join(
                data["output_dir"], f"seqxgpt_all_samples_feat_{m_type}_test.npy"
            )
            np.save(train_cache, np.array(m_train_feats, dtype=object))
            np.save(test_cache, np.array(m_test_feats, dtype=object))
            print(
                f"  Features saved: [{t_name}] {m_type} features saved to: {data['output_dir']}"
            )
        extractor.free_memory()
    print("\n" + "=" * 50)
    print("🚀 Starting SeqXGPT network training")
    print("=" * 50)
    for t_name, data in task_vault.items():
        print(f"\n>>> Starting training for task: {t_name}")
        train_dataset = SeqXGPTDataset(
            data["train_feats"],
            data["train_words"],
            data["train_classes"],
            data["label_processor"],
        )
        test_dataset = SeqXGPTDataset(
            data["test_feats"],
            data["test_words"],
            data["test_classes"],
            data["label_processor"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        model = SeqXGPTModel(
            num_tags=data["label_processor"].num_tags,
            num_proxies=len(Config.PROXY_MODELS),
            seq_len=Config.SEQ_LEN,
        )
        trainer = Trainer(
            model,
            train_loader,
            test_loader,
            data["label_processor"],
            data["output_dir"],
        )
        trainer.train()
        save_path = os.path.join(data["output_dir"], Config.MODEL_SAVE_NAME)
        torch.save(model.state_dict(), save_path)
        print(
            f"[*] Task {t_name} complete; metrics and model saved to: {data['output_dir']}"
        )
    print("\n=== All tasks complete ===")
