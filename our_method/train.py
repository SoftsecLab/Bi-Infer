"""Train the main Bi-Infer classifier and evaluate the best checkpoint."""
import copy
import gc
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from .config import *
from .dataset import MultiModelDataset, collate_fn, load_composed_cache
from .net import M3FNet

MODEL_DISPLAY = MODEL_PATHS

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def json_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def plot_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "g--", label="Val Loss")
    plt.title("Loss Curve (Successful Epochs Only)")
    plt.xlabel("Successful Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-", label="Train Acc (Token)")
    plt.plot(epochs, history["val_acc"], "g--", label="Val Acc (Sentence)")
    plt.title("Accuracy Curve (Successful Epochs Only)")
    plt.xlabel("Successful Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train(train_path, test_path, output_dir, gpu_id=0):
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training.")

    device = torch.device(f"cuda:{gpu_id}")

    train_path, test_path = Path(train_path), Path(test_path)
    domain, config_name = "dataset", "main"
    model_keys = list(MODEL_PATHS)
    feature_mode = FEATURE_MODE

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train cache: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test cache: {test_path}")

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    save_path = run_dir / f"best_model_{feature_mode}.pth"
    plot_path = run_dir / f"training_curves_{feature_mode}.png"
    result_path = run_dir / f"result_{feature_mode}.json"
    report_path = run_dir / f"prediction_report_{feature_mode}.json"

    train_meta, train_raw = load_composed_cache(train_path)
    test_meta, test_raw = load_composed_cache(test_path)

    all_raw_labels = sorted(
        list(
            set(
                [str(d["label"]) for d in train_raw]
                + [str(d["label"]) for d in test_raw]
            )
        )
    )
    raw_label_to_id = {
        name: idx for idx, name in enumerate(all_raw_labels)
    }

    unknown_test_labels = {str(d["label"]) for d in test_raw} - {str(d["label"]) for d in train_raw}
    if unknown_test_labels:
        raise ValueError(f"Unknown test labels: {unknown_test_labels}")
    excluded_raw_names = set()
    for raw_label, raw_id in raw_label_to_id.items():
        if raw_label in EXCLUDED_LABELS or raw_id in EXCLUDED_LABELS:
            excluded_raw_names.add(raw_label)

    remaining_labels = sorted(
        list(
            {
                str(d["label"])
                for d in train_raw
                if str(d["label"]) not in excluded_raw_names
            }
        )
    )
    final_label_map = {
        old_name: new_idx
        for new_idx, old_name in enumerate(remaining_labels)
    }
    num_classes = len(remaining_labels)

    if num_classes == 0:
        raise ValueError(f"No valid classes: {domain}/{config_name}")

    sample_feat_key = (
        "features" if feature_mode == "normalized" else "raw_features"
    )
    base_channels = int(
        torch.tensor(train_raw[0][sample_feat_key]).shape[0]
    )

    full_train_dataset = MultiModelDataset(
        train_path,
        label_map=final_label_map,
        feature_type=feature_mode,
        target_channels=base_channels,
    )
    test_dataset = MultiModelDataset(
        test_path,
        label_map=final_label_map,
        feature_type=feature_mode,
        target_channels=base_channels,
    )

    val_size = int(len(full_train_dataset) * VAL_RATIO)
    if val_size <= 0:
        raise ValueError("Validation set is empty.")

    train_size = len(full_train_dataset) - val_size
    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = M3FNet(
        input_dim=base_channels,
        d_model=MODEL_DIM,
        num_classes=num_classes,
        dropout=DROPOUT,
        num_layers=NUM_LAYERS,
        nhead=NHEAD,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=MIN_LR,
    )
    class_weights = None
    if USE_CLASS_WEIGHTS:
        counts = torch.zeros(num_classes, device=device)
        for idx in train_subset.indices:
            sample = full_train_dataset[idx]
            counts[sample["label"].item()] += min(sample["length"], MAX_SEQ_LEN)
        if (counts == 0).any():
            raise ValueError("Training split lacks a class; supply a larger dataset or a different split.")
        class_weights = counts.sum() / (num_classes * counts)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=-100, label_smoothing=LABEL_SMOOTHING,
        reduction="sum"
    )

    print("\n" + "=" * 100)
    print(f"DATASET={domain} | CONFIG={config_name}")
    print(
        "Models: "
        + " + ".join(MODEL_DISPLAY.get(k, k) for k in model_keys)
    )
    print(f"Raw labels: {raw_label_to_id}")
    print(f"Final label map: {final_label_map}")
    print(
        f"GPU=cuda:{gpu_id} | input_dim={base_channels} | "
        f"train={len(train_subset)} | val={len(val_subset)} | "
        f"test={len(test_dataset)}"
    )
    print(
        f"Batch={BATCH_SIZE} | Epochs={MAX_SUCCESSFUL_EPOCHS} | "
        f"LR={LR} | WD={WEIGHT_DECAY} | Dropout={DROPOUT} | "
        f"MaxSeq={MAX_SEQ_LEN}"
    )
    print("=" * 100)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    successful_epochs = 0
    prev_val_loss = float("inf")
    best_overall_loss = float("inf")
    best_epoch = None
    best_val_acc = None
    total_attempts = 0
    current_epoch_retries = 0

    while successful_epochs < MAX_SUCCESSFUL_EPOCHS:
        total_attempts += 1
        current_lr = optimizer.param_groups[0]["lr"]

        current_margin = min(
            MAX_SPIKE_MARGIN,
            BASE_SPIKE_MARGIN + current_epoch_retries * RELAX_STEP,
        )
        spike_threshold = 1.0 + current_margin

        print(
            f"\n[Attempt #{total_attempts}] Trying Epoch "
            f"{successful_epochs + 1} (LR: {current_lr:.2e})"
        )
        print(
            f"Spike tolerance: +{current_margin*100:.1f}% | "
            f"Retries: {current_epoch_retries}"
        )

        last_safe_state = copy.deepcopy(model.state_dict())
        last_optimizer_state = copy.deepcopy(optimizer.state_dict())

        model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0

        loop = tqdm(
            train_loader,
            desc=f"Training {domain}/{config_name}",
            leave=False,
        )

        for x, _, mask, token_targets, _, _ in loop:
            x = x.to(device)
            mask = mask.to(device)
            token_targets = token_targets.to(device)

            optimizer.zero_grad()
            logits = model(x, mask)

            current_len = logits.size(1)
            aligned_targets = token_targets[:, :current_len]

            loss = criterion(
                logits.reshape(-1, num_classes),
                aligned_targets.reshape(-1),
            )
            loss = loss / (aligned_targets != -100).sum().clamp_min(1)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP_NORM
            )
            optimizer.step()

            epoch_loss += float(loss.item())

            preds = torch.argmax(logits, dim=-1)
            flat_preds = preds.reshape(-1)
            flat_targets = aligned_targets.reshape(-1)
            valid_mask = flat_targets != -100

            if valid_mask.sum() > 0:
                train_correct += (
                    flat_preds[valid_mask] == flat_targets[valid_mask]
                ).sum().item()
                train_total += valid_mask.sum().item()

            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = (
            train_correct / train_total if train_total > 0 else 0.0
        )

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for x, label, mask, token_targets, _, _ in val_loader:
                x = x.to(device)
                mask = mask.to(device)
                token_targets = token_targets.to(device)

                _, raw_logits = model(x, mask)
                current_val_len = raw_logits.size(1)
                aligned_val_targets = token_targets[:, :current_val_len]

                loss_val = criterion(
                    raw_logits.reshape(-1, num_classes),
                    aligned_val_targets.reshape(-1),
                )
                loss_val = loss_val / (aligned_val_targets != -100).sum().clamp_min(1)
                val_loss += float(loss_val.item())

                final_preds, _ = model(x, mask)
                val_preds.extend(final_preds.cpu().numpy().tolist())
                val_labels.extend(label.cpu().numpy().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = float(accuracy_score(val_labels, val_preds))

        print(
            f"Val Loss: {avg_val_loss:.4f} "
            f"(Base: {prev_val_loss:.4f}) | "
            f"Val Acc: {val_acc:.4f}"
        )

        is_spike = False
        if not np.isfinite(avg_val_loss):
            is_spike = True
        elif prev_val_loss != float("inf"):
            if avg_val_loss > prev_val_loss * spike_threshold:
                is_spike = True

        if is_spike:
            print(
                "[ROLLBACK] Invalidating this epoch and restoring model."
            )
            model.load_state_dict(last_safe_state)
            optimizer.load_state_dict(last_optimizer_state)
            current_epoch_retries += 1
            if current_epoch_retries > MAX_RETRIES:
                raise RuntimeError("Rollback retry limit reached; inspect losses and extraction caches.")

            for param_group in optimizer.param_groups:
                new_lr = max(current_lr * 0.5, MIN_LR)
                param_group["lr"] = new_lr

            print(f"[DYNAMIC LR] New LR = {new_lr:.2e}")
            continue

        successful_epochs += 1
        current_epoch_retries = 0
        prev_val_loss = avg_val_loss

        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_overall_loss:
            best_overall_loss = float(avg_val_loss)
            best_epoch = int(successful_epochs)
            best_val_acc = float(val_acc)

            torch.save(model.state_dict(), save_path)

            print(
                f"[SAVE] Best model updated | epoch={best_epoch} | "
                f"val_loss={best_overall_loss:.4f} | "
                f"val_acc={best_val_acc:.4f}"
            )

        print(
            f"[ACCEPTED] {successful_epochs}/{MAX_SUCCESSFUL_EPOCHS} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"train_acc(token)={train_acc:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_acc(doc)={val_acc:.4f}"
        )

    plot_training_curves(history, plot_path)

    print("\nFinal Evaluation (Loading Best Model)...")
    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )
    model.eval()

    reverse_map = {v: k for k, v in final_label_map.items()}
    analysis_report = []

    def evaluate_and_record(loader, split_name):
        preds_list = []
        labels_list = []

        with torch.no_grad():
            for x, label, mask, _, sample_ids, texts in tqdm(
                loader, desc=f"Evaluating {split_name}"
            ):
                x = x.to(device)
                mask = mask.to(device)

                preds, _ = model(x, mask)

                preds_np = preds.cpu().numpy().tolist()
                labels_np = label.cpu().numpy().tolist()

                preds_list.extend(preds_np)
                labels_list.extend(labels_np)

                for i in range(len(preds_np)):
                    analysis_report.append(
                        {
                            "split": split_name,
                            "sample_id": sample_ids[i],
                            "predicted_label": reverse_map.get(
                                int(preds_np[i]), "Unknown"
                            ),
                            "true_label": reverse_map.get(
                                int(labels_np[i]), "Unknown"
                            ),
                            "is_correct": bool(
                                preds_np[i] == labels_np[i]
                            ),
                            "text": texts[i],
                        }
                    )

        return labels_list, preds_list

    test_labels, test_preds = evaluate_and_record(
        test_loader, "test"
    )

    target_names = [
        str(reverse_map[i]) for i in range(num_classes)
    ]

    report_text = classification_report(
        test_labels,
        test_preds,
        labels=list(range(num_classes)),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        test_labels,
        test_preds,
        labels=list(range(num_classes)),
        target_names=target_names,
        digits=6,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(
        test_labels, test_preds, labels=list(range(num_classes))
    )
    cm_norm = confusion_matrix(
        test_labels,
        test_preds,
        labels=list(range(num_classes)),
        normalize="true",
    )

    accuracy = float(accuracy_score(test_labels, test_preds))
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            test_labels,
            test_preds,
            average="macro",
            zero_division=0,
        )
    )

    print("\n" + "=" * 72)
    print(f"FINAL TEST | {domain} | {config_name}")
    print("=" * 72)
    print(report_text)
    print("Confusion Matrix:")
    print(cm)
    print(
        f"Accuracy={accuracy:.4f} | Macro-F1={macro_f1:.4f}"
    )
    print("=" * 72)

    evaluate_and_record(full_train_loader, "train")
    json_dump(analysis_report, report_path)

    result = {
        "domain": domain,
        "config_name": config_name,
        "model_keys": list(model_keys),
        "model_names": [
            MODEL_DISPLAY.get(k, k) for k in model_keys
        ],
        "num_surrogates": len(model_keys),
        "feature_mode": feature_mode,
        "input_dim": base_channels,
        "num_classes": num_classes,
        "class_names": target_names,
        "train_samples_total": len(full_train_dataset),
        "train_samples": len(train_subset),
        "val_samples": len(val_subset),
        "test_samples": len(test_dataset),
        "best_epoch": best_epoch,
        "best_val_loss": best_overall_loss,
        "best_val_acc": best_val_acc,
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "history": history,
        "training_config": {
            "batch_size": BATCH_SIZE,
            "max_successful_epochs": MAX_SUCCESSFUL_EPOCHS,
            "lr": LR,
            "min_lr": MIN_LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "d_model": MODEL_DIM,
            "num_layers": NUM_LAYERS,
            "nhead": NHEAD,
            "max_seq_len": MAX_SEQ_LEN,
            "label_smoothing": LABEL_SMOOTHING,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "val_ratio": VAL_RATIO,
            "seed": SEED,
            "gpu_id": gpu_id,
            "class_weights": USE_CLASS_WEIGHTS,
            "rollback_halves_lr": True,
        },
        "source_cache_meta": {
            "train": train_meta,
            "test": test_meta,
        },
    }

    torch.save({"state_dict": model.state_dict(), "label_map": final_label_map,
                "input_dim": base_channels, "d_model": MODEL_DIM,
                "num_classes": num_classes, "dropout": DROPOUT,
                "num_layers": NUM_LAYERS, "nhead": NHEAD,
                "max_seq_len": MAX_SEQ_LEN}, run_dir / "checkpoint.pt")
    json_dump(result, result_path)

    del model, optimizer, scheduler, criterion
    torch.cuda.empty_cache()
    gc.collect()

    return result

