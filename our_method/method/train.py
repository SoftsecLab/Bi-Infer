"""
Training, validation, and evaluation pipeline for Bi-Infer source tracing model.
Implements dynamic loss spike detection with automatic rollback and LR decay.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import json
import copy
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

from models import M3FNet
from data import MultiModelDataset, collate_fn
from utils import plot_training_curves

warnings.filterwarnings('ignore')


def train_model(config: dict):
    """
    Full training and evaluation pipeline.
    
    Args:
        config: Training configuration dictionary from config.py.
    """
    # Unpack configuration
    feature_mode = config["feature_mode"]
    train_path = config["train_data_path"]
    test_path = config["test_data_path"]
    save_path = config["model_save_path"]
    plot_path = config["plot_save_path"]
    report_path = config["report_save_path"]
    
    device = config["device"] if torch.cuda.is_available() else "cpu"
    batch_size = config["batch_size"]
    max_successful_epochs = config["max_successful_epochs"]
    lr = config["learning_rate"]
    min_lr = config["min_learning_rate"]
    weight_decay = config["weight_decay"]
    dropout = config["dropout"]
    model_dim = config["model_dim"]
    num_layers = config["num_layers"]
    nhead = config["num_heads"]
    max_seq_len = config["max_seq_len"]
    excluded_labels = config["excluded_labels"]

    print(f"Loading dataset... [Feature mode: {feature_mode.upper()}]")
    
    # Load training dataset
    full_train_dataset = MultiModelDataset(train_path, feature_type=feature_mode)
    
    # Get base channel count from training data
    feat_key = 'features' if feature_mode == 'normalized' else 'raw_features'
    base_channels = torch.tensor(full_train_dataset.data[0][feat_key]).shape[0]
    print(f"Base channel count (training set): {base_channels}")
    
    full_train_dataset.target_channels = base_channels
    
    # Load test dataset and align channels
    test_dataset = MultiModelDataset(test_path, feature_type=feature_mode, target_channels=base_channels)
    
    # Build raw label mapping
    all_raw_labels = sorted(list(set(
        [d['label'] for d in full_train_dataset.data] + 
        [d['label'] for d in test_dataset.data]
    )))
    raw_label_to_id = {name: idx for idx, name in enumerate(all_raw_labels)}
    print(f"Global raw label mapping: {raw_label_to_id}")

    # Filter excluded classes
    excluded_raw_names = set()
    for raw_label, raw_id in raw_label_to_id.items():
        if raw_label in excluded_labels or raw_id in excluded_labels:
            excluded_raw_names.add(raw_label)
            
    if excluded_raw_names:
        print(f"Excluded classes: {excluded_raw_names}")

    full_train_dataset.data = [
        d for d in full_train_dataset.data if d['label'] not in excluded_raw_names
    ]
    test_dataset.data = [
        d for d in test_dataset.data if d['label'] not in excluded_raw_names
    ]

    # Build final contiguous label mapping
    remaining_labels = sorted(list(set([d['label'] for d in full_train_dataset.data])))
    num_classes = len(remaining_labels)
    final_label_map = {old_name: new_idx for new_idx, old_name in enumerate(remaining_labels)}
    print(f"Final class index mapping: {final_label_map}")

    # Apply label mapping
    for d in full_train_dataset.data: 
        d['label'] = final_label_map[d['label']]
        
    valid_test_data = []
    for d in test_dataset.data:
        if d['label'] in final_label_map:
            d['label'] = final_label_map[d['label']]
            valid_test_data.append(d)
    test_dataset.data = valid_test_data
    
    # Split training set into train/validation
    val_size = int(len(full_train_dataset) * 0.1)
    train_subset, val_subset = random_split(
        full_train_dataset, 
        [len(full_train_dataset)-val_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    full_train_loader = DataLoader(
        full_train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    # Initialize model
    model = M3FNet(
        input_dim=base_channels, d_model=model_dim, 
        num_classes=num_classes, dropout=dropout, num_layers=num_layers, nhead=nhead,
        max_seq_len=max_seq_len
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=min_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    successful_epochs = 0
    prev_val_loss = float('inf')
    best_val_loss = float('inf')
    
    total_attempts = 0        
    epoch_retries = 0 
    
    print(f"\nStarting training... (Max sequence length: {max_seq_len})")
    
    while successful_epochs < max_successful_epochs:
        total_attempts += 1
        current_lr = optimizer.param_groups[0]['lr']
        
        # Dynamic loss spike tolerance
        base_margin = 0.02
        relax_step = 0.005
        max_margin = 0.04
        current_margin = min(max_margin, base_margin + (epoch_retries * relax_step))
        spike_threshold = 1.0 + current_margin
        
        print(f"\n[Attempt #{total_attempts}] Epoch {successful_epochs + 1} (LR: {current_lr:.2e})")
        print(f"   Loss spike tolerance: +{current_margin*100:.1f}% | Retries: {epoch_retries}")
        
        # Save safe checkpoint before epoch
        last_safe_state = copy.deepcopy(model.state_dict())
        
        # Training phase
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc="Training", leave=False)
        for x, label, mask, token_targets, _ in loop: 
            x, mask, token_targets = x.to(device), mask.to(device), token_targets.to(device)
            
            optimizer.zero_grad()
            logits = model(x, mask)
            
            current_len = logits.size(1)
            aligned_targets = token_targets[:, :current_len]
            
            loss = criterion(logits.reshape(-1, num_classes), aligned_targets.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate token-level training accuracy
            preds = torch.argmax(logits, dim=-1)   
            flat_preds = preds.reshape(-1)            
            flat_targets = aligned_targets.reshape(-1)  
            valid_mask = flat_targets != -100       
            
            if valid_mask.sum() > 0:
                train_correct += (flat_preds[valid_mask] == flat_targets[valid_mask]).sum().item()
                train_total += valid_mask.sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for x, label, mask, token_targets, _ in val_loader: 
                x, mask, token_targets = x.to(device), mask.to(device), token_targets.to(device)
                
                _, raw_logits = model(x, mask)
                current_val_len = raw_logits.size(1)
                aligned_val_targets = token_targets[:, :current_val_len]
                
                loss_val = criterion(raw_logits.reshape(-1, num_classes), aligned_val_targets.reshape(-1))
                val_loss += loss_val.item()
                
                final_preds, _ = model(x, mask)
                val_preds.extend(final_preds.cpu().numpy())
                val_labels.extend(label.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        print(f"   Val Loss: {avg_val_loss:.4f} (Previous: {prev_val_loss:.4f}) | Val Acc: {val_acc:.4f}")

        # Detect loss spikes
        is_spike = False
        if np.isnan(avg_val_loss):
            print("   [FAILURE] Loss is NaN!")
            is_spike = True
        elif prev_val_loss != float('inf'):
            if avg_val_loss > prev_val_loss * spike_threshold:
                diff = (avg_val_loss - prev_val_loss) / prev_val_loss * 100
                print(f"   [WARNING] Loss spiked by {diff:.2f}% (Limit: {current_margin*100:.1f}%).")
                is_spike = True
        
        if is_spike:
            print("   [ROLLBACK] Reverting to last safe checkpoint...")
            model.load_state_dict(last_safe_state) 
            epoch_retries += 1
            
            # Reduce learning rate for next retry
            for param_group in optimizer.param_groups:
                new_lr = max(param_group['lr'] * 0.5, min_lr)
                param_group['lr'] = new_lr
            print(f"   [Dynamic LR] Reduced to {new_lr:.2e} for next retry.")
            
            continue 
        else:
            successful_epochs += 1
            epoch_retries = 0  
            prev_val_loss = avg_val_loss
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                print(f"   [SAVE] Best model updated (Loss: {best_val_loss:.4f})")
            
            print(f"   [ACCEPTED] Epoch {successful_epochs}/{max_successful_epochs} completed.")

    print(f"\nTraining complete! {max_successful_epochs} successful epochs collected.")
    plot_training_curves(history, plot_path)
    
    # Final evaluation on test set
    print("\nGenerating final evaluation report...")
    model.load_state_dict(torch.load(save_path)) 
    model.eval()
    
    analysis_report = []
    reverse_label_map = {v: k for k, v in final_label_map.items()}

    def evaluate_and_record(loader, split_name):
        preds_list, labels_list = [], []
        with torch.no_grad():
            for x, label, mask, _, texts in tqdm(loader, desc=f"Evaluating {split_name}"):
                x, mask = x.to(device), mask.to(device)
                preds, _ = model(x, mask)
                preds_np = preds.cpu().numpy()
                labels_np = label.numpy()
                preds_list.extend(preds_np)
                labels_list.extend(labels_np)
                
                for i in range(len(preds_np)):
                    pred_name = reverse_label_map.get(int(preds_np[i]), "Unknown")
                    true_name = reverse_label_map.get(int(labels_np[i]), "Unknown")
                    
                    analysis_report.append({
                        "split": split_name,
                        "predicted_label": pred_name, 
                        "true_label": true_name,
                        "is_correct": bool(preds_np[i] == labels_np[i]),
                        "text": texts[i]
                    })
        return labels_list, preds_list

    print(">> Evaluating test set...")
    test_labels, test_preds = evaluate_and_record(test_loader, "test")
    
    target_names = [reverse_label_map[i] for i in range(num_classes)]
    
    print("\n" + "="*50)
    print(f"Classification Report [{feature_mode.upper()} Features]")
    print("="*50)
    print(classification_report(test_labels, test_preds, target_names=target_names, digits=4))
    
    print("\n" + "="*50)
    print("Confusion Matrix")
    print("="*50)
    print(confusion_matrix(test_labels, test_preds))

    print(">> Evaluating full training set...")
    evaluate_and_record(full_train_loader, "train")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(analysis_report, f, ensure_ascii=False, indent=2)
    print(f"Full prediction report saved to: {report_path}")


if __name__ == "__main__":
    from config import TRAIN_CONFIG
    train_model(TRAIN_CONFIG)