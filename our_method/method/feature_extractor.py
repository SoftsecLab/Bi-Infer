"""
Feature extraction module for Bi-Infer.
Implements adaptive K-value calculation and bidirectional PMI feature extraction
with surrogate language models.
"""

import torch
import numpy as np
import json
import os
import gc
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Tuple, Optional, List, Dict


# ==========================================
# 1. Text Topology Preprocessing
# ==========================================

def get_text_k_metrics(text: str) -> float:
    """
    Calculate recommended window K value based on text linguistic density.
    
    Args:
        text: Input raw text string.
    
    Returns:
        float: Recommended K value for span interaction window.
    """
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return 2.0
    
    # Count sentences by sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    num_sentences = max(len(sentences), 1)
    avg_sentence_len = total_words / num_sentences
    
    # Count intra-sentence breakpoints (supports both Chinese and English)
    breakpoints = re.findall(r'[,;:—\-(){}\[\]…。、；：]', text)
    punct_density = len(breakpoints) / total_words
    punct_density = min(0.9, punct_density) 
    
    # Compute effective correlation length
    effective_len = avg_sentence_len * (1 - punct_density)
    return float(np.sqrt(effective_len) + 1)


# ==========================================
# 2. Core Feature Extractor Class
# ==========================================

class DynamicPMIBaseExtractor:
    """
    Base class for dynamic bidirectional PMI feature extraction.
    Outputs both entropy-normalized and raw feature matrices.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self.epsilon = 1e-6
        self.max_surprisal = 20.0
        self.tokenizer = None
        self.model = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load tokenizer and causal LM from local checkpoint."""
        print(f"   [Model Loader] Loading model: {os.path.basename(self.model_path)}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float16, 
            device_map={"": self.device}, 
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).eval()

    def get_pmi_matrix(self, text: str, k_fixed: int) -> Tuple[Optional[Tuple], Optional[Tuple], int]:
        """
        Extract full bidirectional PMI feature matrices for input text.
        
        Args:
            text: Input text string.
            k_fixed: Fixed global window size.
        
        Returns:
            Tuple of (normalized_feature_tuple, raw_feature_tuple, word_count).
            Returns (None, None, 0) if text is too short for the window.
        """
        words = text.split()
        num_words = len(words)
        if num_words < k_fixed + 1: 
            return None, None, 0

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", return_offsets_mapping=True, 
                truncation=True, max_length=1024
            ).to(self.device)
            input_ids = inputs.input_ids[0]
            seq_len = len(input_ids)
            offset_mapping = inputs.offset_mapping[0].cpu().numpy()
            
            with torch.no_grad():
                outputs = self.model(inputs.input_ids)
                probs = torch.softmax(outputs.logits[0], dim=-1)

            # Compute Shannon entropy for normalization
            log_probs = torch.log(probs.clamp(min=self.epsilon))
            shannon_entropy = -torch.sum(probs * log_probs, dim=-1).clamp(min=self.epsilon)

            # Compute base token-level surprisal
            target_ids = input_ids[1:]
            token_probs = probs[:-1].gather(1, target_ids.unsqueeze(1)).squeeze(1)
            token_surprisal = -torch.log(token_probs.to(torch.float32).clamp(min=self.epsilon))
            token_surprisal = torch.cat(
                [torch.tensor([0.0], device=self.device), token_surprisal]
            ).clamp(max=self.max_surprisal)

            # Initialize result matrices: n=normalized, r=raw; f=forward, b=backward, i=bidirectional
            result_mats = {
                key: torch.zeros((k_fixed, seq_len), device=self.device)
                for key in ['nf', 'nb', 'ni', 'rf', 'rb', 'ri']
            }

            for k in range(1, k_fixed + 1):
                if seq_len <= k:
                    continue
                
                # Forward span surprisal
                p_forward = probs[:seq_len-k].gather(1, input_ids[k:].unsqueeze(1)).squeeze(1)
                s_forward = -torch.log(p_forward.to(torch.float32).clamp(min=self.epsilon)).clamp(max=self.max_surprisal)
                
                # Backward span surprisal
                p_backward = probs[k-1:-1].gather(1, input_ids[:seq_len-k].unsqueeze(1)).squeeze(1)
                s_backward = -torch.log(p_backward.to(torch.float32).clamp(min=self.epsilon)).clamp(max=self.max_surprisal)
                
                min_len = min(len(s_forward), len(s_backward))
                entropy_window = shannon_entropy[k : k+min_len]
                
                # Raw probability fluctuation differences
                diff_forward = token_surprisal[k : k+min_len] - s_forward[:min_len]
                diff_backward = token_surprisal[0 : min_len] - s_backward[:min_len]
                diff_bidir = (diff_forward + diff_backward) / 2.0
                
                # Store both raw and entropy-normalized values
                result_mats['rf'][k-1, k:k+min_len] = diff_forward
                result_mats['nf'][k-1, k:k+min_len] = diff_forward / entropy_window
                result_mats['rb'][k-1, k:k+min_len] = diff_backward
                result_mats['nb'][k-1, k:k+min_len] = diff_backward / entropy_window
                result_mats['ri'][k-1, k:k+min_len] = diff_bidir
                result_mats['ni'][k-1, k:k+min_len] = diff_bidir / entropy_window

            # Aggregate token-level features to word-level
            word_features = {
                key: np.zeros((k_fixed, num_words), dtype=np.float32)
                for key in result_mats.keys()
            }
            word_spans = []
            curr_char = 0
            for word in words:
                start = text.find(word, curr_char)
                if start == -1:
                    start = curr_char
                end = start + len(word)
                word_spans.append((start, end))
                curr_char = end

            token_ptr = 0
            for w_idx, (w_start, w_end) in enumerate(word_spans):
                token_indices = []
                temp_ptr = token_ptr
                while temp_ptr < seq_len:
                    t_start, t_end = offset_mapping[temp_ptr]
                    if t_start >= w_end:
                        break
                    if max(w_start, t_start) < min(w_end, t_end):
                        token_indices.append(temp_ptr)
                    if t_end <= w_end:
                        token_ptr = temp_ptr + 1
                    temp_ptr += 1
                if token_indices:
                    for key in result_mats.keys():
                        word_features[key][:, w_idx] = torch.mean(
                            result_mats[key][:, token_indices], dim=1
                        ).cpu().numpy()

            return (word_features['nf'], word_features['nb'], word_features['ni']), \
                   (word_features['rf'], word_features['rb'], word_features['ri']), num_words

        except Exception:
            return None, None, 0

    def unload(self):
        """Unload model from memory and free GPU resources."""
        if self.model is not None:
            del self.model, self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()


# --- Model-specific extractor subclasses ---
class LlamaPMIExtractor(DynamicPMIBaseExtractor):
    """Extractor for LLaMA-family models."""
    pass

class QwenPMIExtractor(DynamicPMIBaseExtractor):
    """Extractor for Qwen-family models."""
    pass


def get_extractor_class(model_type: str):
    """
    Dispatch the corresponding extractor class by model type.
    
    Args:
        model_type: String identifier of model architecture.
    
    Returns:
        Extractor class object.
    """
    m_type = model_type.lower()
    if "llama" in m_type:
        return LlamaPMIExtractor
    if "qwen" in m_type:
        return QwenPMIExtractor
    return DynamicPMIBaseExtractor


# ==========================================
# 3. Global K Selection & Batch Extraction Pipeline
# ==========================================

def process_multimodel_global_k(tasks: List[Dict], model_configs: List[Dict]):
    """
    Full feature extraction workflow:
    1. Calculate global K value across all datasets
    2. Extract features with multiple surrogate models
    3. Align and save combined features
    
    Args:
        tasks: List of dataset task configurations.
        model_configs: List of surrogate model configurations.
    """
    # --- Stage 1: Global K-value statistics (ceiling rounding) ---
    print("\n" + "="*75)
    print("Stage 1: Global Text Topology Scan (K-Value Selection)")
    print("="*75)
    
    all_k_values = []
    for task in tasks:
        with open(task['input'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            for label in data:
                texts = data[label][:task['max_samples']]
                for text in texts:
                    all_k_values.append(get_text_k_metrics(text))
    
    global_k_avg = np.mean(all_k_values)
    global_k_fixed = int(max(2, np.ceil(global_k_avg)))
    
    print(f"   [Parameters Log]")
    print(f"   >> Total samples analyzed: {len(all_k_values)}")
    print(f"   >> Raw average K value: {global_k_avg:.4f}")
    print(f"   >> K value std deviation: {np.std(all_k_values):.4f}")
    print(f"   >> Locked global K (ceiling rounded): {global_k_fixed}")
    print("="*75 + "\n")

    # --- Stage 2: Process each task sequentially ---
    for task in tasks:
        task_name = os.path.basename(task['input'])
        print(f"Processing task: {task_name}")
        with open(task['input'], 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Preload all samples
        samples = []
        for label in raw_data:
            texts = raw_data[label][:task['max_samples']]
            for text in texts:
                samples.append({
                    'text': text, 'label': label, 'k_opt': global_k_fixed,
                    'nf_parts': [], 'nb_parts': [], 'ni_parts': [],
                    'rf_parts': [], 'rb_parts': [], 'ri_parts': []
                })

        # Extract features with each model sequentially
        for cfg in model_configs:
            print(f"   [Model Active] {cfg['type'].upper()} | Path: {cfg['path']}")
            extractor = get_extractor_class(cfg['type'])(cfg['path'])
            
            for i in tqdm(range(len(samples)), desc=f"   Extraction Progress"):
                norms, raws, n_words = extractor.get_pmi_matrix(samples[i]['text'], global_k_fixed)
                if n_words > 0:
                    for idx, key in enumerate(['nf_parts', 'nb_parts', 'ni_parts']):
                        samples[i][key].append(norms[idx])
                    for idx, key in enumerate(['rf_parts', 'rb_parts', 'ri_parts']):
                        samples[i][key].append(raws[idx])
                else:
                    # Mark failed samples with None
                    for key in ['nf_parts', 'nb_parts', 'ni_parts', 'rf_parts', 'rb_parts', 'ri_parts']: 
                        samples[i][key].append(None)
            extractor.unload()

        # --- Stage 3: Stack and save dual-track features ---
        def save_combined(suffix: str, norm_key: str, raw_key: str):
            final_list = []
            for sample in samples:
                # Skip samples with failed extraction from any model
                if any(v is None for v in sample[norm_key]):
                    continue
                
                min_len = min([p.shape[1] for p in sample[norm_key]])
                if min_len < 7:  # Filter excessively short texts
                    continue
                
                final_list.append({
                    'features': np.vstack([p[:, :min_len] for p in sample[norm_key]]),
                    'raw_features': np.vstack([p[:, :min_len] for p in sample[raw_key]]),
                    'label': sample['label'],
                    'length': min_len,
                    'k_opt': global_k_fixed,
                    'text': sample['text']
                })
            
            out_path = f"{task['output_base']}_{suffix}.pt"
            torch.save(final_list, out_path)
            print(f"      [{suffix.upper()}] Saved | Valid samples: {len(final_list)} | Path: {out_path}")

        print(f"   [Storage Operation] Aligning multi-model features...")
        save_combined('fwd', 'nf_parts', 'rf_parts')
        save_combined('bwd', 'nb_parts', 'rb_parts')
        save_combined('bi', 'ni_parts', 'ri_parts')
        print("-" * 60)


if __name__ == "__main__":
    from config import FEATURE_EXTRACT_CONFIG
    process_multimodel_global_k(
        FEATURE_EXTRACT_CONFIG["tasks"],
        FEATURE_EXTRACT_CONFIG["model_configs"]
    )