"""Surrogate extraction, adapted from the supplied November test extractor."""
import gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import MAX_TOKEN_LENGTH, MAX_SURPRISAL, EPSILON
EXTRACTION_DTYPE = torch.float16

class DynamicPMIBaseExtractor:
    def __init__(self, model_path: str, device: str):
        self.device = device
        self.model_path = model_path
        self.epsilon = EPSILON
        self.max_surprisal = MAX_SURPRISAL
        self.tokenizer = None
        self.model = None
        self.last_error = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        print(f"\n[Model Loader] Loading: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True,
        )

        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                f"Fast tokenizer is required for offset_mapping: {self.model_path}"
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=EXTRACTION_DTYPE,
            device_map={"": self.device},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()

    def get_pmi_matrix(self, text: str, k_fixed: int):
        words = text.split()
        num_words = len(words)

        if num_words < k_fixed + 1:
            return None, None, 0

        try:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
            )

            offset_mapping = (
                encoded.pop("offset_mapping")[0].cpu().numpy()
            )

            input_ids = encoded["input_ids"].to(self.device)

            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            ids = input_ids[0]
            seq_len = int(ids.numel())
            covered_end = int(offset_mapping[:, 1].max())
            covered_words = []
            import re
            for match in re.finditer(r"\S+", text):
                if match.end() <= covered_end:
                    covered_words.append(match.group())
            words = covered_words
            num_words = len(words)
            if num_words < k_fixed + 1:
                return None, None, 0

            if seq_len <= k_fixed:
                return None, None, 0

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                probs = torch.softmax(outputs.logits[0].float(), dim=-1)

            log_probs = torch.log(
                probs.clamp(min=self.epsilon)
            )

            H_t = (-torch.sum(
                probs * log_probs,
                dim=-1,
            )).clamp(min=self.epsilon)

            target_ids = ids[1:]

            token_probs = probs[:-1].gather(
                1,
                target_ids.unsqueeze(1),
            ).squeeze(1)

            token_surprisal = -torch.log(
                token_probs.float().clamp(min=self.epsilon)
            )

            token_surprisal = torch.cat(
                [
                    torch.tensor([0.0], device=self.device),
                    token_surprisal,
                ]
            ).clamp(max=self.max_surprisal)

            res_mats = {
                key: torch.zeros(
                    (k_fixed, seq_len),
                    device=self.device,
                )
                for key in ["nf", "nb", "ni", "rf", "rb", "ri"]
            }

            for k in range(1, k_fixed + 1):
                if seq_len <= k:
                    continue

                p_fwd = probs[: seq_len - k].gather(
                    1,
                    ids[k:].unsqueeze(1),
                ).squeeze(1)

                s_fwd = -torch.log(
                    p_fwd.float().clamp(min=self.epsilon)
                ).clamp(max=self.max_surprisal)

                p_bwd = probs[k - 1 : -1].gather(
                    1,
                    ids[: seq_len - k].unsqueeze(1),
                ).squeeze(1)

                s_bwd = -torch.log(
                    p_bwd.float().clamp(min=self.epsilon)
                ).clamp(max=self.max_surprisal)

                min_len = min(
                    len(s_fwd),
                    len(s_bwd),
                )

                H_t_win = H_t[
                    k - 1 : k - 1 + min_len
                ]

                diff_fwd = (
                    token_surprisal[
                        k : k + min_len
                    ]
                    - s_fwd[:min_len]
                )

                diff_bwd = (
                    token_surprisal[:min_len]
                    - s_bwd[:min_len]
                )

                diff_bi = (
                    diff_fwd + diff_bwd
                ) / 2.0

                res_mats["rf"][
                    k - 1,
                    k : k + min_len,
                ] = diff_fwd

                res_mats["nf"][
                    k - 1,
                    k : k + min_len,
                ] = diff_fwd / H_t_win

                res_mats["rb"][
                    k - 1,
                    k : k + min_len,
                ] = diff_bwd

                res_mats["nb"][
                    k - 1,
                    k : k + min_len,
                ] = diff_bwd / H_t_win

                res_mats["ri"][
                    k - 1,
                    k : k + min_len,
                ] = diff_bi

                res_mats["ni"][
                    k - 1,
                    k : k + min_len,
                ] = diff_bi / H_t_win

            # token -> whitespace word mapping
            word_features = {
                key: np.zeros(
                    (k_fixed, num_words),
                    dtype=np.float32,
                )
                for key in res_mats.keys()
            }

            word_spans = []
            curr_char = 0

            for w in words:
                start = text.find(w, curr_char)

                if start == -1:
                    start = curr_char

                end = start + len(w)

                word_spans.append(
                    (start, end)
                )

                curr_char = end

            token_ptr = 0

            for w_idx, (
                w_start,
                w_end,
            ) in enumerate(word_spans):

                indices = []
                temp_ptr = token_ptr

                while temp_ptr < seq_len:
                    t_start, t_end = (
                        offset_mapping[temp_ptr]
                    )

                    if t_start >= w_end:
                        break

                    if max(
                        w_start,
                        t_start,
                    ) < min(
                        w_end,
                        t_end,
                    ):
                        indices.append(
                            temp_ptr
                        )

                    if t_end <= w_end:
                        token_ptr = (
                            temp_ptr + 1
                        )

                    temp_ptr += 1

                if indices:
                    for key in res_mats.keys():
                        word_features[
                            key
                        ][:, w_idx] = (
                            torch.mean(
                                res_mats[key][
                                    :, indices
                                ],
                                dim=1,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )

            return (
                word_features["nf"],
                word_features["nb"],
                word_features["ni"],
            ), (
                word_features["rf"],
                word_features["rb"],
                word_features["ri"],
            ), num_words

        except Exception as exc:
            self.last_error = repr(exc)
            return None, None, 0

    def unload(self):
        if self.model is not None:
            del self.model

        if self.tokenizer is not None:
            del self.tokenizer

        self.model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()


def extract_features(train_json, test_json, output_dir, model_paths, device='cuda:0'):
    """Extract both splits with one training-derived window and concatenate surrogates."""
    from pathlib import Path
    from .data import load_rows, dataset_window
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = {'train': load_rows(train_json), 'test': load_rows(test_json)}
    k = dataset_window(rows['train'])
    keys = list(model_paths)
    caches = {}
    for key, path in model_paths.items():
        extractor = DynamicPMIBaseExtractor(path, device)
        try:
            for split, samples in rows.items():
                kept, failures = {}, []
                for row in samples:
                    extractor.last_error = None
                    norms, raws, length = extractor.get_pmi_matrix(row['text'], k)
                    if norms is None or length < 7:
                        failures.append({'sample_id': row['sample_id'], 'reason': extractor.last_error or 'too short'})
                        continue
                    kept[row['sample_id']] = {
                        'sample_id': row['sample_id'], 'label': row['label'],
                        'length': length, 'k_opt': k,
                        'features': norms[2], 'raw_features': raws[2],
                    }
                caches[key, split] = kept
                torch.save({'meta': {'model': path, 'k_fixed': k, 'failures': failures},
                            'samples': list(kept.values())}, output_dir / f'{split}_{key}.pt')
        finally:
            extractor.unload()
    for split in rows:
        common = set.intersection(*(set(caches[key, split]) for key in keys))
        combined = []
        for row in rows[split]:
            sid = row['sample_id']
            if sid not in common:
                continue
            parts = [caches[key, split][sid] for key in keys]
            length = min(part['length'] for part in parts)
            combined.append({
                'sample_id': sid, 'label': row['label'], 'length': length,
                'features': np.vstack([part['features'][:, :length] for part in parts]),
                'raw_features': np.vstack([part['raw_features'][:, :length] for part in parts]),
            })
        if not combined:
            raise ValueError(f'{split}: no common valid samples')
        torch.save({'meta': {'model_paths': model_paths, 'k_fixed': k, 'window_mode': 'mean_per_text_equation1_ceil'},
                    'samples': combined}, output_dir / f'{split}_bi.pt')
        print(f'{split}: {len(combined)} samples, K={k}, channels={len(keys) * k}')
    return output_dir / 'train_bi.pt', output_dir / 'test_bi.pt'
