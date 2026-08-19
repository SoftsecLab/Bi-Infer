# import os
# import sys
# import json
# import gc
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import warnings
# from typing import List, Tuple, Dict, Optional
# from tqdm import tqdm
# from torch.utils.data import DataLoader, Dataset
# from torch.nn import CrossEntropyLoss, TransformerEncoder, TransformerEncoderLayer
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.optimization import get_linear_schedule_with_warmup
# from torch.optim import AdamW
# from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
# from sklearn.preprocessing import label_binarize

# warnings.filterwarnings('ignore')

# # ==========================================
# # 1. 全局配置 (Configuration)
# # ==========================================
# class Config:
#     # 🌟 修改 1: 支持多个代理模型提取序列概率特征并拼接
#     PROXY_MODELS = [
#         {"type": "llama", "path": "/home/share/models/llama-7b"},
#         {"type": "qwen", "path": "/home/share/models/qwen2.5-7b-instruct"}
#         # 可以按需取消注释添加更多模型
#         # {"type": "gpt", "path": "/home/share/models/gpt-j-6b"},
#         # {"type": "falcon", "path": "/home/share/models/falcon-7b-instruct"},
#     ]
    
#     # 🌟 修改 2: 定义批量任务队列 (支持批量处理跨文件夹的数据集)
#     TASKS = [
#        {
#             "task_name": "turn_task",
#             "train": "/home/gsy/project2/TuringBench/train/dataset_train.json",
#             "test": "/home/gsy/project2/TuringBench/test/dataset_test.json",
#             "output_dir": "/home/gsy/project2/m4/turn_task2"
#         }, {
#             "task_name": "arxiv_task",
#             "train": "/home/gsy/project2/m4/arxiv/data_train_arxiv.json",
#             "test": "/home/gsy/project2/m4/arxiv/data_test_arxiv.json",
#             "output_dir": "/home/gsy/project2/m4/arxiv_task2"
#         }, {
#             "task_name": "reddit_task",
#             "train": "/home/gsy/project2/m4/reddit/data_train_reddit.json",
#             "test": "/home/gsy/project2/m4/reddit/data_test_reddit.json",
#             "output_dir": "/home/gsy/project2/m4/reddit_task2"
#         }, {
#             "task_name": "wikihow_task",
#             "train": "/home/gsy/project2/m4/wikihow/data_test_wikihow.json",
#             "test": "/home/gsy/project2/m4/wikihow/data_train_wikihow.json",
#             "output_dir": "/home/gsy/project2/m4/wikihow_task2"
#         },
#         {
#             "task_name": "augpt_task",
#             "train": "/home/gsy/project2/augpt/train/train1.json",
#             "test": "/home/gsy/project2/augpt/test/test1.json",
#             "output_dir": "/home/gsy/project2/m4/augpt_task2"
#         },
#         {
#             "task_name": "wikipedia_task",
#             "train": "/home/gsy/project2/m4/wikipedia/data_train_wikipedia.json",
#             "test": "/home/gsy/project2/m4/wikipedia/data_test_wikipedia.json",
#             "output_dir": "/home/gsy/project2/m4/wikipedia_task2"
#         }
#         # 你可以无限添加需要批量测试的领域
#     ]
    
#     MODEL_SAVE_NAME = "seqxgpt_auto_label.pth"
#     REPORT_SAVE_NAME = "classification_report.txt"
#     METRICS_SAVE_NAME = "test_metrics.json"

#     # 硬件与采样控制
#     GPU_INDEX = 0           
#     DEVICE = f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu"
    
#     MAX_SAMPLES_PER_CLASS_TRAIN = 4000
#     MAX_SAMPLES_PER_CLASS_TEST = 1000 
#     REMOVE_LABEL_INDICES = [] 
    
#     # 训练超参数
#     BATCH_SIZE = 16             
#     SEQ_LEN = 1024              
#     EPOCHS = 10                  
#     LR = 5e-5                   
#     WEIGHT_DECAY = 0.01
#     WARMUP_RATIO = 0.1

# # ==========================================
# # 2. 标签处理与编码 (保持不变)
# # ==========================================
# class LabelProcessor:
#     def __init__(self, raw_data: Dict[str, List[str]], remove_indices: List[int]):
#         raw_classes = [k.strip().lower() for k in raw_data.keys()]
#         self.raw_classes = list(dict.fromkeys(raw_classes))
        
#         self.valid_classes = self.raw_classes.copy()
#         for idx in sorted(remove_indices, reverse=True):
#             if 0 <= idx < len(self.valid_classes):
#                 self.valid_classes.pop(idx)
        
#         self.class2id = {cls: i for i, cls in enumerate(self.valid_classes)}
#         self.id2class = {i: cls for cls, i in self.class2id.items()}
        
#         self.tag2id = {}
#         self.id2tag = {}
#         counter = 0
#         prefixes = ['B-', 'M-', 'E-', 'S-']
#         for cls in self.valid_classes:
#             for pre in prefixes:
#                 tag = f"{pre}{cls}"
#                 self.tag2id[tag] = counter
#                 self.id2tag[counter] = tag
#                 counter += 1
#         self.num_tags = len(self.tag2id)

#     def encode_sentence(self, word_list: List[str], class_name: str) -> List[int]:
#         class_name = class_name.strip().lower()
#         if class_name not in self.class2id: return []
        
#         length = len(word_list)
#         if length == 0: return []
#         if length == 1: return [self.tag2id[f"S-{class_name}"]]
        
#         ids = [0] * length
#         ids[0] = self.tag2id[f"B-{class_name}"]
#         ids[-1] = self.tag2id[f"E-{class_name}"]
#         m_id = self.tag2id[f"M-{class_name}"]
#         for i in range(1, length - 1):
#             ids[i] = m_id
#         return ids

# # ==========================================
# # 3. 数据集解析工具
# # ==========================================
# def load_data(file_path: str, valid_classes: list = None, max_per_class: Optional[int] = None):
#     if not os.path.exists(file_path):
#         print(f"⚠️ 文件不存在: {file_path}")
#         return [], [], {}
        
#     with open(file_path, 'r', encoding='utf-8') as f:
#         raw_data = json.load(f)
    
#     if valid_classes is not None:
#         valid_classes_norm = [cls.strip().lower() for cls in valid_classes]
#         raw_data = {k: v for k, v in raw_data.items() if k.strip().lower() in valid_classes_norm}
    
#     samples, classes = [], []
#     for cls_name, texts in raw_data.items():
#         current_texts = [t for t in texts if t.strip()]
#         if max_per_class is not None and max_per_class > 0:
#             current_texts = current_texts[:max_per_class]
#         for text in current_texts:
#             samples.append(text)
#             classes.append(cls_name)
            
#     return samples, classes, raw_data

# # ==========================================
# # 4. 单模型特征提取器 (解耦，供任务队列调用)
# # ==========================================
# class SeqXGPTFeatureExtractor:
#     def __init__(self, model_config):
#         self.device = Config.DEVICE
#         self.m_type = model_config['type']
#         m_path = model_config['path']
#         print(f"\n[{self.device.upper()}] 正在加载代理模型: {self.m_type}")
        
#         try:
#             trust_remote = True if self.m_type in ['qwen', 'falcon'] else False
#             self.tokenizer = AutoTokenizer.from_pretrained(m_path, trust_remote_code=trust_remote)
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 m_path, 
#                 device_map=self.device, 
#                 torch_dtype=torch.float16, 
#                 trust_remote_code=trust_remote
#             )
#             self.model.eval()
#         except Exception as e:
#             print(f"[!] 加载模型 {self.m_type} 失败: {e}")
#             sys.exit(1)

#     def free_memory(self):
#         """释放显存"""
#         del self.model
#         del self.tokenizer
#         gc.collect()
#         if torch.cuda.is_available(): torch.cuda.empty_cache()
#         print(f"[*] 已卸载模型 {self.m_type} 并清空显存。")

#     def extract_aligned_scores(self, text: str, target_words: List[str]):
#         """获取对齐到单词级别的对数概率"""
#         if not text.strip() or len(target_words) == 0: return []
        
#         try:
#             inputs = self.tokenizer(
#                 text, return_tensors="pt", truncation=True, 
#                 max_length=Config.SEQ_LEN, padding=False
#             ).to(self.device)
#             input_ids = inputs["input_ids"]
            
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
            
#             logits = outputs.logits[0, :-1, :]
#             labels = input_ids[0, 1:]
#             log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
#             token_log_probs = torch.gather(log_probs, dim=1, index=labels.unsqueeze(1)).squeeze(1)
#             token_log_probs = torch.cat([torch.tensor([0.0], device=self.device), token_log_probs])
            
#             offset_mapping = self.tokenizer(
#                 text, return_offsets_mapping=True, truncation=True, 
#                 max_length=Config.SEQ_LEN, return_tensors="pt"
#             )["offset_mapping"][0].cpu().numpy()
            
#             token_log_probs = token_log_probs.cpu().numpy()
#             aligned_scores = []
#             curr_pos = 0
            
#             for word in target_words:
#                 word_start = text.find(word, curr_pos)
#                 if word_start == -1: word_start = curr_pos
#                 word_end = word_start + len(word)
#                 curr_pos = word_end
                
#                 relevant_probs = []
#                 for token_idx, (t_start, t_end) in enumerate(offset_mapping):
#                     if max(word_start, t_start) < min(word_end, t_end):
#                         if token_idx < len(token_log_probs):
#                             relevant_probs.append(token_log_probs[token_idx])
                
#                 score = np.mean(relevant_probs) if relevant_probs else 0.0
#                 aligned_scores.append(score)
#             return aligned_scores
            
#         except Exception:
#             return []

# # ==========================================
# # 5. 模型定义 (动态自适应输入通道)
# # ==========================================
# class CRF(nn.Module):
#     def __init__(self, num_tags):
#         super().__init__()
#         self.num_tags = num_tags
#     def viterbi_decode(self, logits, mask):
#         best_tags = torch.argmax(logits, dim=-1)
#         best_tags[~mask] = -1
#         return best_tags, None

# class SeqXGPTModel(nn.Module):
#     def __init__(self, num_tags, num_proxies, seq_len=1024):
#         super().__init__()
        
#         # CNN 层: 输入1通道(单个模型的特征), 输出64维
#         self.conv = nn.Sequential(
#             nn.Conv1d(1, 64, 5, padding=2), nn.ReLU(),
#             nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
#             nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
#             nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
#             nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(),
#         )
        
#         # 自动根据代理模型的数量计算 Transformer 的输入维度
#         self.cnn_out_dim = 64
#         self.embed_dim = num_proxies * self.cnn_out_dim
        
#         self.pos_enc = nn.Parameter(torch.randn(1, seq_len, self.embed_dim))
#         encoder_layer = TransformerEncoderLayer(
#             d_model=self.embed_dim, nhead=8, dim_feedforward=512, batch_first=True, activation="relu"
#         )
#         self.transformer = TransformerEncoder(encoder_layer, num_layers=2)
#         self.norm = nn.LayerNorm(self.embed_dim)
#         self.classifier = nn.Linear(self.embed_dim, num_tags)
#         self.crf = CRF(num_tags)
#         self.loss_fct = CrossEntropyLoss(ignore_index=-1)

#     def forward(self, features, labels=None):
#         batch_size, seq_len, num_models = features.shape
        
#         cnn_outputs = []
#         for model_idx in range(num_models):
#             single_feature = features[:, :, model_idx:model_idx+1].permute(0, 2, 1)
#             cnn_out = self.conv(single_feature).transpose(1, 2)
#             cnn_outputs.append(cnn_out)
        
#         out = torch.cat(cnn_outputs, dim=-1) # 拼接各模型特征
#         out = out + self.pos_enc[:, :seq_len, :]
#         out = self.norm(out)
        
#         if labels is not None:
#             mask = labels.gt(-1)
#             out = self.transformer(out, src_key_padding_mask=~mask)
#         else:
#             out = self.transformer(out)
        
#         logits = self.classifier(out)
#         output = {"logits": logits}
        
#         if labels is not None:
#             loss = self.loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
#             output["loss"] = loss
#             preds, _ = self.crf.viterbi_decode(logits, mask)
#             output["preds"] = preds
            
#         return output

# # ==========================================
# # 6. 数据集结构
# # ==========================================
# class SeqXGPTDataset(Dataset):
#     def __init__(self, features, words, class_names, label_processor: LabelProcessor):
#         self.features, self.words, self.labels = [], [], []
#         for i in range(len(features)):
#             seq_ids = label_processor.encode_sentence(words[i], class_names[i])
#             if len(seq_ids) == 0 or len(features[i]) == 0 or len(seq_ids) != len(features[i]):
#                 continue
#             self.features.append(features[i])
#             self.words.append(words[i])
#             self.labels.append(seq_ids)

#     def __len__(self): return len(self.features)
#     def __getitem__(self, idx):
#         return {
#             "features": torch.tensor(self.features[idx], dtype=torch.float32),
#             "labels": torch.tensor(self.labels[idx], dtype=torch.long),
#             "text": self.words[idx]
#         }

# def collate_fn(batch):
#     features = [item['features'] for item in batch]
#     labels = [item['labels'] for item in batch]
#     texts = [item['text'] for item in batch]
#     features_padded = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
#     labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
#     return {"features": features_padded, "labels": labels_padded, "text": texts}

# # ==========================================
# # 7. Trainer 训练控制
# # ==========================================
# class Trainer:
#     def __init__(self, model, train_loader, test_loader, label_processor, output_dir):
#         self.model = model.to(Config.DEVICE)
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.label_processor = label_processor
#         self.output_dir = output_dir # 🌟 支持传参区分任务输出目录
        
#         self.optimizer = AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
#         total_steps = len(train_loader) * Config.EPOCHS
#         self.scheduler = get_linear_schedule_with_warmup(
#             self.optimizer, num_warmup_steps=int(Config.WARMUP_RATIO * total_steps), num_training_steps=total_steps
#         )

#     def train_epoch(self, epoch):
#         self.model.train()
#         total_loss = 0.0
#         pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", leave=False)
#         for batch in pbar:
#             for k in batch:
#                 if isinstance(batch[k], torch.Tensor): batch[k] = batch[k].to(Config.DEVICE)
#             output = self.model(batch['features'], batch['labels'])
#             loss = output['loss']
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#             self.scheduler.step()
#             total_loss += loss.item()
#             pbar.set_postfix(loss=f"{loss.item():.4f}")
#         return total_loss / len(self.train_loader)

#     def evaluate(self, final_report=False):
#         self.model.eval()
#         all_preds, all_labels, all_probs = [], [], []
#         with torch.no_grad():
#             for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
#                 for k in batch:
#                     if isinstance(batch[k], torch.Tensor): batch[k] = batch[k].to(Config.DEVICE)
#                 output = self.model(batch['features'], batch['labels'])
#                 logits = output['logits']
#                 preds = output['preds']
#                 labels = batch['labels']
#                 probs = F.softmax(logits, dim=-1)
#                 mask = labels != -1
#                 all_preds.extend(preds[mask].cpu().numpy())
#                 all_labels.extend(labels[mask].cpu().numpy())
#                 all_probs.extend(probs[mask].cpu().numpy())

#         if not all_preds: return
#         accuracy = accuracy_score(all_labels, all_preds)
#         f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

#         if not final_report:
#             print(f"   [Val] ACC: {accuracy:.4f} | F1: {f1:.4f}")
#         else:
#             os.makedirs(self.output_dir, exist_ok=True)
#             def tag_id_to_class_idx(tag_id):
#                 if tag_id == -1: return -1
#                 class_str = self.label_processor.id2tag[tag_id].split('-', 1)[1] 
#                 return self.label_processor.class2id[class_str]

#             cls_preds = [tag_id_to_class_idx(p) for p in all_preds]
#             cls_labels = [tag_id_to_class_idx(l) for l in all_labels]
            
#             target_ids = list(self.label_processor.class2id.values())
#             target_names = list(self.label_processor.class2id.keys())
            
#             report_str = classification_report(cls_labels, cls_preds, labels=target_ids, target_names=target_names, digits=4, zero_division=0)
#             txt_path = os.path.join(self.output_dir, Config.REPORT_SAVE_NAME)
#             with open(txt_path, "w", encoding="utf-8") as f:
#                 f.write(report_str)
#                 f.write(f"\n    accuracy                           {accuracy:.4f}       {len(cls_labels)}\n")
#             print(f"\n[*] 已保存分类报告: {txt_path}")
            
#             num_classes = len(target_names)
#             probs_np = np.array(all_probs)
#             class_probs = np.zeros((len(all_labels), num_classes))
#             for tag_id, tag_name in self.label_processor.id2tag.items():
#                 class_idx = self.label_processor.class2id[tag_name.split('-', 1)[1]]
#                 class_probs[:, class_idx] += probs_np[:, tag_id]
            
#             class_probs = class_probs / (class_probs.sum(axis=1, keepdims=True) + 1e-9)
#             labels_bin = label_binarize(cls_labels, classes=target_ids)
            
#             individual_aucs = {}
#             auc_scores = []
#             for i, class_name in enumerate(target_names):
#                 try:
#                     auc = roc_auc_score(labels_bin[:, i], class_probs[:, i]) if np.sum(labels_bin[:, i]) > 0 else 0.0
#                 except: auc = 0.0
#                 auc_scores.append(auc)
#                 individual_aucs[str(i)] = {"class_name": class_name, "auc": auc}
            
#             json_output = {"accuracy": accuracy, "mean_auc": np.mean(auc_scores), "individual_aucs": individual_aucs}
#             json_path = os.path.join(self.output_dir, Config.METRICS_SAVE_NAME)
#             with open(json_path, "w", encoding="utf-8") as f:
#                 json.dump(json_output, f, indent=4)

#     def train(self):
#         for epoch in range(Config.EPOCHS):
#             self.train_epoch(epoch)
#             self.evaluate(final_report=False)
#         self.evaluate(final_report=True)

# # ==========================================
# # 8. 主流程 (任务队列 + 串行提特征)
# # ==========================================
# def main():
#     print("=== SeqXGPT 复现: 多模型拼接与批量验证 ===")
    
#     # --- 阶段 1: 预加载所有任务的文本数据与标签处理器 ---
#     task_vault = {}
#     for task in Config.TASKS:
#         t_name = task["task_name"]
#         print(f"\n>>> 初始化任务: {t_name}")
        
#         _, _, raw_train = load_data(task["train"], max_per_class=None)
#         if not raw_train: continue
        
#         label_proc = LabelProcessor(raw_train, Config.REMOVE_LABEL_INDICES)
#         tr_txt, tr_cls, _ = load_data(task["train"], label_proc.valid_classes, Config.MAX_SAMPLES_PER_CLASS_TRAIN)
#         te_txt, te_cls, _ = load_data(task["test"], label_proc.valid_classes, Config.MAX_SAMPLES_PER_CLASS_TEST)
        
#         if not tr_txt or not te_txt: continue
        
#         # 预先进行分词定长
#         tr_words = [t.split()[:Config.SEQ_LEN] for t in tr_txt]
#         te_words = [t.split()[:Config.SEQ_LEN] for t in te_txt]
        
#         # 预先分配空 numpy 特征张量 [样本数, 序列长度, 模型数量]
#         num_models = len(Config.PROXY_MODELS)
#         tr_feats = [np.zeros((len(w), num_models), dtype=np.float32) for w in tr_words]
#         te_feats = [np.zeros((len(w), num_models), dtype=np.float32) for w in te_words]
        
#         task_vault[t_name] = {
#             "label_processor": label_proc,
#             "train_classes": tr_cls, "test_classes": te_cls,
#             "train_words": tr_words, "test_words": te_words,
#             "train_texts": tr_txt, "test_texts": te_txt,
#             "train_feats": tr_feats, "test_feats": te_feats,
#             "output_dir": task["output_dir"]
#         }

#     # --- 阶段 2: 串行加载代理模型提取对数概率 ---
#     for model_idx, model_config in enumerate(Config.PROXY_MODELS):
#         extractor = SeqXGPTFeatureExtractor(model_config)
        
#         for t_name, data in task_vault.items():
#             print(f"\n>>> [{model_config['type']}] 提取特征: {t_name}")
            
#             # 提取训练集
#             for i, text in enumerate(tqdm(data["train_texts"], desc="Train", leave=False)):
#                 aligned = extractor.extract_aligned_scores(text, data["train_words"][i])
#                 valid_len = min(len(aligned), data["train_feats"][i].shape[0])
#                 if valid_len > 0:
#                     data["train_feats"][i][:valid_len, model_idx] = aligned[:valid_len]
            
#             # 提取测试集
#             for i, text in enumerate(tqdm(data["test_texts"], desc="Test", leave=False)):
#                 aligned = extractor.extract_aligned_scores(text, data["test_words"][i])
#                 valid_len = min(len(aligned), data["test_feats"][i].shape[0])
#                 if valid_len > 0:
#                     data["test_feats"][i][:valid_len, model_idx] = aligned[:valid_len]
                    
#         extractor.free_memory()

#     # --- 阶段 3: 执行 SeqXGPT 神经网络训练与评估 ---
#     print("\n" + "="*50)
#     print("🚀 开始进行 SeqXGPT 神经网络训练")
#     print("="*50)
    
#     for t_name, data in task_vault.items():
#         print(f"\n>>> 启动任务训练: {t_name}")
#         os.makedirs(data["output_dir"], exist_ok=True)
        
#         train_dataset = SeqXGPTDataset(data["train_feats"], data["train_words"], data["train_classes"], data["label_processor"])
#         test_dataset = SeqXGPTDataset(data["test_feats"], data["test_words"], data["test_classes"], data["label_processor"])
        
#         train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
#         test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
#         # 动态传入模型数量，适应提取的特征通道
#         model = SeqXGPTModel(num_tags=data["label_processor"].num_tags, num_proxies=len(Config.PROXY_MODELS), seq_len=Config.SEQ_LEN)
#         trainer = Trainer(model, train_loader, test_loader, data["label_processor"], data["output_dir"])
        
#         trainer.train()
        
#         save_path = os.path.join(data["output_dir"], Config.MODEL_SAVE_NAME)
#         torch.save(model.state_dict(), save_path)
#         print(f"[*] 模型参数已保存: {save_path}")

#     print("\n=== 全部批处理任务执行完毕 ===")

# if __name__ == "__main__":
#     main()

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
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局配置 (多数据集自动化流水线)
# ==========================================
class Config:
    # 代理模型路径 (白盒特征提取器)
    PROXY_MODELS = [
        {"type": "llama", "path": "/home/share/models/llama-7b"},
        {"type": "qwen", "path": "/home/share/models/qwen2.5-7b-instruct"}
    ]
    
    # 🌟 基础路径统一配置
    INPUT_BASE_DIR = "/home/gsy/project2/TuringBench/two_class"
    OUTPUT_BASE_DIR = "/home/gsy/project2/MAGE/features/squad_othermethod"

    # 🌟 只需要在这里填入你要跑的数据集文件夹名称
    TASK_NAMES = [
        "TT_ctrl",
     "TT_fair_wmt19",   
    "TT_fair_wmt20",
    "TT_gpt1",
    "TT_gpt2_large",
    "TT_gpt2_medium",
    "TT_gpt2_pytorch",
    "TT_gpt2_small",
    "TT_gpt2_xl",
    "TT_gpt3",
    "TT_grover_base",
    "TT_grover_large",
    "TT_grover_mega",
    "TT_pplm_distil",
    "TT_pplm_gpt2",
    "TT_transfo_xl",
    "TT_xlm",
    "TT_xlnet_base",
    "TT_xlnet_large"
    ]
    
    # 自动生成所有数据集的绝对路径配置
    TASKS = []
    for task in TASK_NAMES:
        TASKS.append({
            "task_name": task,
            "train": os.path.join(INPUT_BASE_DIR, task, "train_data.json"),
            "test": os.path.join(INPUT_BASE_DIR, task, "test_data.json"),
            "output_dir": os.path.join(OUTPUT_BASE_DIR, f"seqxgpt_{task}")
        })
    
    MODEL_SAVE_NAME = "seqxgpt_auto_label.pth"
    REPORT_SAVE_NAME = "classification_report.txt"
    METRICS_SAVE_NAME = "test_metrics.json"

    # 硬件与采样控制
    GPU_INDEX = 1           
    DEVICE = f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu"
    
    MAX_SAMPLES_PER_CLASS_TRAIN = 4000
    MAX_SAMPLES_PER_CLASS_TEST = 1000 
    REMOVE_LABEL_INDICES = [] 
    
    # 训练超参数
    BATCH_SIZE = 16             
    SEQ_LEN = 1024              
    EPOCHS = 10                  
    LR = 5e-5                   
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1

# ==========================================
# 2. 标签处理与编码 
# ==========================================
class LabelProcessor:
    def __init__(self, raw_data: Dict[str, List[str]], remove_indices: List[int]):
        raw_classes = [k.strip().lower() for k in raw_data.keys()]
        self.raw_classes = list(dict.fromkeys(raw_classes))
        
        self.valid_classes = self.raw_classes.copy()
        for idx in sorted(remove_indices, reverse=True):
            if 0 <= idx < len(self.valid_classes):
                self.valid_classes.pop(idx)
        
        self.class2id = {cls: i for i, cls in enumerate(self.valid_classes)}
        self.id2class = {i: cls for cls, i in self.class2id.items()}
        
        self.tag2id = {}
        self.id2tag = {}
        counter = 0
        prefixes = ['B-', 'M-', 'E-', 'S-']
        for cls in self.valid_classes:
            for pre in prefixes:
                tag = f"{pre}{cls}"
                self.tag2id[tag] = counter
                self.id2tag[counter] = tag
                counter += 1
        self.num_tags = len(self.tag2id)

    def encode_sentence(self, word_list: List[str], class_name: str) -> List[int]:
        class_name = class_name.strip().lower()
        if class_name not in self.class2id: return []
        
        length = len(word_list)
        if length == 0: return []
        if length == 1: return [self.tag2id[f"S-{class_name}"]]
        
        ids = [0] * length
        ids[0] = self.tag2id[f"B-{class_name}"]
        ids[-1] = self.tag2id[f"E-{class_name}"]
        m_id = self.tag2id[f"M-{class_name}"]
        for i in range(1, length - 1):
            ids[i] = m_id
        return ids

# ==========================================
# 3. 数据集解析工具
# ==========================================
def load_data(file_path: str, valid_classes: list = None, max_per_class: Optional[int] = None):
    if not os.path.exists(file_path):
        return [], [], {}
        
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if valid_classes is not None:
        valid_classes_norm = [cls.strip().lower() for cls in valid_classes]
        raw_data = {k: v for k, v in raw_data.items() if k.strip().lower() in valid_classes_norm}
    
    samples, classes = [], []
    for cls_name, texts in raw_data.items():
        current_texts = [t for t in texts if t.strip()]
        if max_per_class is not None and max_per_class > 0:
            current_texts = current_texts[:max_per_class]
        for text in current_texts:
            samples.append(text)
            classes.append(cls_name)
            
    return samples, classes, raw_data

# ==========================================
# 4. 单模型特征提取器 
# ==========================================
class SeqXGPTFeatureExtractor:
    def __init__(self, model_config):
        self.device = Config.DEVICE
        self.m_type = model_config['type']
        m_path = model_config['path']
        print(f"\n[{self.device.upper()}] 正在加载代理模型: {self.m_type}")
        
        try:
            trust_remote = True if self.m_type in ['qwen', 'falcon'] else False
            self.tokenizer = AutoTokenizer.from_pretrained(m_path, trust_remote_code=trust_remote)
            self.model = AutoModelForCausalLM.from_pretrained(
                m_path, 
                device_map={"": self.device}, 
                torch_dtype=torch.float16, 
                trust_remote_code=trust_remote
            )
            self.model.eval()
        except Exception as e:
            print(f"[!] 加载模型 {self.m_type} 失败: {e}")
            sys.exit(1)

    def free_memory(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"[*] 已卸载模型 {self.m_type} 并清空显存。")

    def extract_aligned_scores(self, text: str, target_words: List[str]):
        if not text.strip() or len(target_words) == 0: return []
        
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, 
                max_length=Config.SEQ_LEN, padding=False
            ).to(self.device)
            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits[0, :-1, :]
            labels = input_ids[0, 1:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, dim=1, index=labels.unsqueeze(1)).squeeze(1)
            token_log_probs = torch.cat([torch.tensor([0.0], device=self.device), token_log_probs])
            
            offset_mapping = self.tokenizer(
                text, return_offsets_mapping=True, truncation=True, 
                max_length=Config.SEQ_LEN, return_tensors="pt"
            )["offset_mapping"][0].cpu().numpy()
            
            token_log_probs = token_log_probs.cpu().numpy()
            aligned_scores = []
            curr_pos = 0
            
            for word in target_words:
                word_start = text.find(word, curr_pos)
                if word_start == -1: word_start = curr_pos
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

# ==========================================
# 5. 模型定义 (SeqXGPT)
# ==========================================
class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
    def viterbi_decode(self, logits, mask):
        best_tags = torch.argmax(logits, dim=-1)
        best_tags[~mask] = -1
        return best_tags, None

class SeqXGPTModel(nn.Module):
    def __init__(self, num_tags, num_proxies, seq_len=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(),
        )
        
        self.cnn_out_dim = 64
        self.embed_dim = num_proxies * self.cnn_out_dim
        
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, self.embed_dim))
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=8, dim_feedforward=512, batch_first=True, activation="relu"
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_tags)
        self.crf = CRF(num_tags)
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)

    def forward(self, features, labels=None):
        batch_size, seq_len, num_models = features.shape
        
        cnn_outputs = []
        for model_idx in range(num_models):
            single_feature = features[:, :, model_idx:model_idx+1].permute(0, 2, 1)
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
            loss = self.loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            output["loss"] = loss
            preds, _ = self.crf.viterbi_decode(logits, mask)
            output["preds"] = preds
            
        return output

# ==========================================
# 6. 数据集结构
# ==========================================
class SeqXGPTDataset(Dataset):
    def __init__(self, features, words, class_names, label_processor: LabelProcessor):
        self.features, self.words, self.labels = [], [], []
        for i in range(len(features)):
            seq_ids = label_processor.encode_sentence(words[i], class_names[i])
            if len(seq_ids) == 0 or len(features[i]) == 0 or len(seq_ids) != len(features[i]):
                continue
            self.features.append(features[i])
            self.words.append(words[i])
            self.labels.append(seq_ids)

    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "text": self.words[idx]
        }

def collate_fn(batch):
    features = [item['features'] for item in batch]
    labels = [item['labels'] for item in batch]
    texts = [item['text'] for item in batch]
    features_padded = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)
    return {"features": features_padded, "labels": labels_padded, "text": texts}

# ==========================================
# 7. Trainer 训练控制
# ==========================================
class Trainer:
    def __init__(self, model, train_loader, test_loader, label_processor, output_dir):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.label_processor = label_processor
        self.output_dir = output_dir 
        
        self.optimizer = AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        total_steps = len(train_loader) * Config.EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(Config.WARMUP_RATIO * total_steps), num_training_steps=total_steps
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", leave=False)
        for batch in pbar:
            for k in batch:
                if isinstance(batch[k], torch.Tensor): batch[k] = batch[k].to(Config.DEVICE)
            output = self.model(batch['features'], batch['labels'])
            loss = output['loss']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / len(self.train_loader)

    def evaluate(self, final_report=False):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
                for k in batch:
                    if isinstance(batch[k], torch.Tensor): batch[k] = batch[k].to(Config.DEVICE)
                output = self.model(batch['features'], batch['labels'])
                logits = output['logits']
                preds = output['preds']
                labels = batch['labels']
                probs = F.softmax(logits, dim=-1)
                mask = labels != -1
                all_preds.extend(preds[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
                all_probs.extend(probs[mask].cpu().numpy())

        if not all_preds: return
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        if not final_report:
            print(f"   [Val] Token-ACC: {accuracy:.4f} | Token-F1: {f1:.4f}")
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            def tag_id_to_class_idx(tag_id):
                if tag_id == -1: return -1
                class_str = self.label_processor.id2tag[tag_id].split('-', 1)[1] 
                return self.label_processor.class2id[class_str]

            cls_preds = [tag_id_to_class_idx(p) for p in all_preds]
            cls_labels = [tag_id_to_class_idx(l) for l in all_labels]
            
            target_ids = list(self.label_processor.class2id.values())
            target_names = list(self.label_processor.class2id.keys())
            
            report_str = classification_report(cls_labels, cls_preds, labels=target_ids, target_names=target_names, digits=4, zero_division=0)
            txt_path = os.path.join(self.output_dir, Config.REPORT_SAVE_NAME)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(report_str)
                f.write(f"\n    accuracy                           {accuracy:.4f}       {len(cls_labels)}\n")
            print(f"\n[*] 已保存分类报告: {txt_path}")
            
            num_classes = len(target_names)
            probs_np = np.array(all_probs)
            class_probs = np.zeros((len(all_labels), num_classes))
            for tag_id, tag_name in self.label_processor.id2tag.items():
                class_idx = self.label_processor.class2id[tag_name.split('-', 1)[1]]
                class_probs[:, class_idx] += probs_np[:, tag_id]
            
            class_probs = class_probs / (class_probs.sum(axis=1, keepdims=True) + 1e-9)
            labels_bin = label_binarize(cls_labels, classes=target_ids)
            
            # Binary classification adjustment
            if num_classes == 2 and labels_bin.shape[1] == 1:
                labels_bin = np.hstack((1 - labels_bin, labels_bin))
            
            individual_aucs = {}
            auc_scores = []
            for i, class_name in enumerate(target_names):
                try:
                    if np.sum(labels_bin[:, i]) > 0:
                        auc = roc_auc_score(labels_bin[:, i], class_probs[:, i]) 
                    else: auc = 0.0
                except: auc = 0.0
                auc_scores.append(auc)
                individual_aucs[str(i)] = {"class_name": class_name, "auc": auc}
            
            json_output = {"accuracy": accuracy, "mean_auc": np.mean(auc_scores), "individual_aucs": individual_aucs}
            json_path = os.path.join(self.output_dir, Config.METRICS_SAVE_NAME)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_output, f, indent=4)

    def train(self):
        for epoch in range(Config.EPOCHS):
            self.train_epoch(epoch)
            self.evaluate(final_report=False)
        self.evaluate(final_report=True)

# ==========================================
# 8. 主流程 (断点续传 + 串行提取 + 批量评估)
# ==========================================
def main():
    if torch.cuda.is_available() and Config.GPU_INDEX in [0, 1]:
        torch.cuda.set_device(Config.GPU_INDEX)
        print(f"\n[*] 使用设备: cuda:{Config.GPU_INDEX} ({torch.cuda.get_device_name(Config.GPU_INDEX)})")
    else:
        print(f"\n[*] 使用设备: {Config.DEVICE}")

    print("="*60)
    print("=== SeqXGPT 自动化批量流水线 ===")
    print("="*60)
    
    # --- 阶段 1: 预加载所有任务数据 ---
    task_vault = {}
    for task in Config.TASKS:
        t_name = task["task_name"]
        print(f"\n>>> 初始化任务: {t_name}")
        
        _, _, raw_train = load_data(task["train"], max_per_class=None)
        if not raw_train: continue
        
        label_proc = LabelProcessor(raw_train, Config.REMOVE_LABEL_INDICES)
        tr_txt, tr_cls, _ = load_data(task["train"], label_proc.valid_classes, Config.MAX_SAMPLES_PER_CLASS_TRAIN)
        te_txt, te_cls, _ = load_data(task["test"], label_proc.valid_classes, Config.MAX_SAMPLES_PER_CLASS_TEST)
        
        if not tr_txt or not te_txt: continue
        
        tr_words = [t.split()[:Config.SEQ_LEN] for t in tr_txt]
        te_words = [t.split()[:Config.SEQ_LEN] for t in te_txt]
        
        num_models = len(Config.PROXY_MODELS)
        tr_feats = [np.zeros((len(w), num_models), dtype=np.float32) for w in tr_words]
        te_feats = [np.zeros((len(w), num_models), dtype=np.float32) for w in te_words]
        
        os.makedirs(task["output_dir"], exist_ok=True)
        
        task_vault[t_name] = {
            "label_processor": label_proc,
            "train_classes": tr_cls, "test_classes": te_cls,
            "train_words": tr_words, "test_words": te_words,
            "train_texts": tr_txt, "test_texts": te_txt,
            "train_feats": tr_feats, "test_feats": te_feats,
            "output_dir": task["output_dir"]
        }

    # --- 阶段 2: 串行特征提取 (支持缓存断点续传) ---
    for model_idx, model_config in enumerate(Config.PROXY_MODELS):
        m_type = model_config['type']
        
        # 检查缓存状态
        need_extract = False
        for t_name, data in task_vault.items():
            train_cache = os.path.join(data["output_dir"], f"seqxgpt_feat_{m_type}_train.npy")
            test_cache = os.path.join(data["output_dir"], f"seqxgpt_feat_{m_type}_test.npy")
            
            if os.path.exists(train_cache) and os.path.exists(test_cache):
                print(f"[*] 检测到缓存: {t_name} 已成功加载 {m_type} 特征。")
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
            print(f"✅ 模型 {m_type} 在所有任务上的特征均已读取，跳过模型加载。")
            continue
            
        extractor = SeqXGPTFeatureExtractor(model_config)
        
        for t_name, data in task_vault.items():
            if data.get(f"cached_{m_type}", False): continue
            
            print(f"\n>>> [{m_type}] 开始提取任务特征: {t_name}")
            
            m_train_feats = []
            for i, text in enumerate(tqdm(data["train_texts"], desc="Train", leave=False)):
                aligned = extractor.extract_aligned_scores(text, data["train_words"][i])
                v_len = min(len(aligned), data["train_feats"][i].shape[0])
                if v_len > 0: data["train_feats"][i][:v_len, model_idx] = aligned[:v_len]
                m_train_feats.append(aligned)
            
            m_test_feats = []
            for i, text in enumerate(tqdm(data["test_texts"], desc="Test", leave=False)):
                aligned = extractor.extract_aligned_scores(text, data["test_words"][i])
                v_len = min(len(aligned), data["test_feats"][i].shape[0])
                if v_len > 0: data["test_feats"][i][:v_len, model_idx] = aligned[:v_len]
                m_test_feats.append(aligned)
                
            # 物理落盘保存
            train_cache = os.path.join(data["output_dir"], f"seqxgpt_feat_{m_type}_train.npy")
            test_cache = os.path.join(data["output_dir"], f"seqxgpt_feat_{m_type}_test.npy")
            np.save(train_cache, np.array(m_train_feats, dtype=object))
            np.save(test_cache, np.array(m_test_feats, dtype=object))
            print(f"  ✅ 即时保存：[{t_name}] 的 {m_type} 特征已保存至 -> {data['output_dir']}")
            
        extractor.free_memory()

    # --- 阶段 3: 执行 SeqXGPT 神经网络训练与评估 ---
    print("\n" + "="*50)
    print("🚀 开始进行 SeqXGPT 神经网络训练")
    print("="*50)
    
    for t_name, data in task_vault.items():
        print(f"\n>>> 启动任务训练: {t_name}")
        
        train_dataset = SeqXGPTDataset(data["train_feats"], data["train_words"], data["train_classes"], data["label_processor"])
        test_dataset = SeqXGPTDataset(data["test_feats"], data["test_words"], data["test_classes"], data["label_processor"])
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        model = SeqXGPTModel(num_tags=data["label_processor"].num_tags, num_proxies=len(Config.PROXY_MODELS), seq_len=Config.SEQ_LEN)
        trainer = Trainer(model, train_loader, test_loader, data["label_processor"], data["output_dir"])
        
        trainer.train()
        
        save_path = os.path.join(data["output_dir"], Config.MODEL_SAVE_NAME)
        torch.save(model.state_dict(), save_path)
        print(f"[*] 任务 {t_name} 完成，所有评估与模型参数已保存至: {data['output_dir']}")

    print("\n=== 全部批处理任务执行完毕 ===")

if __name__ == "__main__":
    main()