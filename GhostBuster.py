# import argparse
# import os

# # ==============================================================================
# # 0. 命令行参数解析 (适配自动化流水线)
# # ==============================================================================
# parser = argparse.ArgumentParser(description="Run GhostBuster Supervised OvR Detection")
# parser.add_argument("--train_data", type=str, required=True, help="Path to training JSON")
# parser.add_argument("--test_data", type=str, required=True, help="Path to testing JSON")
# parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
# parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use (e.g., '0' or '0,1')")
# args = parser.parse_args()

# # ==============================================================================
# # 显存分配设置 (根据传入的 --gpu 参数动态分配)
# # ==============================================================================
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# import json
# import torch
# import numpy as np
# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
# import warnings

# warnings.filterwarnings('ignore')

# # ==============================================================================
# # 1. 用户配置区域 (User Configuration)
# # ==============================================================================
# class UserConfig:
#     # --- 代理模型路径 (使用 LLaMA 和 Qwen 作为概率评估器) ---
#     MODEL_1_PATH = "/home/share/models/llama-7b"
#     MODEL_2_PATH = "/home/share/models/qwen2.5-7b-instruct"

#     # --- 数据集路径 (接收命令行传入的动态路径) ---
#     TRAIN_DATA_PATH = args.train_data  
#     TEST_DATA_PATH = args.test_data   

#     # --- 输出配置 (接收命令行传入的动态目录) ---
#     OUTPUT_DIR = args.output_dir          
#     METRICS_FILENAME = "metrics_result.json"
    
#     # --- 缓存配置 ---
#     USE_CACHE = True
#     TRAIN_CACHE_FILE = "train_ghostbuster_features.csv" 
#     TEST_CACHE_FILE = "test_ghostbuster_features.csv"

#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     MAX_TOKEN_LENGTH = 1024  

# # 动态创建当前数据集的专属输出目录
# os.makedirs(UserConfig.OUTPUT_DIR, exist_ok=True)
# # 确保缓存文件存放在专属目录下，防止不同数据集的特征互相污染
# UserConfig.TRAIN_CACHE_FILE = os.path.join(UserConfig.OUTPUT_DIR, UserConfig.TRAIN_CACHE_FILE)
# UserConfig.TEST_CACHE_FILE = os.path.join(UserConfig.OUTPUT_DIR, UserConfig.TEST_CACHE_FILE)

# # ==============================================================================
# # 2. GhostBuster 特征提取器
# # ==============================================================================
# class GhostBusterFeatureExtractor:
#     def __init__(self):
#         self.device = UserConfig.DEVICE
#         if self.device == "cuda":
#             print(f"[Init] Using GPU(s). Primary GPU: {torch.cuda.get_device_name(0)}")
#         else:
#             print("[Init] Using CPU (GPU not available)")
        
#         # 加载模型 1 (LLaMA)
#         print(f"[Init] Loading Model 1 from: {UserConfig.MODEL_1_PATH} ...")
#         self.tokenizer_1 = AutoTokenizer.from_pretrained(
#             UserConfig.MODEL_1_PATH, local_files_only=True, trust_remote_code=True, use_fast=False, padding_side="right"
#         )
#         if self.tokenizer_1.pad_token is None:
#             self.tokenizer_1.pad_token = self.tokenizer_1.eos_token

#         self.model_1 = AutoModelForCausalLM.from_pretrained(
#             UserConfig.MODEL_1_PATH, trust_remote_code=True, 
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#             local_files_only=True, device_map="auto" 
#         ).eval()

#         # 加载模型 2 (Qwen)
#         print(f"[Init] Loading Model 2 from: {UserConfig.MODEL_2_PATH} ...")
#         self.tokenizer_2 = AutoTokenizer.from_pretrained(
#             UserConfig.MODEL_2_PATH, local_files_only=True, trust_remote_code=True, padding_side="right"
#         )
#         if self.tokenizer_2.pad_token is None:
#             self.tokenizer_2.pad_token = self.tokenizer_2.eos_token

#         self.model_2 = AutoModelForCausalLM.from_pretrained(
#             UserConfig.MODEL_2_PATH, trust_remote_code=True, 
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#             local_files_only=True, device_map="auto" 
#         ).eval()

#     def _get_ghostbuster_stats(self, logits, input_ids, pad_token_id):
#         """核心数学实现：提取 GhostBuster 需要的5维概率统计特征"""
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = input_ids[..., 1:].contiguous()
        
#         # 计算对数概率
#         log_probs_dist = torch.log_softmax(shift_logits, dim=-1)
#         # 提取真实生成词的对数概率
#         token_log_probs = log_probs_dist.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
#         # 排除 Padding
#         mask = (shift_labels != pad_token_id)
#         if mask.sum().item() == 0:
#             return [0.0, 0.0, 0.0, 0.0, 0.0]
            
#         valid_log_probs = token_log_probs[mask]
        
#         # 提取 5 维特征: [均值, 方差, 最小值, 最大值, 中位数]
#         mean_lp = valid_log_probs.mean().item()
#         var_lp = valid_log_probs.var().item() if valid_log_probs.size(0) > 1 else 0.0
#         min_lp = valid_log_probs.min().item()
#         max_lp = valid_log_probs.max().item()
#         median_lp = valid_log_probs.median().item()
        
#         return [mean_lp, var_lp, min_lp, max_lp, median_lp]

#     def extract_batch(self, text_list, desc="Extracting"):
#         records = []
#         for text in tqdm(text_list, desc=desc):
#             inputs_1 = self.tokenizer_1(text, return_tensors="pt", truncation=True, max_length=UserConfig.MAX_TOKEN_LENGTH).to(self.device)
#             inputs_2 = self.tokenizer_2(text, return_tensors="pt", truncation=True, max_length=UserConfig.MAX_TOKEN_LENGTH).to(self.device)

#             if inputs_1.input_ids.size(1) < 3 or inputs_2.input_ids.size(1) < 3:
#                 continue

#             with torch.no_grad():
#                 logits_1 = self.model_1(**inputs_1).logits
#                 logits_2 = self.model_2(**inputs_2).logits

#                 stats_1 = self._get_ghostbuster_stats(logits_1, inputs_1.input_ids, self.tokenizer_1.pad_token_id)
#                 stats_2 = self._get_ghostbuster_stats(logits_2, inputs_2.input_ids, self.tokenizer_2.pad_token_id)

#                 # 将两个模型的特征拼接在一起 (总共 10 维特征)
#                 combined_features = stats_1 + stats_2
            
#             records.append(combined_features)
        
#         return records

# # ==============================================================================
# # 3. 数据处理工具
# # ==============================================================================
# def load_and_process_data(json_path, cache_path, extractor=None):
#     if UserConfig.USE_CACHE and os.path.exists(cache_path):
#         print(f"[Data] Found cache at {cache_path}, loading...")
#         return pd.read_csv(cache_path)
    
#     if not os.path.exists(json_path):
#         raise FileNotFoundError(f"Dataset not found: {json_path}")
        
#     print(f"[Data] Reading dataset: {json_path}")
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     if extractor is None:
#         extractor = GhostBusterFeatureExtractor()
        
#     all_records = []
#     for label, texts in data.items():
#         print(f"Processing label: {label} ({len(texts)} samples)")
#         features = extractor.extract_batch(texts, desc=f"Feat Extract [{label}]")
#         for feat in features:
#             record = {"label": label}
#             # 动态生成列名 f_0 到 f_9
#             for i, val in enumerate(feat):
#                 record[f"f_{i}"] = val
#             all_records.append(record)
            
#     df = pd.DataFrame(all_records)
    
#     if UserConfig.USE_CACHE:
#         df.to_csv(cache_path, index=False)
#         print(f"[Data] Features cached to {cache_path}")
        
#     return df

# # ==============================================================================
# # 4. 主程序流程 (Supervised Multi-class Detection)
# # ==============================================================================
# def main():
#     print("="*75)
#     print(f"🚀 GhostBuster Supervised OvR Detection -> Saving to: {UserConfig.OUTPUT_DIR}")
#     print("="*75)
    
#     need_model = not (UserConfig.USE_CACHE and os.path.exists(UserConfig.TRAIN_CACHE_FILE) and os.path.exists(UserConfig.TEST_CACHE_FILE))
#     extractor = GhostBusterFeatureExtractor() if need_model else None

#     print("\n--- Phase 1: Processing Training Data ---")
#     df_train = load_and_process_data(UserConfig.TRAIN_DATA_PATH, UserConfig.TRAIN_CACHE_FILE, extractor)
    
#     print("\n--- Phase 2: Processing Test Data ---")
#     df_test = load_and_process_data(UserConfig.TEST_DATA_PATH, UserConfig.TEST_CACHE_FILE, extractor)

#     # --- Phase 3: Classifier Training ---
#     print("\n--- Phase 3: Training Logistic Regression Classifier ---")
#     # 提取特征列
#     feature_cols = [col for col in df_train.columns if col.startswith('f_')]
    
#     X_train = df_train[feature_cols].values
#     y_train = df_train["label"].astype(str).values
#     X_test = df_test[feature_cols].values
#     y_test = df_test["label"].astype(str).values
    
#     classes = np.unique(y_test)
    
#     # 填补 NaN 并进行标准化 (极为重要，GhostBuster各特征的数值差异很大)
#     X_train = np.nan_to_num(X_train)
#     X_test = np.nan_to_num(X_test)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # 训练逻辑回归 (由于是有监督方法，模型会自动学到如何划分不同来源)
#     clf = LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=1000, random_state=42)
#     clf.fit(X_train_scaled, y_train)
    
#     print("\n--- Phase 4: Evaluation & Logging ---")
#     # 预测概率和离散标签
#     y_prob = clf.predict_proba(X_test_scaled)
#     y_pred = clf.predict(X_test_scaled)
    
#     individual_metrics = {}
#     auc_list, acc_list, f1_list = [], [], []

#     print(f"\n{'='*60}")
#     print(f"{'Class (Target vs Rest)':<20} | {'AUC':<10} | {'ACC':<10} | {'F1':<10}")
#     print(f"{'-'*60}")

#     for i, class_name in enumerate(clf.classes_):
#         y_test_bin = (y_test == class_name).astype(int)
#         y_pred_bin = (y_pred == class_name).astype(int)
#         pred_scores = y_prob[:, i]
            
#         try:
#             # 由于这是有监督分类器的概率输出（必定介于0~1），我们直接算即可，不需要再翻转人类的正负号了
#             auc = roc_auc_score(y_test_bin, pred_scores)
#             acc = accuracy_score(y_test_bin, y_pred_bin)
#             f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
#         except Exception as e:
#             print(f"[Warning] Failed for class {class_name}: {e}")
#             auc, acc, f1 = 0.0, 0.0, 0.0
            
#         auc_list.append(auc)
#         acc_list.append(acc)
#         f1_list.append(f1)
        
#         individual_metrics[str(class_name)] = {
#             "auc": round(auc, 4),
#             "accuracy": round(acc, 4),
#             "f1_score": round(f1, 4)
#         }
        
#         print(f"{str(class_name):<20} | {auc:<10.4f} | {acc:<10.4f} | {f1:<10.4f}")

#     print(f"{'='*60}")
#     mean_auc = np.mean(auc_list)
#     mean_acc = np.mean(acc_list)
#     mean_f1 = np.mean(f1_list)
    
#     print(f"🎯 Macro Avg Metrics  | AUC: {mean_auc:.4f} | ACC: {mean_acc:.4f} | F1: {mean_f1:.4f}")
#     print(f"{'='*60}")

#     final_metrics = {
#         "macro_avg_auc": round(mean_auc, 4),
#         "macro_avg_accuracy": round(mean_acc, 4),
#         "macro_avg_f1": round(mean_f1, 4),
#         "individual_metrics": individual_metrics
#     }

#     metrics_path = os.path.join(UserConfig.OUTPUT_DIR, UserConfig.METRICS_FILENAME)
#     with open(metrics_path, "w", encoding="utf-8") as f:
#         json.dump(final_metrics, f, indent=4, ensure_ascii=False)
        
#     print(f"📁 Detailed JSON metrics saved to: {metrics_path}\n")

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print(f"\n[Error] Runtime Error: {e}")
#         import traceback
#         traceback.print_exc()


import os
import json
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 配置部分（多数据集自动化配置）
# ==============================================================================
class Config:
    # 代理模型路径 (概率评估器)
    PROXY_MODELS = [
       {"type": "llama", "path": "/home/share/models/llama-7b"},
       {"type": "qwen", "path": "/home/share/models/qwen2.5-7b-instruct"}
    ]

    # 🌟 基础路径统一配置
    INPUT_BASE_DIR = "/home/gsy/project2/TuringBench"
    OUTPUT_BASE_DIR = "/home/gsy/project2/MAGE/features/squad_othermethod"

    # 🌟 只需要在这里填入你要跑的数据集文件夹名称
    TASK_NAMES = [
    "TT_ctrl",
    # "TT_fair_wmt20",
    # "TT_gpt1",
    # "TT_gpt2_large",
    # "TT_gpt2_medium",
    # "TT_gpt2_pytorch",
    # "TT_gpt2_small",
    # "TT_gpt2_xl",
    # "TT_gpt3",
    # "TT_grover_base",
    # "TT_grover_large",
    # "TT_grover_mega",
    # "TT_pplm_distil",
    # "TT_pplm_gpt2",
    # "TT_transfo_xl",
    # "TT_xlm",
    # "TT_xlnet_base",
    # "TT_xlnet_large"
    ]
    
    # 每类的样本条数限制（-1表示使用全部）
    PER_CLASS_TRAIN_SAMPLES = 3000 
    PER_CLASS_TEST_SAMPLES = 1000    
    
    MAX_TOKEN_LENGTH = 1024  
    
    # 指定GPU编号
    GPU_ID = 1 
    DEVICE = f"cuda:{GPU_ID}" if (torch.cuda.is_available() and GPU_ID in [0, 1]) else "cpu"

# ==============================================================================
# 2. 数据处理与辅助工具
# ==============================================================================
def convert_to_python_type(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, (list, tuple)): return [convert_to_python_type(item) for item in obj]
    elif isinstance(obj, dict): return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, bool): return obj
    else: return obj

def sample_per_class(samples, labels, per_class_samples):
    class2samples = {}
    class2labels = {}
    for s, l in zip(samples, labels):
        if l not in class2samples:
            class2samples[l] = []
            class2labels[l] = []
        class2samples[l].append(s)
        class2labels[l].append(l)
    
    new_samples, new_labels = [], []
    for cls in class2samples:
        cls_samples = class2samples[cls]
        cls_labels = class2labels[cls]
        take = len(cls_samples) if (per_class_samples == -1 or len(cls_samples) <= per_class_samples) else per_class_samples
        new_samples.extend(cls_samples[:take])
        new_labels.extend(cls_labels[:take])
    
    return new_samples, new_labels

def load_raw_data_from_json(path):
    if not os.path.exists(path): return {}
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

# ==============================================================================
# 3. 核心特征提取器 (GhostBuster 串行化改造)
# ==============================================================================
class GhostBusterSingleExtractor:
    """每次只加载一个模型，提取 5 维 GhostBuster 特征"""
    def __init__(self, model_config):
        self.device = Config.DEVICE
        self.model_type = model_config['type']
        self.model_path = model_config['path']
        
        print(f"\n[{self.device.upper()}] Loading {self.model_type.upper()} from: {os.path.basename(self.model_path)} ...")
        
        trust_remote = True if self.model_type in ['qwen', 'falcon'] else False
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=trust_remote, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            trust_remote_code=trust_remote, 
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map={"": self.device} 
        ).eval()

    def unload(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_stats(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=Config.MAX_TOKEN_LENGTH).to(self.device)
        
        if inputs.input_ids.size(1) < 3:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()
        
        log_probs_dist = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs_dist.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        mask = (shift_labels != self.tokenizer.pad_token_id)
        if mask.sum().item() == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
            
        valid_log_probs = token_log_probs[mask]
        
        # 提取 5 维特征
        mean_lp = valid_log_probs.mean().item()
        var_lp = valid_log_probs.var().item() if valid_log_probs.size(0) > 1 else 0.0
        min_lp = valid_log_probs.min().item()
        max_lp = valid_log_probs.max().item()
        median_lp = valid_log_probs.median().item()
        
        return [mean_lp, var_lp, min_lp, max_lp, median_lp]

# ==============================================================================
# 4. 主程序流程
# ==============================================================================
def main():
    if torch.cuda.is_available() and Config.GPU_ID in [0, 1]:
        torch.cuda.set_device(Config.GPU_ID)
        print(f"\n[*] 使用设备: cuda:{Config.GPU_ID} ({torch.cuda.get_device_name(Config.GPU_ID)})")
    else:
        print(f"\n[*] 使用设备: {Config.DEVICE}")

    print("="*75)
    print(f"🚀 GhostBuster Multi-Task Supervised OvR Detection Pipeline")
    print("="*75)

    # --- Phase 1: 预加载并处理所有任务数据 ---
    print("\n>>> [Phase 1] Pre-loading all datasets...")
    all_tasks_meta = {}
    datasets_dict = {} 
    
    for task_name in Config.TASK_NAMES:
        train_path = os.path.join(Config.INPUT_BASE_DIR, task_name, "train_data.json")
        test_path = os.path.join(Config.INPUT_BASE_DIR, task_name, "test_data.json")
        output_dir = os.path.join(Config.OUTPUT_BASE_DIR, f"ghostbuster_{task_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        train_raw_data = load_raw_data_from_json(train_path)
        test_raw_data = load_raw_data_from_json(test_path)
        
        if not train_raw_data or not test_raw_data:
            print(f"[!] 跳过任务 {task_name}: 未找到完整的数据集。")
            continue
            
        # 展平数据
        def flatten_data(raw_dict):
            samples, labels = [], []
            for cls_name, texts in raw_dict.items():
                for t in texts:
                    if t.strip():
                        samples.append(t)
                        labels.append(cls_name)
            return samples, labels
            
        train_samples, train_labels = flatten_data(train_raw_data)
        test_samples, test_labels = flatten_data(test_raw_data)
        
        train_samples, train_labels = sample_per_class(train_samples, train_labels, Config.PER_CLASS_TRAIN_SAMPLES)
        test_samples, test_labels = sample_per_class(test_samples, test_labels, Config.PER_CLASS_TEST_SAMPLES)
        
        # 记录元数据
        all_tasks_meta[task_name] = {
            "train_labels": train_labels,
            "test_labels": test_labels,
            "test_samples": test_samples,
            "output_dir": output_dir,
            "train_feats": [], # 用于收集不同模型的特征
            "test_feats": []
        }
        
        datasets_dict[f"{task_name}_train"] = train_samples
        datasets_dict[f"{task_name}_test"] = test_samples

    if not datasets_dict:
        print("\n[!] 没有有效任务可执行，程序退出。")
        return

    # --- Phase 2: 串行模型推理与特征即时落盘 ---
    print("\n>>> [Phase 2] Serial Feature Extraction & Checkpointing...")
    
    for model_idx, model_config in enumerate(Config.PROXY_MODELS):
        m_type = model_config['type']
        
        # 检查是否需要加载该模型 (如果所有任务都缓存了该模型，则跳过)
        datasets_to_process = {}
        for data_name, samples in datasets_dict.items():
            is_train = data_name.endswith("_train")
            task_name = data_name[:-6] if is_train else data_name[:-5]
            split = "train" if is_train else "test"
            out_dir = all_tasks_meta[task_name]["output_dir"]
            
            save_path = os.path.join(out_dir, f"gb_feat_{m_type}_{split}.npy")
            
            if os.path.exists(save_path):
                print(f"[*] 检测到缓存: 快速加载 [{task_name}-{split}] 上的 {m_type} 特征。")
                loaded_feats = np.load(save_path)
                if is_train: all_tasks_meta[task_name]["train_feats"].append(loaded_feats)
                else: all_tasks_meta[task_name]["test_feats"].append(loaded_feats)
            else:
                datasets_to_process[data_name] = samples

        if not datasets_to_process:
            print(f"[*] ✅ 模型 {m_type} 在所有任务上的特征已完整读取，跳过物理加载。")
            continue

        # 物理加载模型进行推理
        extractor = GhostBusterSingleExtractor(model_config)
        
        for data_name, samples in datasets_to_process.items():
            is_train = data_name.endswith("_train")
            task_name = data_name[:-6] if is_train else data_name[:-5]
            split = "train" if is_train else "test"
            out_dir = all_tasks_meta[task_name]["output_dir"]
            save_path = os.path.join(out_dir, f"gb_feat_{m_type}_{split}.npy")
            
            extracted_feats = []
            for text in tqdm(samples, desc=f"   Inferencing {data_name} with {m_type}", leave=False):
                try:
                    stats = extractor.get_stats(text)
                except Exception:
                    stats = [0.0, 0.0, 0.0, 0.0, 0.0]
                extracted_feats.append(stats)
            
            feat_array = np.array(extracted_feats, dtype=np.float32)
            
            # 即时保存落盘
            np.save(save_path, feat_array)
            print(f"  ✅ 阶段性保存：[{data_name}] 的 {m_type} 特征已保存至 -> {save_path}")
            
            # 加入内存矩阵
            if is_train: all_tasks_meta[task_name]["train_feats"].append(feat_array)
            else: all_tasks_meta[task_name]["test_feats"].append(feat_array)
            
        extractor.unload()

    # --- Phase 3: 分类器训练与评估 ---
    print("\n>>> [Phase 3] Training Classifiers & Evaluation...")
    
    for task_name, meta in all_tasks_meta.items():
        print(f"\n" + "="*60)
        print(f" 评估任务: {task_name} ")
        print("="*60)
        
        # 拼接多个模型的特征 (Llama 5维 + Qwen 5维 = 10维)
        X_train = np.concatenate(meta["train_feats"], axis=1)
        X_test = np.concatenate(meta["test_feats"], axis=1)
        
        y_train = np.array(meta["train_labels"])
        y_test = np.array(meta["test_labels"])
        
        # NaN 填补与标准化 (极度重要，GB特征量级差异大)
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练 OvR 逻辑回归
        clf = LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        y_prob = clf.predict_proba(X_test_scaled)
        y_pred = clf.predict(X_test_scaled)
        
        # --- 详细指标计算 ---
        individual_metrics = {}
        auc_list, acc_list, f1_list = [], [], []

        print(f"{'Class (Target vs Rest)':<25} | {'AUC':<8} | {'ACC':<8} | {'F1':<8}")
        print(f"{'-'*60}")

        for i, class_name in enumerate(clf.classes_):
            y_test_bin = (y_test == class_name).astype(int)
            y_pred_bin = (y_pred == class_name).astype(int)
            pred_scores = y_prob[:, i]
                
            try:
                if np.sum(y_test_bin) > 0:
                    auc = roc_auc_score(y_test_bin, pred_scores)
                else: auc = 0.0
                acc = accuracy_score(y_test_bin, y_pred_bin)
                f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
            except Exception:
                auc, acc, f1 = 0.0, 0.0, 0.0
                
            auc_list.append(auc)
            acc_list.append(acc)
            f1_list.append(f1)
            
            individual_metrics[str(class_name)] = {
                "auc": round(auc, 4), "accuracy": round(acc, 4), "f1_score": round(f1, 4)
            }
            print(f"{str(class_name):<25} | {auc:<8.4f} | {acc:<8.4f} | {f1:<8.4f}")

        print(f"{'-'*60}")
        mean_auc, mean_acc, mean_f1 = np.mean(auc_list), np.mean(acc_list), np.mean(f1_list)
        print(f"🎯 Macro Avg Metrics       | {mean_auc:<8.4f} | {mean_acc:<8.4f} | {mean_f1:<8.4f}")
        print(f"{'='*60}")

        # --- 结果保存 ---
        out_dir = meta["output_dir"]
        
        # 1. 保存 Metrics
        final_metrics = {
            "macro_avg_auc": round(mean_auc, 4),
            "macro_avg_accuracy": round(mean_acc, 4),
            "macro_avg_f1": round(mean_f1, 4),
            "individual_metrics": individual_metrics
        }
        metrics_path = os.path.join(out_dir, "metrics_result.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(convert_to_python_type(final_metrics), f, indent=4, ensure_ascii=False)
            
        # 2. 保存 Predictions (类似于 Sniffer 的 Bad Case 记录文件)
        pred_path = os.path.join(out_dir, "predictions.json")
        results = []
        for i in range(len(meta["test_samples"])):
            results.append(convert_to_python_type({
                "sample_text": meta["test_samples"][i],
                "true_label_name": y_test[i],
                "pred_label_name": y_pred[i],
                "is_correct": bool(y_test[i] == y_pred[i])
            }))
        with open(pred_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        print(f"📁 任务 {task_name} 的指标与预测结果已保存至: {out_dir}")

if __name__ == "__main__":
    main()