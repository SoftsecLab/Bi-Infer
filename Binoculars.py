# import os
# import json
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
# import warnings

# warnings.filterwarnings('ignore')

# # ==============================================================================
# # 1. 手动配置区域 (Manual Configuration)
# # ==============================================================================
# GPU_ID = "1"  # 设置使用的 GPU 编号，若显存不足请改为 "0,1"

# # 显存分配设置
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# class UserConfig:
#     # 代理模型路径
#     OBSERVER_MODEL_PATH = "/home/share/models/llama-7b"
#     PERFORMER_MODEL_PATH = "/home/share/models/qwen2.5-7b-instruct"
    
#     # 统一管理多个数据集 (零样本测试无需训练集)
#     DATASETS = [
#         #  {
#         #     "name": "TT_fair_wmt20",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_fair_wmt20/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_gpt1",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_gpt1/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_gpt2_large",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_gpt2_large/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_gpt2_medium",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_gpt2_medium/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_gpt2_pytorch",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_gpt2_pytorch/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_gpt2_small",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_gpt2_small/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_gpt2_xl",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_gpt2_xl/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_gpt3",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_gpt3/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_grover_base",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_grover_base/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_grover_large",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_grover_large/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_grover_mega",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_grover_mega/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_pplm_distil",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_pplm_distil/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_pplm_gpt2",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_pplm_gpt2/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_transfo_xl",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_transfo_xl/test_data.json",
#         #     "max_samples": 4000  
#         # }, {
#         #     "name": "TT_xlm",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_xlm/test_data.json",
#         #     "max_samples": 4000  
#         # },{
#         #     "name": "TT_xlnet_base",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_xlnet_base/test_data.json",
#         #     "max_samples": 4000  
#         # },{
#         #     "name": "TT_xlnet_large",
#         #     "test_path": "/home/gsy/project2/TuringBench/two_class/TT_xlnet_large/test_data.json",
#         #     "max_samples": 4000  
#         # },
#         # # {
#         # #     "name": "arxiv",
#         # #     "test_path": "/home/gsy/project2/m4/arxiv/data_test_arxiv.json",
#         # #     "max_samples": 4000  
#         # # },
#         # # {
#         # #     "name": "reddit",
#         # #     "test_path": "/home/gsy/project2/m4/reddit/data_test_reddit.json",
#         # #     "max_samples": 4000
#         # # },
#         # # {
#         # #     "name": "wikihow",
#         # #     "test_path": "/home/gsy/project2/m4/wikihow/data_test_wikihow.json",
#         # #     "max_samples": 4000
#         # # },
#         {
#             "name": "wikipedia",
#             "test_path": "/home/gsy/project2/m4/wikipedia/data_test_wikipedia.json",
#             "max_samples": 4000
#         },
#     ]

#     OUTPUT_DIR = "/home/gsy/project2/TuringBench/Binoculars_OvR_Results"
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     MAX_TOKEN_LENGTH = 1024  

# # 确保输出目录存在
# os.makedirs(UserConfig.OUTPUT_DIR, exist_ok=True)

# # ==============================================================================
# # 2. Binoculars 特征提取器
# # ==============================================================================
# class BinocularsFeatureExtractor:
#     def __init__(self):
#         self.device = UserConfig.DEVICE
#         if self.device == "cuda":
#             gpu_count = torch.cuda.device_count()
#             print(f"[Init] Using {gpu_count} GPUs.")
        
#         # 加载 Observer 模型 (LLaMA)
#         print(f"[Init] Loading Observer Model (LLaMA)...")
#         self.tokenizer_obs = AutoTokenizer.from_pretrained(
#             UserConfig.OBSERVER_MODEL_PATH, local_files_only=True, trust_remote_code=True, use_fast=False, padding_side="right"
#         )
#         if self.tokenizer_obs.pad_token is None:
#             self.tokenizer_obs.pad_token = self.tokenizer_obs.eos_token

#         self.observer = AutoModelForCausalLM.from_pretrained(
#             UserConfig.OBSERVER_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#             local_files_only=True, device_map="auto" 
#         ).eval()

#         # 加载 Performer 模型 (Qwen)
#         print(f"[Init] Loading Performer Model (Qwen)...")
#         self.tokenizer_perf = AutoTokenizer.from_pretrained(
#             UserConfig.PERFORMER_MODEL_PATH, local_files_only=True, trust_remote_code=True, padding_side="right"
#         )
#         if self.tokenizer_perf.pad_token is None:
#             self.tokenizer_perf.pad_token = self.tokenizer_perf.eos_token

#         self.performer = AutoModelForCausalLM.from_pretrained(
#             UserConfig.PERFORMER_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
#             local_files_only=True, device_map="auto" 
#         ).eval()

#     def extract_batch(self, text_list, label_name):
#         records = []
#         for text in tqdm(text_list, desc=f"Feat Extract [{label_name}]"):
#             inputs_obs = self.tokenizer_obs(text, return_tensors="pt", truncation=True, max_length=UserConfig.MAX_TOKEN_LENGTH, padding="max_length").to(self.device)
#             inputs_perf = self.tokenizer_perf(text, return_tensors="pt", truncation=True, max_length=UserConfig.MAX_TOKEN_LENGTH, padding="max_length").to(self.device)

#             input_ids_obs, input_ids_perf = inputs_obs.input_ids, inputs_perf.input_ids
#             if (input_ids_obs != self.tokenizer_obs.pad_token_id).sum().item() < 2:
#                 continue 

#             with torch.no_grad():
#                 shift_logits_obs = self.observer(**inputs_obs).logits[..., :-1, :].contiguous()
#                 shift_labels_obs = input_ids_obs[..., 1:].contiguous()
#                 mask_obs = (shift_labels_obs != self.tokenizer_obs.pad_token_id)

#                 shift_logits_perf = self.performer(**inputs_perf).logits[..., :-1, :].contiguous()
#                 shift_labels_perf = input_ids_perf[..., 1:].contiguous()
#                 mask_perf = (shift_labels_perf != self.tokenizer_perf.pad_token_id)

#                 loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                
#                 # Log Perplexity (Observer)
#                 loss_obs = loss_fct(shift_logits_obs.view(-1, shift_logits_obs.size(-1)), shift_labels_obs.view(-1)).view(shift_labels_obs.shape)
#                 log_ppl = loss_obs[mask_obs].mean().item() if mask_obs.sum().item() > 0 else 0.0

#                 # Log Cross-Perplexity (Performer)
#                 loss_perf = loss_fct(shift_logits_perf.view(-1, shift_logits_perf.size(-1)), shift_labels_perf.view(-1)).view(shift_labels_perf.shape)
#                 log_x_ppl = loss_perf[mask_perf].mean().item() if mask_perf.sum().item() > 0 else 0.0

#                 score = log_ppl / log_x_ppl if log_x_ppl != 0 else 0
            
#             records.append({"label": label_name, "binoculars_score": score})
#         return records

# # ==============================================================================
# # 3. 数据处理与报告生成工具
# # ==============================================================================
# def load_and_process_test_data(json_path, max_samples, extractor, cache_path):
#     if os.path.exists(cache_path):
#         print(f"[Data] Loading cached features from {cache_path}")
#         return pd.read_csv(cache_path)

#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
        
#     all_records = []
#     for label, texts in data.items():
#         selected_texts = texts[:max_samples] if max_samples else texts
#         print(f"Processing {label}: {len(selected_texts)} samples")
#         features = extractor.extract_batch(selected_texts, label_name=label)
#         all_records.extend(features)
            
#     df = pd.DataFrame(all_records)
#     df.to_csv(cache_path, index=False)
#     return df

# def save_report_as_image(report_dict, dataset_name):
#     df_report = pd.DataFrame(report_dict).transpose()
#     fig, ax = plt.subplots(figsize=(8, len(df_report)*0.6 + 2))
#     ax.axis('off')
#     tbl = ax.table(cellText=df_report.values.round(4), colLabels=df_report.columns, rowLabels=df_report.index, loc='center', cellLoc='center')
#     tbl.auto_set_font_size(False)
#     tbl.set_fontsize(12)
#     tbl.scale(1.2, 1.2)
#     plt.title(f"Zero-Shot OvR Report (Binoculars): {dataset_name}", fontsize=14, pad=20)
#     img_path = os.path.join(UserConfig.OUTPUT_DIR, f"{dataset_name}_ovr_report.png")
#     plt.savefig(img_path, bbox_inches='tight', dpi=300)
#     print(f"📊 Report image saved: {img_path}")
#     plt.close()

# # ==============================================================================
# # 4. 评估流程：零样本一对多独立测试 (Zero-Shot OvR)
# # ==============================================================================
# def main():
#     print("="*60)
#     print(f"🚀 Starting Binoculars (LLaMA+Qwen) Zero-Shot OvR Pipeline")
#     print(f"📁 Output Directory: {UserConfig.OUTPUT_DIR}")
#     print("="*60)

#     # 智能检查缓存：如果所有数据集都已经提取过特征，就不再将两个 7B 模型加载到显存中
#     need_model = False
#     for ds_cfg in UserConfig.DATASETS:
#         cache_path = os.path.join(UserConfig.OUTPUT_DIR, f"{ds_cfg['name']}_test_feat.csv")
#         if not os.path.exists(cache_path):
#             need_model = True
#             break
            
#     extractor = BinocularsFeatureExtractor() if need_model else None

#     # 遍历所有配置的数据集
#     for ds_cfg in UserConfig.DATASETS:
#         name = ds_cfg["name"]
#         print(f"\n{'#'*40}\n# Task: {name} (Zero-Shot OvR)\n{'#'*40}")

#         cache_path = os.path.join(UserConfig.OUTPUT_DIR, f"{name}_test_feat.csv")
#         df_test = load_and_process_test_data(ds_cfg["test_path"], ds_cfg["max_samples"], extractor, cache_path)

#         y_test = df_test["label"].astype(str).values
#         scores = df_test["binoculars_score"].values
#         classes = np.unique(y_test)
        
#         ovr_metrics = {}
#         auc_list, acc_list, f1_list = [], [], []

#         print(f"\n{'Class (Binary OvR)':<25} | {'AUC':<10} | {'Binary ACC':<10} | {'Binary F1':<10}")
#         print("-" * 70)

#         for cls in classes:
#             # 当前类为正例(1)，其他所有为负例(0)
#             y_true_bin = (y_test == cls).astype(int)
            
#             # 极性调整：根据您之前的逻辑，人类得分较高，AI 得分较低
#             # 为了统一 "分数越高越判定为目标类别" 的 OvR 规则：
#             # 若检测目标是人类，保持分数不变；若是检测 AI，将分数取负。
#             current_scores = scores if "human" in str(cls).lower() else -scores
            
#             try:
#                 # 评估 AUC (不依赖阈值)
#                 auc = roc_auc_score(y_true_bin, current_scores)
                
#                 # 零样本核心：使用测试集当前得分的中位数作为阈值
#                 thresh = np.median(current_scores)
#                 y_pred_bin = (current_scores >= thresh).astype(int)
                
#                 # 计算 F1 和 ACC
#                 acc = accuracy_score(y_true_bin, y_pred_bin)
#                 f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
#             except Exception as e:
#                 print(f"[Warning] Failed for {cls}: {e}")
#                 auc, acc, f1 = 0.5, 0.0, 0.0
                
#             ovr_metrics[str(cls)] = {"AUC": round(auc, 4), "ACC": round(acc, 4), "F1": round(f1, 4)}
#             auc_list.append(auc)
#             acc_list.append(acc)
#             f1_list.append(f1)
            
#             print(f"{str(cls):<25} | {auc:<10.4f} | {acc:<10.4f} | {f1:<10.4f}")

#         # 结果聚合
#         mean_auc = np.mean(auc_list)
#         mean_acc = np.mean(acc_list)
#         mean_f1 = np.mean(f1_list)
        
#         ovr_metrics["Macro Average"] = {
#             "AUC": round(mean_auc, 4),
#             "ACC": round(mean_acc, 4),
#             "F1": round(mean_f1, 4)
#         }
        
#         print("-" * 70)
#         print(f"⭐ Overall Accuracy (Mean Binary F1): {mean_f1:.4f}")
        
#         # 导出为 JSON 和图片
#         json_path = os.path.join(UserConfig.OUTPUT_DIR, f"{name}_ovr_metrics.json")
#         with open(json_path, 'w', encoding='utf-8') as f:
#             json.dump(ovr_metrics, f, indent=4)
            
#         save_report_as_image(ovr_metrics, name)

# if __name__ == "__main__":
#     main()

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 🌟 新增：引入 seaborn 以对齐右图的排版引擎
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 手动配置区域 (Manual Configuration)
# ==============================================================================
GPU_ID = "1"  

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

class UserConfig:
    OBSERVER_MODEL_PATH = "/home/share/models/llama-7b"
    PERFORMER_MODEL_PATH = "/home/share/models/qwen2.5-7b-instruct"
    
    DATASETS = [
        {
            "name": "arxiv",
            "test_path": "/home/gsy/project2/m4/arxiv/data_test_arxiv.json",
            "max_samples": 4000
        },
    ]

    OUTPUT_DIR = "/home/gsy/project2/TuringBench/Binoculars_OvR_Results"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_TOKEN_LENGTH = 1024  

os.makedirs(UserConfig.OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. Binoculars 特征提取器
# ==============================================================================
class BinocularsFeatureExtractor:
    def __init__(self):
        self.device = UserConfig.DEVICE
        if self.device == "cuda":
            gpu_count = torch.cuda.device_count()
            print(f"[Init] Using {gpu_count} GPUs.")
        
        print(f"[Init] Loading Observer Model (LLaMA)...")
        self.tokenizer_obs = AutoTokenizer.from_pretrained(
            UserConfig.OBSERVER_MODEL_PATH, local_files_only=True, trust_remote_code=True, use_fast=False, padding_side="right"
        )
        if self.tokenizer_obs.pad_token is None:
            self.tokenizer_obs.pad_token = self.tokenizer_obs.eos_token

        self.observer = AutoModelForCausalLM.from_pretrained(
            UserConfig.OBSERVER_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            local_files_only=True, device_map="auto" 
        ).eval()

        print(f"[Init] Loading Performer Model (Qwen)...")
        self.tokenizer_perf = AutoTokenizer.from_pretrained(
            UserConfig.PERFORMER_MODEL_PATH, local_files_only=True, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer_perf.pad_token is None:
            self.tokenizer_perf.pad_token = self.tokenizer_perf.eos_token

        self.performer = AutoModelForCausalLM.from_pretrained(
            UserConfig.PERFORMER_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            local_files_only=True, device_map="auto" 
        ).eval()

    def extract_batch(self, text_list, label_name):
        records = []
        for text in tqdm(text_list, desc=f"Feat Extract [{label_name}]"):
            inputs_obs = self.tokenizer_obs(text, return_tensors="pt", truncation=True, max_length=UserConfig.MAX_TOKEN_LENGTH, padding="max_length").to(self.device)
            inputs_perf = self.tokenizer_perf(text, return_tensors="pt", truncation=True, max_length=UserConfig.MAX_TOKEN_LENGTH, padding="max_length").to(self.device)

            input_ids_obs, input_ids_perf = inputs_obs.input_ids, inputs_perf.input_ids
            if (input_ids_obs != self.tokenizer_obs.pad_token_id).sum().item() < 2:
                continue 

            with torch.no_grad():
                shift_logits_obs = self.observer(**inputs_obs).logits[..., :-1, :].contiguous()
                shift_labels_obs = input_ids_obs[..., 1:].contiguous()
                mask_obs = (shift_labels_obs != self.tokenizer_obs.pad_token_id)

                shift_logits_perf = self.performer(**inputs_perf).logits[..., :-1, :].contiguous()
                shift_labels_perf = input_ids_perf[..., 1:].contiguous()
                mask_perf = (shift_labels_perf != self.tokenizer_perf.pad_token_id)

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                
                loss_obs = loss_fct(shift_logits_obs.view(-1, shift_logits_obs.size(-1)), shift_labels_obs.view(-1)).view(shift_labels_obs.shape)
                log_ppl = loss_obs[mask_obs].mean().item() if mask_obs.sum().item() > 0 else 0.0

                loss_perf = loss_fct(shift_logits_perf.view(-1, shift_logits_perf.size(-1)), shift_labels_perf.view(-1)).view(shift_labels_perf.shape)
                log_x_ppl = loss_perf[mask_perf].mean().item() if mask_perf.sum().item() > 0 else 0.0

                score = log_ppl / log_x_ppl if log_x_ppl != 0 else 0
            
            records.append({"label": label_name, "binoculars_score": score})
        return records

# ==============================================================================
# 3. 数据处理与报告生成工具
# ==============================================================================
def load_and_process_test_data(json_path, max_samples, extractor, cache_path):
    if os.path.exists(cache_path):
        print(f"[Data] Loading cached features from {cache_path}")
        return pd.read_csv(cache_path)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    all_records = []
    for label, texts in data.items():
        selected_texts = texts[:max_samples] if max_samples else texts
        print(f"Processing {label}: {len(selected_texts)} samples")
        features = extractor.extract_batch(selected_texts, label_name=label)
        all_records.extend(features)
            
    df = pd.DataFrame(all_records)
    df.to_csv(cache_path, index=False)
    return df

def save_report_as_image(report_dict, dataset_name):
    df_report = pd.DataFrame(report_dict).transpose()
    fig, ax = plt.subplots(figsize=(8, len(df_report)*0.6 + 2))
    ax.axis('off')
    tbl = ax.table(cellText=df_report.values.round(4), colLabels=df_report.columns, rowLabels=df_report.index, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.2)
    plt.title(f"Zero-Shot OvR Report (Binoculars): {dataset_name}", fontsize=14, pad=20)
    img_path = os.path.join(UserConfig.OUTPUT_DIR, f"{dataset_name}_ovr_report.png")
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    print(f"📊 Report image saved: {img_path}")
    plt.close()

# ==============================================================================
# 🌟 修改点：生成学术分数分布直方图，参数完全对齐 PROFILER 绘图代码
# ==============================================================================
def plot_binoculars_distribution(df, dataset_name):
    print("🎨 Generating Score Distribution Plot...")
    
    LABEL_MAPPING = {
        "human": "gpt-3.5-turbo",
        "text-davinci-003": "text-davinci-003",
        "bigscience/bloomz": "bloomz",
        "gpt-3.5-turbo": "human",
        "cohere": "cohere",
        "dolly-v2-12b": "dolly-v2-12b",
    }
    
    plot_df = df.copy()
    plot_df['plot_label'] = plot_df['label'].map(lambda x: LABEL_MAPPING.get(x, x))
    
    # 🌟 核心对齐 1：使用相同的 seaborn 主题和字体
    plt.rcParams['font.family'] = 'serif'
    sns.set_theme(style="ticks", font="serif") 
    
    # 🌟 核心对齐 2：使用相同的 figsize
    fig, ax = plt.subplots(figsize=(6, 4))
    
    labels = plot_df['plot_label'].unique()
    colors = sns.color_palette("tab10", n_colors=max(10, len(labels)))
    color_idx = 0
    
    sorted_labels = sorted(labels, key=lambda x: 0 if 'human' in str(x).lower() else 1)

    for label in sorted_labels:
        subset = plot_df[plot_df['plot_label'] == label]['binoculars_score'].dropna().values
        
        weights = np.ones_like(subset) / len(subset) * 100
        
        if 'human' in str(label).lower():
            c = 'dimgray' # 对齐右图的 dimgray
        else:
            c = colors[color_idx]
            color_idx += 1
            
        ax.hist(
            subset, 
            bins=80,
            weights=weights, 
            alpha=0.6, 
            label=label, 
            color=c, 
            edgecolor='none'
        )
        
    # 🌟 核心对齐 3：统一字号和粗细参数
    ax.set_xlabel('Binoculars Score', fontsize=16, fontweight='bold')
    ax.set_ylabel('Frequency (%)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 🌟 核心对齐 4：使用 seaborn 原生的 despine 去除多余边框
    sns.despine()
    
    # 🌟 核心对齐 5：完全一致的图例参数 (ncol=2 保持 3行 布局)
    ax.legend(
        title=None, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.2), 
        frameon=False, 
        fontsize=12, 
        ncol=2, 
        borderaxespad=0.
    )
    
    pdf_path = os.path.join(UserConfig.OUTPUT_DIR, f"{dataset_name}_score_dist.pdf")
    png_path = os.path.join(UserConfig.OUTPUT_DIR, f"{dataset_name}_score_dist.png")
    
    # 使用 bbox_inches='tight' 防止图例被裁切
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"✅ Distribution plot saved to:\n  - {pdf_path}")
    plt.close()

# ==============================================================================
# 4. 评估流程：零样本一对多独立测试 (Zero-Shot OvR)
# ==============================================================================
def main():
    print("="*60)
    print(f"🚀 Starting Binoculars (LLaMA+Qwen) Zero-Shot OvR Pipeline")
    print(f"📁 Output Directory: {UserConfig.OUTPUT_DIR}")
    print("="*60)

    need_model = False
    for ds_cfg in UserConfig.DATASETS:
        cache_path = os.path.join(UserConfig.OUTPUT_DIR, f"{ds_cfg['name']}_test_feat.csv")
        if not os.path.exists(cache_path):
            need_model = True
            break
            
    extractor = BinocularsFeatureExtractor() if need_model else None

    for ds_cfg in UserConfig.DATASETS:
        name = ds_cfg["name"]
        print(f"\n{'#'*40}\n# Task: {name} (Zero-Shot OvR)\n{'#'*40}")

        cache_path = os.path.join(UserConfig.OUTPUT_DIR, f"{name}_test_feat.csv")
        df_test = load_and_process_test_data(ds_cfg["test_path"], ds_cfg["max_samples"], extractor, cache_path)

        plot_binoculars_distribution(df_test, name)

        y_test = df_test["label"].astype(str).values
        scores = df_test["binoculars_score"].values
        classes = np.unique(y_test)
        
        ovr_metrics = {}
        auc_list, acc_list, f1_list = [], [], []

        print(f"\n{'Class (Binary OvR)':<25} | {'AUC':<10} | {'Binary ACC':<10} | {'Binary F1':<10}")
        print("-" * 70)

        for cls in classes:
            y_true_bin = (y_test == cls).astype(int)
            current_scores = scores if "human" in str(cls).lower() else -scores
            
            try:
                auc = roc_auc_score(y_true_bin, current_scores)
                thresh = np.median(current_scores)
                y_pred_bin = (current_scores >= thresh).astype(int)
                
                acc = accuracy_score(y_true_bin, y_pred_bin)
                f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            except Exception as e:
                print(f"[Warning] Failed for {cls}: {e}")
                auc, acc, f1 = 0.5, 0.0, 0.0
                
            ovr_metrics[str(cls)] = {"AUC": round(auc, 4), "ACC": round(acc, 4), "F1": round(f1, 4)}
            auc_list.append(auc)
            acc_list.append(acc)
            f1_list.append(f1)
            
            print(f"{str(cls):<25} | {auc:<10.4f} | {acc:<10.4f} | {f1:<10.4f}")

        mean_auc = np.mean(auc_list)
        mean_acc = np.mean(acc_list)
        mean_f1 = np.mean(f1_list)
        
        ovr_metrics["Macro Average"] = {
            "AUC": round(mean_auc, 4),
            "ACC": round(mean_acc, 4),
            "F1": round(mean_f1, 4)
        }
        
        print("-" * 70)
        print(f"⭐ Overall Accuracy (Mean Binary F1): {mean_f1:.4f}")
        
        json_path = os.path.join(UserConfig.OUTPUT_DIR, f"{name}_ovr_metrics.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ovr_metrics, f, indent=4)
            
        save_report_as_image(ovr_metrics, name)

if __name__ == "__main__":
    main()