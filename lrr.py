import json
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 用户配置区域
# ==============================================================================
class UserConfig:
    MODEL_PATH = "/home/share/models/llama-7b"
    
    # 定义多个任务/数据集 (移除 train_path，仅保留 test_path)
    DATASETS = [
        {
            "name": "TT_xlnet_large",
            "test_path": "/home/gsy/project2/TuringBench/two_class/TT_xlnet_large/test_data.json",
            "max_samples": 4000  # 每个类别最多取多少样本，设为 None 则取全部
        },
        # {
        #     "name": "reddit",
        #     "test_path": "/home/gsy/project2/m4/reddit/data_test_reddit.json",
        #     "max_samples": 4000  # 每个类别最多取多少样本，设为 None 则取全部
        # },
        # {
        #     "name": "wikihow",
        #     "test_path": "/home/gsy/project2/m4/wikihow/data_test_wikihow.json",
        #     "max_samples": 4000  # 每个类别最多取多少样本，设为 None 则取全部
        # },
        # {
        #     "name": "wikipedia",
        #     "test_path": "/home/gsy/project2/m4/wikipedia/data_test_wikipedia.json",
        #     "max_samples": 4000  # 每个类别最多取多少样本，设为 None 则取全部
        # },
        # 你可以在这里添加更多数据集
    ]

    OUTPUT_DIR = "/home/gsy/project2/TuringBench/detectllm_lrr_results"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_TOKEN_LENGTH = 512 

os.makedirs(UserConfig.OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. LRR 特征提取器
# ==============================================================================
class LRRFeatureExtractor:
    def __init__(self):
        self.device = UserConfig.DEVICE
        print(f"[Init] Loading Model from: {UserConfig.MODEL_PATH} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(UserConfig.MODEL_PATH, local_files_only=True, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            UserConfig.MODEL_PATH,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" 
        ).eval()

    def _compute_lrr_score(self, logits, input_ids):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        sorted_indices = shift_logits.argsort(dim=-1, descending=True)
        matches = (sorted_indices == shift_labels.unsqueeze(-1))
        ranks = matches.float().argmax(dim=-1) + 1  
        log_ranks = torch.log(ranks.float())
        
        mask = (shift_labels != self.tokenizer.pad_token_id)
        if mask.sum().item() > 0:
            avg_log_prob = token_log_probs[mask].mean().item()
            avg_log_rank = log_ranks[mask].mean().item()
            nll = -avg_log_prob
            return nll / (avg_log_rank + 1e-8)
        return 0.0

    def extract_batch(self, text_list, desc="Extracting"):
        records = []
        for text in tqdm(text_list, desc=desc):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=UserConfig.MAX_TOKEN_LENGTH).to(self.device)
            if inputs.input_ids.size(1) < 10: continue # 过滤过短文本
            with torch.no_grad():
                logits = self.model(**inputs).logits
                lrr_score = self._compute_lrr_score(logits, inputs.input_ids)
            records.append({"lrr_score": lrr_score})
        return records

# ==============================================================================
# 3. 数据处理与报告生成
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
        features = extractor.extract_batch(selected_texts, desc=f"[{label}]")
        for feat in features:
            all_records.append({"label": label, "lrr_score": feat["lrr_score"]})
            
    df = pd.DataFrame(all_records)
    df.to_csv(cache_path, index=False)
    return df

def save_report_as_image(report_dict, dataset_name):
    """ 将 OvR 评估结果的字典转换成图片 """
    df_report = pd.DataFrame(report_dict).transpose()
    
    fig, ax = plt.subplots(figsize=(8, len(df_report)*0.6 + 2))
    ax.axis('off')
    tbl = ax.table(cellText=df_report.values.round(4), 
                   colLabels=df_report.columns, 
                   rowLabels=df_report.index, 
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.2)
    plt.title(f"Zero-Shot OvR Report: {dataset_name}", fontsize=14, pad=20)
    
    img_path = os.path.join(UserConfig.OUTPUT_DIR, f"{dataset_name}_ovr_report.png")
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    print(f"📊 Report image saved: {img_path}")
    plt.close()

# ==============================================================================
# 4. 主程序 (零样本一对多独立测试)
# ==============================================================================
def main():
    # 智能判别是否需要加载大模型 (如果全部数据集都有缓存了，就不占用显存加载模型)
    need_model = False
    for ds_cfg in UserConfig.DATASETS:
        cache_path = os.path.join(UserConfig.OUTPUT_DIR, f"{ds_cfg['name']}_test_feat.csv")
        if not os.path.exists(cache_path):
            need_model = True
            break
            
    extractor = LRRFeatureExtractor() if need_model else None

    for ds_cfg in UserConfig.DATASETS:
        name = ds_cfg["name"]
        print(f"\n{'#'*40}\n# Task: {name} (Zero-Shot OvR)\n{'#'*40}")

        # 1. 提取或加载测试集特征
        cache_path = os.path.join(UserConfig.OUTPUT_DIR, f"{name}_test_feat.csv")
        df_test = load_and_process_test_data(ds_cfg["test_path"], ds_cfg["max_samples"], extractor, cache_path)

        # 2. 独立的多二元检测器评估 (One-vs-Rest)
        classes = df_test["label"].unique()
        y_test = df_test["label"].astype(str).values
        scores = df_test["lrr_score"].values
        
        ovr_metrics = {}
        auc_list, acc_list, f1_list = [], [], []

        print(f"\n{'Class (Binary OvR)':<25} | {'AUC':<10} | {'Binary ACC':<10} | {'Binary F1':<10}")
        print("-" * 70)

        for cls in classes:
            # 当前类为正例(1)，其他所有为负例(0)
            y_true_bin = (y_test == cls).astype(int)
            
            # 分数极性调整：通常 AI 的 LRR 特征异于人类
            # 若评估的是 Human 类，我们将分数倒置以符合常规的“得分越高越可能是该类”直觉
            current_scores = -scores if "human" in str(cls).lower() else scores
            
            try:
                # 评估 AUC
                auc = roc_auc_score(y_true_bin, current_scores)
                
                # 零样本核心步骤：使用测试集得分分布的中位数作为无监督分类阈值
                thresh = np.median(current_scores)
                y_pred_bin = (current_scores >= thresh).astype(int)
                
                # 计算针对该类别的独立二分指标
                acc = accuracy_score(y_true_bin, y_pred_bin)
                f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            except Exception as e:
                print(f"[Warning] Failed for {cls}: {e}")
                auc, acc, f1 = 0.5, 0.0, 0.0
                
            # 存入字典供画图
            ovr_metrics[str(cls)] = {"AUC": round(auc, 4), "ACC": round(acc, 4), "F1": round(f1, 4)}
            auc_list.append(auc)
            acc_list.append(acc)
            f1_list.append(f1)
            
            print(f"{str(cls):<25} | {auc:<10.4f} | {acc:<10.4f} | {f1:<10.4f}")
            
        # 3. 结果聚合与图像生成
        mean_auc = np.mean(auc_list)
        mean_acc = np.mean(acc_list)
        mean_f1 = np.mean(f1_list)
        
        # 将宏平均指标也加入字典中，方便在生成的图片最下面看到
        ovr_metrics["Macro Average"] = {
            "AUC": round(mean_auc, 4),
            "ACC": round(mean_acc, 4),
            "F1": round(mean_f1, 4)
        }
        
        print("-" * 70)
        print(f"⭐ Overall Accuracy (Mean Binary F1): {mean_f1:.4f}")
        
        # 保存详细指标为 JSON
        json_path = os.path.join(UserConfig.OUTPUT_DIR, f"{name}_ovr_metrics.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ovr_metrics, f, indent=4)
            
        # 渲染出美观的表格图片
        save_report_as_image(ovr_metrics, name)

if __name__ == "__main__":
    main()