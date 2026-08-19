# import json
# import torch
# import numpy as np
# import os
# import sys
# import scipy.stats
# import gc
# from itertools import combinations
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
# from sklearn.preprocessing import label_binarize
# from sklearn.impute import SimpleImputer
# from tqdm import tqdm
# import warnings

# warnings.filterwarnings('ignore')

# # ==========================================
# # 1. 全局配置 (Configuration)
# # ==========================================
# class Config:
#     # 🌟 修改1: 在这里填入你的 Llama 和 Qwen 模型的实际绝对路径
#     PROXY_MODEL_PATHS = [
#         "/home/share/models/llama-7b",  # 替换为实际的 Llama 路径
#         "/home/share/models/qwen2.5-7b-instruct"    # 替换为实际的 Qwen 路径
#     ] 
    
#     # 数据集路径
#     TRAIN_DATA_PATH = "/home/gsy/project2/TuringBench/two_class/TT_xlnet_large/train_data.json"
#     TEST_DATA_PATH = "/home/gsy/project2/TuringBench/two_class/TT_xlnet_large/test_data.json"
    
#     # 结果保存目录
#     OUTPUT_DIR = "/home/gsy/project2/MAGE/features/cmv_othermethod/profiler_llama_qwen_reddit"
    
#     # 移除的类别索引 (0-based)
#     REMOVE_LABEL_INDICES = [] 
    
#     # PROFILER 超参数
#     CONTEXT_WINDOW_SIZE = 4 # 论文推荐 W=4 或 6
#     MAX_SEQ_LEN = 512       # 截断长度，防止显存溢出
    
#     # 训练参数
#     SAMPLES_PER_CLASS_TRAIN = 4000
#     SAMPLES_PER_CLASS_TEST = 1000 
    
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ==========================================
# # 2. PROFILER 特征提取核心
# # ==========================================
# class ProfilerFeatureExtractor:
#     def __init__(self, model_path, device, window_size=4):
#         print(f"\n[{device.upper()}] 正在加载代理模型: {model_path} ...")
#         self.device = device
#         self.window_size = window_size
#         self.half_window = window_size // 2
        
#         # 预计算特征维度，用于特征提取失败时的 NaN 填充对齐
#         # Independent Patterns 维度: 18 * W
#         # Correlated Patterns 维度: W * (W - 1) / 2
#         self.expected_dim = (self.window_size * 18) + (self.window_size * (self.window_size - 1)) // 2
        
#         try:
#             # 🌟 修改2: 兼容 Qwen 的 Tokenizer，trust_remote_code=True 是必须的
#             self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
#             # 兼容不同模型的 pad_token 缺失问题
#             if self.tokenizer.pad_token is None:
#                 if self.tokenizer.eos_token is not None:
#                     self.tokenizer.pad_token = self.tokenizer.eos_token
#                 else:
#                     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
#             # 🌟 修改3: 统一使用 float16 加载，兼容 Llama 和 Qwen 降低显存占用
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 torch_dtype=torch.float16,
#                 device_map={"": self.device},
#                 trust_remote_code=True
#             )
#             self.model.eval()
            
#             # 如果新增了特殊 token，调整 embedding 大小
#             if self.tokenizer.pad_token == '[PAD]':
#                 self.model.resize_token_embeddings(len(self.tokenizer))
                
#         except Exception as e:
#             print(f"模型加载失败: {e}")
#             sys.exit(1)
            
#     def _compute_context_losses(self, input_ids, logits):
#         """计算上下文损失矩阵 (Context Losses)"""
#         ids = input_ids[0]      # [seq_len]
#         logs = logits[0][:-1]   # [seq_len-1, vocab]
        
#         seq_len = logs.size(0)
#         valid_start = self.half_window
#         valid_end = seq_len - self.half_window
        
#         if valid_start >= valid_end:
#             return None 
            
#         context_losses = []
#         loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
#         # 将 Float16 的 logits 转为 Float32，防止计算 Loss 时下溢出产生 NaN
#         valid_logits = logs[valid_start:valid_end].to(torch.float32) # [N, vocab]
        
#         # 遍历窗口内的每个偏移量
#         for j in range(-self.half_window, self.half_window + 1):
#             if j == 0 and self.window_size % 2 == 0: 
#                 continue
                
#             # 对齐 Logits 和 目标 Token
#             target_indices = torch.arange(valid_start, valid_end) + j + 1 
#             target_tokens = ids[target_indices.long()].to(self.device)
            
#             loss = loss_fct(valid_logits, target_tokens) # [N]
#             context_losses.append(loss.cpu().numpy())
            
#             if len(context_losses) >= self.window_size:
#                 break
                
#         if not context_losses:
#             return None
            
#         return np.stack(context_losses, axis=1) # [N, W]

#     def _extract_independent_patterns(self, loss_matrix):
#         """提取独立模式特征 (IP)"""
#         features = []
#         N, W = loss_matrix.shape
        
#         for w in range(W):
#             col = loss_matrix[:, w]
#             # 统计量
#             stats = [np.mean(col), np.std(col), np.min(col), np.max(col), np.median(col), np.var(col)]
#             features.extend(stats)
            
#             # 一阶差分
#             if len(col) > 1:
#                 diff1 = np.diff(col)
#                 stats_d1 = [np.mean(diff1), np.std(diff1), np.min(diff1), np.max(diff1), np.median(diff1), np.var(diff1)]
#                 features.extend(stats_d1)
#             else:
#                 features.extend([0]*6)
                
#             # 二阶中心差分
#             if len(col) > 2:
#                 diff2 = np.diff(col, n=2) / 2.0
#                 stats_d2 = [np.mean(diff2), np.std(diff2), np.min(diff2), np.max(diff2), np.median(diff2), np.var(diff2)]
#                 features.extend(stats_d2)
#             else:
#                 features.extend([0]*6)
#         return np.array(features)

#     def _extract_correlated_patterns(self, loss_matrix):
#         """提取相关模式特征 (CP)"""
#         features = []
#         probs_matrix = scipy.special.softmax(loss_matrix, axis=0)
        
#         for i, j in combinations(range(loss_matrix.shape[1]), 2):
#             p = probs_matrix[:, i] + 1e-9
#             q = probs_matrix[:, j] + 1e-9
#             kl_sym = np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))
#             features.append(kl_sym)
            
#         return np.array(features)

#     def get_features(self, text):
#         if not text or not text.strip(): return None
#         # 添加 padding=True 确保输入格式对齐
#         inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=Config.MAX_SEQ_LEN, padding=False).to(self.device)
#         if inputs.input_ids.size(1) < self.window_size + 5: return None
            
#         with torch.no_grad():
#             outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            
#         loss_matrix = self._compute_context_losses(inputs.input_ids, outputs.logits)
#         if loss_matrix is None: return None
            
#         feat_ip = self._extract_independent_patterns(loss_matrix)
#         feat_cp = self._extract_correlated_patterns(loss_matrix)
        
#         final_feats = np.concatenate([feat_ip, feat_cp])
#         final_feats = np.nan_to_num(final_feats, nan=0.0, posinf=0.0, neginf=0.0)
#         return final_feats

# # ==========================================
# # 3. 数据处理与保存逻辑 
# # ==========================================
# class DataProcessor:
#     def __init__(self, raw_data, remove_indices):
#         all_classes = sorted(list(raw_data.keys()))
#         print(f"[*] 原始类别: {all_classes}")
#         self.valid_classes = [c for i, c in enumerate(all_classes) if i not in remove_indices]
#         print(f"[*] 保留类别: {self.valid_classes}")
#         self.class2id = {c: i for i, c in enumerate(self.valid_classes)}
#         self.id2class = {i: c for c, i in self.class2id.items()}
        
#     def load_samples(self, file_path, max_per_class=None):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         samples, labels = [], []
#         for cls_name, texts in data.items():
#             norm_cls = cls_name.strip()
#             if norm_cls not in self.class2id: continue
#             label_idx = self.class2id[norm_cls]
#             if max_per_class: texts = texts[:max_per_class]
#             for t in texts:
#                 if t.strip():
#                     samples.append(t)
#                     labels.append(label_idx)
#         return samples, labels

# def save_evaluation_results(y_true, y_pred, y_prob, label_map, output_dir):
#     """保存评估结果"""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     target_names = [label_map[i] for i in range(len(label_map))]
    
#     report_str = classification_report(
#         y_true, y_pred, target_names=target_names, digits=4, zero_division=0
#     )
#     txt_path = os.path.join(output_dir, "classification_report.txt")
#     with open(txt_path, "w", encoding="utf-8") as f:
#         f.write(report_str)
#     print(f"[*] 已保存分类报告: {txt_path}")
#     print(report_str)
    
#     acc = accuracy_score(y_true, y_pred)
#     classes = list(range(len(label_map)))
#     y_true_bin = label_binarize(y_true, classes=classes)
    
#     if len(classes) == 2 and y_true_bin.shape[1] == 1:
#         y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        
#     individual_aucs = {}
#     auc_scores = []
    
#     for i, class_name in enumerate(target_names):
#         try:
#             if np.sum(y_true_bin[:, i]) > 0:
#                 auc_val = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
#             else:
#                 auc_val = 0.0
#         except:
#             auc_val = 0.0
        
#         auc_scores.append(auc_val)
#         individual_aucs[str(i)] = {"class_name": class_name, "auc": float(auc_val)}
        
#     mean_auc = np.mean(auc_scores)
#     json_output = {
#         "accuracy": float(acc),
#         "mean_auc": float(mean_auc),
#         "individual_aucs": individual_aucs
#     }
    
#     json_path = os.path.join(output_dir, "test_metrics.json")
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(json_output, f, indent=4)
#     print(f"[*] 已保存 JSON 指标: {json_path}")

# # ==========================================
# # 4. 主程序
# # ==========================================
# def main():
#     print("=== PROFILER 复现: Llama & Qwen 多模型特征融合 ===")
    
#     # 1. 预处理
#     with open(Config.TRAIN_DATA_PATH, 'r') as f: raw_train = json.load(f)
#     processor = DataProcessor(raw_train, Config.REMOVE_LABEL_INDICES)
    
#     # 2. 加载数据
#     print("\n>>> 加载数据...")
#     train_texts, train_labels = processor.load_samples(Config.TRAIN_DATA_PATH, Config.SAMPLES_PER_CLASS_TRAIN)
#     test_texts, test_labels = processor.load_samples(Config.TEST_DATA_PATH, Config.SAMPLES_PER_CLASS_TEST)
    
#     if not train_texts or not test_texts:
#         print("[!] 数据不足，退出")
#         return

#     # 3. 串行特征提取
#     all_train_feats = []
#     all_test_feats = []

#     for model_path in Config.PROXY_MODEL_PATHS:
#         extractor = ProfilerFeatureExtractor(model_path, Config.DEVICE, Config.CONTEXT_WINDOW_SIZE)
#         model_name = os.path.basename(model_path)
        
#         # 提取训练集
#         model_train_f = []
#         for t in tqdm(train_texts, desc=f"Extracting Train - {model_name}"):
#             f = extractor.get_features(t)
#             if f is None:
#                 f = np.full(extractor.expected_dim, np.nan)
#             model_train_f.append(f)
#         all_train_feats.append(np.array(model_train_f))
        
#         # 提取测试集
#         model_test_f = []
#         for t in tqdm(test_texts, desc=f"Extracting Test - {model_name}"):
#             f = extractor.get_features(t)
#             if f is None:
#                 f = np.full(extractor.expected_dim, np.nan)
#             model_test_f.append(f)
#         all_test_feats.append(np.array(model_test_f))

#         # 🌟 修改4: 彻底清理显存，安全切换到下一个模型
#         print(f"\n[*] 卸载模型 {model_name} 并清理显存...")
#         del extractor.model
#         del extractor.tokenizer
#         del extractor
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     # 4. 拼接特征
#     print("\n>>> 融合 Llama 和 Qwen 特征...")
#     X_train = np.concatenate(all_train_feats, axis=1) # 维度翻倍
#     X_test  = np.concatenate(all_test_feats, axis=1)  
    
#     y_train = np.array(train_labels)
#     y_test  = np.array(test_labels)
    
#     # 使用均值填充因文本过短导致的局部 NaN
#     print(">>> 处理缺失值 (Imputation)...")
#     imputer = SimpleImputer(strategy='mean')
#     X_train = imputer.fit_transform(X_train)
#     X_test = imputer.transform(X_test)
    
#     # 5. 训练与评估
#     print(f"\n>>> 训练分类器 (Random Forest, 融合后特征维度={X_train.shape[1]})...")
#     clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#     clf.fit(X_train, y_train)
    
#     print("\n>>> 生成预测结果...")
#     y_pred = clf.predict(X_test)
#     y_prob = clf.predict_proba(X_test)
    
#     # 6. 保存结果
#     save_evaluation_results(y_test, y_pred, y_prob, processor.id2class, Config.OUTPUT_DIR)
    
#     print("\n=== 程序执行完毕 ===")

# if __name__ == "__main__":
#     main()



import json
import torch
import numpy as np
import os
import sys
import scipy.stats
import gc
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
class Config:
    # 🌟 替换为你的 Llama 和 Qwen 模型的实际绝对路径
    PROXY_MODEL_PATHS = [
        "/home/share/models/llama3.1-8b-instruct",  
        "/home/share/models/qwen2.5-7b-instruct"    
    ] 
    
    # 数据集路径
    TRAIN_DATA_PATH = "/home/gsy/project2/m4/wikipedia/data_test_wikipedia.json"
    
    # 结果保存目录
    OUTPUT_DIR = "/home/gsy/project2/MAGE/features/cmv_othermethod/profiler_llama_qwen_reddit_viz"
    
    # 移除的类别索引 (0-based)
    REMOVE_LABEL_INDICES = [] 
    
    # PROFILER 超参数
    CONTEXT_WINDOW_SIZE = 4 # 论文推荐 W=4 或 6
    MAX_SEQ_LEN = 512       # 截断长度，防止显存溢出
    
    # 🌟 配置：每类抽取的文本条数（用于平均特征可视化）
    NUM_SAMPLES_PER_CLASS = 20 
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. PROFILER 特征提取核心
# ==========================================
class ProfilerFeatureExtractor:
    def __init__(self, model_path, device, window_size=4):
        print(f"\n[{device.upper()}] 正在加载代理模型: {model_path} ...")
        
        # ⚠️ 启动前检查本地路径是否有效
        if not os.path.exists(model_path):
            print(f"[错误] 找不到指定的模型路径: {model_path}")
            print("请检查路径拼写，或者确认您有读取该目录的权限。")
            sys.exit(1)
            
        self.device = device
        self.window_size = window_size
        self.half_window = window_size // 2
        self.expected_dim = (self.window_size * 18) + (self.window_size * (self.window_size - 1)) // 2
        
        try:
            # ⭐ 核心修改：强制只使用本地文件，禁止联网
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                local_files_only=True  # <--- 强制本地加载
            )
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map={"": self.device},
                trust_remote_code=True,
                local_files_only=True  # <--- 强制本地加载
            )
            self.model.eval()
            
            if self.tokenizer.pad_token == '[PAD]':
                self.model.resize_token_embeddings(len(self.tokenizer))
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("\n💡 提示: 如果仍然报错，请检查目录中是否包含 config.json 和模型权重文件。")
            sys.exit(1)
            
    def _compute_context_losses(self, input_ids, logits):
        ids = input_ids[0]      
        logs = logits[0][:-1]   
        
        seq_len = logs.size(0)
        valid_start = self.half_window
        valid_end = seq_len - self.half_window
        
        if valid_start >= valid_end:
            return None 
            
        context_losses = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
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
        features = []
        N, W = loss_matrix.shape
        for w in range(W):
            col = loss_matrix[:, w]
            stats = [np.mean(col), np.std(col), np.min(col), np.max(col), np.median(col), np.var(col)]
            features.extend(stats)
            if len(col) > 1:
                diff1 = np.diff(col)
                stats_d1 = [np.mean(diff1), np.std(diff1), np.min(diff1), np.max(diff1), np.median(diff1), np.var(diff1)]
                features.extend(stats_d1)
            else:
                features.extend([0]*6)
            if len(col) > 2:
                diff2 = np.diff(col, n=2) / 2.0
                stats_d2 = [np.mean(diff2), np.std(diff2), np.min(diff2), np.max(diff2), np.median(diff2), np.var(diff2)]
                features.extend(stats_d2)
            else:
                features.extend([0]*6)
        return np.array(features)

    def _extract_correlated_patterns(self, loss_matrix):
        features = []
        probs_matrix = scipy.special.softmax(loss_matrix, axis=0)
        for i, j in combinations(range(loss_matrix.shape[1]), 2):
            p = probs_matrix[:, i] + 1e-9
            q = probs_matrix[:, j] + 1e-9
            kl_sym = np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))
            features.append(kl_sym)
        return np.array(features)

    def get_features_with_matrix(self, text):
        if not text or not text.strip(): return None, None
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=Config.MAX_SEQ_LEN, padding=False).to(self.device)
        if inputs.input_ids.size(1) < self.window_size + 5: return None, None
            
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            
        loss_matrix = self._compute_context_losses(inputs.input_ids, outputs.logits)
        if loss_matrix is None: return None, None
            
        feat_ip = self._extract_independent_patterns(loss_matrix)
        feat_cp = self._extract_correlated_patterns(loss_matrix)
        
        final_feats = np.concatenate([feat_ip, feat_cp])
        final_feats = np.nan_to_num(final_feats, nan=0.0, posinf=0.0, neginf=0.0)
        return final_feats, loss_matrix

# ==========================================
# 3. 数据处理与可视化绘制逻辑 
# ==========================================
class DataProcessor:
    def __init__(self, raw_data, remove_indices):
        all_classes = sorted(list(raw_data.keys()))
        self.valid_classes = [c for i, c in enumerate(all_classes) if i not in remove_indices]
        self.class2id = {c: i for i, c in enumerate(self.valid_classes)}
        self.id2class = {i: c for c, i in self.class2id.items()}
        
    def load_samples(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        samples, labels = [], []
        for cls_name, texts in data.items():
            norm_cls = cls_name.strip()
            if norm_cls not in self.class2id: continue
            label_idx = self.class2id[norm_cls]
            for t in texts:
                if t.strip():
                    samples.append(t)
                    labels.append(label_idx)
        return samples, labels

def visualize_classes_profiler_features(class_matrices, model_name, output_dir):
    """
    生成学术级别、极简且体现趋势的特征对比折线图
    """
    if not class_matrices:
        return

    # 1. 学术极简排版设置
    plt.rcParams['font.family'] = 'serif'
    sns.set_theme(style="ticks", font="serif") 
    
    plt.figure(figsize=(6, 5))

    # 生成足够数量的颜色调色板
    palette = sns.color_palette("tab10", n_colors=max(10, len(class_matrices)))
    color_idx = 0

    # 2. 强制排序：让人类文本优先绘制并固定样式，其他模型跟后
    sorted_classes = sorted(list(class_matrices.keys()), key=lambda x: 0 if 'human' in str(x).lower() else 1)

    w = 0
    x_axis = None

    for cls_name in sorted_classes:
        loss_matrix = class_matrices[cls_name]
        if loss_matrix is None:
            continue
            
        # 计算窗口位置上的平均值以观察趋势
        window_means = np.mean(loss_matrix, axis=0)
        w = window_means.shape[0]
        x_axis = np.arange(w)
        
        # 3. Human 基线特殊样式处理
        if 'human' in str(cls_name).lower():
            color = 'dimgray'      # 深灰色
            linestyle = '--'       # 虚线
            marker = 's'           # 方块
            zorder = 5             # 确保画在最上层
            alpha = 1.0
        else:
            color = palette[color_idx]
            color_idx += 1
            linestyle = '-'        # 实线
            marker = 'o'           # 圆点
            zorder = 3
            alpha = 0.85
        
        plt.plot(x_axis, window_means, marker=marker, linestyle=linestyle, 
                 linewidth=2, markersize=7, alpha=alpha,
                 label=cls_name, color=color, zorder=zorder)

    if w > 0:
        # 4. 坐标轴及细节打磨
        plt.xlabel("Context Window Position", fontsize=16, fontweight='bold')
        plt.ylabel("Avg. Cross-Entropy Loss", fontsize=16, fontweight='bold')
        
        # 将 X 轴刻度设为 Pos 1, Pos 2 ...
        plt.xticks(x_axis, [f"Pos {i+1}" for i in range(w)], fontsize=12)
        plt.yticks(fontsize=12)
        
        # 去除上方和右侧的边框线 (学术图标准)
        sns.despine()

        # ⭐ 修改点 2：图例移至下方居中对齐
        plt.legend(
            title=None, 
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.2), # 放置在 X 轴下方
            frameon=False, 
            fontsize=12, 
            ncol=3,                     # 根据类别数量可微调列数
            borderaxespad=0.
        )
        
        os.makedirs(output_dir, exist_ok=True)
        # 获取干净的模型名称（去掉路径前缀）
        clean_model_name = os.path.basename(model_name)
        
        pdf_path = os.path.join(output_dir, f"loss_trend_{clean_model_name}.pdf")
        png_path = os.path.join(output_dir, f"loss_trend_{clean_model_name}.png")
        
        # bbox_inches='tight' 自动裁减四周白边，保证底部的图例被完整保存
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] 成功！学术特征趋势图已保存至: {pdf_path}")


# ==========================================
# 4. 主程序
# ==========================================
def main():
    print("=== PROFILER: 多样本平均特征提取与可视化 ===")
    
    # 1. 预处理
    with open(Config.TRAIN_DATA_PATH, 'r') as f: raw_train = json.load(f)
    processor = DataProcessor(raw_train, Config.REMOVE_LABEL_INDICES)
    
    # 2. 加载数据并根据配置抽取样本
    print(f"\n>>> 加载数据并每类抽取 {Config.NUM_SAMPLES_PER_CLASS} 条样本...")
    train_texts, train_labels = processor.load_samples(Config.TRAIN_DATA_PATH)
    
    # 初始化字典，用列表存储多条文本
    samples_per_class = {cls: [] for cls in processor.valid_classes}
    
    for text, label in zip(train_texts, train_labels):
        cls_name = processor.id2class[label]
        if len(samples_per_class[cls_name]) < Config.NUM_SAMPLES_PER_CLASS:
            samples_per_class[cls_name].append(text)
            
        if all(len(texts) == Config.NUM_SAMPLES_PER_CLASS for texts in samples_per_class.values()):
            break

    print(f"[*] 已提取类别样本，每类 {Config.NUM_SAMPLES_PER_CLASS} 条。")

    # 3. 提取特征并可视化
    for model_path in Config.PROXY_MODEL_PATHS:
        extractor = ProfilerFeatureExtractor(model_path, Config.DEVICE, Config.CONTEXT_WINDOW_SIZE)
        model_name = os.path.basename(model_path)
        
        class_avg_matrices = {}
        
        for cls_name, texts in samples_per_class.items():
            print(f"[{model_name}] 正在提取 '{cls_name}' 类别的特征矩阵...")
            
            cls_loss_matrices = []
            for text in tqdm(texts, desc=f"  Processing {cls_name}", leave=False):
                features, loss_matrix = extractor.get_features_with_matrix(text)
                if loss_matrix is not None:
                    cls_loss_matrices.append(loss_matrix)
            
            if cls_loss_matrices:
                combined_matrix = np.concatenate(cls_loss_matrices, axis=0)
                class_avg_matrices[cls_name] = combined_matrix
                print(f"    - 成功提取 {len(cls_loss_matrices)}/{Config.NUM_SAMPLES_PER_CLASS} 条有效文本")
            else:
                print(f"    - 特征提取失败 (所有文本过短或无效)")
                
        # 执行可视化绘制
        print(f"\n>>> 正在生成 {model_name} 的特征趋势对比图...")
        visualize_classes_profiler_features(class_avg_matrices, model_name, Config.OUTPUT_DIR)

        # 彻底清理显存，安全切换到下一个模型
        print(f"[*] 卸载模型 {model_name} 并清理显存...\n" + "-"*40)
        del extractor.model
        del extractor.tokenizer
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== 可视化程序执行完毕 ===")

if __name__ == "__main__":
    main()
# import torch
# import numpy as np
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# # ==========================================
# # 1. 路径配置 (请确认路径与你本地一致)
# # ==========================================
# # 特征缓存文件路径
# FEATURES_PATH = "/home/gsy/project2/MAGE/features/cmv_othermethod/profiler_llama_qwen_reddit/saved_test_features_for_tsne.pt"

# # 训练好的随机森林模型路径
# RF_MODEL_PATH = "/home/gsy/project2/MAGE/features/cmv_othermethod/profiler_llama_qwen_reddit/random_forest_model.pkl"

# # 输出的 PDF 图片路径
# OUTPUT_PDF_PATH = "/home/gsy/project2/MAGE/features/cmv_othermethod/profiler_llama_qwen_reddit/tsne_rf_predictions_final.pdf"

# # 输出的分类报告 TXT 路径
# OUTPUT_REPORT_PATH = "/home/gsy/project2/MAGE/features/cmv_othermethod/profiler_llama_qwen_reddit/rf_classification_report.txt"

# # ==========================================
# # 2. 核心加载、预测与评估逻辑
# # ==========================================
# def main():
#     # --- 1. 加载特征数据 ---
#     print(f"📥 正在加载特征缓存文件: {FEATURES_PATH}")
#     if not os.path.exists(FEATURES_PATH):
#         raise FileNotFoundError(f"找不到特征文件: {FEATURES_PATH}")
    
#     # 绕过 PyTorch 安全拦截
#     data = torch.load(FEATURES_PATH, weights_only=False)
#     X_test = data['X_test']
#     y_test_true = data['y_test']  # 真实的正确标签
#     id2class = data['id2class']
    
#     # --- 2. 加载训练好的随机森林模型 ---
#     print(f"📥 正在加载随机森林模型: {RF_MODEL_PATH}")
#     if not os.path.exists(RF_MODEL_PATH):
#         raise FileNotFoundError(f"找不到模型文件: {RF_MODEL_PATH}")
    
#     rf_model = joblib.load(RF_MODEL_PATH)
    
#     # --- 3. 在完整测试集上进行预测与评估 ---
#     print("\n🤖 正在使用随机森林模型进行全局预测...")
#     y_pred_all = rf_model.predict(X_test)
    
#     # 提取类别名称列表 (严格按照 id 0, 1, 2... 的顺序排列)
#     target_names = [id2class[i] for i in range(len(id2class))]
    
#     print("\n📊 正在生成分类评估报告...")
#     report_str = classification_report(
#         y_test_true, y_pred_all, 
#         target_names=target_names, 
#         digits=4, # 保留4位小数，符合顶会要求
#         zero_division=0
#     )
    
#     # 打印到控制台
#     print("="*60)
#     print("Random Forest Classification Report (Test Set)")
#     print("="*60)
#     print(report_str)
#     print(f"Overall Accuracy: {accuracy_score(y_test_true, y_pred_all):.4f}")
#     print("="*60)
    
#     # 将报告保存到本地 txt 文件
#     with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
#         f.write("Random Forest Classification Report (Test Set)\n")
#         f.write("="*60 + "\n")
#         f.write(report_str)
#         f.write(f"\nOverall Accuracy: {accuracy_score(y_test_true, y_pred_all):.4f}\n")
#     print(f"[*] 分类报告已安全保存至: {OUTPUT_REPORT_PATH}\n")

#     # --- 4. 为 t-SNE 画图准备数据 (控制点数) ---
#     max_samples = 11000
#     if len(y_test_true) > max_samples:
#         print(f"✂️ 数据点过多，正在随机采样 {max_samples} 个点以保证画图质量...")
#         np.random.seed(42)
#         indices = np.random.choice(len(y_test_true), max_samples, replace=False)
#         X_plot = X_test[indices]
#         y_pred_plot = y_pred_all[indices]
#     else:
#         X_plot = X_test
#         y_pred_plot = y_pred_all
    
#     # 将预测的数字标签映射回字符串
#     mapped_pred_labels = [id2class[lbl] for lbl in y_pred_plot]

#     # --- 5. 运行 t-SNE 降维 ---
#     print("⏳ 正在对特征进行 t-SNE 降维 (这可能需要几分钟)...")
#     tsne = TSNE(
#         n_components=2, 
#         perplexity=30, 
#         n_iter=1000, 
#         random_state=42, 
#         init='pca', 
#         learning_rate='auto'
#     )
#     X_embedded = tsne.fit_transform(X_plot)

#     # ==========================================
#     # 3. 论文级高质量可视化排版
#     # ==========================================
#     print("🎨 正在生成基于模型预测结果的 t-SNE 图表...")
    
#     df_plot = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'Predicted_Label': mapped_pred_labels})
    
#     # 强制固定图例顺序，保证和论文其他图一致
#     target_order = ["ctrl", "fair_wmt20", "gpt3", "grover_large", "human", "pplm_gpt2", "xlm", "xlnet_large"]
#     actual_labels = list(id2class.values())
#     hue_order = [t for t in target_order if t in actual_labels]
#     hue_order.extend([t for t in actual_labels if t not in target_order])

#     # 扩大画布
#     plt.figure(figsize=(10, 8))
    
#     # 绘制散点图 (注意这里 hue 使用的是 Predicted_Label)
#     sns.scatterplot(
#         data=df_plot, 
#         x='x', 
#         y='y', 
#         hue='Predicted_Label', 
#         hue_order=hue_order, 
#         palette='tab10', 
#         s=30,          # 点大小
#         alpha=0.8,     # 透明度
#         edgecolor='w', # 白色描边
#         linewidths=0.3 # 描边粗细
#     )
    
#     # 设置坐标轴标签
#     plt.xlabel('t-SNE Dimension 1', fontsize=16, fontweight='bold')
#     plt.ylabel('t-SNE Dimension 2', fontsize=16, fontweight='bold')
#     plt.tick_params(axis='both', which='major', labelsize=12)
    
#     # 设置图例
#     plt.legend(
#         title=None, 
#         bbox_to_anchor=(1.02, 1), 
#         loc='upper left', 
#         fontsize=14, 
#         frameon=False, 
#         markerscale=1.5 
#     )
    
#     # 保存图片
#     plt.savefig(OUTPUT_PDF_PATH, format='pdf', dpi=300, bbox_inches='tight')
#     print(f"✅ 大功告成！基于预测结果的 PDF 图表已保存至:\n{OUTPUT_PDF_PATH}")

# if __name__ == "__main__":
#     main()
# import json
# import torch
# import numpy as np
# import os
# import sys
# import scipy.stats
# import gc
# import joblib  # 新增：用于保存分类器模型
# import pandas as pd # 新增：用于绘图数据处理
# import matplotlib.pyplot as plt # 新增：绘图
# import seaborn as sns # 新增：绘图
# from itertools import combinations
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
# from sklearn.preprocessing import label_binarize
# from sklearn.impute import SimpleImputer
# from sklearn.manifold import TSNE # 新增：t-SNE降维
# from tqdm import tqdm
# import warnings

# warnings.filterwarnings('ignore')

# # ==========================================
# # 1. 全局配置 (Configuration)
# # ==========================================
# class Config:
#     # 🌟 在这里填入你的 Llama 和 Qwen 模型的实际绝对路径
#     PROXY_MODEL_PATHS = [
#         "/home/share/models/llama-7b",  
#         "/home/share/models/qwen2.5-7b-instruct"    
#     ] 
    
#     # 数据集路径
#     TRAIN_DATA_PATH = "/home/gsy/project2/TuringBench/train/data_train.json"
#     TEST_DATA_PATH = "/home/gsy/project2/TuringBench/test/dataset_test.json"
    
#     # 结果保存目录
#     OUTPUT_DIR = "/home/gsy/project2/MAGE/features/cmv_othermethod/profiler_llama_qwen_reddit"
    
#     # 特征保存文件 (用于随时重新画图)
#     SAVED_FEATURES_PATH = os.path.join(OUTPUT_DIR, "saved_test_features_for_tsne.pt")
#     # PDF 画图输出路径
#     TSNE_PDF_PATH = os.path.join(OUTPUT_DIR, "tsne_comparison_profiler.pdf")
    
#     # 移除的类别索引 (0-based)
#     REMOVE_LABEL_INDICES = [] 
    
#     # PROFILER 超参数
#     CONTEXT_WINDOW_SIZE = 4 
#     MAX_SEQ_LEN = 512       
    
#     # 训练参数
#     SAMPLES_PER_CLASS_TRAIN = 4000
#     SAMPLES_PER_CLASS_TEST = 1000 
    
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ==========================================
# # 2. PROFILER 特征提取核心 (保持不变)
# # ==========================================
# class ProfilerFeatureExtractor:
#     def __init__(self, model_path, device, window_size=4):
#         print(f"\n[{device.upper()}] 正在加载代理模型: {model_path} ...")
#         self.device = device
#         self.window_size = window_size
#         self.half_window = window_size // 2
        
#         self.expected_dim = (self.window_size * 18) + (self.window_size * (self.window_size - 1)) // 2
        
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
#             if self.tokenizer.pad_token is None:
#                 if self.tokenizer.eos_token is not None:
#                     self.tokenizer.pad_token = self.tokenizer.eos_token
#                 else:
#                     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 torch_dtype=torch.float16,
#                 device_map={"": self.device},
#                 trust_remote_code=True
#             )
#             self.model.eval()
            
#             if self.tokenizer.pad_token == '[PAD]':
#                 self.model.resize_token_embeddings(len(self.tokenizer))
                
#         except Exception as e:
#             print(f"模型加载失败: {e}")
#             sys.exit(1)
            
#     def _compute_context_losses(self, input_ids, logits):
#         ids = input_ids[0]      
#         logs = logits[0][:-1]   
        
#         seq_len = logs.size(0)
#         valid_start = self.half_window
#         valid_end = seq_len - self.half_window
        
#         if valid_start >= valid_end:
#             return None 
            
#         context_losses = []
#         loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
#         valid_logits = logs[valid_start:valid_end].to(torch.float32) 
        
#         for j in range(-self.half_window, self.half_window + 1):
#             if j == 0 and self.window_size % 2 == 0: 
#                 continue
                
#             target_indices = torch.arange(valid_start, valid_end) + j + 1 
#             target_tokens = ids[target_indices.long()].to(self.device)
            
#             loss = loss_fct(valid_logits, target_tokens) 
#             context_losses.append(loss.cpu().numpy())
            
#             if len(context_losses) >= self.window_size:
#                 break
                
#         if not context_losses:
#             return None
            
#         return np.stack(context_losses, axis=1) 

#     def _extract_independent_patterns(self, loss_matrix):
#         features = []
#         N, W = loss_matrix.shape
        
#         for w in range(W):
#             col = loss_matrix[:, w]
#             stats = [np.mean(col), np.std(col), np.min(col), np.max(col), np.median(col), np.var(col)]
#             features.extend(stats)
            
#             if len(col) > 1:
#                 diff1 = np.diff(col)
#                 stats_d1 = [np.mean(diff1), np.std(diff1), np.min(diff1), np.max(diff1), np.median(diff1), np.var(diff1)]
#                 features.extend(stats_d1)
#             else:
#                 features.extend([0]*6)
                
#             if len(col) > 2:
#                 diff2 = np.diff(col, n=2) / 2.0
#                 stats_d2 = [np.mean(diff2), np.std(diff2), np.min(diff2), np.max(diff2), np.median(diff2), np.var(diff2)]
#                 features.extend(stats_d2)
#             else:
#                 features.extend([0]*6)
#         return np.array(features)

#     def _extract_correlated_patterns(self, loss_matrix):
#         features = []
#         probs_matrix = scipy.special.softmax(loss_matrix, axis=0)
        
#         for i, j in combinations(range(loss_matrix.shape[1]), 2):
#             p = probs_matrix[:, i] + 1e-9
#             q = probs_matrix[:, j] + 1e-9
#             kl_sym = np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))
#             features.append(kl_sym)
            
#         return np.array(features)

#     def get_features(self, text):
#         if not text or not text.strip(): return None
#         inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=Config.MAX_SEQ_LEN, padding=False).to(self.device)
#         if inputs.input_ids.size(1) < self.window_size + 5: return None
            
#         with torch.no_grad():
#             outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            
#         loss_matrix = self._compute_context_losses(inputs.input_ids, outputs.logits)
#         if loss_matrix is None: return None
            
#         feat_ip = self._extract_independent_patterns(loss_matrix)
#         feat_cp = self._extract_correlated_patterns(loss_matrix)
        
#         final_feats = np.concatenate([feat_ip, feat_cp])
#         final_feats = np.nan_to_num(final_feats, nan=0.0, posinf=0.0, neginf=0.0)
#         return final_feats

# # ==========================================
# # 3. 数据处理与评估保存逻辑 
# # ==========================================
# class DataProcessor:
#     def __init__(self, raw_data, remove_indices):
#         all_classes = sorted(list(raw_data.keys()))
#         print(f"[*] 原始类别: {all_classes}")
#         self.valid_classes = [c for i, c in enumerate(all_classes) if i not in remove_indices]
#         print(f"[*] 保留类别: {self.valid_classes}")
#         self.class2id = {c: i for i, c in enumerate(self.valid_classes)}
#         self.id2class = {i: c for c, i in self.class2id.items()}
        
#     def load_samples(self, file_path, max_per_class=None):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         samples, labels = [], []
#         for cls_name, texts in data.items():
#             norm_cls = cls_name.strip()
#             if norm_cls not in self.class2id: continue
#             label_idx = self.class2id[norm_cls]
#             if max_per_class: texts = texts[:max_per_class]
#             for t in texts:
#                 if t.strip():
#                     samples.append(t)
#                     labels.append(label_idx)
#         return samples, labels

# def save_evaluation_results(y_true, y_pred, y_prob, label_map, output_dir):
#     """保存评估结果"""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     target_names = [label_map[i] for i in range(len(label_map))]
    
#     report_str = classification_report(
#         y_true, y_pred, target_names=target_names, digits=4, zero_division=0
#     )
#     txt_path = os.path.join(output_dir, "classification_report.txt")
#     with open(txt_path, "w", encoding="utf-8") as f:
#         f.write(report_str)
#     print(f"[*] 已保存分类报告: {txt_path}")
    
#     acc = accuracy_score(y_true, y_pred)
#     classes = list(range(len(label_map)))
#     y_true_bin = label_binarize(y_true, classes=classes)
    
#     if len(classes) == 2 and y_true_bin.shape[1] == 1:
#         y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        
#     individual_aucs = {}
#     auc_scores = []
    
#     for i, class_name in enumerate(target_names):
#         try:
#             if np.sum(y_true_bin[:, i]) > 0:
#                 auc_val = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
#             else:
#                 auc_val = 0.0
#         except:
#             auc_val = 0.0
        
#         auc_scores.append(auc_val)
#         individual_aucs[str(i)] = {"class_name": class_name, "auc": float(auc_val)}
        
#     mean_auc = np.mean(auc_scores)
#     json_output = {
#         "accuracy": float(acc),
#         "mean_auc": float(mean_auc),
#         "individual_aucs": individual_aucs
#     }
    
#     json_path = os.path.join(output_dir, "test_metrics.json")
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(json_output, f, indent=4)
#     print(f"[*] 已保存 JSON 指标: {json_path}")


# # ==========================================
# # 4. 论文级 t-SNE 绘图函数
# # ==========================================
# def draw_paper_tsne(features_path, output_pdf_path):
#     print("\n🎨 正在基于保存的特征生成学术 t-SNE 散点图...")
#     if not os.path.exists(features_path):
#         print(f"[!] 找不到特征文件 {features_path}，跳过绘图。")
#         return
        
#     # 加载已保存的数据
#     data = torch.load(features_path, weights_only=False)
#     X = data['X_test']
#     y = data['y_test']
#     id2class = data['id2class']
    
#     # 截断数据以防绘图过慢，最多画 11000 个点以保证视觉效果与之前一致
#     max_samples = 11000
#     if len(y) > max_samples:
#         indices = np.random.choice(len(y), max_samples, replace=False)
#         X = X[indices]
#         y = y[indices]
        
#     print("⏳ 正在计算 t-SNE 降维 (可能需要几分钟)...")
#     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
#     X_embedded = tsne.fit_transform(X)
    
#     # 构建绘图 DataFrame
#     mapped_labels = [id2class[lbl] for lbl in y]
#     df_plot = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'Label': mapped_labels})
    
#     # 与之前的目标顺序对齐 (如果数据集中包含这些类的话)
#     target_order = ["ctrl", "fair_wmt20", "gpt3", "grover_large", "human", "pplm_gpt2", "xlm", "xlnet_large"]
#     actual_labels = list(id2class.values())
#     hue_order = [t for t in target_order if t in actual_labels]
#     # 把不在 target_order 中的其他类追加在后面
#     hue_order.extend([t for t in actual_labels if t not in target_order])
    
#     # --- 画布与视觉参数配置 (与前置代码完全一致) ---
#     plt.figure(figsize=(10, 8))
    
#     sns.scatterplot(
#         data=df_plot, x='x', y='y', hue='Label', 
#         hue_order=hue_order, 
#         palette='tab10', 
#         s=30,          # 小点，制造分散感
#         alpha=0.8,     # 高透明度
#         edgecolor='w', # 保留白色描边提供质感
#         linewidths=0.3 # 细描边
#     )
    
#     plt.xlabel('t-SNE Dimension 1', fontsize=16, fontweight='bold')
#     plt.ylabel('t-SNE Dimension 2', fontsize=16, fontweight='bold')
    
#     # 保留刻度并放大刻度字体
#     plt.tick_params(axis='both', which='major', labelsize=12)
    
#     # 图例去框线、放大字体、去除标题
#     plt.legend(
#         title=None, 
#         bbox_to_anchor=(1.02, 1), 
#         loc='upper left', 
#         fontsize=14, 
#         frameon=False, 
#         markerscale=1.5 
#     )
    
#     plt.savefig(output_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
#     print(f"✅ 学术分散图表已成功保存至: {output_pdf_path}")

# # ==========================================
# # 5. 主程序
# # ==========================================
# def main():
#     print("=== PROFILER 复现: Llama & Qwen 多模型特征融合 ===")
    
#     if not os.path.exists(Config.OUTPUT_DIR):
#         os.makedirs(Config.OUTPUT_DIR)
    
#     # 1. 预处理
#     with open(Config.TRAIN_DATA_PATH, 'r') as f: raw_train = json.load(f)
#     processor = DataProcessor(raw_train, Config.REMOVE_LABEL_INDICES)
    
#     # 2. 加载数据
#     print("\n>>> 加载数据...")
#     train_texts, train_labels = processor.load_samples(Config.TRAIN_DATA_PATH, Config.SAMPLES_PER_CLASS_TRAIN)
#     test_texts, test_labels = processor.load_samples(Config.TEST_DATA_PATH, Config.SAMPLES_PER_CLASS_TEST)
    
#     if not train_texts or not test_texts:
#         print("[!] 数据不足，退出")
#         return

#     # 3. 串行特征提取
#     all_train_feats = []
#     all_test_feats = []

#     for model_path in Config.PROXY_MODEL_PATHS:
#         extractor = ProfilerFeatureExtractor(model_path, Config.DEVICE, Config.CONTEXT_WINDOW_SIZE)
#         model_name = os.path.basename(model_path)
        
#         # 提取训练集
#         model_train_f = []
#         for t in tqdm(train_texts, desc=f"Extracting Train - {model_name}"):
#             f = extractor.get_features(t)
#             if f is None:
#                 f = np.full(extractor.expected_dim, np.nan)
#             model_train_f.append(f)
#         all_train_feats.append(np.array(model_train_f))
        
#         # 提取测试集
#         model_test_f = []
#         for t in tqdm(test_texts, desc=f"Extracting Test - {model_name}"):
#             f = extractor.get_features(t)
#             if f is None:
#                 f = np.full(extractor.expected_dim, np.nan)
#             model_test_f.append(f)
#         all_test_feats.append(np.array(model_test_f))

#         print(f"\n[*] 卸载模型 {model_name} 并清理显存...")
#         del extractor.model
#         del extractor.tokenizer
#         del extractor
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     # 4. 拼接特征
#     print("\n>>> 融合 Llama 和 Qwen 特征...")
#     X_train = np.concatenate(all_train_feats, axis=1)
#     X_test  = np.concatenate(all_test_feats, axis=1)  
    
#     y_train = np.array(train_labels)
#     y_test  = np.array(test_labels)
    
#     # 使用均值填充因文本过短导致的局部 NaN
#     print(">>> 处理缺失值 (Imputation)...")
#     imputer = SimpleImputer(strategy='mean')
#     X_train = imputer.fit_transform(X_train)
#     X_test = imputer.transform(X_test)
    
#     # 🌟 修改点 1：持久化保存特征（方便后续直接拿来画图）
#     print("\n>>> 持久化保存 Test 阶段的特征以供随时画图...")
#     save_dict = {
#         'X_test': X_test,
#         'y_test': y_test,
#         'id2class': processor.id2class
#     }
#     torch.save(save_dict, Config.SAVED_FEATURES_PATH)
#     print(f"[*] 特征已保存至: {Config.SAVED_FEATURES_PATH}")
    
#     # 5. 训练与评估
#     print(f"\n>>> 训练分类器 (Random Forest, 融合后特征维度={X_train.shape[1]})...")
#     clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#     clf.fit(X_train, y_train)
    
#     # 🌟 修改点 2：保存训练好的随机森林模型 (以备不时之需)
#     model_save_path = os.path.join(Config.OUTPUT_DIR, "random_forest_model.pkl")
#     joblib.dump(clf, model_save_path)
#     print(f"[*] 模型已保存至: {model_save_path}")
    
#     print("\n>>> 生成预测结果...")
#     y_pred = clf.predict(X_test)
#     y_prob = clf.predict_proba(X_test)
    
#     # 6. 保存结果
#     save_evaluation_results(y_test, y_pred, y_prob, processor.id2class, Config.OUTPUT_DIR)
    
#     # 🌟 修改点 3：自动调用画图模块
#     draw_paper_tsne(Config.SAVED_FEATURES_PATH, Config.TSNE_PDF_PATH)
    
#     print("\n=== 程序执行完毕 ===")

# if __name__ == "__main__":
#     main()