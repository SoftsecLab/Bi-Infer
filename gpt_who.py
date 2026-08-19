import json
import torch
import numpy as np
import os
import sys
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize 
from sklearn.impute import SimpleImputer 
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 用户配置区域 (请修改此处)
# ==========================================
MODEL_PATHS = [
    r"/home/share/models/llama-7b",              # 替换为实际的 Llama 路径
    r"/home/share/models/qwen2.5-7b-instruct"    # 替换为实际的 Qwen 路径
] 

# 🌟 基础路径统一配置
INPUT_BASE_DIR = "/home/gsy/project2/TuringBench/two_class"
OUTPUT_BASE_DIR = "/home/gsy/project2/MAGE/features/squad_othermethod"

# 🌟 只需要在这里填入你要跑的子文件夹名称即可
TASK_NAMES = [
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
    # 你可以随时在这里添加或注释掉不需要的任务
]

# 自动生成所有数据集的绝对路径配置，无需再手动写长串路径
DATASETS_CONFIG = []
for task in TASK_NAMES:
    DATASETS_CONFIG.append({
        "task_name": task,
        "train_file": os.path.join(INPUT_BASE_DIR, task, "train_data.json"),
        "test_file": os.path.join(INPUT_BASE_DIR, task, "test_data.json"),
        "output_dir": os.path.join(OUTPUT_BASE_DIR, f"llama_qwen_{task}")
    })

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 样本数量控制 (每类最大样本数)
TRAIN_SAMPLES_PER_CLASS = 4000  
TEST_SAMPLES_PER_CLASS = 1000   

# 需要移除的类别索引 (索引从0开始)
REMOVE_CLASS_IDXS = []

# ==========================================
# 2. 核心特征提取器 (Surprisal / LLM-who)
# ==========================================
class SurprisalFeatureExtractor:
    def __init__(self, model_path, device):
        print(f"\n[{device.upper()}] 正在加载模型: {os.path.basename(model_path)} ...")
        
        if not os.path.exists(model_path):
            print(f"错误: 路径不存在 -> {model_path}")
            sys.exit(1)

        self.device = device
        self.window_size = 20  # 窗口大小
        self.feature_dim = 4 + self.window_size * 2 # 固定特征维度: 44

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # 兼容不同模型的 pad_token 缺失问题
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
            # 统一使用 float16 加载，兼容 Llama 和 Qwen 降低显存占用
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map={"": self.device},
                trust_remote_code=True
            )
            self.model.eval()
            
            # 如果新增了特殊 token，调整 embedding 大小
            if self.tokenizer.pad_token == '[PAD]':
                self.model.resize_token_embeddings(len(self.tokenizer))
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)

    def calculate_surprisals_gpu(self, text):
        """在 GPU 上计算 Surprisal"""
        encodings = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
        input_ids = encodings.input_ids.to(self.device)
        
        if input_ids.size(1) < self.window_size + 1:
            return None

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # 防止 float16 精度下溢，转为 float32 计算 loss
        valid_logits = shift_logits.view(-1, shift_logits.size(-1)).to(torch.float32)
        surprisals_tensor = loss_fct(valid_logits, shift_labels.view(-1))
        
        return surprisals_tensor.cpu().numpy()

    def extract_uid_features(self, surprisals):
        """提取 44 维特征"""
        if surprisals is None: return None

        mean_s = np.mean(surprisals)
        var_s = np.var(surprisals)
        diffs = np.diff(surprisals)
        mean_diff = np.mean(np.abs(diffs))
        mean_sq_diff = np.mean(diffs ** 2)
        
        num_tokens = len(surprisals)
        spans = [surprisals[i : i + self.window_size] for i in range(num_tokens - self.window_size + 1)]
        span_vars = [np.var(s) for s in spans]

        if not span_vars: return None

        min_idx = np.argmin(span_vars)
        max_idx = np.argmax(span_vars)

        return np.concatenate([[mean_s, var_s, mean_diff, mean_sq_diff], spans[min_idx], spans[max_idx]])

# ==========================================
# 3. 数据加载辅助函数
# ==========================================
def load_raw_dataset(file_path, label_map=None, is_train=True, max_samples_per_class=None):
    if not os.path.exists(file_path):
        return [], [], label_map
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if is_train:
        original_classes = sorted(list(data.keys()))
        filtered_classes = [cls for idx, cls in enumerate(original_classes) if idx not in REMOVE_CLASS_IDXS]
        current_label_map = {cls: i for i, cls in enumerate(filtered_classes)}
    else:
        current_label_map = label_map

    texts_list, labels_list = [], []
    for label_name, texts in data.items():
        if label_name not in current_label_map: continue
        
        if max_samples_per_class is not None:
            texts = texts[:max_samples_per_class]
            
        label_id = current_label_map[label_name]
        for text in texts:
            if text.strip():  
                texts_list.append(text)
                labels_list.append(label_id)

    return texts_list, labels_list, current_label_map

# ==========================================
# 4. 主流程
# ==========================================
def main():
    print(f"=== LLM-who 多模型多任务特征融合提取工具 (运行设备: {DEVICE}) ===")
    
    # 🌟 Phase 1: 将所有数据集的任务提前加载进内存
    print("\n[Phase 1] 预加载所有数据集文本...")
    tasks_data = {}
    for config in DATASETS_CONFIG:
        task_name = config["task_name"]
        print(f"  -> 正在加载任务: {task_name}")
        
        if not os.path.exists(config["output_dir"]):
            os.makedirs(config["output_dir"], exist_ok=True)
            
        train_texts, train_labels, label_map = load_raw_dataset(
            config["train_file"], is_train=True, max_samples_per_class=TRAIN_SAMPLES_PER_CLASS
        )
        test_texts, test_labels, _ = load_raw_dataset(
            config["test_file"], label_map=label_map, is_train=False, max_samples_per_class=TEST_SAMPLES_PER_CLASS
        )
        
        if not train_texts or not test_texts:
            print(f"  ⚠️ 警告: 任务 {task_name} 的 JSON 文件不存在或为空，已自动跳过该任务。")
            continue
            
        tasks_data[task_name] = {
            "config": config,
            "train_texts": train_texts, "train_labels": train_labels,
            "test_texts": test_texts, "test_labels": test_labels,
            "label_map": label_map,
            "all_train_feats": [], # 用于收集不同模型的训练特征
            "all_test_feats": []   # 用于收集不同模型的测试特征
        }

    if not tasks_data:
        print("错误: 所有任务加载失败或为空，退出程序。")
        return

    # 🌟 Phase 2: 模型主导的特征提取循环 (最外层是模型，内层是数据集)
    print("\n[Phase 2] 开始模型特征提取...")
    for model_path in MODEL_PATHS:
        extractor = SurprisalFeatureExtractor(model_path, DEVICE)
        model_name = os.path.basename(model_path)
        
        # 依次对所有加载的数据集进行该模型的特征提取
        for task_name, task in tasks_data.items():
            # 提取 Train
            model_train_f = []
            for text in tqdm(task["train_texts"], desc=f"[{task_name}] Train - {model_name}"):
                surprisals = extractor.calculate_surprisals_gpu(text)
                feats = extractor.extract_uid_features(surprisals)
                if feats is None:
                    feats = np.full(extractor.feature_dim, np.nan)
                model_train_f.append(feats)
            task["all_train_feats"].append(np.array(model_train_f))
            
            # 提取 Test
            model_test_f = []
            for text in tqdm(task["test_texts"], desc=f"[{task_name}] Test - {model_name}"):
                surprisals = extractor.calculate_surprisals_gpu(text)
                feats = extractor.extract_uid_features(surprisals)
                if feats is None:
                    feats = np.full(extractor.feature_dim, np.nan)
                model_test_f.append(feats)
            task["all_test_feats"].append(np.array(model_test_f))

        # 当前模型在所有数据集上提取完毕，卸载以释放显存给下一个模型
        print(f"[*] 卸载模型 {model_name} 并清理显存...")
        del extractor.model
        del extractor.tokenizer
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 🌟 Phase 3: 针对每个数据集分别训练并评估
    print("\n[Phase 3] 训练分类器并保存结果...")
    for task_name, task in tasks_data.items():
        print(f"\n" + "="*50)
        print(f" 开始处理任务: {task_name} ")
        print("="*50)
        
        config = task["config"]
        label_map = task["label_map"]
        
        # 融合该任务所有模型的特征 (axis=1 拼接)
        X_train = np.concatenate(task["all_train_feats"], axis=1)  
        X_test  = np.concatenate(task["all_test_feats"], axis=1)
        y_train = np.array(task["train_labels"])
        y_test  = np.array(task["test_labels"])

        print(f"[*] 处理缺失值 (Imputation)...")
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test  = imputer.transform(X_test)

        print(f"[*] 训练分类器 (Logistic Regression, 融合特征维度={X_train.shape[1]})...")
        clf = LogisticRegression(max_iter=10000, multi_class='multinomial', solver='lbfgs')
        clf.fit(X_train, y_train)

        # 评估
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test) 

        sorted_items = sorted(label_map.items(), key=lambda x: x[1])
        target_names = [item[0] for item in sorted_items]
        target_ids = [item[1] for item in sorted_items]
        num_classes = len(target_names)

        acc = accuracy_score(y_test, y_pred)
        report_str = classification_report(
            y_test, y_pred, 
            target_names=target_names, digits=4, zero_division=0
        )
        
        # 1. 写入分类报告 TXT
        txt_filename = os.path.join(config["output_dir"], "classification_report.txt")
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(report_str)
        print(f"✅ 分类报告已保存: {txt_filename}")
        print(report_str)

        # 2. 计算 AUC 并写入 JSON
        y_test_bin = label_binarize(y_test, classes=target_ids)
        if num_classes == 2 and y_test_bin.shape[1] == 1:
            y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))
        
        individual_aucs = {}
        auc_scores = []
        for i, class_name in enumerate(target_names):
            try:
                if np.sum(y_test_bin[:, i]) > 0: 
                    auc = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
                else:
                    auc = 0.0
            except Exception:
                auc = 0.0
            
            auc_scores.append(auc)
            individual_aucs[str(i)] = {"class_name": class_name, "auc": auc}

        json_output = {
            "accuracy": acc,
            "mean_auc": np.mean(auc_scores),
            "individual_aucs": individual_aucs
        }

        json_filename = os.path.join(config["output_dir"], "test_metrics.json")
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=4)
        print(f"✅ 评估指标 JSON 已保存: {json_filename}")

if __name__ == "__main__":
    main()