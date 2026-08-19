import os
import json
import gc
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, 
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置部分（多数据集自动化配置）
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
        # 可继续添加...
    ]
    
    # 要删除的类别索引
    REMOVE_LABEL_INDICES = [] 
    
    # 每类的样本条数限制（-1表示使用全部）
    PER_CLASS_TRAIN_SAMPLES = 3000 
    PER_CLASS_TEST_SAMPLES = 1000    
    
    SEQ_LEN = 512  # 截断长度，Sniffer不需要太长即可提取统计特征
    
    # 指定GPU编号
    GPU_ID = 1 
    DEVICE = f"cuda:{GPU_ID}" if (torch.cuda.is_available() and GPU_ID in [0, 1]) else "cpu"

# ==========================================
# 2. 工具函数：修复JSON序列化问题
# ==========================================
def convert_to_python_type(obj):
    """将NumPy类型、特殊布尔类型转换为Python原生类型"""
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, (list, tuple)): return [convert_to_python_type(item) for item in obj]
    elif isinstance(obj, dict): return {k: convert_to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, bool): return obj
    else: return obj

# ==========================================
# 3. 工具函数：按类别抽取指定条数的样本
# ==========================================
def sample_per_class(samples, labels, per_class_samples, data_type="train"):
    class2samples = {}
    class2labels = {}
    for s, l in zip(samples, labels):
        if l not in class2samples:
            class2samples[l] = []
            class2labels[l] = []
        class2samples[l].append(s)
        class2labels[l].append(l)
    
    new_samples = []
    new_labels = []
    for cls in class2samples:
        cls_samples = class2samples[cls]
        cls_labels = class2labels[cls]
        cls_count = len(cls_samples)
        
        if per_class_samples == -1 or cls_count <= per_class_samples:
            take = cls_count
        else:
            take = per_class_samples
        
        new_samples.extend(cls_samples[:take])
        new_labels.extend(cls_labels[:take])
    
    return new_samples, new_labels

def check_class_distribution(samples, labels):
    unique_classes = list(set(labels))
    num_classes = len(unique_classes)
    return unique_classes, num_classes

# ==========================================
# 4. 标签自动处理系统
# ==========================================
class LabelProcessor:
    def __init__(self, raw_data: dict, remove_indices: list):
        self.raw_classes = [k.strip().lower() for k in raw_data.keys()]
        self.raw_classes = list(dict.fromkeys(self.raw_classes)) 
        
        self.valid_classes = self.raw_classes.copy()
        for idx in sorted(remove_indices, reverse=True):
            if 0 <= idx < len(self.valid_classes):
                self.valid_classes.pop(idx)
        
        self.class2id = {cls: i for i, cls in enumerate(self.valid_classes)}
        self.id2class = {i: cls for cls, i in self.class2id.items()}

    def get_valid_class_names(self):
        return self.valid_classes

    def encode_label(self, class_name: str):
        class_name = class_name.strip().lower()
        return self.class2id.get(class_name, -1) 

    def get_num_classes(self):
        return len(self.valid_classes)

# ==========================================
# 5. 基础特征提取器 (计算 Perplexity)
# ==========================================
class PerplexityExtractor:
    def __init__(self, model_configs):
        self.model_configs = model_configs
        self.device = Config.DEVICE

    def _clean_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_token_log_probs(self, model, tokenizer, text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=Config.SEQ_LEN).to(self.device)
        input_ids = inputs["input_ids"]
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits[0, :-1, :]
        labels = input_ids[0, 1:]
        
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs_all, dim=1, index=labels.unsqueeze(1)).squeeze(1)
        token_log_probs = torch.cat([torch.tensor([0.0], device=self.device), token_log_probs])
        
        offset_mapping = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=Config.SEQ_LEN, return_tensors="pt")["offset_mapping"][0]
        
        return token_log_probs.cpu().numpy(), offset_mapping.cpu().numpy()

    def extract_aligned_ppl(self, datasets_dict):
        print("\n>>> [Phase 2.1] Pre-tokenizing base words for all tasks...")
        base_words_map = {}
        for key, samples in datasets_dict.items():
            base_words_map[key] = [t.split()[:Config.SEQ_LEN] for t in samples]
        
        raw_ppl_data = {k: [[] for _ in range(len(v))] for k, v in datasets_dict.items()}
        
        print("\n>>> [Phase 2.2] Serial Feature Extraction across all tasks...")
        for model_idx, config in enumerate(self.model_configs):
            m_type = config['type']
            m_path = config['path']
            print(f"\n[Model {model_idx+1}/{len(self.model_configs)}] Loading {m_type}...")
            
            try:
                trust_remote = True if m_type in ['qwen', 'falcon'] else False
                tokenizer = AutoTokenizer.from_pretrained(m_path, trust_remote_code=trust_remote)
                model = AutoModelForCausalLM.from_pretrained(
                    m_path,
                    device_map={"": self.device}, 
                    torch_dtype=torch.float16,
                    trust_remote_code=trust_remote
                )
                model.eval()
                
                for data_name, samples in datasets_dict.items():
                    target_words_list = base_words_map[data_name]
                    target_container = raw_ppl_data[data_name]
                    
                    for i, text in enumerate(tqdm(samples, desc=f"   Inferencing {data_name}", leave=False)):
                        if not text.strip():
                            if len(target_container[i]) <= model_idx:
                                target_container[i].append(np.array([]))
                            continue
                            
                        try:
                            token_log_probs, offsets = self._get_token_log_probs(model, tokenizer, text)
                            words = target_words_list[i]
                            aligned_ppl = []
                            curr = 0
                            
                            for w in words:
                                start = text.find(w, curr)
                                if start == -1: start = curr
                                end = start + len(w)
                                curr = end
                                
                                relevant = []
                                for t_i, (ts, te) in enumerate(offsets):
                                    if max(start, ts) < min(end, te):
                                        relevant.append(token_log_probs[t_i])
                                
                                score = np.mean(relevant) if relevant else 0.0
                                aligned_ppl.append(score)
                            
                            if len(target_container[i]) <= model_idx:
                                target_container[i].append(np.array(aligned_ppl))
                            else:
                                target_container[i][model_idx] = np.array(aligned_ppl)
                                
                        except Exception:
                            if len(target_container[i]) <= model_idx:
                                target_container[i].append(np.array([]))
                            continue
                
                del model
                del tokenizer
                self._clean_memory()
                
            except Exception as e:
                print(f"!!! Error with {m_type}: {e}")
                self._clean_memory()
                continue
        
        return raw_ppl_data

# ==========================================
# 6. 特征工程 (Sniffer 核心逻辑)
# ==========================================
class SnifferFeatureEngineer:
    def __init__(self, num_models):
        self.num_models = num_models
        
    def compute_features(self, sample_ppl_list):
        features = []
        for ppl_arr in sample_ppl_list:
            if len(ppl_arr) == 0: features.append(0.0)
            else: features.append(np.mean(ppl_arr))
        
        for i in range(self.num_models):
            for j in range(i + 1, self.num_models):
                ppl_i = sample_ppl_list[i]
                ppl_j = sample_ppl_list[j]
                
                min_len = min(len(ppl_i), len(ppl_j))
                if min_len < 2:
                    features.extend([0.0, 0.0, 0.0])
                    continue
                
                p_i = ppl_i[:min_len]
                p_j = ppl_j[:min_len]
                
                pct = np.mean(p_i > p_j)
                features.append(pct)
                
                pearson_corr, _ = pearsonr(p_i, p_j)
                features.append(pearson_corr if not np.isnan(pearson_corr) else 0.0)
                
                spearman_corr, _ = spearmanr(p_i, p_j)
                features.append(spearman_corr if not np.isnan(spearman_corr) else 0.0)
        
        return np.array(features, dtype=np.float32)

    def process_dataset(self, raw_ppl_data):
        X, valid_indices = [], []
        for idx, sample_ppls in enumerate(raw_ppl_data):
            if len(sample_ppls) != self.num_models: continue
            feat_vec = self.compute_features(sample_ppls)
            X.append(feat_vec)
            valid_indices.append(idx)
        return np.array(X), valid_indices

# ==========================================
# 7. 结果保存与指标计算工具
# ==========================================
def save_predictions(test_samples, y_test, y_pred, y_test_names, y_pred_names, output_dir):
    save_path = os.path.join(output_dir, "predictions.json")
    results = []
    for i in range(len(test_samples)):
        is_correct = bool(y_test[i] == y_pred[i])
        result = {
            "sample_text": test_samples[i],
            "true_label_id": convert_to_python_type(y_test[i]),
            "pred_label_id": convert_to_python_type(y_pred[i]),
            "true_label_name": y_test_names[i],
            "pred_label_name": y_pred_names[i],
            "is_correct": is_correct
        }
        results.append(convert_to_python_type(result))
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return save_path

def calculate_auc_metrics(y_test, y_pred, y_pred_proba, label_processor):
    accuracy = accuracy_score(y_test, y_pred)
    num_classes = label_processor.get_num_classes()
    class_names = label_processor.get_valid_class_names()
    
    y_test_binarized = label_binarize(y_test, classes=range(num_classes))
    if num_classes == 2 and y_test_binarized.shape[1] == 1:
        y_test_binarized = np.hstack((1 - y_test_binarized, y_test_binarized))
        
    individual_aucs = {}
    for cls_idx in range(num_classes):
        try:
            if np.sum(y_test_binarized[:, cls_idx]) > 0:
                auc = roc_auc_score(y_test_binarized[:, cls_idx], y_pred_proba[:, cls_idx])
            else:
                auc = 0.0
        except:
            auc = 0.0
        individual_aucs[str(cls_idx)] = {"class_name": class_names[cls_idx], "auc": round(auc, 4)}
        
    mean_auc = np.mean([v["auc"] for v in individual_aucs.values()])
    auc_metrics = {
        "accuracy": round(accuracy, 4), "mean_auc": round(mean_auc, 4), "individual_aucs": individual_aucs
    }
    return convert_to_python_type(auc_metrics)

def generate_class_report_txt(y_test, y_pred, label_processor):
    class_names = label_processor.get_valid_class_names()
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    txt_content = []
    header = f"{'':<20} {'precision':<10} {'recall':<10} {'f1-score':<10} {'support':<10}"
    txt_content.append(header)
    txt_content.append("-" * len(header)) 
    
    for cls_name in class_names:
        metrics = report_dict[cls_name]
        txt_content.append(f"{cls_name:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} {metrics['support']:<10}")
    
    txt_content.append("")
    txt_content.append(f"{'accuracy':<20} {'':<10} {'':<10} {report_dict['accuracy']:<10.4f} {report_dict['macro avg']['support']:<10}")
    macro_avg = report_dict["macro avg"]
    txt_content.append(f"{'macro avg':<20} {macro_avg['precision']:<10.4f} {macro_avg['recall']:<10.4f} {macro_avg['f1-score']:<10.4f} {macro_avg['support']:<10}")
    weighted_avg = report_dict["weighted avg"]
    txt_content.append(f"{'weighted avg':<20} {weighted_avg['precision']:<10.4f} {weighted_avg['recall']:<10.4f} {weighted_avg['f1-score']:<10.4f} {weighted_avg['support']:<10}")
    
    return "\n".join(txt_content)

def save_auc_metrics(auc_metrics, output_dir):
    save_path = os.path.join(output_dir, "auc_metrics.json")
    with open(save_path, 'w', encoding='utf-8') as f: json.dump(auc_metrics, f, ensure_ascii=False, indent=4)
    return save_path

def save_class_report_txt(class_report_txt, output_dir):
    save_path = os.path.join(output_dir, "classification_report.txt")
    with open(save_path, 'w', encoding='utf-8') as f: f.write(class_report_txt)
    return save_path

# ==========================================
# 8. 数据加载工具
# ==========================================
def load_raw_data_from_json(path):
    if not os.path.exists(path): return {}
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def load_filtered_data_from_json(path, valid_classes):
    raw_data = load_raw_data_from_json(path)
    samples, raw_labels = [], []
    valid_classes_norm = [cls.strip().lower() for cls in valid_classes]
    for cls_name, texts in raw_data.items():
        cls_name_norm = cls_name.strip().lower()
        if cls_name_norm not in valid_classes_norm: continue
        for t in texts:
            if t.strip():
                samples.append(t)
                raw_labels.append(cls_name_norm)
    return samples, raw_labels

# ==========================================
# 9. 主流程
# ==========================================
def main():
    if torch.cuda.is_available() and Config.GPU_ID in [0, 1]:
        torch.cuda.set_device(Config.GPU_ID)
        print(f"\n[*] 使用设备: cuda:{Config.GPU_ID} ({torch.cuda.get_device_name(Config.GPU_ID)})")
    else:
        print(f"\n[*] 使用设备: {Config.DEVICE}")

    # --- Phase 1: 加载所有任务的数据 ---
    print("\n>>> [Phase 1] Pre-loading and pre-processing all tasks...")
    all_tasks_meta = {}
    datasets_dict = {} # 提供给提取器的巨型字典
    
    for task_name in Config.TASK_NAMES:
        train_path = os.path.join(Config.INPUT_BASE_DIR, task_name, "train_data.json")
        test_path = os.path.join(Config.INPUT_BASE_DIR, task_name, "test_data.json")
        output_dir = os.path.join(Config.OUTPUT_BASE_DIR, f"sniffer_{task_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        train_raw_data = load_raw_data_from_json(train_path)
        if not train_raw_data:
            print(f"[!] 跳过任务 {task_name}: 未找到训练数据。")
            continue
            
        label_processor = LabelProcessor(train_raw_data, Config.REMOVE_LABEL_INDICES)
        valid_classes = label_processor.get_valid_class_names()
        
        train_samples, train_labels_raw = load_filtered_data_from_json(train_path, valid_classes)
        test_samples, test_labels_raw = load_filtered_data_from_json(test_path, valid_classes)
        
        train_labels = [label_processor.encode_label(cls) for cls in train_labels_raw]
        test_labels = [label_processor.encode_label(cls) for cls in test_labels_raw]
        
        train_samples, train_labels = sample_per_class(train_samples, train_labels, Config.PER_CLASS_TRAIN_SAMPLES, f"[{task_name}] Train")
        test_samples, test_labels = sample_per_class(test_samples, test_labels, Config.PER_CLASS_TEST_SAMPLES, f"[{task_name}] Test")
        
        _, train_num_classes = check_class_distribution(train_samples, train_labels)
        if train_num_classes < 2:
            print(f"[!] 跳过任务 {task_name}: 类别数不足 2。")
            continue
            
        all_tasks_meta[task_name] = {
            "label_processor": label_processor,
            "train_samples": train_samples, "train_labels": train_labels,
            "test_samples": test_samples, "test_labels": test_labels,
            "output_dir": output_dir
        }
        
        datasets_dict[f"{task_name}_train"] = train_samples
        datasets_dict[f"{task_name}_test"] = test_samples

    if not datasets_dict:
        print("\n[!] 没有有效任务可执行，程序退出。")
        return

    # --- Phase 2: 模型主导的特征提取 ---
    extractor = PerplexityExtractor(Config.PROXY_MODELS)
    raw_ppl_dict = extractor.extract_aligned_ppl(datasets_dict)
    
    # --- Phase 3: 特征工程与分类评估 ---
    print("\n>>> [Phase 3] Constructing Features & Training Classifiers...")
    engineer = SnifferFeatureEngineer(len(Config.PROXY_MODELS))
    
    for task_name, meta in all_tasks_meta.items():
        print(f"\n" + "="*50)
        print(f" 开始评估任务: {task_name} ")
        print("="*50)
        
        label_processor = meta["label_processor"]
        
        X_train, train_valid_idx = engineer.process_dataset(raw_ppl_dict[f"{task_name}_train"])
        X_test, test_valid_idx = engineer.process_dataset(raw_ppl_dict[f"{task_name}_test"])
        
        y_train = [meta["train_labels"][i] for i in train_valid_idx]
        y_test = [meta["test_labels"][i] for i in test_valid_idx]
        
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test) 
        
        y_test_names = [label_processor.id2class[lbl] for lbl in y_test]
        y_pred_names = [label_processor.id2class[lbl] for lbl in y_pred]
        
        auc_metrics = calculate_auc_metrics(y_test, y_pred, y_pred_proba, label_processor)
        class_report_txt = generate_class_report_txt(y_test, y_pred, label_processor)
        
        test_samples_filtered = [meta["test_samples"][i] for i in test_valid_idx]
        
        out_dir = meta["output_dir"]
        save_predictions(test_samples_filtered, y_test, y_pred, y_test_names, y_pred_names, out_dir)
        save_auc_metrics(auc_metrics, out_dir)
        save_class_report_txt(class_report_txt, out_dir)
        
        print(f"✅ 任务 {task_name} 完成。结果已保存至: {out_dir}")

if __name__ == "__main__":
    main()