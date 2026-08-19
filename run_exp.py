import os
import subprocess
from datetime import datetime

# ==============================================================================
# 1. 数据集配置区 (Datasets Configuration)
# ==============================================================================
# 你可以在这里无限添加你需要跑的数据集
DATASETS = {
    # "TuringBench": {
    #     "train": "/home/gsy/project2/TuringBench/train/dataset_train.json",
    #     "test":  "/home/gsy/project2/TuringBench/test/dataset_test.json"
    # },
    # "Arxiv": {
    #     "train": "/home/gsy/project2/m4/arxiv/data_train_arxiv.json",
    #     "test":  "/home/gsy/project2/m4/arxiv/data_test_arxiv.json"
    # },
    # "Wikipedia": {
    #     "train": "/home/gsy/project2/m4/wikipedia/data_train_wikipedia.json",
    #     "test":  "/home/gsy/project2/m4/wikipedia/data_test_wikipedia.json"
    # }, 
    "Wikihow1": {
        "train": "/home/gsy/project2/augpt/train/train1.json",
        "test":  "/home/gsy/project2/augpt/test/test1.json"
    },
    # "reddit": {
    #     "train": "/home/gsy/project2/m4/reddit/data_train_reddit.json",
    #     "test":  "/home/gsy/project2/m4/reddit/data_test_reddit.json"
    # }

}

# ==============================================================================
# 2. 方法脚本与硬件分配区 (Methods & Hardware Configuration)
# ==============================================================================
# 键名是方法名称，"script" 是你保存的 python 文件名，"gpu" 是为该方法分配的显卡
METHODS = {
    # "Fast_DetectGPT": {
    #     "script": "/home/gsy/project2/my_method2/other_method/fast_detectgpt.py", 
    #     "gpu": "1"   # 你之前配置了使用卡 1
    # },
    # "RoBERTa_Large": {
    #     "script": "/home/gsy/project2/my_method2/other_method/reoberta_large.py",        
    #     "gpu": "0"   # 单卡 0 跑微调即可
    # },
    "GhostBuster": {
        "script": "/home/gsy/project2/my_method2/other_method/GhostBuster.py",    
        "gpu": "1"   # 单卡 0 跑特征提取即可（若显存不够可改 "0,1"）
    }
}

# ==============================================================================
# 3. 全局输出配置 (Global Output Configuration)
# ==============================================================================
# 所有实验结果的统一存放总目录
BASE_OUTPUT_DIR = "/home/gsy/project2/All_Experiments_Results"

# ==============================================================================
# 4. 核心调度引擎 (Master Scheduler)
# ==============================================================================
def main():
    total_experiments = len(DATASETS) * len(METHODS)
    print("*" * 80)
    print(f"🚀 [Master Runner] 自动化实验流水线启动")
    print(f"📦 计划执行: {len(DATASETS)} 个数据集 × {len(METHODS)} 种方法 = 共 {total_experiments} 组实验")
    print(f"📁 总输出目录: {BASE_OUTPUT_DIR}")
    print("*" * 80)

    experiment_count = 1

    # 外层循环：遍历数据集
    for dataset_name, data_paths in DATASETS.items():
        print(f"\n\n{'='*80}")
        print(f"📂 当前处理数据集 [{dataset_name}]")
        print(f"{'='*80}")
        
        # 内层循环：遍历检测方法
        for method_name, config in METHODS.items():
            script_path = config["script"]
            allocated_gpu = config["gpu"]
            
            # 为当前 [数据集 + 方法] 组合创建绝对独立的输出文件夹
            current_out_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name, method_name)
            os.makedirs(current_out_dir, exist_ok=True)
            
            print(f"\n[{experiment_count}/{total_experiments}] 正在运行方法: {method_name}")
            print(f"  ▶ 分配显卡: GPU {allocated_gpu}")
            print(f"  ▶ 报告目录: {current_out_dir}")
            
            # 构造命令行指令
            command = [
                "python", script_path,
                "--train_data", data_paths["train"],
                "--test_data", data_paths["test"],
                "--output_dir", current_out_dir,
                "--gpu", allocated_gpu
            ]
            
            start_time = datetime.now()
            
            try:
                # 阻塞式运行子进程（必须跑完当前这个，才会进入下一个循环）
                # check=True 表示如果脚本报错退出，会触发 CalledProcessError
                subprocess.run(command, check=True)
                
                elapsed = datetime.now() - start_time
                print(f"  ✅ [成功] {method_name} on {dataset_name} 运行完毕! 耗时: {elapsed}")
                
            except subprocess.CalledProcessError as e:
                # 容错机制：如果某个方法报错（如 OOM），打印错误但继续跑下一个方法
                elapsed = datetime.now() - start_time
                print(f"  ❌ [失败] {method_name} on {dataset_name} 运行崩溃! 错误代码: {e.returncode}，耗时: {elapsed}")
                print(f"  ⚠️  跳过该实验，继续执行下一个任务...")
                continue
            except FileNotFoundError:
                print(f"  ❌ [文件丢失] 找不到脚本 {script_path}，请检查文件名是否正确！")
                continue
                
            experiment_count += 1

    print("\n\n" + "*" * 80)
    print("🎉 [Master Runner] 所有数据集与检测方法的实验流水线均已执行完毕！")
    print(f"📁 最终结果请前往查看: {BASE_OUTPUT_DIR}")
    print("*" * 80)

if __name__ == "__main__":
    main()



# import os
# import subprocess
# from datetime import datetime

# # ==============================================================================
# # 1. 数据集配置区 (仅保留测试集路径)
# # ==============================================================================
# # 适配零样本检测任务，移除了 train 路径
# DATASETS = {
#     "TuringBench": {
#         "test":  "/home/gsy/project2/TuringBench/test/dataset_test.json"
#     },
#     "Arxiv": {
#         "test":  "/home/gsy/project2/m4/arxiv/data_test_arxiv.json"
#     },
#     "Wikipedia": {
#         "test":  "/home/gsy/project2/m4/wikipedia/data_test_wikipedia.json"
#     }, 
#     "Wikihow": {
#         "test":  "/home/gsy/project2/m4/Wikihow/data_test_Wikihow.json"
#     },
#     "reddit": {
#         "test":  "/home/gsy/project2/m4/reddit/data_test_reddit.json"
#     }
# }

# # ==============================================================================
# # 2. 方法脚本与硬件分配区
# # ==============================================================================
# METHODS = {
#     "Fast_DetectGPT": {
#         "script": "/home/gsy/project2/my_method2/other_method/GhostBuster.py", 
#         "gpu": "1" 
#     }
# }

# # ==============================================================================
# # 3. 全局输出配置
# # ==============================================================================
# BASE_OUTPUT_DIR = "/home/gsy/project2/All_Experiments_Results"

# # ==============================================================================
# # 4. 核心调度引擎 (零样本适配版)
# # ==============================================================================
# def main():
#     total_experiments = len(DATASETS) * len(METHODS)
#     print("*" * 80)
#     print(f"🚀 [Master Runner] 零样本(Zero-Shot)自动化实验流水线启动")
#     print(f"📦 计划执行: {len(DATASETS)} 个数据集 × {len(METHODS)} 种方法 = 共 {total_experiments} 组实验")
#     print(f"📁 总输出目录: {BASE_OUTPUT_DIR}")
#     print("*" * 80)

#     experiment_count = 1

#     for dataset_name, data_paths in DATASETS.items():
#         print(f"\n\n{'='*80}")
#         print(f"📂 当前处理数据集 [{dataset_name}] (Zero-Shot Mode)")
#         print(f"{'='*80}")
        
#         for method_name, config in METHODS.items():
#             script_path = config["script"]
#             allocated_gpu = config["gpu"]
            
#             # 创建独立的输出文件夹
#             current_out_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name, method_name)
#             os.makedirs(current_out_dir, exist_ok=True)
            
#             print(f"\n[{experiment_count}/{total_experiments}] 正在运行: {method_name}")
#             print(f"  ▶ 分配显卡: GPU {allocated_gpu}")
#             print(f"  ▶ 测试数据: {data_paths['test']}")
            
#             # 构造命令行指令：移除了 --train_data
#             command = [
#                 "python", script_path,
#                 "--test_data", data_paths["test"],
#                 "--output_dir", current_out_dir,
#                 "--gpu", allocated_gpu
#             ]
            
#             start_time = datetime.now()
            
#             try:
#                 # 阻塞式运行子进程
#                 subprocess.run(command, check=True)
                
#                 elapsed = datetime.now() - start_time
#                 print(f"  ✅ [成功] {method_name} 运行完毕! 耗时: {elapsed}")
                
#             except subprocess.CalledProcessError as e:
#                 elapsed = datetime.now() - start_time
#                 print(f"  ❌ [失败] {method_name} 运行崩溃! 错误代码: {e.returncode}，耗时: {elapsed}")
#                 continue
#             except FileNotFoundError:
#                 print(f"  ❌ [文件丢失] 找不到脚本 {script_path}")
#                 continue
                
#             experiment_count += 1

#     print("\n\n" + "*" * 80)
#     print("🎉 [Master Runner] 零样本对比实验流水线执行完毕！")
#     print("*" * 80)

# if __name__ == "__main__":
#     main()