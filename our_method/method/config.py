"""
Global configuration for Bi-Infer pipeline.
All hyperparameters, paths, and model settings are centralized here for easy modification.
"""

import os

# ==============================
# Base Directory Configuration
# ==============================
BASE_DIR = "/home/gsy/project2/m4"
CACHE_DIR = os.path.join(BASE_DIR, "cache_dynamic_pmi_full")
os.makedirs(CACHE_DIR, exist_ok=True)

# ==============================
# Feature Extraction Configuration
# ==============================
FEATURE_EXTRACT_CONFIG = {
    "model_configs": [
        {"type": "llama", "path": "/home/share/models/llama-7b"},
        {"type": "qwen", "path": "/home/share/models/qwen2.5-7b-instruct"}
    ],
    "tasks": [
        {
            "input": os.path.join(BASE_DIR, "arxiv", "data_test_arxiv.json"),
            "output_base": os.path.join(CACHE_DIR, "test_data"),
            "max_samples": 4000
        },
        # Uncomment to add more extraction tasks
        # {
        #     "input": os.path.join(BASE_DIR, "wikihow", "data_train_wikihow.json"),
        #     "output_base": os.path.join(CACHE_DIR, "train_dynamic_wikihow"),
        #     "max_samples": 3000
        # },
    ]
}

# ==============================
# Training Configuration
# ==============================
TRAIN_CONFIG = {
    "feature_mode": "normalized",  # Options: "normalized", "raw"
    "train_data_path": os.path.join(CACHE_DIR, "train_dynamic_wikipedia_bi.pt"),
    "test_data_path": os.path.join(CACHE_DIR, "test_dynamic_wikipedia_bi.pt"),
    "model_save_path": os.path.join(CACHE_DIR, "m3f_conformer_adaptive_normalized_all1.pth"),
    "plot_save_path": os.path.join(CACHE_DIR, "training_curves_adaptive_normalized_all.png"),
    "report_save_path": os.path.join(CACHE_DIR, "full_prediction_report_normalized_all.json"),

    "device": "cuda:0",  # Fallback to CPU automatically if CUDA is unavailable
    "batch_size": 32,
    "max_successful_epochs": 4,
    "learning_rate": 1e-4,
    "min_learning_rate": 1e-6,
    "weight_decay": 1e-3,
    "dropout": 0.1,
    "model_dim": 128,
    "num_layers": 4,
    "num_heads": 4,
    "max_seq_len": 2000,
    "excluded_labels": set()
}