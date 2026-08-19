"""
End-to-end pipeline for Bi-Infer: feature extraction + model training.
This is the main entry point for the full workflow.
"""

from config import FEATURE_EXTRACT_CONFIG, TRAIN_CONFIG
from feature_extractor import process_multimodel_global_k
from train import train_model


def main():
    print("="*80)
    print("Bi-Infer Full Pipeline: Feature Extraction -> Source Tracing Model Training")
    print("="*80)

    # Step 1: Extract bidirectional PMI features
    print("\n[Step 1/2] Running multi-model feature extraction...")
    process_multimodel_global_k(
        FEATURE_EXTRACT_CONFIG["tasks"],
        FEATURE_EXTRACT_CONFIG["model_configs"]
    )

    # Step 2: Train the Conformer-based tracing model
    print("\n[Step 2/2] Running model training and evaluation...")
    train_model(TRAIN_CONFIG)

    print("\n" + "="*80)
    print("Full pipeline completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()