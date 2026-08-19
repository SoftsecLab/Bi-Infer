"""
Visualization utilities for training monitoring.
"""

import matplotlib.pyplot as plt


def plot_training_curves(history: dict, save_path: str):
    """
    Plot training and validation loss & accuracy curves.
    
    Args:
        history: Dictionary containing training metrics.
        save_path: File path to save the output plot.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'g--', label='Val Loss')
    plt.title('Loss Curve (Successful Epochs Only)')
    plt.xlabel('Successful Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc (Token)')
    plt.plot(epochs, history['val_acc'], 'g--', label='Val Acc (Document)')
    plt.title('Accuracy Curve (Successful Epochs Only)')
    plt.xlabel('Successful Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training curves saved to: {save_path}")
    plt.close()