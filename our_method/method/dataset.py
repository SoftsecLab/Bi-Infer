"""
Dataset and collate function for Bi-Infer training and evaluation.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MultiModelDataset(Dataset):
    """
    Dataset for pre-extracted multi-model PMI features.
    Supports both normalized and raw feature modes.
    """
    def __init__(
        self, data_path: str, feature_type: str = 'normalized',
        target_channels: int = None
    ):
        self.data = torch.load(data_path, weights_only=False)
        assert feature_type in ['normalized', 'raw'], \
            "feature_type must be either 'normalized' or 'raw'"
        self.feature_type = feature_type
        self.target_channels = target_channels 
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        if self.feature_type == 'normalized':
            selected_features = item['features']
        else:
            selected_features = item['raw_features']
            
        feat_tensor = torch.tensor(selected_features, dtype=torch.float32)
        
        # Align channel count
        if self.target_channels is not None:
            current_channels = feat_tensor.shape[0]
            if current_channels > self.target_channels:
                feat_tensor = feat_tensor[:self.target_channels, :]
            elif current_channels < self.target_channels:
                pad_size = self.target_channels - current_channels
                feat_tensor = F.pad(feat_tensor, (0, 0, 0, pad_size))
            
        return {
            'features': feat_tensor,
            'label': torch.tensor(item['label'], dtype=torch.long),
            'length': item['length'],
            'text': item.get('text', '')
        }


def collate_fn(batch: list) -> tuple:
    """
    Custom collate function for variable-length sequences.
    
    Args:
        batch: List of dataset items.
    
    Returns:
        Tuple: (padded_features, labels, padding_masks, token_labels, texts)
    """
    max_len = max([item['length'] for item in batch])
    num_channels = batch[0]['features'].shape[0]
    batch_size = len(batch)
    
    padded_x = torch.zeros(batch_size, num_channels, max_len)
    labels = torch.zeros(batch_size, dtype=torch.long)
    masks = torch.ones(batch_size, max_len, dtype=torch.bool)
    token_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    texts = []
    
    for i, item in enumerate(batch):
        seq_len = item['length']
        padded_x[i, :, :seq_len] = item['features'][:, :seq_len]
        labels[i] = item['label']
        masks[i, :seq_len] = False
        token_labels[i, :seq_len] = item['label']
        texts.append(item['text'])
        
    return padded_x, labels, masks, token_labels, texts
