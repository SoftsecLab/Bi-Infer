"""Feature loading and padding masks."""
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def load_composed_cache(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "samples" in obj:
        return obj.get("meta", {}), obj["samples"]
    if isinstance(obj, list):
        return {}, obj
    raise ValueError(f"Unsupported cache format: {path}")

class MultiModelDataset(Dataset):
    def __init__(
        self,
        data_path,
        label_map,
        feature_type="normalized",
        target_channels=None,
    ):
        self.meta, self.data = load_composed_cache(data_path)
        assert feature_type in ["normalized", "raw"]
        self.feature_type = feature_type
        self.target_channels = target_channels
        self.label_map = label_map
        self.indices = [
            i
            for i, item in enumerate(self.data)
            if str(item["label"]) in self.label_map
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.data[self.indices[idx]]

        if self.feature_type == "normalized":
            selected_features = item["features"]
        else:
            selected_features = item["raw_features"]

        feat_tensor = torch.tensor(
            selected_features, dtype=torch.float32
        )

        if self.target_channels is not None:
            current_channels = feat_tensor.shape[0]
            if current_channels > self.target_channels:
                feat_tensor = feat_tensor[: self.target_channels, :]
            elif current_channels < self.target_channels:
                pad_size = self.target_channels - current_channels
                feat_tensor = F.pad(
                    feat_tensor, (0, 0, 0, pad_size)
                )

        raw_length = int(item["length"])
        length = min(raw_length, feat_tensor.shape[1])

        return {
            "features": feat_tensor[:, :length],
            "label": torch.tensor(
                self.label_map[str(item["label"])], dtype=torch.long
            ),
            "length": length,
            "sample_id": item.get("sample_id", str(idx)),
            "text": item.get("text", ""),
        }

def collate_fn(batch):
    max_len = max(item["length"] for item in batch)
    C = batch[0]["features"].shape[0]
    B = len(batch)

    padded_x = torch.zeros(B, C, max_len)
    labels = torch.zeros(B, dtype=torch.long)
    masks = torch.ones(B, max_len, dtype=torch.bool)
    token_labels = torch.full(
        (B, max_len), -100, dtype=torch.long
    )

    sample_ids = []
    texts = []

    for i, item in enumerate(batch):
        l = item["length"]
        padded_x[i, :, :l] = item["features"][:, :l]
        labels[i] = item["label"]
        masks[i, :l] = False
        token_labels[i, :l] = item["label"]
        sample_ids.append(item["sample_id"])
        texts.append(item["text"])

    return padded_x, labels, masks, token_labels, sample_ids, texts

