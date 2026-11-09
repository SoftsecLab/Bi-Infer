"""Conformer classifier with masked majority voting."""
import torch
from torch import nn
from .conformer import ConformerBlock

class M3FNet(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        num_classes=2,
        dropout=0.1,
        num_layers=4,
        nhead=4,
        max_seq_len=1024,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Sequential(
            nn.Conv1d(self.input_dim, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.max_seq_len, d_model) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model, nhead, kernel_size=15, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        if x.size(2) > self.max_seq_len:
            x = x[:, :, : self.max_seq_len]
            if mask is not None:
                mask = mask[:, : self.max_seq_len]

        x = self.input_proj(x)
        x = x.transpose(1, 2)
        seq_len = x.size(1)

        x = x + self.pos_embedding[:, :seq_len, :]
        for layer in self.layers:
            x = layer(x, mask=mask)

        logits = self.classifier(x)
        if self.training:
            return logits
        return self._vote(logits, mask)

    def _vote(self, logits, mask):
        preds = torch.argmax(logits, dim=-1)
        if mask is not None:
            preds = preds.masked_fill(mask, -1)

        final_labels = []
        for i in range(preds.size(0)):
            valid_votes = preds[i][preds[i] != -1]
            if len(valid_votes) == 0:
                final_labels.append(0)
            else:
                vals, counts = torch.unique(
                    valid_votes, return_counts=True
                )
                mode_idx = torch.argmax(counts)
                final_labels.append(vals[mode_idx].item())

        return torch.tensor(final_labels, device=logits.device), logits

