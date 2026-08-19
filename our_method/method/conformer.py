"""
Conformer module implementation.
Includes feed-forward module, depthwise convolution module, and full Conformer block.
"""

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    """Gated Linear Unit activation."""
    def __init__(self, dim: int = 1):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)


class DepthwiseConv1d(nn.Module):
    """1D Depthwise Convolution layer."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super(DepthwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, groups=in_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConformerConvModule(nn.Module):
    """Convolution module inside the Conformer block."""
    def __init__(self, embedding_dim: int, kernel_size: int = 15, dropout: float = 0.1):
        super(ConformerConvModule, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.pointwise_conv1 = nn.Conv1d(embedding_dim, embedding_dim * 2, kernel_size=1)
        self.glu = GLU(dim=1)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = DepthwiseConv1d(embedding_dim, embedding_dim, kernel_size, padding)
        self.batch_norm = nn.BatchNorm1d(embedding_dim)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return residual + x


class FeedForwardModule(nn.Module):
    """Feed-forward module with pre-norm and residual connection."""
    def __init__(self, embedding_dim: int, expansion_factor: int = 4, dropout: float = 0.1):
        super(FeedForwardModule, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim * expansion_factor)
        self.activation = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embedding_dim * expansion_factor, embedding_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x


class ConformerBlock(nn.Module):
    """
    Full Conformer block: FFN -> Multi-Head Self-Attention -> Conv Module -> FFN
    Follows the Macaron-Net structure with half-step residual scaling.
    """
    def __init__(
        self, embedding_dim: int, num_heads: int,
        kernel_size: int = 15, dropout: float = 0.1
    ):
        super(ConformerBlock, self).__init__()
        self.ffn1 = FeedForwardModule(embedding_dim, expansion_factor=4, dropout=dropout)
        self.attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.conv_module = ConformerConvModule(embedding_dim, kernel_size=kernel_size, dropout=dropout)
        self.ffn2 = FeedForwardModule(embedding_dim, expansion_factor=4, dropout=dropout)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.ffn1(x)
        residual = x
        x_norm = self.attn_layer_norm(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = residual + x_attn
        x = self.conv_module(x)
        x = self.ffn2(x)
        x = self.final_layer_norm(x)
        return x