"""Conformer building blocks."""
import torch
from torch import nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GLU(nn.Module):
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=1)
        return out * torch.sigmoid(gate)

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, x):
        return self.conv(x)

class ConformerConvModule(nn.Module):
    def __init__(self, embedding_dim, kernel_size=15, dropout=0.1):
        super(ConformerConvModule, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.pointwise_conv1 = nn.Conv1d(
            embedding_dim, embedding_dim * 2, kernel_size=1
        )
        self.glu = GLU(embedding_dim)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = DepthwiseConv1d(
            embedding_dim, embedding_dim, kernel_size, padding
        )
        self.batch_norm = nn.BatchNorm1d(embedding_dim)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            embedding_dim, embedding_dim, kernel_size=1
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
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
    def __init__(self, embedding_dim, expansion_factor=4, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.linear1 = nn.Linear(
            embedding_dim, embedding_dim * expansion_factor
        )
        self.activation = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(
            embedding_dim * expansion_factor, embedding_dim
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x

class ConformerBlock(nn.Module):
    def __init__(
        self, embedding_dim, num_heads, kernel_size=15, dropout=0.1
    ):
        super(ConformerBlock, self).__init__()
        self.ffn1 = FeedForwardModule(
            embedding_dim, expansion_factor=4, dropout=dropout
        )
        self.attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.conv_module = ConformerConvModule(
            embedding_dim, kernel_size=kernel_size, dropout=dropout
        )
        self.ffn2 = FeedForwardModule(
            embedding_dim, expansion_factor=4, dropout=dropout
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        x = self.ffn1(x)
        residual = x
        x_norm = self.attn_layer_norm(x)
        x_attn, _ = self.attn(
            x_norm, x_norm, x_norm, key_padding_mask=mask
        )
        x = residual + x_attn
        x = self.conv_module(x)
        x = self.ffn2(x)
        x = self.final_layer_norm(x)
        return x

