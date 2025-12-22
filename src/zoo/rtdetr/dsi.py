# In rtdetrv2_pytorch/src/zoo/rtdetr/dsi_pan.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """A lightweight cross-attention module to fuse raw features into enhanced features, using nn.MultiheadAttention."""
    def __init__(self, in_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, enhanced_feat, raw_feat):
        """
        Args:
            enhanced_feat (torch.Tensor): The feature from the enhanced image branch. (Query)
            raw_feat (torch.Tensor): The feature from the raw image branch. (Key, Value)
        Returns:
            torch.Tensor: The fused feature map, with the same shape as enhanced_feat.
        """
        B, C, H, W = enhanced_feat.shape

        query = enhanced_feat.flatten(2).permute(0, 2, 1)
        key = raw_feat.flatten(2).permute(0, 2, 1)
        value = raw_feat.flatten(2).permute(0, 2, 1)

        attn_output, _ = self.multihead_attn(query=query, key=key, value=value)

        fused_feat = attn_output
        fused_feat = fused_feat.permute(0, 2, 1).reshape(B, C, H, W)

        return raw_feat + fused_feat

class GatedFusion(nn.Module):
    """Adaptive Gating Fusion Module"""
    def __init__(self, in_dim):
        super().__init__()
        self.gate_conv = nn.Conv2d(in_dim * 2, in_dim, kernel_size=1)

    def forward(self, original_feat, interactive_feat):
        gate = torch.sigmoid(self.gate_conv(torch.cat([original_feat, interactive_feat], dim=1)))
        fused_feat = original_feat + gate * interactive_feat
        return fused_feat

