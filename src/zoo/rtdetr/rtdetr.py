"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'enhancer']

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
        enhancer: nn.Module = None,
    ):
        super().__init__()
        self.enhancer = enhancer
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        
        if self.enhancer is not None:
            for param in self.enhancer.parameters():
                param.requires_grad = False
        
    def forward(self, x, targets=None):
        enhanced = self.enhancer(x) if self.enhancer is not None else x
        # enhanced = F.interpolate(enhanced, size=x.shape[2:], mode='bilinear', align_corners=False)
        enhanced = self.backbone(enhanced)
        x = self.backbone(x)
        x, enhanced = self.encoder(x, enhanced)        
        x = self.decoder(x, enhanced, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
