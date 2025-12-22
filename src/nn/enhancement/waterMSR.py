import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List

from ...core import register

__all__ = ['MSREnhancement']

class GaussianBlurConv(object):
    """
    Fast Gaussian Blur using Pyramid approximation.
    Based on the provided optimized code.
    """
    def FastFilter(self, img, sigma):
        """
        Recursive fast gaussian blur.
        Args:
            img: Input image (H, W, C) numpy array.
            sigma: Standard deviation.
        """
        # reject unreasonable demands
        if sigma > 300:
            sigma = 300
            
        # Kernel size must be odd
        kernel_size = round(sigma * 3 * 2 + 1) | 1
        
        # If kernel size is small, return or use standard gaussian
        if kernel_size < 3:
            return img
            
        # Base case: Use standard cv2.GaussianBlur for small kernels
        if kernel_size < 10:
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        else:
            # Recursive step: Downsample -> Blur with half sigma -> Upsample
            if img.shape[1] < 2 or img.shape[0] < 2:
                return img
                
            sub_img = np.zeros_like(img)
            # cv2.pyrDown performs Gaussian blur and downsampling (1/2 size)
            sub_img = cv2.pyrDown(img)
            
            # Recursive call with half sigma
            sub_img = self.FastFilter(sub_img, sigma / 2.0)
            
            # Resize back to original size
            # Note: cv2.resize dsize is (width, height)
            res_img = cv2.resize(sub_img, (img.shape[1], img.shape[0]))
            return res_img

    def __call__(self, x, sigma):
        return self.FastFilter(x, sigma)


@register()
class MSREnhancement(nn.Module):
    """
    Multi-Scale Retinex with Color Restoration (MSR).
    
    Optimized using Pyramid Gaussian Blur for speed (GaussianBlurConv).
    """
    
    def __init__(self, 
                 sigma: List[float] = [30, 150, 300], 
                 restore_factor: float = 2.0, 
                 color_gain: float = 10.0, 
                 gain: float = 270.0, 
                 offset: float = 128.0):
        super(MSREnhancement, self).__init__()
        self.sigma = sigma
        self.restore_factor = restore_factor
        self.color_gain = color_gain
        self.gain = gain
        self.offset = offset
        # Initialize the optimized blur operator
        self.gaussian_conv = GaussianBlurConv()

    def _gaussian_blur_optimized(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply Fast Gaussian Blur.
        
        Args:
            x: Input tensor (N, 3, H, W).
            sigma: Standard deviation.
            
        Returns:
            Blurred tensor (N, 3, H, W).
        """
        device = x.device
        
        # Move to CPU and convert to numpy (N, H, W, 3)
        img_np = x.detach().cpu().permute(0, 2, 3, 1).numpy()
        
        blurred_np = np.empty_like(img_np)
        
        for i in range(img_np.shape[0]):
            # Apply FastFilter per image
            blurred_np[i] = self.gaussian_conv(img_np[i], sigma)
            
        # Convert back to tensor (N, 3, H, W)
        output_tensor = torch.from_numpy(blurred_np).permute(0, 3, 1, 2).to(device)
        
        return output_tensor

    def _ssr(self, img, sigma):
        """Single Scale Retinex"""
        # Use the optimized blur function
        filter_img = self._gaussian_blur_optimized(img, sigma)
        return torch.log10(img + 1e-6) - torch.log10(filter_img + 1e-6)

    def _msr(self, img):
        """Multi Scale Retinex"""
        retinex = torch.zeros_like(img)
        
        for sig in self.sigma:
            retinex += self._ssr(img, sig)
            
        return retinex / len(self.sigma)

    def _color_restoration(self, img, retinex):
        """Color Restoration Step"""
        # Sum over channels (dim 1)
        img_sum = torch.sum(img, dim=1, keepdim=True)
        
        color_restoration = torch.log10((img * self.restore_factor / (img_sum + 1e-6)) + 1.0)
        img_merge = retinex * color_restoration * self.color_gain
        return img_merge * self.gain + self.offset

    def forward(self, x):
        """
        Args:
            x: Input tensor (N, 3, H, W), typically normalized [0, 1]
        Returns:
            enhanced: Tensor (N, 3, H, W), normalized [0, 1]
        """
        
        img = x * 255.0
        img_float = img + 1.0
        retinex = self._msr(img_float)
        enhanced = self._color_restoration(img_float, retinex)
        enhanced = torch.clamp(enhanced, 0, 255)
        
        return enhanced / 255.0