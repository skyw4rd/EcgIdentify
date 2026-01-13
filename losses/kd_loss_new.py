"""
蒸馏损失
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
import logging
from .loss import Loss


from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

logger = logging.getLogger()


KD_POINTS = {
    'resnet34.a1_in1k': dict(kd_points=['backbone.layer4'], channels=[512], feature_map_size=(7, 7)),
    'mobilenetv3_small_100.lamb_in1k': dict(kd_points=['blocks'], channels=[576], feature_map_size=(7, 7)),
    'deit_tiny_patch16_224': dict(kd_points=['blocks.11'], channels=[192], feature_map_size=None)
}

__all__ = ["KDLoss"]


def standardize_features(tensor, eps=1e-8):
    """标准化特征"""
    # This works for both 4D CNN features and 4D reshaped ViT features
    mean = torch.mean(tensor, dim=[1, 2, 3], keepdim=True)
    std = torch.std(tensor, dim=[1, 2, 3], keepdim=True)
    return (tensor - mean) / (std + eps)


class KDLoss(Loss):
    def __init__(
        self,
        student: Tuple[str, nn.Module],
        teacher: Tuple[str, nn.Module],
        base_criterion,
        cls_loss_w=0.35,
        feat_loss_w=1.0,
    ):
        super().__init__()
        self.stu_name, self.stu_model = student[0], student[1]
        self.tea_name, self.tea_model = teacher[0], teacher[1]

        self.base_criterion = base_criterion
        
        # 损失权重
        self.cls_loss_w = cls_loss_w
        self.kl_loss_w = 1 - cls_loss_w
        self.feat_loss_w = feat_loss_w

        stu_kd_points = KD_POINTS[self.stu_name]['kd_points'][:1]
        tea_kd_points = KD_POINTS[self.tea_name]['kd_points'][:1]
        stu_channels = KD_POINTS[self.stu_name]['channels'][:1]
        tea_channels = KD_POINTS[self.tea_name]['channels'][:1]
        
        self.is_vit_student = 'deit' in self.stu_name

        # kd_method = mse
        self.kd_loss = nn.MSELoss()

        # Create alignment layer based on student model type
        # 对齐方式
        if self.is_vit_student:
            # For ViT, we align the feature dimension
            self.align = nn.Linear(stu_channels[0], tea_channels[0])
            # And create a pooling layer to match spatial dimensions
            teacher_map_size = KD_POINTS[self.tea_name].get('feature_map_size')
            if teacher_map_size:
                self.pool = nn.AdaptiveAvgPool2d(teacher_map_size)
            else:
                # Fallback if teacher map size is not specified (e.g. 7x7 for resnet on 224 input)
                self.pool = nn.AdaptiveAvgPool2d((7, 7))
        else:
            # For CNN, we use Conv2d for alignment
            self.align = nn.Conv2d(stu_channels[0], tea_channels[0], 1)
        
        self.align.cuda() # Assuming GPU is available
        # self.stu_model._align = self.align # This is not used, better to handle in loss call

        self._tea_out = {}
        self._stu_out = {}

        # register hook in tea and stu model
        for stu_point, tea_point in zip(stu_kd_points, tea_kd_points):
            self._register_forward_hook(
            self.stu_model, stu_point, is_tea=False)
            self._register_forward_hook(self.tea_model, tea_point, is_tea=True)

            self.stu_kd_points = stu_kd_points
            self.tea_kd_points = tea_kd_points

            self.tea_model.eval()
            # self._iter = 0

    def __call__(self, x, targets):
        self._stu_out = {}
        self._tea_out = {}

        # with torch.no_grad():
        t_logits = self.tea_model.forward(x)
        s_logits = self.stu_model.forward(x)

        T = 5.0

        cls_loss = self.base_criterion(s_logits, targets)

        feat_loss = 0
        for sp, tp in zip(self.stu_kd_points, self.tea_kd_points):
            
            stu_feature_orig = self._stu_out[sp]
            tea_feature_orig = self._tea_out[tp]

            # Align student features to match teacher features
            if self.is_vit_student:
                # Input is (B, N, D), e.g., (B, 197, 384)
                # Align feature dimension
                aligned_feat = self.align(stu_feature_orig) # (B, N, 512)
                
                # Separate CLS token from patch tokens and reshape patches to a 2D feature map
                B, N, D = aligned_feat.shape
                # Assuming patch tokens are all but the first token (CLS token)
                # If N=197, N-1=196 patches
                patch_tokens = aligned_feat[:, 1:, :]
                
                # Calculate H, W from number of patch tokens.
                num_patches = N - 1
                H = W = int(math.sqrt(num_patches))
                if H * W != num_patches:
                    logger.warning(f"Number of patch tokens ({num_patches}) is not a perfect square. Check feature point for ViT student.")
                    # Fallback or error if not perfect square
                    # For now, let's just make it HxW=1, if not perfect square
                    H = W = 1
                    patch_tokens_reshaped = patch_tokens.mean(dim=1, keepdim=True).permute(0, 2, 1).reshape(B, D, H, W)
                else:
                    # Reshape: (B, N-1, D) -> (B, D, N-1) -> (B, D, H, W)
                    patch_tokens_reshaped = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
                
                # Downsample spatial dimensions to match the teacher's feature map
                stu_feature_aligned = self.pool(patch_tokens_reshaped)
            else: # CNN student
                stu_feature_aligned = self.align(stu_feature_orig)

            # Standardize features before comparing
            stu_feature = standardize_features(stu_feature_aligned)
            tea_feature = standardize_features(tea_feature_orig)

            feat_loss_ = self.kd_loss(stu_feature, tea_feature)
            feat_loss += feat_loss_
        
        # 计算kl散度
        kl_loss = F.kl_div(
            F.log_softmax(s_logits / T, dim=1),  # 学生
            F.softmax(t_logits / T, dim=1),  # 教师
            reduction='batchmean',
        ) * (T ** 2)

        return cls_loss + self.kl_loss_w * kl_loss + self.feat_loss_w * feat_loss, s_logits

    def _register_forward_hook(self, model: nn.Module, point_name: str, is_tea=False):
        if point_name == "":
            model.register_forward_hook(
                partial(self._forward_hook, name=point_name, is_tea=is_tea)
            )
        else:
            module = None
            # Find the module using its name
            try:
                module = dict(model.named_modules())[point_name]
            except KeyError:
                logger.error(f"Module '{point_name}' not found in model '{self.stu_name if not is_tea else self.tea_name}'")
                raise
            
            module.register_forward_hook(
                partial(self._forward_hook, name=point_name, is_tea=is_tea)
            )

    def _forward_hook(self, module, input, output, name: str, is_tea=False):
        # DeiT block output is a tuple in some timm versions, take the first element
        if isinstance(output, tuple):
            output = output[0]
            
        if is_tea:
            self._tea_out[name] = output
        else:
            self._stu_out[name] = output