"""
蒸馏损失
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
import logging
from .loss import Loss


from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

logger = logging.getLogger()


KD_POINTS = {
    "resnet50": dict(kd_points=["layer4", "fc"], channels=[2048, 1000]),
    # 'resnet34': dict(kd_points=['layer4', 'fc'])
    "mobilenet_v1": dict(kd_points=["model.13", "classifier"], channels=[1024, 1000]),
    "regnety_040": dict(kd_points=["backbone.final_conv"], channels=[1088]),
    "deit": dict(kd_points=[""], channels=[]),
}

__all__ = ["KDLoss"]


def standardize_features(tensor, eps=1e-8):
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
        self.stu_name, self.stu_model = student[0], student[1]
        self.tea_name, self.tea_model = teacher[0], teacher[1]

        self.base_criterion = base_criterion

        self.cls_loss_w = cls_loss_w
        self.kl_loss_w = 1 - cls_loss_w
        self.feat_loss_w = feat_loss_w

        # kd_method = mse
        # stu_kd_points = KD_POINTS[self.stu_name]['kd_points'][:1]
        # tea_kd_points = KD_POINTS[self.tea_name]['kd_points'][:1]
        # stu_channels = KD_POINTS[self.stu_name]['channels'][:1]
        # tea_channels = KD_POINTS[self.tea_name]['channels'][:1]

        # self.kd_loss = nn.MSELoss()

        # self.align = nn.Conv2d(stu_channels[0], tea_channels[0], 1)
        # self.align.cuda()
        # self.stu_model._align = self.align

        self._tea_out = {}
        self._stu_out = {}

        # register hook in tea and stu model
        # for stu_point, tea_point in zip(stu_kd_points, tea_kd_points):
        # self._register_forward_hook(
        # self.stu_model, stu_point, is_tea=False)
        # self._register_forward_hook(self.tea_model, tea_point, is_tea=True)

        # self.stu_kd_points = stu_kd_points
        # self.tea_kd_points = tea_kd_points

        self.tea_model.eval()
        self._iter = 0

    def __call__(self, x, targets):
        self._stu_out = {}
        self._tea_out = {}

        # with torch.no_grad():
        # t_logits = self.tea_model.forward(x)

        s_logits = self.stu_model(x)

        # T = 6.0

        cls_loss = self.base_criterion(s_logits, targets)

        # feat_loss = 0
        # for sp, tp in zip(self.stu_kd_points, self.tea_kd_points):
        # if hasattr(self, 'align'):
        # self._stu_out[sp] = self.align(self._stu_out[sp])

        # stu_feature = standardize_features(self._stu_out[sp])
        # tea_feature = standardize_features(self._tea_out[tp])

        # feat_loss_ = self.kd_loss(stu_feature, tea_feature)
        # feat_loss += feat_loss_

        # kl_loss = F.kl_div(
        # F.log_softmax(s_logits / T, dim=1),  # 学生
        # F.softmax(t_logits / T, dim=1),  # 教师
        # reduction='batchmean',
        # ) * (T ** 2)

        return cls_loss, s_logits

    def _register_forward_hook(self, model: nn.Module, point_name: str, is_tea=False):
        # register_points = []
        if point_name == "":
            # use the output of model
            model.register_forward_hook(
                partial(self._forward_hook, name=point_name, is_tea=is_tea)
            )
        else:
            module = None
            for k, m in model.named_modules():
                if k == point_name:
                    print("ok")
                    module = m
                    break
            module.register_forward_hook(
                partial(self._forward_hook, name=point_name, is_tea=is_tea)
            )
        # return register_points

    def _forward_hook(self, module, input, output, name: str, is_tea=False):
        if is_tea:
            self._tea_out[name] = output[0] if len(output) == 1 else output
        else:
            self._stu_out[name] = output[0] if len(output) == 1 else output
