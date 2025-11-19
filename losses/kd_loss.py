import math
import torch
import torch.nn as nn

from functools import partial

import logging
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
logger = logging.getLogger()

KD_POINTS= {
    'resnet50': dict(kd_points=['layer4', 'fc'], channels=[2048, 1000]),
    'mobilenet_v1': dict(kd_points=['model.13', 'classifier'], channels=[1024, 1000])
}

__all__ = ['KDLoss']

class KDLoss():
    def __init__(
            self,
            student: Tuple[str, nn.Module],
            teacher: Tuple[str, nn.Module],
            ori_loss,
            ori_loss_w=1.0,
            kd_loss_w=1.0
        ):

        self.stu_name, self.stu_model = student[0], student[1]
        self.tea_name, self.tea_model = teacher[0], teacher[1]

        self.ori_loss = ori_loss
        self.ori_loss_w = ori_loss_w
        self.kd_loss_w = kd_loss_w

        # kd_method = mse
        stu_kd_points = KD_POINTS[self.stu_name]['kd_points'][:1]
        tea_kd_points = KD_POINTS[self.tea_name]['kd_points'][:1]
        stu_channels = KD_POINTS[self.stu_name]['channels'][:1]
        tea_channels = KD_POINTS[self.tea_name]['channels'][:1]
        self.kd_loss = nn.MSELoss()
        self.align = nn.Conv2d(stu_channels[0], tea_channels[0], 1)
        self.align.cuda()
        self.stu_model._align = self.align
        
        self._tea_out = {}
        self._stu_out = {}
        
        # register hook in tea and stu model
        for stu_point, tea_point in zip(stu_kd_points, tea_kd_points):
            self._register_forward_hook(self.stu_model, stu_point, is_tea=False)
            self._register_forward_hook(self.tea_model, tea_point, is_tea=True)
            # print('student register points:', res_stu)
            # print('teacher register points', res_tea)
        
        self.stu_kd_points = stu_kd_points
        self.tea_kd_points = tea_kd_points
        
        self.tea_model.eval()
        self._iter = 0
    
    def __call__(self, x, targets):
        with torch.no_grad():
            t_logits = self.tea_model(x)

        # compute stu ori loss
        logits = self.stu_model(x)
        ori_loss = self.ori_loss(logits, targets)
        
        kd_loss = 0

        for sp, tp in zip(self.stu_kd_points, self.tea_kd_points):
            if hasattr(self, 'align'):
                self._stu_out[sp] = self.align(self._stu_out[sp])

            print(self._stu_out[sp].shape)
            print(self._tea_out[tp].shape)

            kd_loss_ = self.kd_loss(self._stu_out[sp], self._tea_out[tp])
            kd_loss += kd_loss_
        
        self._stu_out = {}
        self._tea_out = {}

        return ori_loss * self.ori_loss_w + kd_loss * self.kd_loss_w
    
    def _register_forward_hook(self, model: nn.Module, point_name: str, is_tea=False):
        # register_points = []
        if point_name == '':
            # use the output of model
            model.register_forward_hook(partial(self._forward_hook, name=point_name, is_tea=is_tea))
        else:
            module = None
            for k, m in model.named_modules():
                if k == point_name:
                    module = m
                    # register_points.append(m)
                    break
            module.register_forward_hook(partial(self._forward_hook, name=point_name, is_tea=is_tea))
        # return register_points

    def _forward_hook(self, module: nn.Module, input, output, name: str, is_tea=False):
        if is_tea:
            self._tea_out[name] = output[0] if len(output) == 1 else output
        else:
            self._stu_out[name] = output[0] if len(output) == 1 else output