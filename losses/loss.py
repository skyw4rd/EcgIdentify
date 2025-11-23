import numpy as np
from abc import ABC, abstractmethod

KD_POINTS = {
    'resnet50': dict(kd_points=['layer4', 'fc'], channels=[2048, 1000]),
    'mobilenet_v1': dict(kd_points=['model.13', 'classifier'], channels=[1024, 1000])
}

class Loss():
    """
    损失
    """
    def __init__(self):
        pass

    def __call__(self, x, targets):
        pass
