"""
三元损失
"""
import torch
from torch import nn
from .loss import Loss


class TripletPlusCe(Loss):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def __call__(self):
        pass


def get_mask(batch_shape):
    """
    加速正负样本的查找
    """
    classes_num, embedding_num = batch_shape
    batch_size = classes_num * embedding_num
    positive_mask, negative_mask = torch.full(
        (batch_size, batch_size), True), torch.full((batch_size, batch_size), False)
    for s in range(0, batch_size, embedding_num):
        for i in range(embedding_num):
            for j in range(embedding_num):
                if i != j:
                    positive_mask[s + i][s + j] = False
    for s in range(0, batch_size, embedding_num):
        for i in range(embedding_num):
            for j in range(embedding_num):
                negative_mask[s + i][s + j] = True
    return positive_mask, negative_mask
