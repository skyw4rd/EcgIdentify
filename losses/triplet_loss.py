"""
三元损失
"""
import sys
from typing import Tuple
import torch
from torch import nn
from .loss import Loss


sys.path.append('..')
from teacher_model import TeacherModel


class TripletLoss(Loss):
    """
    三元损失类
    """
    def __init__(self, model: TeacherModel):
        super().__init__()
        self.model = model
        self.feature_out = None
        self.triplet_loss = nn.TripletMarginLoss(margin=1.2, p=2)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(self, x, targets):
        # 特征向量，概率分布输出
        embeddings = self.model.forward_embeddings(x)
        logits = self.model.forward(x)

        # 相似度矩阵
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)

        # 生成掩码
        positive_mask, negative_mask = get_mask((embeddings.shape[0] // 4, 4))

        # 正样本与锚点的距离掩码
        pos_masked_matrix = distance_matrix.clone()
        pos_masked_matrix[positive_mask] = float('-inf')

        # 负样本与锚点的距离掩码
        neg_masked_matrix = distance_matrix.clone()
        neg_masked_matrix[negative_mask] = float('inf')

        # 32个锚点的对应的正样本
        _, hardest_positive_idxs = torch.max(pos_masked_matrix, dim=1)
        positives = embeddings[hardest_positive_idxs]

        # 32个锚点对应的负样本
        _, hardest_negative_idxs = torch.min(neg_masked_matrix, dim=1)
        negatives = embeddings[hardest_negative_idxs]

        # 总损失 = 交叉熵 + 三元损失
        t_loss = self.triplet_loss(embeddings, positives, negatives)
        task_loss = self.cross_entropy_loss(logits, targets)
        return t_loss + task_loss, logits 

def get_mask(batch_shape: Tuple[int, int]):
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
