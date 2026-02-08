"""
三元损失（包含embedding压缩）
"""
from typing import Tuple, Optional

import torch
from torch import nn

from .loss import Loss


class EmbeddingHead(nn.Module):
    """
    Embedding compression head (dim reduction + normalization).
    """
    def __init__(self, in_dim: int, target_dim: int = 128):
        super().__init__()
        self.dim_reduction = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, target_dim),
            nn.BatchNorm1d(target_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        emb = nn.functional.normalize(self.dim_reduction(x), p=2, dim=1)
        return emb


class TripletLoss(Loss):
    """
    三元损失类（内置embedding压缩）
    """
    def __init__(self, model: nn.Module, target_dim: int = 128, feature_dim: Optional[int] = None):
        super().__init__()
        self.model = model
        self.feature_out = None
        if feature_dim is None:
            feature_dim = getattr(model, 'num_features', None)
        if feature_dim is None and hasattr(model, 'fc'):
            feature_dim = model.fc.in_features
        if feature_dim is None:
            raise ValueError('Cannot infer feature_dim from model; please pass feature_dim')
        self.embedding_head = EmbeddingHead(feature_dim, target_dim)
        self.triplet_loss = nn.TripletMarginLoss(margin=0.7, p=2)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def parameters(self):
        return self.embedding_head.parameters()

    def _extract_features(self, x):
        if hasattr(self.model, 'forward_features'):
            x = self.model.forward_features(x)
            return self.model.global_pool(x)

        if all(hasattr(self.model, name) for name in [
            'conv1', 'bn1', 'relu', 'maxpool',
            'layer1', 'layer2', 'layer3', 'layer4', 'avgpool'
        ]):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        raise ValueError('Model does not support feature extraction; add forward_features or pass a compatible model')

    def __call__(self, x, targets):
        # 特征向量，概率分布输出
        self.embedding_head = self.embedding_head.to(x.device)
        features = self._extract_features(x)

        embeddings = nn.functional.normalize(features, p=2, dim=1)
        # embeddings = self.embedding_head(features)
        logits = self.model(x)

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
        # print(positives.shape, negatives.shape, embeddings.shape)

        # 总损失 = 交叉熵 + 三元损失
        t_loss = self.triplet_loss(embeddings, positives, negatives)
        task_loss = self.cross_entropy_loss(logits, targets)
        
        # print("dist_pos", (embeddings - positives).norm(dim=1).mean().item())
        # print("dist_neg", (embeddings - negatives).norm(dim=1).mean().item())

        return t_loss + task_loss, logits

def get_mask(batch_shape):
    """
    加速正负样本的查找
    """
    classes_num, embedding_num = batch_shape
    batch_size = classes_num * embedding_num
    negative_mask, positive_mask = torch.full(
        (batch_size, batch_size), False), torch.full((batch_size, batch_size), True)
    for s in range(0, batch_size, embedding_num):
        for i in range(embedding_num):
            for j in range(embedding_num):
                negative_mask[s + i][s + j] = True 
    for s in range(0, batch_size, embedding_num):
        for i in range(embedding_num):
            for j in range(embedding_num):
                positive_mask[s + i][s + j] = False
    return positive_mask, negative_mask

if __name__ == '__main__':
    pm, nm = get_mask((2, 2))
    print(pm)
    print(nm)