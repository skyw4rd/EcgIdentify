"""
增加教师模型的降维和特征
"""
from ast import mod
import torch.nn.functional as F
from torch import nn
import timm

# 一个batch的构成32个人一个人1张图


class TeacherModel(nn.Module):
    """
    教师模型
    """
    def __init__(self, model_name, num_classes=90, target_dim=128, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, num_classes=num_classes, pretrained=pretrained)
        self.original_fc = self.backbone.fc
        original_dim = self.original_fc.in_features
        self.backbone.fc = nn.Identity()

        self.dim_reduction = nn.Sequential(
            nn.Linear(original_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, target_dim),
            nn.BatchNorm1d(target_dim),
            nn.ReLU(inplace=True)
        )

    def forward_embeddings(self, data):
        """
        输出embedding
        """
        embeddings = self.backbone(data)
        # 降维，归一化
        reduced_embeddings = F.normalize(
            self.dim_reduction(embeddings), p=2, dim=1)
        return reduced_embeddings

    def forward(self, data):
        """
        输出logit
        """
        embeddings = self.backbone(data)
        return self.original_fc(embeddings)


def create_test_teacher_model(model_name, nb_classes):
    ecg_model = TeacherModel(
        model_name=model_name,
        pretrained=True,
        num_classes=nb_classes
    )

    return ecg_model

def create_teacher_model(args):
    """
    创建教师模型
    """
    ecg_model = TeacherModel(
        model_name=args.teacher_model,
        pretrained=True,
        num_classes=args.nb_classes
    )

    return ecg_model
