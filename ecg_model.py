import torch.nn.functional as F
import torch.nn as nn
import timm

# 一个batch的构成32个人一个人1张图

class EcgModel(nn.Module):
    def __init__(self, model_name, num_classes=90, target_dim=128, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained)

        self.original_fc = self.backbone.head.fc
        original_dim = self.original_fc.in_features

        self.backbone.head.fc = nn.Identity()

        self.dim_reduction = nn.Sequential(
            nn.Linear(original_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(256, target_dim),
            nn.BatchNorm1d(target_dim),
            nn.ReLU(inplace=True)
        )

    # def get_embedding(self, data):
        # embeddings = self.base_model.head.global_pool(
            # self.base_model.forward_features(data)
        # )
        # return embeddings
    
    # def get_outputs(self, embeddings):
        # return self.base_model.head.fc(embeddings)
    def forward_embeddings(self, data):
        embeddings = self.backbone(data)
        # 降维，归一化
        reduced_embeddings = F.normalize(self.dim_reduction(embeddings), p=2, dim=1)
        return reduced_embeddings

    def forward(self, data):
        embeddings = self.backbone(data)
        return self.original_fc(embeddings)

def create_feature_ecg_model(args):
    model_features = timm.create_model(
                                    args.model,
                                    features_only=True,
                                    num_classes=args.nb_classes,
                                    pretrained=True,
                                    out_indices=[1, 2, 3, 4]).to('cuda')
    return model_features

def create_ecg_model(args):
    ecg_model = EcgModel(
        model_name=args.model,
        pretrained=True,
        num_classes=args.nb_classes
    )

    return ecg_model