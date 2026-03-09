import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowCNN(nn.Module):
    def __init__(self, num_classes=300, in_ch=3, emb_dim=256):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),             # 下采样，减计算
                nn.Dropout2d(0.1)            # 小数据集很有用
            )

        self.b1 = block(in_ch, 32)         # H,W /2
        self.b2 = block(32, 64)            # /4
        self.b3 = block(64, 128)           # /8

        self.gap = nn.AdaptiveAvgPool2d(1) # -> [B,128,1,1]
        self.feat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.cls = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.gap(x)
        x = self.feat(x)
        return self.cls(x)
