"""
单次训练
"""
from typing import Iterable

import torch
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from losses import *

# progress = Progress(
    # # [动态文本列] 显示 Epoch 信息，使用 {task.description} 占位
    # TextColumn("[bold white]{task.description}", justify="right"),

    # # [进度条列] 设定轨道为暗白(灰)，进度为纯白
    # BarColumn(
        # bar_width=40,
        # style="dim white",       # 轨道颜色
        # complete_style="white",  # 完成部分颜色
        # finished_style="white"   # 完成后的颜色
    # ),

    # # [百分比列] 强制白色
    # TaskProgressColumn(style="white"),

    # # [时间列] 自定义格式：只显示秒数，例如 "12.5s"
    # # 这里直接调用 task.elapsed 获取耗时
    # TextColumn("[white]{task.elapsed:.1f}s"),
# )


def train_one_epoch(model: torch.nn.Module,
                    loss_fn: Loss,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    args=None
                    ):
    """一个epoch训练"""
    model.train()
    # header = 'Epoch: [{}]'.format(epoch)

    epoch_loss = 0.0
    epoch_acc = 0.0
    
    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        x = data.shape[0]
        if x != 32:
            print(targets)
            break
        continue
        loss, logits = loss_fn(data, targets)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=1) == targets).float().mean()
        epoch_acc += acc.cpu().item() / len(data_loader)
        epoch_loss += loss.cpu().item() / len(data_loader)

    print(
        f"Epoch {epoch + 1}: train_loss : {epoch_loss:.4f} - train_acc: {epoch_acc:.4f}\n")
    return epoch_loss, epoch_acc


def val_one_epoch(data_loader, model, device, epoch):
    """验证一个epoch"""
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model.forward(data)

        loss = criterion(outputs, targets)
        # acc1 = accuracy(outputs, targets, topk=(1,))[0]

        acc = (outputs.argmax(dim=1) == targets).float().mean()
        epoch_loss += loss.cpu().item() / len(data_loader)
        epoch_acc += acc.cpu().item() / len(data_loader)
    print(
        f"Epoch {epoch + 1}: val_loss : {epoch_loss:.4f} - val_acc: {epoch_acc:.4f}\n\n")

    return epoch_loss, epoch_acc