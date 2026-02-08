"""
单次训练
"""
from typing import Iterable

try:
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    _RICH_AVAILABLE = True
except Exception:
    _RICH_AVAILABLE = False

import torch
from losses import *


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

    if _RICH_AVAILABLE:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        with progress:
            task = progress.add_task(f"Epoch {epoch + 1}", total=len(data_loader))
            for data, targets in data_loader:
                data, targets = data.to(device), targets.to(device)
                loss, logits = loss_fn(data, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = (logits.argmax(dim=1) == targets).float().mean()
                epoch_acc += acc.cpu().item() / len(data_loader)
                epoch_loss += loss.cpu().item() / len(data_loader)
                progress.advance(task)
    else:
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
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
