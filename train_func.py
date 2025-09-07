import torch

from typing import Iterable, Optional
from ecg_dataset import EcgImage
from losses import TripletPlusCe

from timm.utils import accuracy

def get_mask(batch_shape):
    classes_num, embedding_num = batch_shape
    batch_size = classes_num * embedding_num
    positive_mask, negative_mask = torch.full((batch_size, batch_size), True), torch.full((batch_size, batch_size), False)
    for s in range(0, batch_size, embedding_num):
        for i in range(embedding_num):
            for j in range(embedding_num):
                if i != j:
                    positive_mask[s + i][s + j] = False
    for s in range (0, batch_size, embedding_num):
        for i in range(embedding_num):
            for j in range(embedding_num):
                negative_mask[s + i][s + j] = True
    return positive_mask, negative_mask

def train_one_epoch(model: torch.nn.Module,
                    criterion: TripletPlusCe,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    args=None
                    ):
    model.train()
    # header = 'Epoch: [{}]'.format(epoch)

    epoch_loss = 0.0
    epoch_acc = 0.0

    for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            embeddings = model.forward_embeddings(data)
            outputs = model.forward(data)
            # outputs = ecg_model.get_outputs(embeddings)
            # embeddings = embeddings.reshape(8, 4, 368)
            # print(embeddings)
            distance_matrix = torch.cdist(embeddings, embeddings, p=2)
            # print(distance_matrix)

            # 生成掩码
            positive_mask, negative_mask = get_mask((embeddings.shape[0] // 4, 4))
            # print(positive_mask)
            # print(negative_mask)
            # 正样本与锚点的距离掩码
            pos_masked_matrix = distance_matrix.clone()
            pos_masked_matrix[positive_mask] = float('-inf')

            # 负样本与锚点的距离掩码
            neg_masked_matrix = distance_matrix.clone()
            neg_masked_matrix[negative_mask] = float('inf')

            # 32个锚点的对应的正样本
            _, hardest_positive_idxs = torch.max(pos_masked_matrix, dim=1)
            # print(hardest_positive_idxs)
            positives = embeddings[hardest_positive_idxs]

            # 32个锚点对应的负样本
            _, hardest_negative_idxs = torch.min(neg_masked_matrix, dim=1)
            # print(hardest_negative_idxs)
            negatives = embeddings[hardest_negative_idxs]
        
            # 总损失 = 交叉熵 + 三元损失
            loss = criterion(embeddings, positives, negatives, outputs, targets)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # preds = outputs.argmax(dim=1)
            # error_mask = (preds != label)
            # error_indices = torch.nonzero(error_mask).squeeze()
            # # 提取错误样本的真实标签和预测标签
            # error_true_labels = label[error_mask]
            # error_pred_labels = preds[error_mask]
            # for idx, true, pred in zip(error_indices, error_true_labels, error_pred_labels):
                # data_set.priority_class_counters[true.item()][pred.item()] += 1
            acc = (outputs.argmax(dim=1) == targets).float().mean()
            epoch_acc += acc.cpu().item() / len(data_loader)
            epoch_loss += loss.cpu().item() / len(data_loader)

    print(f"Epoch {epoch + 1}: train_loss : {epoch_loss:.4f} - train_acc: {epoch_acc:.4f}\n")

    return epoch_loss, epoch_acc
    
def val_one_epoch(data_loader, model, device, epoch):
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
    print(f"Epoch {epoch + 1}: val_loss : {epoch_loss:.4f} - val_acc: {epoch_acc:.4f}\n\n")

    return epoch_loss, epoch_acc