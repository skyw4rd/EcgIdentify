import torch
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from typing import Tuple, Dict, List
from sklearn.metrics import classification_report, confusion_matrix

@torch.no_grad()
def eval_sample_level_metrics(
    model,
    dataset_val,
    device: str = "cuda",
    num_worker: int = 4,
) -> Tuple[Dict[str, float], torch.Tensor, List[int]]:
    """
    样本级（sample-level）评估：直接对验证集每个样本的预测结果计算 ACC / Precision_macro / Recall_macro / F1_macro。

    返回：
      metrics: dict
      conf_mat: torch.Tensor [K,K]          # 仅包含验证集中出现的类别
      class_ids: list[int]                  # conf_mat 行列对应的真实类别id列表（排序）
    """
    model.eval()
    model.to(device)

    dataloader_test = DataLoader(
        dataset_val,
        batch_size=64,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=False,
    )

    y_true_list = []
    y_pred_list = []

    for batch in dataloader_test:
        x, y = batch[0], batch[1]

        # 强制为 long，避免 float 标签导致类别截断
        if torch.is_tensor(y):
            y = y.to(device, dtype=torch.long, non_blocking=True)
        else:
            y = torch.tensor(y, device=device, dtype=torch.long)

        x = x.to(device, non_blocking=True)

        logits = model(x)
        pred = torch.argmax(logits, dim=1).to(torch.long)

        y_true_list.append(y.detach())
        y_pred_list.append(pred.detach())

    if len(y_true_list) == 0:
        raise ValueError("Validation set is empty; cannot compute sample-level metrics.")

    y_true = torch.cat(y_true_list).cpu().view(-1).long()
    y_pred = torch.cat(y_pred_list).cpu().view(-1).long()

    # 统一标签集合：真实 ∪ 预测，确保混淆矩阵和 report 对齐
    class_ids = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    # sklearn 计算报告和混淆矩阵
    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(y_true, y_pred, labels=class_ids)

    metrics = {
        "ACC_sample_level": report.get("accuracy", 0.0),
        "P_macro_sample_level": report.get("macro avg", {}).get("precision", 0.0),
        "R_macro_sample_level": report.get("macro avg", {}).get("recall", 0.0),
        "F1_macro_sample_level": report.get("macro avg", {}).get("f1-score", 0.0),
        "num_classes_in_val": len(class_ids),
        "num_samples_in_val": int(conf_mat.sum()),
    }

    # 转成 torch.Tensor 与原接口兼容
    conf_mat = torch.tensor(conf_mat, dtype=torch.long)

    return metrics, conf_mat, class_ids

@torch.no_grad()
def eval_class_level_metrics(
    model,
    dataset_val,
    device="cuda",
    num_workers=4,
    target_success_ratio: float = 0.9,
):
    """
    类别级（identity-level）评估：
    - 对每个真实类别，统计其样本被识别为该类别（target=true_cls）的比例
    - 当比例 >= target_success_ratio（默认0.9）时，该类别判定为识别成功（pred=true_cls）
    - 否则判定为失败（pred 设为最常见的非 target 类）
    - 形成“类别级样本”（每类1条记录），计算 ACC / Precision_macro / Recall_macro / F1_macro

    返回：
      metrics: dict
      vote_pred_per_class: dict[int->int]  # 每个真实类的类别级最终预测结果
      conf_mat: torch.Tensor [K,K]         # 仅包含出现过的类别
      class_ids: list[int]                 # conf_mat 行列对应的真实类别id列表（排序）
    """
    model.eval()
    model.to(device)

    dataloader_test = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # 1) 收集：每个真实类别 -> 所有样本的预测类别列表
    preds_by_true = defaultdict(list)

    for batch in dataloader_test:
        # 兼容 batch 返回 (x,y) 或 (x,y,...) 的情况
        x, y = batch[0], batch[1]
        y = int(y.item()) if torch.is_tensor(y) else int(y)

        x = x.to(device, non_blocking=True)

        logits = model(x)  # [1, C]
        pred = int(torch.argmax(logits, dim=1).item())

        preds_by_true[y].append(pred)

    # 2) 对每个真实类别按 target 命中率阈值做类别级判定
    vote_pred_per_class = {}
    target_hit_ratio_per_class = {}
    for true_cls, pred_list in preds_by_true.items():
        total = len(pred_list)
        target_hits = sum(1 for p in pred_list if p == true_cls)
        target_ratio = (target_hits / total) if total > 0 else 0.0
        target_hit_ratio_per_class[true_cls] = target_ratio

        if target_ratio >= target_success_ratio:
            vote_pred = true_cls
        else:
            non_target_cnt = Counter(p for p in pred_list if p != true_cls)
            if len(non_target_cnt) == 0:
                # 极端兜底：没有非 target 预测时（理论上不应出现），保持 true_cls。
                vote_pred = true_cls
            else:
                max_count = max(non_target_cnt.values())
                tied = [c for c, v in non_target_cnt.items() if v == max_count]
                vote_pred = min(tied)

        vote_pred_per_class[true_cls] = vote_pred

    # 3) 构造“类别级”混淆矩阵
    #    只对验证集中实际出现过的类别计算（K = len(preds_by_true)）
    class_ids = sorted(preds_by_true.keys())
    idx = {cid: i for i, cid in enumerate(class_ids)}
    K = len(class_ids)

    conf_mat = torch.zeros((K, K), dtype=torch.long)
    # 行：真实类别；列：预测类别（投票结果）
    for true_cls in class_ids:
        pred_cls = vote_pred_per_class[true_cls]

        # 注意：pred_cls 可能是验证集中未出现过的类别。
        # 为保证每个 true_cls 都能计入一次，若不在 class_ids，则映射为一个确定的“错误类别”。
        if pred_cls not in idx:
            wrong_candidates = [cid for cid in class_ids if cid != true_cls]
            pred_cls = wrong_candidates[0] if len(wrong_candidates) > 0 else true_cls

        conf_mat[idx[true_cls], idx[pred_cls]] += 1

    # 4) 类别级 ACC（= 对角线之和 / 类别数）
    correct = conf_mat.diag().sum().item()
    total = conf_mat.sum().item()  # 理论上等于K
    acc = correct / total if total > 0 else 0.0

    # 5) 类别级 Precision/Recall/F1（macro）
    # 对每个类别 i：
    # TP = conf[i,i]
    # FP = sum(conf[:,i]) - TP
    # FN = sum(conf[i,:]) - TP
    tp = conf_mat.diag().to(torch.float32)
    fp = conf_mat.sum(dim=0).to(torch.float32) - tp
    fn = conf_mat.sum(dim=1).to(torch.float32) - tp

    eps = 1e-12
    precision_per_class = tp / (tp + fp + eps)
    recall_per_class = tp / (tp + fn + eps)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)

    precision_macro = precision_per_class.mean().item() if K > 0 else 0.0
    recall_macro = recall_per_class.mean().item() if K > 0 else 0.0
    f1_macro = f1_per_class.mean().item() if K > 0 else 0.0

    metrics = {
        "ACC_class_level": acc,
        "P_macro_class_level": precision_macro,
        "R_macro_class_level": recall_macro,
        "F1_macro_class_level": f1_macro,
        "num_classes_in_val": K,
        "target_success_ratio_threshold": target_success_ratio,
        "target_success_rate_class_level": (
            sum(1 for cid in class_ids if target_hit_ratio_per_class.get(cid, 0.0) >= target_success_ratio) / K
            if K > 0 else 0.0
        ),
    }
    return metrics, vote_pred_per_class, conf_mat, class_ids


# 你原来的函数可以这样包一下
def test_one_fold(model, dataset_val, device="cuda"):
    metrics, vote_pred_per_class , conf_mat, class_ids = eval_class_level_metrics(
        model=model,
        dataset_val=dataset_val,
        device=device,
    )
    return metrics
