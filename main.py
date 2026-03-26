"""
主训练
"""
import logging
import os
import random
from glob import glob

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python <=3.10

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import models as tv_models
from torchvision import transforms
import timm

from losses import TripletLoss 
from losses.kd_loss_new import KDLoss

from dataset import build_dataset
from train_func import train_one_epoch, val_one_epoch
# from timm.optim import create_optimizer
from models.simple_cnn import ShallowCNN
from models.simple_vit import SimpleViT
from models.ecg_baselines import EDITHNet, ECGIoTNet, ECGXtractorNet
from test_func import test_one_fold

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


_TORCHVISION_BASELINE_FACTORIES = {
    "squeezenet1_0": tv_models.squeezenet1_0,
    "squeezenet1_1": tv_models.squeezenet1_1,
    "shufflenet_v2_x0_5": tv_models.shufflenet_v2_x0_5,
    "shufflenet_v2_x1_0": tv_models.shufflenet_v2_x1_0,
    "shufflenet_v2_x1_5": tv_models.shufflenet_v2_x1_5,
    "shufflenet_v2_x2_0": tv_models.shufflenet_v2_x2_0,
    "mobilenet_v3_small": tv_models.mobilenet_v3_small,
    "mobilenet_v3_large": tv_models.mobilenet_v3_large,
    "densenet121": tv_models.densenet121,
    "densenet161": tv_models.densenet161,
    "densenet169": tv_models.densenet169,
    "densenet201": tv_models.densenet201,
    "resnet18": tv_models.resnet18,
    "resnet34": tv_models.resnet34,
    "resnet50": tv_models.resnet50,
}


def _flatten_config(config):
    flat = {}
    if not isinstance(config, dict):
        return flat
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    for sub_key2, sub_value2 in sub_value.items():
                        flat[sub_key2] = sub_value2
                else:
                    flat[sub_key] = sub_value
        else:
            flat[key] = value
    return flat


def load_config_toml(path):
    if not path:
        return {}
    if not os.path.isfile(path):
        return {}
    with open(path, 'rb') as f:
        data = tomllib.load(f)
    return _flatten_config(data)


def get_args(config_path='config.toml'):
    # All hyperparameters come from TOML
    config_defaults = load_config_toml(config_path)
    if not config_defaults:
        raise FileNotFoundError(f'Config not found or empty: {config_path}')
    from types import SimpleNamespace
    return SimpleNamespace(**config_defaults)


def _resolve_cv_source_root(dataset_root):
    train_root = os.path.join(dataset_root, 'train')
    if os.path.isdir(train_root):
        return train_root
    return dataset_root


def _build_cv_path_splits(dataset_root, folds=5, fold_idx=0, seed=3407):
    if folds <= 1:
        raise ValueError(f'Invalid CV setup: folds={folds}')
    if not (0 <= fold_idx < folds):
        raise ValueError(f'fold_idx out of range: {fold_idx}, expected [0, {folds - 1}]')

    train_paths, val_paths = {}, {}
    source_root = _resolve_cv_source_root(dataset_root)
    class_names = sorted(entry.name for entry in os.scandir(source_root) if entry.is_dir())
    if not class_names:
        raise FileNotFoundError(f'No class folders found under: {source_root}')

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(source_root, cls_name)
        cls_files = []
        for root, _, fnames in os.walk(cls_dir, followlinks=True):
            for fname in sorted(fnames):
                cls_files.append(os.path.join(root, fname))

        if len(cls_files) < folds:
            raise ValueError(
                f'Class "{cls_name}" has {len(cls_files)} samples, fewer than folds={folds}.'
            )

        rng = random.Random(seed + cls_idx * 97)
        shuffled = list(cls_files)
        rng.shuffle(shuffled)

        fold_sizes = [len(shuffled) // folds] * folds
        for i in range(len(shuffled) % folds):
            fold_sizes[i] += 1

        boundaries = [0]
        for s in fold_sizes:
            boundaries.append(boundaries[-1] + s)
        start, end = boundaries[fold_idx], boundaries[fold_idx + 1]

        val_cls = shuffled[start:end]
        train_cls = shuffled[:start] + shuffled[end:]
        train_paths[cls_name] = train_cls
        val_paths[cls_name] = val_cls

    return train_paths, val_paths


def _build_train_val_for_cv(args, data_transform):
    dataset_root = os.path.join(args.data_path, args.dataset)
    folds = 5
    fold_idx = int(getattr(args, 'cv_fold_idx', 0))
    seed = int(getattr(args, 'cv_seed', 3407))
    source_root = _resolve_cv_source_root(dataset_root)

    train_paths, val_paths = _build_cv_path_splits(
        dataset_root=dataset_root,
        folds=folds,
        fold_idx=fold_idx,
        seed=seed
    )

    full_dataset = datasets.ImageFolder(root=source_root, transform=data_transform)
    path_to_index = {p: i for i, (p, _) in enumerate(full_dataset.samples)}
    train_indices, val_indices = [], []

    for cls_name in full_dataset.classes:
        train_indices.extend(path_to_index[p] for p in train_paths.get(cls_name, []))
        val_indices.extend(path_to_index[p] for p in val_paths.get(cls_name, []))

    val_dataset = Subset(full_dataset, val_indices)

    if not args.kd and not args.baseline:
        train_dataset = build_dataset(
            args=args,
            root=source_root,
            samples_dict=train_paths
        )
    else:
        train_dataset = Subset(full_dataset, train_indices)

    cv_tag = f"f{fold_idx + 1}"
    print(
        f'Using {folds}-fold CV for {args.dataset}: '
        f'fold={fold_idx + 1}/{folds}, '
        f'train={len(train_indices)}, val={len(val_indices)}'
    )
    return train_dataset, val_dataset, cv_tag


def _resolve_teacher_checkpoint(args):
    """Resolve fold-specific teacher checkpoint path for KD runs."""
    env_path = os.getenv('TEACHER_CKPT_PATH', '').strip()
    if env_path:
        if not os.path.isfile(env_path):
            raise FileNotFoundError(f'TEACHER_CKPT_PATH does not exist: {env_path}')
        return env_path

    fold_idx = int(getattr(args, 'cv_fold_idx', 0))
    fold_human = fold_idx + 1
    teacher_dir = os.path.join('models_para', args.dataset, 'resnet34')
    teacher_name = str(getattr(args, 'teacher_model', '')).strip()

    bsl_or_kd = 'baseline' if args.scheme == 'B' else 'kd'
    expected = [
        os.path.join(teacher_dir, f'{teacher_name}_{args.dataset}_f{fold_human}_{bsl_or_kd}.pth'),
        os.path.join(teacher_dir, f'{teacher_name}_{args.dataset}_f{fold_human}.pth'),
        os.path.join(teacher_dir, f'triplet_f{fold_human}.pth'),
    ]
    for candidate in expected:
        if os.path.isfile(candidate):
            return candidate

    matched = sorted(glob(os.path.join(teacher_dir, f'*_f{fold_human}_*.pth')))
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        raise FileNotFoundError(
            f'Multiple teacher checkpoints found for fold f{fold_human} under {teacher_dir}: {matched}'
        )

    raise FileNotFoundError(
        f'No teacher checkpoint found for fold f{fold_human} under {teacher_dir}. '
        f'Tried: {expected}'
    )


def _replace_classifier_head(model, num_classes):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if hasattr(model, "classifier"):
        classifier = model.classifier

        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
            return model

        if isinstance(classifier, nn.Sequential) and len(classifier) > 0:
            last = classifier[-1]
            if isinstance(last, nn.Linear):
                classifier[-1] = nn.Linear(last.in_features, num_classes)
                return model
            if isinstance(last, nn.Conv2d):
                classifier[-1] = nn.Conv2d(last.in_channels, num_classes, kernel_size=1)
                model.num_classes = num_classes
                return model

    raise ValueError(f"Unsupported classifier head for model type: {type(model)}")


def _create_baseline_model(args, device):
    model_name = str(args.baseline_model).strip()
    model_key = model_name.lower()
    pretrained = bool(getattr(args, "pretrained", True))

    if model_key.startswith("shallow_cnn"):
        return ShallowCNN(num_classes=args.nb_classes).to(device)

    if model_key in {"simple_vit", "vit_encoder"}:
        model = SimpleViT(
            image_size=int(args.input_size),
            patch_size=int(getattr(args, "vit_patch_size", 16)),
            in_chans=3,
            num_classes=args.nb_classes,
            embed_dim=int(getattr(args, "vit_embed_dim", 256)),
            depth=int(getattr(args, "vit_depth", 4)),
            num_heads=int(getattr(args, "vit_num_heads", 8)),
            mlp_ratio=float(getattr(args, "vit_mlp_ratio", 4.0)),
            dropout=float(getattr(args, "vit_dropout", 0.1)),
        )
        return model.to(device)

    if model_key in {"edith", "edith_cnn"}:
        model = EDITHNet(
            in_chans=3,
            num_classes=args.nb_classes,
            emb_dim=int(getattr(args, "edith_emb_dim", 256)),
        )
        return model.to(device)

    if model_key in {"ecgiot", "ecg_iot", "ecgiot_cnn"}:
        model = ECGIoTNet(
            in_chans=3,
            num_classes=args.nb_classes,
            width_mult=float(getattr(args, "ecgiot_width_mult", 1.0)),
        )
        return model.to(device)

    if model_key in {"ecgxtractor", "ecgxtractor_cnn"}:
        return ECGXtractorNet(in_chans=3, num_classes=args.nb_classes).to(device)

    if model_key in _TORCHVISION_BASELINE_FACTORIES:
        model = _TORCHVISION_BASELINE_FACTORIES[model_key](pretrained=pretrained)
        model = _replace_classifier_head(model, args.nb_classes)
        return model.to(device)

    return timm.create_model(
        model_name, pretrained=pretrained, num_classes=args.nb_classes
    ).to(device)


def main(args):
    """主函数"""
    print(args)

    device = torch.device(args.device)
    
    if args.baseline:
        model_name = args.baseline_model
    if not args.baseline:
        model_name = args.teacher_model if not args.kd else args.student_model


    data_transform = transforms.Compose([
        transforms.Resize([args.input_size, args.input_size]),
        transforms.ToTensor(),
    ])

    dataset_train, dataset_val, cv_tag = _build_train_val_for_cv(
        args=args, data_transform=data_transform
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False if not args.kd and not args.baseline else True,
        num_workers=4,
        pin_memory=False
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )

    # 创建教师模型
    print(f'Creating teacher model: {args.teacher_model}, Dataset: {args.dataset}')
    
    # 损失函数
    if args.baseline:
        model = _create_baseline_model(args=args, device=device)
        optimizer = optim.Adam(model.parameters(), args.lr)
        criterion = nn.CrossEntropyLoss()

        def baseline_loss_fn(x, targets):
            logits = model(x)
            loss = criterion(logits, targets)
            return loss, logits

        loss_fn = baseline_loss_fn

    teacher_pretrained = bool(getattr(args, "teacher_pretrained", False))
    student_pretrained = bool(getattr(args, "student_pretrained", True))

    teacher_model = timm.create_model(
        args.teacher_model,
        pretrained=teacher_pretrained,
        num_classes=args.nb_classes
    ).to(device)
    if not args.baseline and not args.kd:
        model = teacher_model
        optimizer = optim.Adam(model.parameters(), args.lr)
        loss_fn = TripletLoss(model=model)
    
    if not args.baseline and args.kd:
        teacher_ckpt_path = _resolve_teacher_checkpoint(args)
        print(f'Loading teacher checkpoint: {teacher_ckpt_path}')
        teacher_model.load_state_dict(torch.load(teacher_ckpt_path, map_location=device), strict=False)
        student_model = timm.create_model(
            args.student_model,
            pretrained=student_pretrained,
            num_classes=args.nb_classes
        ).to(device)
        optimizer = optim.Adam(student_model.parameters(), args.lr)
        loss_fn = KDLoss(student=(args.student_model, student_model), teacher=(
            args.teacher_model, teacher_model), base_criterion=nn.CrossEntropyLoss(), cls_loss_w=0.3, feat_loss_w=1)
        model = student_model

    # 计算FLOPs
    # from thop import profile
    # dummy_input = torch.randn(1, 3, args.input_size, args.input_size).to(device)
    # model_flops, model_params = profile(model, inputs=(dummy_input,))
    # print(f"Model FLOPs: {model_flops/1e9:.2f} GFLOPs, Student PARAMS: {model_params/1e6:.2f} M")

    # 优化器
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print(f'Strat training for {args.epochs} epochs')
    # 开始训练
    import time
    start_time = time.time()

    t_loss_vec, t_acc_vec, v_loss_vec, v_acc_vec = [], [], [], []

    # 训练
    ma_acc = 0
    model_suffix = f"_{cv_tag}" if cv_tag else ""
    model_path = f"models_para/{model_name}_{args.dataset}{model_suffix}_{'baseline' if args.baseline else 'kd'}.pth"
    for epoch in range(args.epochs):
        t_loss, t_acc = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=dataloader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args
        )
        v_loss, v_acc = val_one_epoch(
            data_loader=dataloader_val,
            model=model,
            device=device,
            epoch=epoch,
        )
        
        # 保存最优模型
        if v_acc > ma_acc: 
            ma_acc = v_acc
            print("save model")
            torch.save(model.state_dict(), model_path)
        # 保存loss acc
        t_loss_vec.append(t_loss)
        t_acc_vec.append(t_acc)
        v_loss_vec.append(v_loss)
        v_acc_vec.append(v_acc)

        # 更新Dataset
        if not args.kd and not args.baseline:
            dataset_train.set_samples()
        # 更新学习率
        scheduler.step()
    
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f}s")

    model.load_state_dict(torch.load(model_path), strict=False)
    metric = test_one_fold(model, dataset_val=dataset_val)
    print(metric)

    # 保存每个epoch的指标到txt
    os.makedirs(args.output_dir, exist_ok=True)
    tag = "baseline" if args.baseline else ("kd" if args.kd else "teacher")
    if cv_tag:
        tag = f"{tag}_{cv_tag}"
    metrics_path = os.path.join(
        args.output_dir,
        f"{model_name}_{args.dataset}_{tag}_metrics.txt"
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc\n")
        for i in range(args.epochs):
            f.write(
                f"{i + 1}\t"
                f"{t_loss_vec[i]:.6f}\t{t_acc_vec[i]:.6f}\t"
                f"{v_loss_vec[i]:.6f}\t{v_acc_vec[i]:.6f}\n"
            )
        f.write("\n[test_metric]\n")
        for metric_name, metric_value in metric.items():
            if isinstance(metric_value, float):
                f.write(f"{metric_name}\t{metric_value:.6f}\n")
            else:
                f.write(f"{metric_name}\t{metric_value}\n")

if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
