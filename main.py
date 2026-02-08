"""
主训练
"""
import logging
import os
import tomllib

import torch
from torch import logit, nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models as tv_models
from torchvision import transforms
import timm

from losses import TripletLoss 
from losses.kd_loss_new import KDLoss

from dataset import build_dataset
from train_func import train_one_epoch, val_one_epoch
# from timm.optim import create_optimizer

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


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

    # build dataloader
    if not args.kd and not args.baseline:
        dataset_train = build_dataset(args=args)
    else:
        dataset_train = datasets.ImageFolder(
            root=args.data_path + args.dataset + '/train',
            transform=data_transform
        )

    dataset_val = datasets.ImageFolder(
        root=args.data_path + args.dataset + '/val',
        transform=data_transform
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
        if args.baseline_model.startswith("squeezenet"):
            if args.baseline_model == "squeezenet1_1":
                model = tv_models.squeezenet1_1(pretrained=True)
            elif args.baseline_model == "squeezenet1_0":
                model = tv_models.squeezenet1_0(pretrained=True)
            else:
                raise ValueError(f"Unsupported SqueezeNet variant: {args.baseline_model}")
            model.classifier[1] = nn.Conv2d(512, args.nb_classes, kernel_size=1)
            model.num_classes = args.nb_classes
            model = model.to(device)
        elif args.baseline_model.startswith("shufflenet_v2_"):
            if args.baseline_model == "shufflenet_v2_x1_0":
                model = tv_models.shufflenet_v2_x1_0(pretrained=True)
            elif args.baseline_model == "shufflenet_v2_x0_5":
                model = tv_models.shufflenet_v2_x0_5(pretrained=True)
            elif args.baseline_model == "shufflenet_v2_x1_5":
                model = tv_models.shufflenet_v2_x1_5(pretrained=True)
            elif args.baseline_model == "shufflenet_v2_x2_0":
                model = tv_models.shufflenet_v2_x2_0(pretrained=True)
            else:
                raise ValueError(f"Unsupported ShuffleNet variant: {args.baseline_model}")
            model.fc = nn.Linear(model.fc.in_features, args.nb_classes)
            model = model.to(device)
        else:
            model = timm.create_model(args.baseline_model, pretrained=True, num_classes=args.nb_classes).to(device)
        optimizer = optim.Adam(model.parameters(), args.lr)
        criterion = nn.CrossEntropyLoss()

        def baseline_loss_fn(x, targets):
            logits = model(x)
            loss = criterion(logits, targets)
            return loss, logits

        loss_fn = baseline_loss_fn

    teacher_model = timm.create_model(args.teacher_model, pretrained=True, num_classes=args.nb_classes).to(device)
    if not args.baseline and not args.kd:
        model = teacher_model
        optimizer = optim.Adam(model.parameters(), args.lr)
        loss_fn = TripletLoss(model=model)
    
    if not args.baseline and args.kd:
        teacher_model.load_state_dict(torch.load(f'models_para/resnet34.a1_in1k_{args.dataset}_kd.pth'), strict=False)
        student_model = timm.create_model(args.student_model, pretrained=True, num_classes=args.nb_classes).to(device)
        optimizer = optim.Adam(student_model.parameters(), args.lr)
        loss_fn = KDLoss(student=(args.student_model, student_model), teacher=(
            args.teacher_model, teacher_model), base_criterion=nn.CrossEntropyLoss(), cls_loss_w=1, feat_loss_w=1)
        model = student_model

    # 计算FLOPs
    from thop import profile
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size).to(device)
    model_flops, model_params = profile(model, inputs=(dummy_input,))
    print(f"Model FLOPs: {model_flops/1e9:.2f} GFLOPs, Student PARAMS: {model_params/1e6:.2f} M")

    # 优化器
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print(f'Strat training for {args.epochs} epochs')
    # 开始训练
    import time
    start_time = time.time()

    t_loss_vec, t_acc_vec, v_loss_vec, v_acc_vec = [], [], [], []

    # 训练
    min_loss = 100
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
        if v_loss < min_loss:
            min_loss = v_loss
            print("save model")
            torch.save(model.state_dict(),
                       f"models_para/{model_name}_{args.dataset}_{'baseline' if args.baseline else 'kd'}.pth")

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

    # 保存每个epoch的指标到txt
    os.makedirs(args.output_dir, exist_ok=True)
    tag = "baseline" if args.baseline else ("kd" if args.kd else "teacher")
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


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
