import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from losses import TripletPlusCe
import matplotlib.pyplot as plt

from ecg_dataset import build_dataset
from ecg_model import create_ecg_model
from train_func import train_one_epoch, val_one_epoch
# from timm.optim import create_optimizer


def get_args():
    # 超参数
    parser = argparse.ArgumentParser('teacher model ecg training script', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--batch_classes_num', default=8, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    
    # 模型参数
    parser.add_argument('--model', default='regnety_002', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    
    # 优化器参数
    parser.add_argument('--opt', default='adamw', type=str)

    # 学习率参数
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--gamma', default=0.5, type=float)
    
    # 数据集参数
    # -- dataset/
    #    -- train/
    #    -- val/
    #    -- test/
    parser.add_argument('--data-path', default='ecg_id_img/', type=str)
    parser.add_argument('--output-dir', default='', type=str)
    parser.add_argument('-device', default='cuda', type=str)

    return parser.parse_args()

def main(args : argparse.Namespace):
    print(args)
    device = torch.device(args.device)

    # 训练集和验证集 
    dataset_train, args.nb_classes = build_dataset(args=args)

    data_transform = transforms.Compose([
        transforms.Resize([args.input_size, args.input_size]),
        transforms.ToTensor(),
    ])
    dataset_val = datasets.ImageFolder(
        root=args.data_path + 'val', 
        transform=data_transform
    )
    
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
    
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
    
    # 创建模型
    print(f'Creating model: {args.model}')
    model = create_ecg_model(args=args)
    model.to(device)

    # 优化器 
    # optimizer = create_optimizer(args, model)
    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # 损失函数
    criterion = TripletPlusCe(
        triplet_loss=nn.TripletMarginLoss(margin=1.2, p=2),
        cross_entropy_loss=nn.CrossEntropyLoss()
    )
    print(f'Strat training for {args.epochs} epochs')
    start_time = time.time()
    max_acc = 0.0
    
    t_loss_vec, t_acc_vec, v_loss_vec, v_acc_vec = [], [], [], []

    # 训练
    for epoch in range(args.epochs): 
        t_loss, t_acc = train_one_epoch(
            model=model,
            criterion=criterion,
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
        
        # 更新Dataset
        dataset_train.set_samples()
        # 更新学习率
        scheduler.step()
    
    # 画图
    xr = args.epochs
    plt.figure()
    plt.plot(list(range(xr)), t_acc_vec, label='train')
    plt.plot(list(range(xr)), v_acc_vec, label='valid', ls='--')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.xticks()
    plt.yticks()
    plt.title(f'{args.model} acc')
    plt.legend()
    plt.grid(ls='--')
    plt.savefig(f'{args.output_dir}{args.model}_acc.png')
    plt.show()

    # 损失函数图
    plt.figure()
    plt.plot(list(range(xr)), t_loss_vec, label='train')
    plt.plot(list(range(xr)), v_loss_vec, label='valid', ls='--')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks()
    plt.yticks()
    plt.title(f'{args.model} loss')
    plt.legend()
    plt.grid(ls='--')
    plt.savefig(f'{args.output_dir}{args.model}_loss.png')

if __name__ == '__main__':
    args = get_args()
    main(args)