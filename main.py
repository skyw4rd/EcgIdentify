"""
主训练
"""
import argparse
import logging

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import timm

from losses import TripletLoss 
from losses.kd_loss_new import KDLoss
import matplotlib.pyplot as plt

from dataset import build_dataset
from teacher_model import create_teacher_model
from train_func import train_one_epoch, val_one_epoch
# from timm.optim import create_optimizer

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_args():
    # 超参数
    parser = argparse.ArgumentParser(
        'teacher model ecg training script', add_help=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--batch-classes-num', default=8, type=int)
    parser.add_argument('--epochs', default=20, type=int)

    # 模型参数
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--teacher-model', default='resnet34.a1_in1k', type=str)
    parser.add_argument('--student-model',
                        default='deit_small_patch16_224', type=str)

    # mobilenetv3_small_100.lamb_in1k
    # 优化器参数
    parser.add_argument('--opt', default='adamw', type=str)

    # 学习率参数
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--step-size', default=10, type=int)
    parser.add_argument('--gamma', default=0.5, type=float)

    # 是否开启蒸馏
    parser.add_argument('--kd', default=False, type=bool)

    # 数据集参数
    # -- dataset/
    #    -- train/
    #    -- val/
    #    -- test/
    parser.add_argument('--dataset', default='ptb', type=str)
    parser.add_argument('--data-path', default='data/', type=str)
    parser.add_argument('--output-dir', default='runs', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    return parser.parse_args()


def main(args):
    """主函数"""
    print(args)

    device = torch.device(args.device)
    model_name = args.teacher_model if not args.kd else args.student_model

    data_transform = transforms.Compose([
        transforms.Resize([args.input_size, args.input_size]),
        transforms.ToTensor(),
    ])

    # build dataloader
    if not args.kd:
        dataset_train, args.nb_classes = build_dataset(args=args)
    else:
        dataset_train = datasets.ImageFolder(
            root=args.data_path + args.dataset + '/train',
            transform=data_transform
        )
        args.nb_classes = 289

    dataset_val = datasets.ImageFolder(
        root=args.data_path + args.dataset + '/val',
        transform=data_transform
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
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
    teacher_model = create_teacher_model(args=args).to(device)

    # 损失函数
    if not args.kd:
        optimizer = optim.Adam(teacher_model.parameters(), args.lr)
        loss_fn = TripletLoss(model=teacher_model)
        model = teacher_model
    else:
        teacher_model.load_state_dict(torch.load(
            'models_para/resnet34.a1_in1k_ptb.pth'))
        student_model = timm.create_model(args.student_model, pretrained=True, num_classes=args.nb_classes).to(device)
        optimizer = optim.Adam(student_model.parameters(), args.lr)
        loss_fn = KDLoss(student=(args.student_model, student_model), teacher=(
            args.teacher_model, teacher_model), base_criterion=nn.CrossEntropyLoss())
        model = student_model

    # 优化器
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    print(f'Strat training for {args.epochs} epochs')

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
                       f"models_para/{model_name}_{args.dataset}.pth")

        # 保存loss acc
        t_loss_vec.append(t_loss)
        t_acc_vec.append(t_acc)
        v_loss_vec.append(v_loss)
        v_acc_vec.append(v_acc)

        # 更新Dataset
        if not args.kd:
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
    plt.title(f'{model_name} acc')
    plt.legend()
    plt.grid(ls='--')
    plt.savefig(f'{args.output_dir}/{model_name}_{args.dataset}_acc.png')

    # 损失函数图
    plt.figure()
    plt.plot(list(range(xr)), t_loss_vec, label='train')
    plt.plot(list(range(xr)), v_loss_vec, label='valid', ls='--')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks()
    plt.yticks()
    plt.title(f'{model_name} loss')
    plt.legend()
    plt.grid(ls='--')
    plt.savefig(f'{args.output_dir}/{model_name}_{args.dataset}_loss.png')


if __name__ == '__main__':
    args = get_args()
    main(args)
