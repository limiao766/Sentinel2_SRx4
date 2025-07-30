import sys
sys.path.append('/root')
sys.path.append('/root/SR')

import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import argparse
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import cv2
import csv
import datetime
import glob

from SR.SR_x4.dataset import NYUDDataset, readTif
from SR.SR_x4.rrdbnet_arch import RRDBNet, UNetDiscriminatorSN, USMSharp
from SR.SR_x4.srloss import PerceptualLoss, GANLoss

def parse_args():
    parser = argparse.ArgumentParser(description='训练超分辨率GAN网络')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/dataset', help='数据集根目录')
    parser.add_argument('--train_dir', type=str, default='train', help='训练集目录')
    parser.add_argument('--val_dir', type=str, default='val', help='验证集目录')
    
    parser.add_argument('--lr_dir', type=str, default='lr', help='低分辨率图像目录')
    parser.add_argument('--hr_dir', type=str, default='hr', help='高分辨率图像目录')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    
    # 模型相关参数
    parser.add_argument('--pretrain_g_path', type=str, default='/root/SR/SR_x4/pretrained/RealESRGAN_x4plus.pth', help='生成器预训练模型路径')
    parser.add_argument('--pretrain_d_path', type=str, default='/root/SR/SR_x4/pretrained/RealESRGAN_x4plus_netD.pth', help='判别器预训练模型路径')
    parser.add_argument('--num_feat', type=int, default=64, help='特征通道数')
    parser.add_argument('--num_block', type=int, default=23, help='RRDB块数量')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=12, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='生成器学习率')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='判别器学习率')
    parser.add_argument('--weight_decay', type=float, default=0, help='权重衰减')
    parser.add_argument('--lambda_pix', type=float, default=1.0, help='像素损失权重')
    parser.add_argument('--lambda_perceptual', type=float, default=1.0, help='感知损失权重')
    parser.add_argument('--lambda_gan', type=float, default=0.1, help='GAN损失权重')
    
    # 设备和保存相关参数
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/model/sr_gan1', help='所有输出的根目录')
    parser.add_argument('--save_dir', type=str, default='checkpoint', help='保存模型的目录')
    parser.add_argument('--save_interval', type=int, default=5, help='保存模型的间隔轮次')
    parser.add_argument('--num_workers', type=int, default=10, help='数据加载的工作线程数')
    
    # 训练策略参数
    parser.add_argument('--gan_start_epoch', type=int, default=5, help='开始GAN训练的epoch')
    parser.add_argument('--d_update_ratio', type=int, default=1, help='每训练多少次G更新一次D')
    parser.add_argument('--apply_usm', type=bool, default=True, help='是否对HR图像应用USM锐化')
    
    return parser.parse_args()

def create_dataloaders(args):
    """创建训练和验证数据加载器"""
    # 设置随机种子
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # 训练集路径
    train_lr_paths = sorted(glob.glob(os.path.join(args.data_dir, args.train_dir, args.lr_dir, '*.tif')))
    train_hr_paths = sorted(glob.glob(os.path.join(args.data_dir, args.train_dir, args.hr_dir, '*.tif')))
    
    # 验证集路径
    val_lr_paths = sorted(glob.glob(os.path.join(args.data_dir, args.val_dir, args.lr_dir, '*.tif')))
    val_hr_paths = sorted(glob.glob(os.path.join(args.data_dir, args.val_dir, args.hr_dir, '*.tif')))
    
    # 检查文件数量
    train_count = len(train_lr_paths)
    val_count = len(val_lr_paths)
    
    if len(train_hr_paths) != train_count:
        raise ValueError(f"训练集LR和HR图像数量不一致: {train_count} vs {len(train_hr_paths)}")
    
    if len(val_hr_paths) != val_count:
        raise ValueError(f"验证集LR和HR图像数量不一致: {val_count} vs {len(val_hr_paths)}")
    
    print(f"训练集: {train_count} 个样本, 验证集: {val_count} 个样本")
    
    # 创建数据集
    train_dataset = NYUDDataset(
        lr1_paths=train_lr_paths,
        hr_paths=train_hr_paths,
        apply_sharpening=False  # 不使用dataset中的锐化，而是使用USM模块
    )
    
    val_dataset = NYUDDataset(
        lr1_paths=val_lr_paths,
        hr_paths=val_hr_paths,
        apply_sharpening=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_one_epoch(netG, netD, train_loader, criterion_pix, criterion_perceptual, criterion_gan, 
                  optimizer_g, optimizer_d, device, epoch, args, usm_sharpener=None):
    netG.train()
    netD.train()
    
    # 损失统计
    running_g_loss = 0.0
    running_pix_loss = 0.0
    running_perceptual_loss = 0.0
    running_gan_loss = 0.0
    running_d_loss = 0.0
    
    # 是否使用GAN训练
    use_gan = epoch >= args.gan_start_epoch
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1} Training')):
        # 获取输入和目标
        lr = batch['lr1'].to(device)
        hr = batch['hr'].to(device)
        
        # 应用USM锐化
        if args.apply_usm and usm_sharpener is not None:
            hr_usm = usm_sharpener(hr, weight=0.5, threshold=10)
        else:
            hr_usm = hr
        
        # ---------------------
        # 训练生成器
        # ---------------------
        optimizer_g.zero_grad()
        
        # 前向传播
        sr = netG(lr)
        
        # 像素损失
        l_pix = criterion_pix(sr, hr_usm)
        
        # 感知损失
        l_percep = criterion_perceptual(sr, hr_usm)
        
        # GAN损失
        l_g_gan = 0
        if use_gan:
            pred_g_fake = netD(sr)
            l_g_gan = criterion_gan(pred_g_fake, True, is_disc=False)
        
        # 生成器总损失
        l_g = args.lambda_pix * l_pix + args.lambda_perceptual * l_percep
        if use_gan:
            l_g += args.lambda_gan * l_g_gan
        
        # 反向传播和优化
        l_g.backward()
        optimizer_g.step()
        
        # 记录生成器损失
        running_g_loss += l_g.item()
        running_pix_loss += l_pix.item()
        running_perceptual_loss += l_percep.item()
        if use_gan:
            running_gan_loss += l_g_gan.item()
        
        # ---------------------
        # 训练判别器
        # ---------------------
        if use_gan and batch_idx % args.d_update_ratio == 0:
            optimizer_d.zero_grad()
            
            # 真实图像判别
            pred_d_real = netD(hr_usm)
            l_d_real = criterion_gan(pred_d_real, True, is_disc=True)
            
            # 生成图像判别 (需要分离梯度以避免影响生成器)
            with torch.no_grad():
                sr_detach = netG(lr).detach().clone()  # clone for pt1.9
            
            pred_d_fake = netD(sr_detach)
            l_d_fake = criterion_gan(pred_d_fake, False, is_disc=True)
            
            # 判别器总损失
            l_d = l_d_real + l_d_fake
            
            # 反向传播和优化
            l_d.backward()
            optimizer_d.step()
            
            # 记录判别器损失
            running_d_loss += l_d.item()
    
    # 计算平均损失
    epoch_g_loss = running_g_loss / len(train_loader)
    epoch_pix_loss = running_pix_loss / len(train_loader)
    epoch_perceptual_loss = running_perceptual_loss / len(train_loader)
    epoch_gan_loss = running_gan_loss / len(train_loader) if use_gan else 0
    epoch_d_loss = running_d_loss / (len(train_loader) / args.d_update_ratio + 1e-8) if use_gan else 0
    
    return epoch_g_loss, epoch_pix_loss, epoch_perceptual_loss, epoch_gan_loss, epoch_d_loss

def validate(netG, val_loader, criterion_pix, criterion_perceptual, device, usm_sharpener=None, apply_usm=False):
    netG.eval()
    
    running_psnr = 0.0
    running_pix_loss = 0.0
    running_perceptual_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # 获取输入和目标
            lr = batch['lr1'].to(device)
            hr = batch['hr'].to(device)
            
            # 应用USM锐化
            if apply_usm and usm_sharpener is not None:
                hr_usm = usm_sharpener(hr)
            else:
                hr_usm = hr
                
            # 前向传播
            sr = netG(lr)
            
            # 计算损失
            l_pix = criterion_pix(sr, hr_usm).item()
            l_percep = criterion_perceptual(sr, hr_usm).item()
            
            # 计算PSNR（相对于原始HR，而不是锐化版本）
            mse = torch.mean((sr - hr) ** 2)
            psnr = 10 * torch.log10(1.0 / mse).item()
            
            # 累计指标
            running_psnr += psnr
            running_pix_loss += l_pix
            running_perceptual_loss += l_percep
    
    # 计算平均指标
    avg_psnr = running_psnr / len(val_loader)
    avg_pix_loss = running_pix_loss / len(val_loader)
    avg_perceptual_loss = running_perceptual_loss / len(val_loader)
    
    return avg_psnr, avg_pix_loss, avg_perceptual_loss

def save_checkpoint(netG, netD, optimizer_g, optimizer_d, scheduler_g, scheduler_d, epoch, best_psnr, args, is_best=False):
    """保存检查点"""
    checkpoint_dir = os.path.join(args.output_dir, args.save_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'scheduler_g_state_dict': scheduler_g.state_dict() if scheduler_g else None,
        'scheduler_d_state_dict': scheduler_d.state_dict() if scheduler_d else None,
        'best_psnr': best_psnr,
        'args': args
    }
    
    # 只保存当前的模型，用于恢复训练
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_latest.pth'))
    torch.save({'params': netG.state_dict()}, os.path.join(checkpoint_dir, 'netG_latest.pth'))
    
    # 如果是最佳模型，则额外保存一份
    if is_best:
        print(f"发现新的最佳模型 (PSNR: {best_psnr:.2f}dB)，正在保存...")
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
        torch.save({'params': netG.state_dict()}, os.path.join(checkpoint_dir, 'netG_best.pth'))

def load_pretrained_model(model, path, device):
    """加载预训练模型权重"""
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        # 检查是否为完整的state_dict或嵌套在字典中
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']
        
        # 尝试加载
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"成功加载预训练模型: {path}")
            return True
        except Exception as e:
            print(f"加载预训练模型时出错: {e}")
            return False
    
    print(f"预训练模型文件不存在: {path}")
    return False

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出根目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(args)
    
    # 创建网络模型
    netG = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        scale=4,
        num_feat=args.num_feat,
        num_block=args.num_block
    ).to(device)
    
    netD = UNetDiscriminatorSN(
        num_in_ch=3,
        num_feat=args.num_feat
    ).to(device)
    
    # 创建USM锐化器
    usm_sharpener = USMSharp().to(device)
    
    # 加载预训练模型
    if args.pretrain_g_path:
        load_pretrained_model(netG, args.pretrain_g_path, device)
    if args.pretrain_d_path:
        load_pretrained_model(netD, args.pretrain_d_path, device)
    
    # 定义损失函数
    criterion_pix = nn.L1Loss().to(device)
    criterion_perceptual = PerceptualLoss(use_input_norm=True).to(device)
    criterion_gan = GANLoss(gan_type='vanilla').to(device)
    
    # 定义优化器
    optimizer_g = optim.Adam(netG.parameters(), lr=args.lr_g, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    optimizer_d = optim.Adam(netD.parameters(), lr=args.lr_d, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    
    # 定义学习率调度器
    milestones = [10, 20]
    scheduler_g = MultiStepLR(optimizer_g, milestones=milestones, gamma=0.5)
    scheduler_d = MultiStepLR(optimizer_d, milestones=milestones, gamma=0.5)
    
    # 创建CSV日志文件
    csv_log_path = os.path.join(args.output_dir, 'training_log.csv')
    with open(csv_log_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        header = ['Epoch', 'G_Loss', 'Pix_Loss', 'Percep_Loss', 'GAN_Loss', 'D_Loss', 
                  'Val_PSNR', 'Val_Pix_Loss', 'Val_Percep_Loss', 
                  'LR_G', 'LR_D', 'Best_Model', 'Timestamp']
        writer_csv.writerow(header)
    
    # 初始化训练状态
    start_epoch = 0
    best_psnr = -float('inf')
    
    # 恢复训练（如果指定）
    if args.resume and os.path.isfile(args.resume):
        print(f'加载检查点: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['best_psnr']
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        if checkpoint['scheduler_g_state_dict']:
            scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        if checkpoint['scheduler_d_state_dict']:
            scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        print(f'恢复训练，从epoch {start_epoch}开始')
    
    # 打印开始训练信息
    print(f'开始训练，共 {args.num_epochs} 个epoch')
    print(f'GAN训练将从第 {args.gan_start_epoch+1} 个epoch开始')
    print(f'USM锐化: {"启用" if args.apply_usm else "禁用"}')
    
    # 训练循环
    for epoch in range(start_epoch, args.num_epochs):
        # 训练一个epoch
        g_loss, pix_loss, percep_loss, gan_loss, d_loss = train_one_epoch(
            netG, netD, train_loader, criterion_pix, criterion_perceptual, criterion_gan,
            optimizer_g, optimizer_d, device, epoch, args, usm_sharpener
        )
        
        # 验证
        val_psnr, val_pix_loss, val_percep_loss = validate(
            netG, val_loader, criterion_pix, criterion_perceptual, device, 
            usm_sharpener, args.apply_usm
        )
        
        # 更新学习率调度器
        scheduler_g.step()
        scheduler_d.step()
        
        # 记录当前学习率
        current_lr_g = optimizer_g.param_groups[0]['lr']
        current_lr_d = optimizer_d.param_groups[0]['lr']
        
        # 是否为最佳模型
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr
        
        # 记录到CSV
        with open(csv_log_path, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            row = [epoch + 1, g_loss, pix_loss, percep_loss, gan_loss, d_loss,
                   val_psnr, val_pix_loss, val_percep_loss,
                   current_lr_g, current_lr_d,
                   "Yes" if is_best else "No",
                   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            writer_csv.writerow(row)
        
        # 保存模型
        save_checkpoint(netG, netD, optimizer_g, optimizer_d, scheduler_g, scheduler_d, epoch, best_psnr, args, is_best)
        
        # 打印进度
        print(f'Epoch {epoch+1}/{args.num_epochs} | '
              f'G Loss: {g_loss:.4f} (Pix: {pix_loss:.4f}, Percep: {percep_loss:.4f}, GAN: {gan_loss:.4f}) | '
              f'D Loss: {d_loss:.4f} | Val PSNR: {val_psnr:.2f}dB | '
              f'LR G: {current_lr_g:.1e}, LR D: {current_lr_d:.1e}')
    
    print(f'训练完成！最佳PSNR: {best_psnr:.2f}dB')

if __name__ == '__main__':
    main() 