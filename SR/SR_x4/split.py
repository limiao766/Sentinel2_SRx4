import os
import random
import shutil
from pathlib import Path
import numpy as np
import argparse

def split_dataset(lr_dir, hr_dir, output_dir, train_ratio=0.6, val_ratio=0.2, random_seed=42):
    """
    将LR和HR图像数据集按照指定比例划分为训练集、验证集和测试集
    
    参数:
        lr_dir: 低分辨率图像目录路径
        hr_dir: 高分辨率图像目录路径
        output_dir: 输出目录的根路径
        train_ratio: 训练集比例，默认0.6 (60%)
        val_ratio: 验证集比例，默认0.2 (20%)，测试集将占用剩余20%
        random_seed: 随机种子，用于可重复的随机划分
    """
    # 设置随机种子确保可重复性
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 创建输出目录结构
    train_lr_dir = os.path.join(output_dir, 'train', 'lr')
    train_hr_dir = os.path.join(output_dir, 'train', 'hr')
    val_lr_dir = os.path.join(output_dir, 'val', 'lr')
    val_hr_dir = os.path.join(output_dir, 'val', 'hr')
    test_lr_dir = os.path.join(output_dir, 'test', 'lr')
    test_hr_dir = os.path.join(output_dir, 'test', 'hr')
    
    # 创建所有需要的目录
    for directory in [train_lr_dir, train_hr_dir, val_lr_dir, val_hr_dir, test_lr_dir, test_hr_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # 获取所有LR图像文件名
    lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    
    # 将文件名混洗
    random.shuffle(lr_files)
    
    # 计算分割点
    total_files = len(lr_files)
    train_split = int(total_files * train_ratio)
    val_split = int(total_files * (train_ratio + val_ratio))
    
    # 划分数据集
    train_files = lr_files[:train_split]
    val_files = lr_files[train_split:val_split]
    test_files = lr_files[val_split:]
    
    # 复制文件到目标目录
    def copy_files(file_list, lr_src, hr_src, lr_dst, hr_dst):
        copied_count = 0
        skipped_count = 0
        
        for lr_filename in file_list:
            # 从LR文件名构造HR文件名 (将_64替换为_256)
            hr_filename = lr_filename.replace('_64.', '_256.')
            
            lr_src_path = os.path.join(lr_src, lr_filename)
            hr_src_path = os.path.join(hr_src, hr_filename)
            
            # 确保HR文件存在
            if not os.path.exists(hr_src_path):
                print(f"警告: 对应的HR文件不存在: {hr_src_path}")
                skipped_count += 1
                continue
            
            # 复制文件
            shutil.copy2(lr_src_path, os.path.join(lr_dst, lr_filename))
            shutil.copy2(hr_src_path, os.path.join(hr_dst, hr_filename))
            copied_count += 1
        
        return copied_count, skipped_count
    
    # 执行复制
    train_copied, train_skipped = copy_files(train_files, lr_dir, hr_dir, train_lr_dir, train_hr_dir)
    val_copied, val_skipped = copy_files(val_files, lr_dir, hr_dir, val_lr_dir, val_hr_dir)
    test_copied, test_skipped = copy_files(test_files, lr_dir, hr_dir, test_lr_dir, test_hr_dir)
    
    total_copied = train_copied + val_copied + test_copied
    total_skipped = train_skipped + val_skipped + test_skipped
    
    # 打印统计信息
    print(f"数据集划分完成!")
    print(f"总文件数: {total_files}")
    print(f"成功复制的文件对: {total_copied}")
    print(f"跳过的文件(HR不存在): {total_skipped}")
    print(f"训练集: {train_copied} 文件对 ({train_copied/total_copied*100:.1f}%)")
    print(f"验证集: {val_copied} 文件对 ({val_copied/total_copied*100:.1f}%)")
    print(f"测试集: {test_copied} 文件对 ({test_copied/total_copied*100:.1f}%)")
    
    return {
        'train': train_copied,
        'val': val_copied,
        'test': test_copied
    }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将图像数据集按照指定比例划分为训练集、验证集和测试集')
    parser.add_argument('--lr_dir', type=str, default='/root/autodl-tmp/data/crop_64', help='低分辨率图像目录路径')
    parser.add_argument('--hr_dir', type=str, default='/root/autodl-tmp/data/crop_256', help='高分辨率图像目录路径')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/dataset', help='输出目录的根路径')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='训练集比例，默认0.6 (60%%)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例，默认0.2 (20%%)')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子，用于可重复的随机划分')
    
    args = parser.parse_args()
    
    # 执行划分
    split_dataset(
        args.lr_dir, 
        args.hr_dir, 
        args.output_dir, 
        train_ratio=args.train_ratio, 
        val_ratio=args.val_ratio, 
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main()