<<<<<<< HEAD
# Sentinel2_SRx4
=======
# 遥感图像超分辨率GAN项目（SR_x4）

本项目基于Real-ESRGAN架构，针对遥感TIF影像进行4倍超分辨率重建，支持大图分块、地理信息保持、USM锐化等功能，适合遥感影像增强与科研应用。

---

## 目录结构

```
SR/SR_x4/
├── train_sr_gan.py      # 训练主程序
├── predict.py           # 推理/预测脚本
├── dataset.py           # 数据集处理与增强
├── rrdbnet_arch.py      # RRDBNet生成器与判别器结构
├── split.py             # 数据集分割工具
├── __init__.py
../model/
    └── netG_latest.pth  # 训练生成器权重
../pretrained/
    ├── RealESRGAN_x4plus.pth         # 生成器预训练权重
    └── RealESRGAN_x4plus_netD.pth    # 判别器预训练权重
```

---

## 环境依赖

- Python 3.7+
- torch >= 1.8.0
- torchvision >= 0.9.0
- numpy >= 1.19.0
- opencv-python >= 4.5.0
- gdal >= 3.0.0
- matplotlib >= 3.3.0
- tqdm >= 4.60.0


## 数据准备

数据集目录结构建议如下：

```
dataset/
├── train/
│   ├── lr/    # 低分辨率训练影像（TIF，8位无符号整型）
│   └── hr/    # 高分辨率训练影像（TIF，8位无符号整型）
└── val/
    ├── lr/    # 低分辨率验证影像（TIF，8位无符号整型）
    └── hr/    # 高分辨率验证影像（TIF，8位无符号整型）
```

---

## 训练方法

基本训练命令：

```bash
python train_sr_gan.py \
    --data_dir /path/to/dataset \
    --output_dir /path/to/output \
    --batch_size 12 \
    --num_epochs 30 \
    --lr_g 1e-4 \
    --lr_d 1e-4
```

常用参数说明：

- `--data_dir`：数据集根目录
- `--output_dir`：输出模型与日志目录
- `--batch_size`：批次大小（默认12）
- `--num_epochs`：训练轮数（默认30）
- `--lr_g`：生成器学习率
- `--lr_d`：判别器学习率
- `--lambda_pix`：像素损失权重
- `--lambda_perceptual`：感知损失权重
- `--lambda_gan`：GAN损失权重
- `--pretrain_g_path`：生成器预训练权重路径
- `--pretrain_d_path`：判别器预训练权重路径

支持断点续训：

```bash
python train_sr_gan.py --resume /path/to/checkpoint.pth
```

---

## 推理/预测

1. 修改 `predict.py` 中的模型路径和输入输出目录：

```python
MODEL_PATH = '/path/to/your/model.pth'
INPUT_DIR = '/path/to/input/images'
OUTPUT_DIR = '/path/to/output/images'
```

2. 运行预测：

```bash
python predict.py
```

- 支持批量处理TIF影像，自动保持地理参考信息，结果输出到指定目录。

---

## 主要特性

- 基于RRDBNet的生成器，UNet判别器，谱归一化
- 支持TIF遥感影像，自动地理信息保持
---

## 评估与工具

- PSNR/SSIM评估（可扩展）
- split.py：数据集自动分割
- dataset.py：数据加载与增强
- rrdbnet_arch.py：网络结构定义

---

## 联系与贡献

- 欢迎提交Issue或PR改进项目
- 联系邮箱：lph5878@163.com

---

**注意**：请确保已正确安装所有依赖，并准备好格式规范的数据集。 
>>>>>>> d9a2264 (Temporary commit for local changes)
