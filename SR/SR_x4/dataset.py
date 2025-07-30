from torch.utils.data import Dataset
import numpy as np
import torch
from osgeo import gdal
import os
import cv2

def readTif(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(fileName + "文件无法打开")
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(xoff, yoff, data_width or width, data_height or height)
    return data

def sharpen_image(image, kernel_size=3, sigma=1.0, amount=1.0, threshold=0):
    """
    使用USM（Unsharp Masking）方法锐化图像
    
    参数:
        image: 输入图像
        kernel_size: 高斯模糊核大小
        sigma: 高斯模糊的标准差
        amount: 锐化强度
        threshold: 锐化阈值，只有差异大于阈值的像素才会被锐化
    
    返回:
        锐化后的图像
    """
    # 确保图像是float32类型
    image = image.astype(np.float32)
    
    # 如果图像是多通道的，分别处理每个通道
    if len(image.shape) == 3:
        sharpened = np.zeros_like(image)
        for i in range(image.shape[0]):
            # 创建高斯模糊版本
            blurred = cv2.GaussianBlur(image[i], (kernel_size, kernel_size), sigma)
            # 计算锐化图像
            sharpened[i] = image[i] + amount * (image[i] - blurred)
        return sharpened
    else:
        # 创建高斯模糊版本
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        # 计算锐化图像
        sharpened = image + amount * (image - blurred)
        return sharpened

class NYUDDataset(Dataset):
    def __init__(self, lr1_paths, hr_paths, transform=None, 
                 apply_sharpening=False, sharpen_amount=2.5, sharpen_sigma=1.5, sharpen_kernel_size=3):
        super().__init__()
        self.lr1_paths = lr1_paths      # 光学图像
        self.hr_paths = hr_paths        # 高分辨率RGB目标
        self.transform = transform
        self.apply_sharpening = apply_sharpening
        self.sharpen_amount = sharpen_amount
        self.sharpen_sigma = sharpen_sigma
        self.sharpen_kernel_size = sharpen_kernel_size

    def __len__(self):
        return len(self.lr1_paths)

    def __getitem__(self, idx):
        # 读取图像数据
        lr1 = (readTif(self.lr1_paths[idx])).astype(np.float32)/255.0
        hr = (readTif(self.hr_paths[idx])).astype(np.float32)/255.0
        
        # 应用锐化处理（可选）
        if self.apply_sharpening:
            hr = sharpen_image(hr, 
                             kernel_size=self.sharpen_kernel_size, 
                             sigma=self.sharpen_sigma, 
                             amount=self.sharpen_amount)
     
        # 转换为张量
        lr1 = torch.Tensor(lr1)  
        hr = torch.Tensor(hr)
        
        sample = {
            "lr1": lr1,
            "hr": hr
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class NYUDDataset1(Dataset):
    def __init__(self, lr1_paths):
        super().__init__()
        self.lr1_paths = lr1_paths      # 光学图像
        
        
    def __len__(self):
        return len(self.lr1_paths)

    def __getitem__(self, idx):
        # 读取图像数据
        print(self.lr1_paths)
        lr1 = (readTif(self.lr1_paths)).astype(np.float32)/255.0
       
       
        # 转换为张量
        lr1 = torch.Tensor(lr1)
      
        sample = {
            "lr1": lr1
        }

        return sample