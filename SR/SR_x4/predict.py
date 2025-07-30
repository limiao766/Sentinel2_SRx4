import sys
sys.path.append('/root')
sys.path.append('/root/SR')
from osgeo import gdal
import numpy as np
import torch
import os
import glob
from SR.SR_x4.dataset import NYUDDataset1
from SR.SR_x4.rrdbnet_arch import RRDBNet

# =================配置参数=================
# 路径配置
MODEL_PATH = '/root/autodl-tmp/model/sr_gan1/checkpoint/netG_best.pth'
INPUT_DIR = '/root/SR/data'
OUTPUT_DIR = '/root/SR/data/pre'

# 模型配置
MODEL_PARAMS = {
    'num_in_ch': 3,
    'num_out_ch': 3,
    'scale': 4,
    'num_feat': 64,
    'num_block': 23
}

# 裁剪和拼接配置
CROP_SIZE = 64               # 裁剪块大小
REPEAT_LENGTH = 0            # 重叠区域长度
OUTPUT_SIZE = 256            # 输出块大小 (CROP_SIZE * MODEL_PARAMS['scale'])

# 地理信息调整参数
GEO_SCALE = 0.25             # 地理参考比例调整

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

# 模型加载辅助类
class Args:
    def __init__(self):
        pass

# 初始化模型
model = RRDBNet(
    num_in_ch=MODEL_PARAMS['num_in_ch'],
    num_out_ch=MODEL_PARAMS['num_out_ch'],
    scale=MODEL_PARAMS['scale'],
    num_feat=MODEL_PARAMS['num_feat'],
    num_block=MODEL_PARAMS['num_block']
).to(DEVICE)

# 加载模型权重
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
if 'params_ema' in ckpt:
    model.load_state_dict(ckpt['params_ema'], strict=True)
elif 'params' in ckpt:
    model.load_state_dict(ckpt['params'], strict=True)
else:
    model.load_state_dict(ckpt, strict=True)
model.eval()


def readTif1(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    """读取TIF文件并返回数据和地理信息"""
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(f"{fileName} 文件无法打开")
        return None, None, None, None, None, None
    
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    if (data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj

def writeTiff(im_data, im_geotrans, im_proj, path):
    """保存TIF文件，包括地理信息"""
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    
    if len(im_data.shape) == 3:
        # 检查通道维度是否在前面
        if im_data.shape[2] <= 4:  # 如果通道数小于等于4，假设是(H, W, C)格式
            im_bands, im_height, im_width = im_data.shape[2], im_data.shape[0], im_data.shape[1]
            # 转换为(C, H, W)格式
            im_data = np.transpose(im_data, (2, 0, 1))
        else:  # 否则假设是(C, H, W)格式
            im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    # 调整地理变换参数
    im_geotrans = np.array(im_geotrans)
    im_geotrans[1] *= GEO_SCALE
    im_geotrans[5] *= GEO_SCALE

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset:
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def TifCroppingArray(img, SideLength):
    """将图像裁剪成重叠的小块"""
    height, width = img.shape[0], img.shape[1]
    step = CROP_SIZE - SideLength * 2
    
    ColumnNum = int((height - SideLength * 2) / step)
    RowNum = int((width - SideLength * 2) / step)
    
    TifArrayReturn = []
    total_crops = 0
    
    # 处理主体网格
    for i in range(ColumnNum):
        row_crops = []
        for j in range(RowNum):
            y_start = i * step
            x_start = j * step
            cropped = img[y_start:y_start+CROP_SIZE, x_start:x_start+CROP_SIZE]
            row_crops.append(cropped)
            total_crops += 1
        
        # 处理每行最右侧
        cropped = img[i*step:i*step+CROP_SIZE, width-CROP_SIZE:width]
        row_crops.append(cropped)
        total_crops += 1
        
        TifArrayReturn.append(row_crops)
    
    # 处理最底部行
    bottom_row = []
    for j in range(RowNum):
        cropped = img[height-CROP_SIZE:height, j*step:j*step+CROP_SIZE]
        bottom_row.append(cropped)
        total_crops += 1
    
    # 处理右下角
    cropped = img[height-CROP_SIZE:height, width-CROP_SIZE:width]
    bottom_row.append(cropped)
    total_crops += 1
    
    TifArrayReturn.append(bottom_row)
    
    ColumnOver = (height - SideLength * 2) % step + SideLength
    RowOver = (width - SideLength * 2) % step + SideLength
    print(f"总裁剪图像数: {total_crops}")
    
    return TifArrayReturn, RowOver, ColumnOver


def Result(shape, TifArray, npyfile, RepetitiveLength, RowOver, ColumnOver):
    """将预测结果小块拼接回完整图像"""
    # 检查第一个预测结果的维度
    first_pred = npyfile[0]
    is_multichannel = len(first_pred.shape) > 2
    
    # 创建结果数组
    if is_multichannel:
        channels = first_pred.shape[2]
        result = np.zeros((shape[0], shape[1], channels), np.float32)
    else:
        result = np.zeros(shape, np.float32)
    
    # 调整重叠区域大小
    RowOver *= MODEL_PARAMS['scale']
    ColumnOver *= MODEL_PARAMS['scale']
    j = 0  # 行索引
    
    for i, img in enumerate(npyfile):
        row_len = len(TifArray[0])
        
        # 计算当前切片位置信息
        is_leftmost = (i % row_len == 0)
        is_rightmost = (i % row_len == row_len - 1)
        is_top = (j == 0)
        is_bottom = (j == len(TifArray) - 1)
        
        # 拼接逻辑
        if is_leftmost:
            if is_top:
                # 左上角
                if is_multichannel:
                    result[:OUTPUT_SIZE-RepetitiveLength, :OUTPUT_SIZE-RepetitiveLength, :] = img[:OUTPUT_SIZE-RepetitiveLength, :OUTPUT_SIZE-RepetitiveLength, :]
                else:
                    result[:OUTPUT_SIZE-RepetitiveLength, :OUTPUT_SIZE-RepetitiveLength] = img[:OUTPUT_SIZE-RepetitiveLength, :OUTPUT_SIZE-RepetitiveLength]
            elif is_bottom:
                # 左下角
                if is_multichannel:
                    result[shape[0]-ColumnOver-RepetitiveLength:shape[0], :OUTPUT_SIZE-RepetitiveLength, :] = img[OUTPUT_SIZE-ColumnOver-RepetitiveLength:OUTPUT_SIZE, :OUTPUT_SIZE-RepetitiveLength, :]
                else:
                    result[shape[0]-ColumnOver-RepetitiveLength:shape[0], :OUTPUT_SIZE-RepetitiveLength] = img[OUTPUT_SIZE-ColumnOver-RepetitiveLength:OUTPUT_SIZE, :OUTPUT_SIZE-RepetitiveLength]
            else:
                # 左边
                row_start = j * (OUTPUT_SIZE - 2 * RepetitiveLength) + RepetitiveLength
                row_end = (j + 1) * (OUTPUT_SIZE - 2 * RepetitiveLength) + RepetitiveLength
                if is_multichannel:
                    result[row_start:row_end, :OUTPUT_SIZE-RepetitiveLength, :] = img[RepetitiveLength:OUTPUT_SIZE-RepetitiveLength, :OUTPUT_SIZE-RepetitiveLength, :]
                else:
                    result[row_start:row_end, :OUTPUT_SIZE-RepetitiveLength] = img[RepetitiveLength:OUTPUT_SIZE-RepetitiveLength, :OUTPUT_SIZE-RepetitiveLength]
        
        elif is_rightmost:
            if is_top:
                # 右上角
                if is_multichannel:
                    result[:OUTPUT_SIZE-RepetitiveLength, shape[1]-RowOver:shape[1], :] = img[:OUTPUT_SIZE-RepetitiveLength, OUTPUT_SIZE-RowOver:OUTPUT_SIZE, :]
                else:
                    result[:OUTPUT_SIZE-RepetitiveLength, shape[1]-RowOver:shape[1]] = img[:OUTPUT_SIZE-RepetitiveLength, OUTPUT_SIZE-RowOver:OUTPUT_SIZE]
            elif is_bottom:
                # 右下角
                if is_multichannel:
                    result[shape[0]-ColumnOver:shape[0], shape[1]-RowOver:shape[1], :] = img[OUTPUT_SIZE-ColumnOver:OUTPUT_SIZE, OUTPUT_SIZE-RowOver:OUTPUT_SIZE, :]
                else:
                    result[shape[0]-ColumnOver:shape[0], shape[1]-RowOver:shape[1]] = img[OUTPUT_SIZE-ColumnOver:OUTPUT_SIZE, OUTPUT_SIZE-RowOver:OUTPUT_SIZE]
            else:
                # 右边
                row_start = j * (OUTPUT_SIZE - 2 * RepetitiveLength) + RepetitiveLength
                row_end = (j + 1) * (OUTPUT_SIZE - 2 * RepetitiveLength) + RepetitiveLength
                if is_multichannel:
                    result[row_start:row_end, shape[1]-RowOver:shape[1], :] = img[RepetitiveLength:OUTPUT_SIZE-RepetitiveLength, OUTPUT_SIZE-RowOver:OUTPUT_SIZE, :]
                else:
                    result[row_start:row_end, shape[1]-RowOver:shape[1]] = img[RepetitiveLength:OUTPUT_SIZE-RepetitiveLength, OUTPUT_SIZE-RowOver:OUTPUT_SIZE]
            j += 1  # 右边结束后行索引加1
        
        else:
            # 中间区域
            col_start = (i - j * row_len) * (OUTPUT_SIZE - 2 * RepetitiveLength) + RepetitiveLength
            col_end = (i - j * row_len + 1) * (OUTPUT_SIZE - 2 * RepetitiveLength) + RepetitiveLength
            
            if is_top:
                # 上边
                if is_multichannel:
                    result[:OUTPUT_SIZE-RepetitiveLength, col_start:col_end, :] = img[:OUTPUT_SIZE-RepetitiveLength, RepetitiveLength:OUTPUT_SIZE-RepetitiveLength, :]
                else:
                    result[:OUTPUT_SIZE-RepetitiveLength, col_start:col_end] = img[:OUTPUT_SIZE-RepetitiveLength, RepetitiveLength:OUTPUT_SIZE-RepetitiveLength]
            elif is_bottom:
                # 下边
                if is_multichannel:
                    result[shape[0]-ColumnOver:shape[0], col_start:col_end, :] = img[OUTPUT_SIZE-ColumnOver:OUTPUT_SIZE, RepetitiveLength:OUTPUT_SIZE-RepetitiveLength, :]
                else:
                    result[shape[0]-ColumnOver:shape[0], col_start:col_end] = img[OUTPUT_SIZE-ColumnOver:OUTPUT_SIZE, RepetitiveLength:OUTPUT_SIZE-RepetitiveLength]
            else:
                # 中间
                row_start = j * (OUTPUT_SIZE - 2 * RepetitiveLength) + RepetitiveLength
                row_end = (j + 1) * (OUTPUT_SIZE - 2 * RepetitiveLength) + RepetitiveLength
                if is_multichannel:
                    result[row_start:row_end, col_start:col_end, :] = img[RepetitiveLength:OUTPUT_SIZE-RepetitiveLength, RepetitiveLength:OUTPUT_SIZE-RepetitiveLength, :]
                else:
                    result[row_start:row_end, col_start:col_end] = img[RepetitiveLength:OUTPUT_SIZE-RepetitiveLength, RepetitiveLength:OUTPUT_SIZE-RepetitiveLength]
    
    return result


# 主处理流程
def process_images():
    """处理全部图像的主函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    img_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*")))
    
    # 过滤出文件（排除目录）
    img_paths = [path for path in img_paths if os.path.isfile(path)]
    
    for img_path in img_paths:
        # 获取地理信息
        width, height, bands, data, geotrans, proj = readTif1(img_path)
        
        # 检查文件是否成功读取
        if width is None:
            print(f"跳过无法读取的文件: {img_path}")
            continue
        
        # 准备文件名
        img_path = img_path.replace("\\", "/")
        file_name = os.path.basename(img_path)
        output_path = os.path.join(OUTPUT_DIR, file_name)
        
        # 加载和处理数据
        im = NYUDDataset1(img_path)
        image = im[0]['lr1'].permute(1, 2, 0)
        img_shape = (image.shape[0] * MODEL_PARAMS['scale'], image.shape[1] * MODEL_PARAMS['scale'])
        
        # 裁剪数据
        TifArray_img, RowOver, ColumnOver = TifCroppingArray(image, REPEAT_LENGTH)
        print(f"[INFO]: 加载数据完成: {file_name}")
        
        # 预测
        sr_predicts = []
        for i in range(len(TifArray_img)):
            for j in range(len(TifArray_img[0])):
                # 准备输入
                patch = TifArray_img[i][j].permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    # 模型推理
                    sr_pre = model(patch)
                    
                    # 处理预测结果
                    sr_pred = sr_pre.cpu().detach().numpy()[0]
                    
                    # 转换格式为(H,W,C)
                    sr_pred = np.transpose(sr_pred, (1, 2, 0))
                    
                    sr_predicts.append(sr_pred)
        
        # 合并预测结果
        sr_result = Result(img_shape, TifArray_img, sr_predicts, REPEAT_LENGTH * MODEL_PARAMS['scale'], RowOver, ColumnOver)
        
        # 保存结果
        writeTiff(sr_result, geotrans, proj, output_path)
        
        print(f"处理完成: {file_name}")


if __name__ == "__main__":
    process_images()