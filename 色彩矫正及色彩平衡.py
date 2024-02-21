# -*- coding:utf-8 -*-
"""
作者：YJH
日期：2021年11月05日
"""
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from 彩色空间转换 import hsi2rgb              # 从前面写的一个文件里导入自定义的两个函数
from 彩色空间转换 import rgb2hsi

# 显示汉字用
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 定义坐标数字字体及大小
def label_def():
    plt.xticks(fontproperties='Times New Roman', size=8)
    plt.yticks(fontproperties='Times New Roman', size=8)
    # plt.axis('off')                                      # 关坐标，可选


if __name__ == '__main__':
    # 读取图片
    img_orig = cv.imread('caomei.jpg', 1)               # 读取彩色图片
    img_orig = np.power(img_orig.astype(np.float32), 0.5)  # 图像较亮，若采用幂率变换，γ>1，压缩高灰度级
    temp1 = img_orig - np.min(img_orig)
    img_orig = temp1 / np.max(temp1)
# ------------------------------------------------色调校正---------------------------------------------------------#
    # 伽马变换处理
    img_gama = np.power(img_orig.astype(np.float32), 1.5)         # 图像较亮，若采用幂率变换，γ>1，压缩高灰度级
    temp1 = img_gama - np.min(img_gama)
    img_gama = temp1/np.max(temp1)
    # 对比度拉伸变换函数
    med = np.median(img_orig.astype(np.float32))                    # 获取中值M
    img_temp = 1 / (1 + np.power((140/(img_orig+1e-6)), 4.5))       # 4.5为斜率，交互式选择(感觉med效果不如140)
    temp2 = img_temp - np.min(img_temp)                             # 标定到[0~255]，才能进行BGR2RGB
    img_con_str = np.uint8(255*(temp2/np.max(temp2)))
    # 显示所用的变换函数
    x1 = np.linspace(img_orig.min(), img_orig.max(), num=200)
    y1 = np.power(x1, 1.5)                                       # 伽马函数

    x2 = np.linspace(img_orig.min(), img_orig.max(), num=200)
    y2 = 1 / (1 + np.power((med/(x2+1e-6)), 4.5))               # 对比度拉伸函数

    plt.subplot(221), plt.title('原图像'), plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)), plt.axis('off')
    plt.subplot(222), plt.title('伽马变换'), plt.imshow(cv.cvtColor(img_gama, cv.COLOR_BGR2RGB)), plt.axis('off')
    # plt.subplot(233), plt.title('对比度拉伸'), plt.imshow(cv.cvtColor(img_con_str, cv.COLOR_BGR2RGB)), plt.axis('off')
    plt.subplot(224), plt.title('s=r**(1.5)'), plt.plot(x1, y1), plt.grid(), label_def()
    # plt.subplot(236), plt.title('s=1/(1+(M/r)**4)'), plt.plot(x2, y2), plt.grid(), label_def()
    plt.show()
