#coding=utf8
import codecs
import cv2
import numpy as np
from maketwo import Detecttwo
from tqdm import tqdm

datapath = '/unsullied/sharefs/lh/isilon-home/fer2013/ferpic/'  # 数据路径
label_file = '/unsullied/sharefs/lh/isilon-home/fer2013/label.txt'  # 标签文件路径

# 读取标签文件
label_data = codecs.open(label_file, 'r', 'utf8').readlines()
label_data = [i.split('\t') for i in label_data]
label_data = [[i[0], int(i[1])] for i in label_data]

X = []  # 存储图像数据
Y = []  # 存储标签数据

# 遍历标签数据
for i in tqdm(label_data):
    picname = datapath + i[0] + '.png'  # 图像文件名
    img = cv2.imread(picname, 0)  # 读取灰度图像
    img = np.expand_dims(img, axis=2)  # 扩展维度，从 (224, 224) 变为 (224, 224, 1)
    img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_LINEAR)  # 图像尺寸调整为 48x48
    img = np.expand_dims(img, axis=2)  # 再次扩展维度，从 (48, 48) 变为 (48, 48, 1)
    X.append(img)  # 添加图像数据

    y = [0, 0, 0, 0, 0, 0, 0]  # 创建标签向量
    y[i[1]] = 1  # 设置正确的类别索引为 1，其余为 0
    y = np.array(y)  # 转换为 NumPy 数组
    Y.append(y)  # 添加标签数据

X = np.array(X)  # 转换为 NumPy 数组
Y = np.array(Y)  # 转换为 NumPy 数组

print(X.shape, Y.shape)  # 输出数据形状信息
print(X[0], Y[0:5])  # 输出前五个样本的图像数据和标签数据

import h5py

f = h5py.File("Data.hdf5", 'w')  # 创建 HDF5 文件对象
f.create_dataset('X', data=X)  # 创建数据集 'X'，存储图像数据
f.create_dataset('Y', data=Y)  # 创建数据集 'Y'，存储标签数据
f.close()  # 关闭 HDF5 文件对象
