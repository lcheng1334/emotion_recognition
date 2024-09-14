#coding=utf8
import codecs
import cv2
from tqdm import tqdm
import numpy as np

# 读取FER2013数据集的CSV文件（已去掉表头）
f = codecs.open('fer2013.csv', 'r', 'utf8').readlines()[1:]

# 创建标签文件
labelfile = codecs.open('label.txt', 'w', 'utf8')

index = 0  # 图像索引
for line in tqdm(f):  # 遍历CSV文件中的每一行数据
    flist = line.split(',')  # 按逗号分割数据
    label = flist[0]  # 获取标签
    img = flist[1].split(' ')  # 获取图像数据并按空格分割
    img = [int(i) for i in img]  # 将图像数据转换为整数类型
    img = np.array(img)  # 转换为NumPy数组
    img = img.reshape((48, 48))  # 将图像数据转换为48x48的形状
    cv2.imwrite('ferpic/' + str(index) + '.png', img)  # 将图像保存为PNG文件
    labelfile.write(str(index) + '\t' + label + '\n')  # 写入图像索引和标签到标签文件
    index += 1  # 索引自增

# 关闭标签文件
labelfile.close()
