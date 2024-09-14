import sys  # 导入系统模块

import cv2  # 导入OpenCV模块
import os  # 导入操作系统模块
from keras.models import load_model  # 导入Keras模型加载函数
import numpy as np  # 导入NumPy模块
from tqdm import tqdm  # 导入进度条模块
from utils import preprocess_input  # 导入预处理函数
from tensorflow.compat.v1 import ConfigProto  # 导入TensorFlow配置模块
from tensorflow.compat.v1 import InteractiveSession  # 导入TensorFlow会话模块

# 配置TensorFlow会话，允许GPU内存自增长
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

emotion_model_path = 'trained_models/float_models/fer2013_mini_XCEPTION.33-0.65.hdf5'  # 情绪识别模型路径
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                  4: 'sad', 5: 'surprise', 6: 'neutral'}  # 情绪标签字典
detection_model_path = 'trained_models/facemodel/haarcascade_frontalface_default.xml'  # 人脸检测模型路径

# 加载情绪识别模型和人脸检测器
emotion_classifier = load_model(emotion_model_path, compile=False)
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]  # 情绪识别模型输入尺寸

def general_predict(imggray, imgcolor):
    gray_image = np.expand_dims(imggray, axis=2)  # 扩展图像维度，变为 (224, 224, 1)
    faces = face_detection.detectMultiScale(imggray, 1.3, 5)  # 检测图像中的人脸
    res = []  # 存储识别结果
    if len(faces) == 0:  # 若未检测到人脸
        print('No face')  # 输出信息
        return None  # 返回空结果
    else:
        for face_coordinates in faces:  # 遍历每张检测到的人脸
            x1, y1, width, height = face_coordinates  # 获取人脸坐标和尺寸
            x1, y1, x2, y2 = x1, y1, x1 + width, y1 + height  # 计算人脸边界框
            gray_face = gray_image[y1:y2, x1:x2]  # 提取人脸区域
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))  # 将人脸图像大小调整为模型输入大小
            except:
                continue
            gray_face = preprocess_input(gray_face, True)  # 预处理人脸图像
            gray_face = np.expand_dims(gray_face, 0)  # 扩展维度，变为 (1, 224, 224, 1)
            gray_face = np.expand_dims(gray_face, -1)  # 再次扩展维度，变为 (1, 224, 224, 1)
            emotion_prediction = emotion_classifier.predict(gray_face)  # 使用模型预测情绪
            emotion_label_arg = np.argmax(emotion_prediction)  # 获取预测结果中概率最大的类别索引
            res.append([emotion_label_arg, x1, y1, x2, y2])  # 将结果添加到列表中

    return res  # 返回识别结果列表

def save_predict(imgurl, targeturl='images/predicted_test_image.png'):
    imggray = cv2.imread(imgurl, 0)  # 以灰度图像读取输入图像
    imgcolor = cv2.imread(imgurl, 1)  # 以彩色图像读取输入图像
    ress = general_predict(imggray, imgcolor)  # 进行情绪识别
    if ress == None:  # 若未检测到人脸
        print('No face and no image saved')  # 输出信息
    for res in ress:  # 遍历识别结果
        label = emotion_labels[res[0]]  # 获取情绪标签
        lx, ly, rx, ry = res[1], res[2], res[3], res[4]  # 获取人脸边界框坐标
        cv2.rectangle(imgcolor, (lx, ly), (rx, ry), (0, 0, 255), 2)  # 绘制人脸边界框
        cv2.putText(imgcolor, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)  # 在图像上标注情绪

    cv2.imwrite('res.png', imgcolor)  # 保存带有情绪标签的图像

save_predict('test/11.png')  # 对测试图像进行情绪识别和保存
