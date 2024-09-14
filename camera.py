from PyQt5 import QtCore, QtGui, QtWidgets
import cv2, os  # 导入OpenCV和os模块
from keras.models import load_model  # 导入Keras模型加载函数
import numpy as np  # 导入NumPy模块
from utils import preprocess_input  # 导入预处理函数
# parameters for loading data and images
from tensorflow.compat.v1 import ConfigProto  # 导入TensorFlow配置相关模块
from tensorflow.compat.v1 import InteractiveSession
from moviepy.editor import *  # 导入MoviePy模块

# 定义UI类
class Ui_MainWindow2(object):
    # 设置UI界面
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")  # 设置主窗口对象名
        MainWindow.resize(776, 583)  # 设置窗口大小
        self.centralwidget = QtWidgets.QWidget(MainWindow)  # 创建中心窗口部件
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)  # 创建标签部件
        self.label.setGeometry(QtCore.QRect(140, 30, 361, 81))  # 设置标签位置和大小
        self.label.setStyleSheet("font: 18pt \"微软雅黑\";")  # 设置标签样式表
        self.label.setObjectName("label")  # 设置标签对象名
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)  # 创建按钮部件
        self.pushButton.setGeometry(QtCore.QRect(610, 230, 111, 51))  # 设置按钮位置和大小
        self.pushButton.setStyleSheet("font: 10pt \"微软雅黑\";")  # 设置按钮样式表
        self.pushButton.setObjectName("pushButton")  # 设置按钮对象名
        self.label_2 = QtWidgets.QLabel(self.centralwidget)  # 创建标签部件
        self.label_2.setGeometry(QtCore.QRect(50, 140, 501, 401))  # 设置标签位置和大小
        self.label_2.setStyleSheet("background-color: rgb(225, 225, 225);")  # 设置标签样式表
        self.label_2.setText("")  # 设置标签文本内容为空
        self.label_2.setObjectName("label_2")  # 设置标签对象名
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)  # 创建按钮部件
        self.pushButton_2.setGeometry(QtCore.QRect(610, 340, 111, 51))  # 设置按钮位置和大小
        self.pushButton_2.setStyleSheet("font: 10pt \"微软雅黑\";")  # 设置按钮样式表
        self.pushButton_2.setObjectName("pushButton_2")  # 设置按钮对象名
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)  # 创建按钮部件
        self.pushButton_3.setGeometry(QtCore.QRect(670, -10, 111, 51))  # 设置按钮位置和大小
        self.pushButton_3.setStyleSheet("font: 10pt \"黑体\";")  # 设置按钮样式表
        self.pushButton_3.setObjectName("pushButton_3")  # 设置按钮对象名
        MainWindow.setCentralWidget(self.centralwidget)  # 将中心窗口部件设置为主窗口的中心部件
        self.statusbar = QtWidgets.QStatusBar(MainWindow)  # 创建状态栏部件
        self.statusbar.setObjectName("statusbar")  # 设置状态栏对象名
        MainWindow.setStatusBar(self.statusbar)  # 将状态栏设置到主窗口

        self.retranslateUi(MainWindow)  # 调用retranslateUi函数设置界面文本
        QtCore.QMetaObject.connectSlotsByName(MainWindow)  # 连接信号槽

        config = ConfigProto()  # 创建TensorFlow配置对象
        config.gpu_options.allow_growth = True  # 设置GPU内存按需分配
        session = InteractiveSession(config=config)  # 创建TensorFlow交互式会话

        # 定义情感识别模型路径和标签
        emotion_model_path = 'trained_models/float_models/fer2013_mini_XCEPTION.33-0.65.hdf5'
        self.emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                               4: 'sad', 5: 'surprise', 6: 'neutral'}
        detection_model_path = 'trained_models/facemodel/haarcascade_frontalface_default.xml'

        # 加载情感识别模型和人脸检测模型
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.face_detection = cv2.CascadeClassifier(detection_model_path)
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]

        self.retranslateUi(MainWindow)  # 调用retranslateUi函数设置界面文本
        QtCore.QMetaObject.connectSlotsByName(MainWindow)  # 连接信号槽

        self.img_num = 0  # 初始化图片计数器
        self.cap = cv2.VideoCapture()  # 创建视频流对象
        self.CAM_NUM = 0  # 设置默认摄像头编号为0
        self.timer_camera = QtCore.QTimer()  # 创建定时器对象，用于控制视频帧率
        self.timer_camera.timeout.connect(self.show_camera)  # 定时器超时信号连接到show_camera函数

        # 连接按钮点击信号与槽函数
        self.pushButton.clicked.connect(self.button_open_camera_clicked)  # 打开摄像头按钮
        self.pushButton_2.clicked.connect(self.button_close_camera_clicked)  # 关闭摄像头按钮

    # 打开摄像头按钮点击事件
    def button_open_camera_clicked(self):
        self.label_2.setStyleSheet("background: transparent;")  # 设置显示视频区域背景透明
        if self.timer_camera.isActive() == False:  # 若定时器未启动
                flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
                if flag == False:  # flag表示open()成不成功
                # https://www.icode9.com/content-4-96818.html
                    msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确",
                                                                    buttons=QtWidgets.QMessageBox.Ok)
                else:
                    self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示

        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_2.clear()  # 清空视频
    # 关闭摄像头按钮点击事件
    def button_close_camera_clicked(self):
        self.timer_camera.stop()  # 关闭定时器
        self.cap.release()  # 释放视频流
        self.label_2.clear()  # 清空视频显示区域
        self.label_2.setStyleSheet("background-color: rgb(225, 225, 225);")  # 设置视频显示区域背景颜色

    # 显示摄像头画面
    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读取一帧图像
        show = cv2.resize(self.image, (640, 480))  # 将读取到的图像大小调整为 640x480
        cv2.imwrite('./jietu.jpg', show)  # 将图像保存到文件中
        input_img = self.save_predict('./jietu.jpg')  # 对保存的图像进行表情预测

        cv2.imwrite('res2.png', input_img)  # 将预测结果保存为图片文件
        self.label_2.setStyleSheet("image: url(./res2.png)")  # 将检测出的图片放到界面框中
        self.img_num += 1  # 图片计数器加1

    # 进行表情预测
    def general_predict(self, imggray, imgcolor):
        gray_image = np.expand_dims(imggray, axis=2)  # 将灰度图像扩展为三维数组
        faces = self.face_detection.detectMultiScale(imggray, 1.3, 5)  # 检测图像中的人脸
        res = []  # 存储预测结果
        if len(faces) == 0:
            print('No face')  # 若未检测到人脸，打印提示信息
            return None
        else:
            for face_coordinates in faces:
                x1, y1, width, height = face_coordinates
                x1, y1, x2, y2 = x1, y1, x1 + width, y1 + height
                gray_face = gray_image[y1:y2, x1:x2]  # 提取人脸区域
                try:
                    gray_face = cv2.resize(gray_face, (self.emotion_target_size))  # 调整人脸图像大小
                except:
                    continue
                gray_face = preprocess_input(gray_face, True)  # 预处理人脸图像
                gray_face = np.expand_dims(gray_face, 0)  # 增加一个维度
                gray_face = np.expand_dims(gray_face, -1)  # 增加一个维度
                emotion_prediction = self.emotion_classifier.predict(gray_face)  # 对人脸图像进行表情预测
                emotion_label_arg = np.argmax(emotion_prediction)  # 获取最可能的表情标签
                res.append([emotion_label_arg, x1, y1, x2, y2])  # 将预测结果添加到列表中

        return res  # 返回预测结果列表

    # 对图像进行表情预测并保存预测结果
    def save_predict(self, imgurl):
        imggray = cv2.imread(imgurl, 0)  # 以灰度图像读取图像文件
        imgcolor = cv2.imread(imgurl, 1)  # 以彩色图像读取图像文件
        ress = self.general_predict(imggray, imgcolor)  # 对图像进行表情预测
        if ress == None:
            print('No face and no image saved')  # 若未检测到人脸，则打印提示信息
        try:
            for res in ress:
                label = self.emotion_labels[res[0]]  # 获取预测结果对应的表情标签
                lx, ly, rx, ry = res[1], res[2], res[3], res[4]  # 获取人脸区域坐标
                cv2.rectangle(imgcolor, (lx, ly), (rx, ry), (0, 0, 255), 2)  # 在图像上绘制人脸区域框
                cv2.putText(imgcolor, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)  # 在图像上标注表情标签
        except:
            print('no')  # 若出现异常，则打印提示信息
            # cv2
            # cv2.imwrite('images/res_1.png', imgcolor)
        return imgcolor
        # cv2.resize(imgcolor, (741, 421))
        # cv2.imwrite('res.png', imgcolor)
        # self.label_13.setStyleSheet("image: url(res.png)")
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "人脸表情实时检测"))
        self.pushButton.setText(_translate("MainWindow", "打开摄像头"))
        self.pushButton_2.setText(_translate("MainWindow", "关闭摄像头"))
        self.pushButton_3.setText(_translate("MainWindow", "返回菜单"))

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow2()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())