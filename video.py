from moviepy.editor import *
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import cv2, os
from keras.models import load_model
import numpy as np
from tqdm import tqdm
from utils import preprocess_input
# parameters for loading data and images
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



class Ui_MainWindow3(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(817, 663)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(240, 10, 351, 81))
        self.label.setStyleSheet("font: 20pt \"微软雅黑\";")
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(210, 580, 111, 51))
        self.pushButton.setStyleSheet("font: 10pt \"微软雅黑\";")
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(60, 120, 691, 431))
        self.label_2.setStyleSheet("background-color: rgb(225, 225, 225);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(480, 580, 111, 51))
        self.pushButton_2.setStyleSheet("font: 10pt \"微软雅黑\";")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(710, -10, 111, 51))
        self.pushButton_3.setStyleSheet("font: 10pt \"黑体\";")
        self.pushButton_3.setObjectName("pushButton_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        emotion_model_path = 'trained_models/float_models/fer2013_mini_XCEPTION.33-0.65.hdf5'
        self.emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                          4: 'sad', 5: 'surprise', 6: 'neutral'}
        detection_model_path = 'trained_models/facemodel/haarcascade_frontalface_default.xml'

        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.face_detection = cv2.CascadeClassifier(detection_model_path)
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.file_main)  #文件选择按钮  连接file_main函数

    def file_main(self):
        #文件选择
        fname , _ = QFileDialog.getOpenFileName(None, 'open file', '',"*.avi;*.mp4;;All Files(*)")
        print(fname)
        # clip = VideoFileClip(sys.argv[1]) # can be gif or movie
        #加载识别模型
        clip = VideoFileClip(fname)
        # python version
        pyFlag = ''
        if len(sys.argv) < 3:
            pyFlag = '2'  # default to use moviepy to show, this can work on python2.7 and python3.5
        elif len(sys.argv) == 3:
            pyFlag = sys.argv[2]  # python version
        else:
            print('Wrong input!')
            sys.exit()
        img_size = 64
        stage_num = [3, 3, 3]
        lambda_local = 1
        lambda_d = 1
        img_idx = 0

        skip_frame = 5  # every 5 frame do 1 detection and network forward propagation
        for img in clip.iter_frames():
            if self.pushButton_2.isDown():
                self.label_2.setStyleSheet("background-color: rgb(236, 236, 236);")
                cv2.destroyAllWindows()
                self.label_2.setText('')
                break
            img_idx = img_idx + 1

            input_img = img  # using python2.7 with moivepy to show th image without channel flip


            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_h, img_w, _ = np.shape(input_img)
            input_img = cv2.resize(input_img, (691, int(691 * img_h / img_w)))
            img_h, img_w, _ = np.shape(input_img)
            cv2.imwrite('test.png',input_img)
            self.label_2.setStyleSheet("image: url(./test.png)")
            if img_idx == 1 or img_idx % skip_frame == 0:
                input_img=self.save_predict('test.png')

            cv2.imwrite('res.png',input_img)
            # self.label_13.setStyleSheet("image: url(./file_res.png)")
            self.label_2.setStyleSheet("image: url(./res.png)")
            if cv2.waitKey(30) & 0xFF==27:
                break
            # self.label_13.setStyleSheet("image: url(./file_res.png)")
            # self.save_predict('test.png')
            # # cv2.imwrite('file_res.png', cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR))




    def general_predict(self,imggray, imgcolor):
        gray_image = np.expand_dims(imggray, axis=2)  # 224*224*1
        faces = self.face_detection.detectMultiScale(imggray, 1.3, 5)
        res = []
        if len(faces) == 0:
            print('No face')
            return None
        else:
            for face_coordinates in faces:
                x1, y1, width, height = face_coordinates
                x1, y1, x2, y2 = x1, y1, x1 + width, y1 + height
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (self.emotion_target_size))
                except:
                    continue
                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = self.emotion_classifier.predict(gray_face)
                # emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                res.append([emotion_label_arg, x1, y1, x2, y2])

        return res

    def save_predict(self,imgurl):
        imggray = cv2.imread(imgurl, 0)
        imgcolor = cv2.imread(imgurl, 1)
        ress = self.general_predict(imggray, imgcolor)
        if ress == None:
            print('No face and no image saved')
        try:
            for res in ress:
                label = self.emotion_labels[res[0]]
                lx, ly, rx, ry = res[1], res[2], res[3], res[4]
                cv2.rectangle(imgcolor, (lx, ly), (rx, ry), (0, 0, 255), 2)
                cv2.putText(imgcolor, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        except:
            print('no')
            # cv2.imwrite('images/res_1.png', imgcolor)
        return imgcolor
        # cv2.resize(imgcolor, (741, 421))
        # cv2.imwrite('res.png', imgcolor)
        # self.label_13.setStyleSheet("image: url(res.png)")
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "人脸表情视频检测"))
        self.pushButton.setText(_translate("MainWindow", "选择视频"))
        self.pushButton_2.setText(_translate("MainWindow", "关闭视频"))
        self.pushButton_3.setText(_translate("MainWindow", "返回菜单"))

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow3()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())