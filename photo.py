from PyQt5 import QtCore, QtGui, QtWidgets


from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import cv2, os
from keras.models import load_model
import numpy as np

from utils import preprocess_input

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class Ui_MainWindow4(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(776, 583)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(140, 30, 361, 81))
        self.label.setStyleSheet("font: 20pt \"微软雅黑\";")
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(610, 230, 111, 51))
        self.pushButton.setStyleSheet("font: 10pt \"微软雅黑\";")
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 140, 501, 401))
        self.label_2.setStyleSheet("background-color: rgb(225, 225, 225);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(610, 340, 111, 51))
        self.pushButton_2.setStyleSheet("font: 10pt \"微软雅黑\";")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(670, -10, 111, 51))
        self.pushButton_3.setStyleSheet("font: 10pt \"黑体\";")
        self.pushButton_3.setObjectName("pushButton_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.select_img)
        self.pushButton_2.clicked.connect(self.detect_img)
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

    def select_img(self):
        self.img_path, _ = QFileDialog.getOpenFileName(None, 'open img', '', "*.png;*.jpg;;All Files(*)")
        print(self.img_path)
        # img = cv2.imread(self.img_path)
        img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), -1)
        input_img = cv2.resize(img, (501, 401))
        cv2.imwrite('resize_img.png', input_img)
        self.label_2.setStyleSheet("image: url(./resize_img.png)")
    def detect_img(self):
        input_img=self.save_predict('resize_img.png')
        cv2.imwrite('res3.png', input_img)
        self.label_2.setStyleSheet("image: url(./res3.png)")  #将检测出的图片放到界面框中

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

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "人脸表情图像识别"))
        self.pushButton.setText(_translate("MainWindow", "选择图像"))
        self.pushButton_2.setText(_translate("MainWindow", "开始检测"))
        self.pushButton_3.setText(_translate("MainWindow", "返回菜单"))

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow4()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())