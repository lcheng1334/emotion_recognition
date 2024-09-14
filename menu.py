# -*- coding: utf-8 -*-

# 从UI文件生成的窗体实现
# WARNING：手动修改此文件将在再次运行pyuic5时丢失。除非您知道您在做什么，否则不要编辑此文件。

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(441, 483)
        MainWindow.setStyleSheet("font: 20pt \"微软雅黑\";")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 20, 291, 71))
        self.label.setStyleSheet("font: 20pt \"微软雅黑\";")
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(120, 130, 211, 71))
        self.pushButton.setStyleSheet("font: 15pt \"微软雅黑\";")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(120, 230, 211, 71))
        self.pushButton_2.setStyleSheet("font: 15pt \"微软雅黑\";")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(120, 330, 211, 71))
        self.pushButton_3.setStyleSheet("font: 15pt \"微软雅黑\";")
        self.pushButton_3.setObjectName("pushButton_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "人脸表情识别系统"))
        self.pushButton.setText(_translate("MainWindow", "摄像头检测"))
        self.pushButton_2.setText(_translate("MainWindow", "图像识别"))
        self.pushButton_3.setText(_translate("MainWindow", "视频检测"))


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
