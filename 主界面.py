from PyQt5 import QtCore, QtGui, QtWidgets
from menu import Ui_MainWindow  # 导入主界面的UI类
from camera import Ui_MainWindow2  # 导入相机界面的UI类
from video import Ui_MainWindow3  # 导入视频界面的UI类
from photo import Ui_MainWindow4  # 导入图片界面的UI类

# 相机界面类
class jiemian2(QtWidgets.QMainWindow, Ui_MainWindow2):
    def __init__(self):
        super(jiemian2, self).__init__()
        self.setupUi(self)  # 加载相机识别模块
        self.pushButton_3.clicked.connect(self.back)  # 返回主界面功能按钮，连接下面的back函数

    def back(self):
        self.hide()  # 隐藏此窗口
        self.log = loginWindow()
        self.log.show()  # 显示登录窗口

# 视频界面类
class jiemian3(QtWidgets.QMainWindow, Ui_MainWindow3):
    def __init__(self):
        super(jiemian3, self).__init__()
        self.setupUi(self)  # 加载视频文件识别模块
        self.pushButton_3.clicked.connect(self.back)  # 返回主界面功能按钮，连接下面的back函数

    def back(self):
        self.hide()  # 隐藏此窗口
        self.log = loginWindow()
        self.log.show()  # 显示登录窗口

# 图片界面类
class jiemian4(QtWidgets.QMainWindow, Ui_MainWindow4):
    def __init__(self):
        super(jiemian4, self).__init__()
        self.setupUi(self)
        self.pushButton_3.clicked.connect(self.back)  # 返回主界面功能按钮，连接下面的back函数

    def back(self):
        self.hide()  # 隐藏此窗口
        self.log = loginWindow()
        self.log.show()  # 显示登录窗口

# 主界面类
class loginWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(loginWindow, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.camera)  # 相机检测按钮，连接下面的camera_detect功能函数
        self.pushButton_2.clicked.connect(self.photo_detect)  # 视频文件检测按钮，连接下面的file_detect函数
        self.pushButton_3.clicked.connect(self.file_detect)

    def file_detect(self):
        self.hide()
        self.jiemian3 = jiemian3()  # 加载视频文件识别界面
        self.jiemian3.show()

    def camera(self):
        self.hide()
        self.jiemian2 = jiemian2()  # 加载相机界面
        self.jiemian2.show()

    def photo_detect(self):
        self.hide()
        self.jiemian4 = jiemian4()  # 加载图片界面
        self.jiemian4.show()

# 运行窗口Login
if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    login_show = loginWindow()
    login_show.show()
    sys.exit(app.exec_())
