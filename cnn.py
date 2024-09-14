from keras.layers import Activation, Convolution2D, Dropout, Conv2D  # 导入激活函数、卷积层等模块
from keras.layers import AveragePooling2D, BatchNormalization  # 导入平均池化层、批量归一化等模块
from keras.layers import GlobalAveragePooling2D  # 导入全局平均池化层模块
from keras.models import Sequential  # 导入序贯模型
from keras.layers import Flatten  # 导入展平层模块
from keras.models import Model  # 导入模型类
from keras.layers import Input  # 导入输入层模块
from keras.layers import MaxPooling2D  # 导入最大池化层模块
from keras.layers import SeparableConv2D  # 导入可分离卷积层模块
from keras import layers  # 导入层模块
from keras.regularizers import l2  # 导入L2正则化模块

def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):  # 定义mini_XCEPTION函数，接受输入形状、类别数和L2正则化参数
    regularization = l2(l2_regularization)  # 构建L2正则化器

    # 基础部分
    img_input = Input(input_shape)  # 输入层
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(img_input)  # 添加卷积层，8个卷积核，3x3大小，步长为1
    x = BatchNormalization()(x)  # 批量归一化
    x = Activation('relu')(x)  # 激活函数ReLU
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(x)  # 添加卷积层
    x = BatchNormalization()(x)  # 批量归一化
    x = Activation('relu')(x)  # 激活函数ReLU

    # 模块1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)  # 添加1x1卷积核，步长为2的卷积层
    residual = BatchNormalization()(residual)  # 批量归一化

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)  # 可分离卷积层，3x3大小，步长为1
    x = BatchNormalization()(x)  # 批量归一化
    x = Activation('relu')(x)  # 激活函数ReLU
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)  # 可分离卷积层，3x3大小，步长为1
    x = BatchNormalization()(x)  # 批量归一化

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)  # 最大池化层，3x3大小，步长为2
    x = layers.add([x, residual])  # 残差连接

    # 模块2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)  # 添加1x1卷积核，步长为2的卷积层
    residual = BatchNormalization()(residual)  # 批量归一化

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)  # 可分离卷积层，3x3大小，步长为1
    x = BatchNormalization()(x)  # 批量归一化
    x = Activation('relu')(x)  # 激活函数ReLU
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)  # 可分离卷积层，3x3大小，步长为1
    x = BatchNormalization()(x)  # 批量归一化

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)  # 最大池化层，3x3大小，步长为2
    x = layers.add([x, residual])  # 残差连接

    # 模块3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)  # 添加1x1卷积核，步长为2的卷积层
    residual = BatchNormalization()(residual)  # 批量归一化

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)  # 可分离卷积层，3x3大小，步长为1
    x = BatchNormalization()(x)  # 批量归一化
    x = Activation('relu')(x)  # 激活函数ReLU
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)  # 可分离卷积层，3x3大小，步长为1
    x = BatchNormalization()(x)  # 批量归一化

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)  # 最大池化层，3x3大小，步长为2
    x = layers.add([x, residual])  # 残差连接

    # 模块4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)  # 添加1x1卷积核，步长为2的卷积层
    residual = BatchNormalization()(residual)  # 批量归一化

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)  # 可分离卷积层，3x3大小，步长为1
    x = BatchNormalization()(x)  # 批量归一化
    x = Activation('relu')(x)  # 激活函数ReLU
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)  # 可分离卷积层，3x3大小，步长为1
    x = BatchNormalization()(x)  # 批量归一化

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)  # 最大池化层，3x3大小，步长为2
    x = layers.add([x, residual])  # 残差连接

    x = Conv2D(num_classes, (3, 3),
            #kernel_regularizer=regularization,
            padding='same')(x)  # 添加卷积层，输出类别数，3x3大小，步长为1
    x = GlobalAveragePooling2D()(x)  # 全局平均池化层
    output = Activation('softmax',name='predictions')(x)  # 输出层，使用softmax激活函数

    model = Model(img_input, output)  # 构建模型
    return model  # 返回模型

if __name__ == "__main__":
    input_shape = (48, 48, 1)  # 输入形状
    num_classes = 7  # 类别数
    model = mini_XCEPTION(input_shape, num_classes)  # 创建mini_XCEPTION模型
    model.summary()  # 输出模型概要信息
