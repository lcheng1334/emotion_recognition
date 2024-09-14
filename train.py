
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from cnn import mini_XCEPTION
from utils import preprocess_input
import numpy as np
import h5py
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, f1_score
import pandas as pd
import seaborn as sns

# parameters

batch_size = 32
num_epochs = 10
input_shape = (48, 48, 1)
validation_split = 0.1
num_classes = 7
patience = 50
base_path = 'trained_models/float_models/'

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


datasets = ['fer2013']

for dataset_name in datasets:
    print('Training dataset:', dataset_name)
    # callbacks
    log_file_path = base_path + dataset_name + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/4), verbose=1)
    trained_models_path = base_path + dataset_name + '_mini_XCEPTION'
    model_names = trained_models_path + '.{epoch:02d}-{accuracy:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                    save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # loading dataset
    f = h5py.File('Data.hdf5','r')
    X = f['X'][()]
    X = preprocess_input(X)
    Y = f['Y'][()]
    f.close()
    #X = np.load('X.npy')
    #Y = np.load('Y.npy')
    train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=validation_split,random_state=0)

    model.fit_generator(data_generator.flow(train_X, train_Y,
                                            batch_size),
                        steps_per_epoch=len(train_X) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(test_X,test_Y))



# Step 1: 从日志文件中读取准确率和损失数据
log_data = pd.read_csv(log_file_path)

# Step 2: 绘制准确率和损失曲线图
plt.figure(figsize=(12, 5))

# 绘制准确率曲线图
plt.subplot(1, 2, 1)
plt.plot(log_data['epoch'], log_data['accuracy'], label='Training Accuracy')
plt.plot(log_data['epoch'], log_data['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 绘制损失曲线图
plt.subplot(1, 2, 2)
plt.plot(log_data['epoch'], log_data['loss'], label='Training Loss')
plt.plot(log_data['epoch'], log_data['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# 保存图形
plt.savefig('accuracy_loss_curves.png')
plt.show()

# Step 3: 生成混淆矩阵
# 使用训练好的模型对测试集进行预测
predictions = model.predict(test_X)
# 将预测结果转换为类别标签
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_Y, axis=1)
# 定义类别标签
class_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 生成混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 可视化混淆矩阵
plt.close()  # 关闭之前的图形
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')

# 保存图形
plt.savefig('confusion_matrix.png')
plt.show()


# Step 4: 计算召回率和F1分数

# 计算召回率和F1分数
recall = recall_score(true_labels, predicted_labels, average=None)
f1 = f1_score(true_labels, predicted_labels, average=None)

# 计算加权平均召回率和F1分数
weighted_recall = recall_score(true_labels, predicted_labels, average='weighted')
weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')

# 创建包含类别名称的标签
class_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 创建数据框架
metrics_df = pd.DataFrame({'Class': class_labels, 'Recall': recall, 'F1 Score': f1})

# 显示加权平均值
metrics_df.loc[len(metrics_df)] = ['Weighted Average', weighted_recall, weighted_f1]

# Step 5: 绘制召回率和F1分数曲线
plt.close()  # 关闭之前的图形
plt.figure(figsize=(12, 5))

# 绘制召回率曲线图
plt.subplot(1, 2, 1)
plt.bar(metrics_df['Class'], metrics_df['Recall'], color='skyblue')
plt.axhline(y=weighted_recall, color='r', linestyle='--', label='Weighted Recall')
plt.title('Recall by Class')
plt.xlabel('Class')
plt.ylabel('Recall')
plt.xticks(rotation=45)
plt.legend()

# 绘制F1分数曲线图
plt.subplot(1, 2, 2)
plt.bar(metrics_df['Class'], metrics_df['F1 Score'], color='lightgreen')
plt.axhline(y=weighted_f1, color='r', linestyle='--', label='Weighted F1 Score')
plt.title('F1 Score by Class')
plt.xlabel('Class')
plt.ylabel('F1 Score')
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()

# 保存图形
plt.savefig('recall_f1_scores.png')
plt.show()


