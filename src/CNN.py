import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Multiply, Reshape, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Activation, Dropout, BatchNormalization, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import pandas as pd
import os
from datetime import datetime

# 創建模型相關目錄
def create_directories(model_weights_dir, model_name):
    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)
    model_dir = os.path.join(model_weights_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory '{model_name}'.")
    return model_dir

# 載入數據集
def load_dataset(image_dir, labels_file, batch_size=16, target_size=(224, 224)):
    labels_df = pd.read_csv(labels_file)
    labels_df['span'] = labels_df['span'].astype(str)

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest',
        channel_shift_range=0.1)

    train_generator = datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=image_dir,
        x_col='pic_id',
        y_col='span',
        subset="training",
        batch_size=batch_size,
        seed=87,
        shuffle=True,
        class_mode="categorical",
        target_size=target_size,
        workers=4)

    val_generator = datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=image_dir,
        x_col='pic_id',
        y_col='span',
        subset="validation",
        batch_size=batch_size,
        seed=87,
        shuffle=True,
        class_mode="categorical",
        target_size=target_size,
        workers=4)
    
    return train_generator, val_generator

# SE Block（Squeeze-and-Excitation Block）
def squeeze_excite_block(input, ratio=16):
    channel_axis = -1
    filters = input.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(input)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    scale = Multiply()([input, se])

    return scale

# CNN區塊
def conv_block(x, filters, kernel_size, padding, dropout_rate=0.2, use_se=False):
    residual = x

    # 主要卷積路徑
    x = Conv2D(filters, kernel_size, strides=(1, 1), padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  # 減少空間尺寸

    if use_se:
        x = squeeze_excite_block(x)

    # 殘差路徑
    residual = Conv2D(filters, (1, 1), strides=(2, 2), padding='same')(residual)
    
    # 確保空間尺寸匹配以進行特徵相加
    x = Add()([x, residual])
    x = Activation('relu')(x)

    return x

# DNN區塊
def dnn_block(x, units, dropout_rate=0.3, regularizer_rate=0.001):
    x = Dense(units, activation='relu', kernel_regularizer=l2(regularizer_rate))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return x

# 建立模型
def build_model(input_shape=(224, 224, 3), num_classes=6):
    cnn_input = Input(shape=input_shape, name='cnn_input')

    # 添加卷積區塊
    x = conv_block(cnn_input, 32, (3, 3), padding='same')
    x = conv_block(x, 64, (3, 3), padding='same')
    x = conv_block(x, 128, (3, 3), padding='same')
    x = conv_block(x, 256, (3, 3), padding='same')
    x = conv_block(x, 512, (3, 3), padding='same', use_se=True) # Using SE to enhance features

    # 全局平均池化
    x = GlobalAveragePooling2D()(x)

    # 添加DNN
    x = dnn_block(x, 512, 0.3, 0.002)
    x = dnn_block(x, 256, 0.3, 0.002)

    # 最終輸出層
    final_output = Dense(num_classes, activation='softmax', name='price_range')(x)

    return Model(inputs=cnn_input, outputs=final_output)

# 訓練模型
def train_model(model, train_generator, val_generator, model_weights_dir, model_name):
    tensorboard_log_dir = f'./logs/{model_name}/run_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir)

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_weights_dir, f"{model_name}_best_model.h5"),
        save_best_only=True,
        monitor='val_loss',
        mode='min')

    model.fit(
        train_generator,
        epochs=200,
        validation_data=val_generator,
        workers=4,
        callbacks=[tensorboard_callback, checkpoint])
    
    model.save(os.path.join(model_weights_dir, f"{model_name}_final_model"), save_format='tf')

def main():
    # 主程式設定
    model_name = 'CNN-ResNet-01'
    model_weights_dir = "model_weights"
    image_dir = './JPG-224-2/'
    labels_file = './final-2.csv'

    # 創建模型相關目錄
    model_dir = create_directories(model_weights_dir, model_name)

    # 載入訓練和驗證數據
    train_generator, val_generator = load_dataset(image_dir, labels_file)

    # 構建並編譯模型
    model = build_model()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    # 生成模型結構圖
    plot_model(model, to_file=f'{model_name}.png', show_shapes=True, show_layer_names=True)

    # 訓練模型
    train_model(model, train_generator, val_generator, model_dir, model_name)

if __name__ == "__main__":
    main()