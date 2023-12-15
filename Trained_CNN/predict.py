# image_predictor.py
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

class ImagePredictor:
    def __init__(self, model_path, default_image_folder='path_to_images'):
        self.model = load_model(model_path)
        self.image_folder = default_image_folder

    def predict_image(self, image_filename):
        # 轉成224X224
        image_path = f'{self.image_folder}/{image_filename}'
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))

        # 壓縮特徵
        image = image.astype('float32')
        image /= 255.0

        # 模型預測
        image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image)

        # 解析預測結果，返回對應的價格範圍
        predicted_class = np.argmax(prediction)
        if predicted_class == 0:
            return "0-50元"
        elif predicted_class == 1:
            return "50-100元"
        elif predicted_class == 2:
            return "100-150元"
        elif predicted_class == 3:
            return "150-200元"
        elif predicted_class == 4:
            return "200-300元"
        else:  # assumed to be class 5
            return "300元以上"
        

#使用方法
#from image_predictor import ImagePredictor

# 指定模型路徑和默認圖片文件夾
#ImagePredictor(模型位置, 圖片目錄)
predictor = ImagePredictor('CNN-ResNet-DenseNet-01_best_model.h5', '.')
# 預測
result = predictor.predict_image('A1.png')
print(result)