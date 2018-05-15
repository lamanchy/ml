# coding=utf-8
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image as keras_image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False, pooling='avg')


def load_image_features(image_path):
    #                   image is instance of Image from image.py

    image_file = keras_image.load_img(image_path, target_size=(224, 224))

    preprocessed_data = keras_image.img_to_array(image_file)
    preprocessed_data = np.expand_dims(preprocessed_data, axis=0)
    preprocessed_data = preprocess_input(preprocessed_data)

    return model.predict(preprocessed_data)[0]


# test
if __name__ == "__main__":
    print load_image_features('images/2007_000032.jpg')
