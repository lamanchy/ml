# coding=utf-8
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image as keras_image
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
import numpy as np


def load_image_features(image_path, model):
    #                   image is instance of Image from image.py
    image_file = keras_image.load_img(image_path)

    preprocessed_data = keras_image.img_to_array(image_file)
    preprocessed_data = np.expand_dims(preprocessed_data, axis=0)

    if model == VGG16:
        model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        preprocessed_data = preprocess_input_vgg16(preprocessed_data)

        return model.predict(preprocessed_data)[0]

    if model == VGG19:
        model = VGG19(weights='imagenet', include_top=False, pooling='avg')
        preprocessed_data = preprocess_input_vgg19(preprocessed_data)
        print model.predict(preprocessed_data)[0]

        return model.predict(preprocessed_data)[0]


# test
if __name__ == "__main__":
    print load_image_features('images/2007_000032.jpg')
