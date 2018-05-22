# coding=utf-8
import json
import os
import warnings
from PIL.Image import ANTIALIAS
from collections import OrderedDict

import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19, VGG19
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.preprocessing import image as keras_image
from matplotlib import pyplot as p

from libraries.pca.pca import pca
from user_anomalies import user_anomalies


class Image(object):
    images_path = 'images'
    cache_path = '.cache'
    anomaly_detectors = set()
    images = {}
    compare_base = {}
    feature_model_name = None
    feature_model = None
    fmf = None

    current_training_data = None
    current_validation_data = None

    def __init__(self, name, path):
        self.path = path
        self.name = name
        self.anomaly_by = {}
        self.features = None
        self.load_features()

    def load_features(self):
        cache_name = os.path.join(self.cache_path, self.name + '-' + self.feature_model_name + '.features')
        if os.path.exists(cache_name):
            with open(cache_name, 'r') as f:
                self.features = json.load(f)
        else:
            print 'computing ' + self.feature_model_name + ' features for ' + self.name + ' (' + str(len(Image.get_images())) + ')'
            self.features = [float(i) for i in self.load_image_features()]
            with open(cache_name, 'w') as f:
                json.dump(self.features, f)

    def load_image_features(self):
        if self.feature_model is None:
            self.set_feature_model(self.feature_model_name)

        image_file = keras_image.load_img(self.path)
        if image_file.size[0] < 200 or image_file.size[1] < 200:
            ratio = max(200.0/image_file.size[0], 200.0/image_file.size[1])
            image_file = image_file.resize((int(image_file.size[0]*ratio), int(image_file.size[1]*ratio)), resample=ANTIALIAS)

        preprocessed_data = keras_image.img_to_array(image_file)
        preprocessed_data = np.expand_dims(preprocessed_data, axis=0)
        preprocessed_data = self.fmf(preprocessed_data)

        res = self.feature_model.predict(preprocessed_data)[0]
        return res

    def set_as_anomaly(self, who_says_that, description=""):
        self.anomaly_detectors.add(who_says_that)
        self.anomaly_by[who_says_that] = description

    def detected_by(self, anomaly_detector):
        return anomaly_detector in self.anomaly_by

    def is_in_base(self):
        return self.name in self.compare_base

    @classmethod
    def load_images(cls, max_images=None, pca_dimensions=None, feature_model_name='vgg16'):
        cls.feature_model_name = feature_model_name
        cls.feature_model = None

        if not os.path.exists(cls.cache_path):
            os.mkdir(cls.cache_path)

        for i, file_name in enumerate(os.listdir(cls.images_path)):
            if max_images is not None and i >= max_images:
                break

            cls.images[file_name] = Image(file_name, os.path.join(cls.images_path, file_name))

        if pca_dimensions:
            cls.perform_pca(pca_dimensions)

        cls.normalize_features()

    @classmethod
    def set_image_as_anomaly(cls, image_name, who_says_that, description=""):
        if image_name in cls.images:
            cls.images[image_name].set_as_anomaly(who_says_that, description)

    @classmethod
    def load_user_anomalies(cls):
        for who_says_that in user_anomalies:
            for image_name, description in user_anomalies[who_says_that].items():
                Image.set_image_as_anomaly(image_name, who_says_that, description)

        for image in Image.get_images():
            if len(image.anomaly_by) >= 2:
                cls.compare_base[image.name] = image

    @classmethod
    def get_images(cls):
        return cls.images.values()

    @classmethod
    def perform_pca(cls, pca_dimensions):
        images = Image.get_images()
        data = [image.features for image in images]
        data, _, _ = pca(data, pca_dimensions)
        for i, new_features in enumerate(data):
            images[i].features = new_features

        cls.normalize_features()

    @classmethod
    def get_feature_dimensions(cls):
        return len(Image.get_images()[0].features)

    @classmethod
    def create_plot(cls):
        print "Creating graph"

        for image in Image.get_images():
            for anomaly_detector, color in zip(Image.anomaly_detectors,
                                               ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", ]):
                if image.detected_by(anomaly_detector):
                    p.scatter(image.features[0], image.features[1], color=color, label=anomaly_detector, alpha=0.3,
                              s=20.0 * 5)

            if image.is_in_base():
                p.scatter(image.features[0], image.features[1], color="#000000", label="anomaly by two and more people",
                          alpha=1, s=10.0 * 5)
            if sum([(1 if image.detected_by(anomaly_detector) else 0) for anomaly_detector in ['lof', 'z-score', 'autoencoder']]) > 1:
                p.scatter(image.features[0], image.features[1], color="#000FFF", label="anomaly by two and more detectors",
                          alpha=1, s=10.0 * 5)
                if image.is_in_base():
                    p.scatter(image.features[0], image.features[1], color="#FFF000", label="intersection of people and detectors",
                          alpha=1, s=10.0 * 5)

            p.scatter(image.features[0], image.features[1], color="#000000", label="all", alpha=1, s=1 * 5)


        handles, labels = p.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        p.legend(by_label.values(), by_label.keys())
        p.show()

    @classmethod
    def normalize_features(cls):
        features = [image.features for image in cls.get_images()]
        max_values = np.max(features, axis=0)
        min_values = np.min(features, axis=0)

        diff = max_values - min_values
        if not np.all(diff):
            problematic_dimensions = ", ".join(str(i + 1) for i, v
                                               in enumerate(diff) if v == 0)
            warnings.warn("No data variation in dimensions: %s. You should "
                          "check your data or disable normalization."
                          % problematic_dimensions)

        features = (features - min_values) / (max_values - min_values)
        features[np.logical_not(np.isfinite(features))] = 0

        for i, image in enumerate(cls.get_images()):
            image.features = features[i].tolist()

    @classmethod
    def set_feature_model(cls, model_name):
        if model_name == 'vgg16':
            cls.feature_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
            cls.fmf = lambda self, x: preprocess_input_vgg16(x)

        elif model_name == 'vgg19':
            cls.feature_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
            cls.fmf = lambda self, x: preprocess_input_vgg19(x)

        elif model_name == 'xception':
            cls.feature_model = Xception(weights='imagenet', include_top=False, pooling='avg')
            cls.fmf = lambda self, x: preprocess_input_xception(x)

        elif model_name == 'resnet50':
            cls.feature_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            cls.fmf = lambda self, x: preprocess_input_resnet50(x)

        else:
            raise ValueError("Unknown model")

    @classmethod
    def get_training_data(cls, size=10):
        assert size >= 2
        images = cls.get_images()
        minimum = int(len(images)/float(size))
        for i in range(0, size):
            low = i*minimum
            high = (i+1)*minimum
            cls.current_training_data = images[:low] + images[high:]
            cls.current_validation_data = images[low:high]
            yield "data in cls.current_..._data"
