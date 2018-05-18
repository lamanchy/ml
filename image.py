# coding=utf-8
import json
import os
from collections import OrderedDict

from libraries.pca.pca import pca
import warnings
import numpy as np
from load_features import load_image_features
from user_anomalies import user_anomalies
from matplotlib import pyplot as p


class Image(object):
    images_path = 'images'
    cache_path = '.cache'
    anomaly_detectors = set()
    images = {}
    compare_base = {}

    def __init__(self, name, path):
        self.path = path
        self.name = name
        self.anomaly_by = {}
        self.features = None
        self.load_features()

    def load_features(self):
        cache_name = os.path.join(self.cache_path, self.name + '.features')
        if os.path.exists(cache_name):
            with open(cache_name, 'r') as f:
                self.features = json.load(f)
        else:
            print 'computing features for ' + self.name
            self.features = [float(i) for i in load_image_features(self.path)]
            with open(cache_name, 'w') as f:
                json.dump(self.features, f)

    def set_as_anomaly(self, who_says_that, description=""):
        self.anomaly_detectors.add(who_says_that)
        self.anomaly_by[who_says_that] = description

    def detected_by(self, anomaly_detector):
        return anomaly_detector in self.anomaly_by

    def is_in_base(self):
        return self.name in self.compare_base


    @classmethod
    def load_images(cls, max_images=None, pca_dimensions=None):
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
            for anomaly_detector, color in zip(Image.anomaly_detectors, ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", ]):
                if image.detected_by(anomaly_detector):
                    p.scatter(image.features[0], image.features[1], color=color, label=anomaly_detector, alpha=0.3, s=20.0*5)

            if image.is_in_base():
                p.scatter(image.features[0], image.features[1], color="#000000", label="anomaly by two and more people", alpha=1, s=10.0*5)
            p.scatter(image.features[0], image.features[1], color="#000000", label="all", alpha=1, s=1*5)

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
