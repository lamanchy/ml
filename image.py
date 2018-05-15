# coding=utf-8
import os

from load_features import load_image_features
from user_anomalies import user_anomalies


class Image(object):
    images_path = 'images'
    images = {}

    def __init__(self, name, path):
        self.path = path
        self.name = name
        self.anomaly_by = {}
        self.features = load_image_features(self.path)

    def set_as_anomaly(self, who_says_that, description=""):
        self.anomaly_by[who_says_that] = description


    @classmethod
    def load_images(cls, limit=None):
        i = 0
        for file_name in os.listdir(cls.images_path):
            if limit is not None and i >= limit:
                break

            i += 1
            print "loading %d. image: " % i, file_name
            cls.images[file_name] = Image(file_name, os.path.join(cls.images_path, file_name))

    @classmethod
    def set_image_as_anomaly(cls, image_name, who_says_that, description=""):
        if image_name in cls.images:
            cls.images[image_name].set_as_anomaly(who_says_that, description)

    @classmethod
    def load_user_anomalies(cls):
        for who_says_that in user_anomalies:
            for image_name, description in user_anomalies[who_says_that].items():
                Image.set_image_as_anomaly(image_name, who_says_that, description)

    @classmethod
    def get_images(cls):
        return cls.images.values()

