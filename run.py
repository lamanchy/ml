# coding=utf-8
from image import Image
from test_tensorflow import test_tensorflow


def run():
    test_tensorflow()

    Image.load_images(limit=3)

    Image.load_user_anomalies()

    for i in Image.get_images():
        print i.path, i.anomaly_by, list(i.features)


if __name__ == "__main__":
    run()
