# coding=utf-8
from example_of_computing_anomalies import example_of_computing_anomalies
from image import Image
from test_tensorflow import test_tensorflow


def run():
    test_tensorflow()

    Image.load_images(limit=None)

    Image.load_user_anomalies()

    # optional
    # Image do PCA to reduce dimensionality https://cs.wikipedia.org/wiki/Anal%C3%BDza_hlavn%C3%ADch_komponent

    example_of_computing_anomalies()

    # process anomalies, for now only print them
    for i in Image.get_images():
        # print i.path, i.anomaly_by, list(i.features)

        if len(i.anomaly_by) >= 1:
            print i.name


if __name__ == "__main__":
    run()
