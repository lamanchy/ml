# coding=utf-8
from datetime import datetime

from autoencoder import run_autoencoder
from image import Image
from lof import run_lof
from test_tensorflow import test_tensorflow
from zscore import compute_zscore


def run():
    start_time = datetime.now()
    test_tensorflow()

                                                                            # vgg16, vgg19, xception, resnet50
    Image.load_images(max_images=None, pca_dimensions=512, feature_model_name='vgg16')

    Image.load_user_anomalies()

    five_percent = 5.0 * len(Image.get_images()) / 100

    # run_lof(
    #     number_of_neighbours=5,
    #     max_number_of_outliers=five_percent
    # )
    run_autoencoder(
        batch_size=16,
        number_of_outliers=five_percent,
        layers=[(3/4, 'sigmoid'), (2/3, 'relu'), (1/6, 'relu'), (1/12, 'relu')]
    )
    compute_zscore(
        threshold=0,
        max_number_of_outliers=five_percent
    )

    for anomaly_detector in Image.anomaly_detectors:
        tp = tn = fp = fn = total = float(0)
        for image in Image.get_images():
            if image.detected_by(anomaly_detector) and image.is_in_base():
                tp += 1
            if image.detected_by(anomaly_detector) and not image.is_in_base():
                fp += 1
            if not image.detected_by(anomaly_detector) and image.is_in_base():
                fn += 1
            if not image.detected_by(anomaly_detector) and not image.is_in_base():
                tn += 1
            total += 1

        # https://en.wikipedia.org/wiki/Confusion_matrix
        print
        print "Anomaly detector %s:" % anomaly_detector
        print "accuracy:              %20f (how many percent is right)" % ((tp + tn) / total if total > 0 else 0)
        print "precision:             %20f (tp/(tp + fp))" % (tp / (tp + fp) if tp + fp > 0 else 0)
        print "recall:                %20f (probability of detection)" % (tp / (tp + fn) if tp + fn > 0 else 0)
        print "false negative rate:   %20f (miss rate)" % (fn / (tp + fn) if tp + fn > 0 else 1)
        print "fall-out:              %20f (false alarm detection)" % (fp / (fp + tn) if fp + tn > 0 else 1)
        print "f1 score:              %20f (0 - worst, 1 - best)" % \
              (2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0)
        print

    print
    print "execution took %d seconds" % (datetime.now() - start_time).total_seconds()

    # after all computation, reduce dims to 2 and plot result
    Image.perform_pca(2)
    Image.create_plot()


if __name__ == "__main__":
    run()
