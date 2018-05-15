# coding=utf-8
from datetime import datetime

from example_of_computing_anomalies import example_of_computing_anomalies
from image import Image
from lof import compute_lof
from test_tensorflow import test_tensorflow


def run():
    start_time = datetime.now()
    test_tensorflow()

    Image.load_images(limit=None)

    Image.load_user_anomalies()

    # optional
    # Image do PCA to reduce dimensionality https://cs.wikipedia.org/wiki/Anal%C3%BDza_hlavn%C3%ADch_komponent

    compute_lof()

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
        print "accuracy:              %20f (how many percent is right)" % ((tp + tn)/total)
        print "precision:             %20f (tp/(tp + fp))" % (tp/(tp + fp))
        print "recall:                %20f (probability of detection)" % (tp/(tp + fn))
        print "false negative rate:   %20f (miss rate)" % (fn/(tp + fn))
        print "fall-out:              %20f (false alarm detection)" % (fp/(fp + tn))
        print "f1 score:              %20f (0 - worst, 1 - best)" % (2*tp/(2*tp + fp + fn))
        print

    print
    print "execution took %d seconds" % (datetime.now() - start_time).total_seconds()


if __name__ == "__main__":
    run()
