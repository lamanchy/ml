# coding=utf-8
from __future__ import division

from datetime import datetime

from autoencoder import run_autoencoder
import numpy as np
from image import Image
from lof import run_lof
from test_tensorflow import test_tensorflow
from zscore import compute_zscore


def run(
        anomalies="classbased",
        pca_dimensions=32,
        feature_model_name='xception',
        base_as='intersection',
        filtering=None,
        crossvalidation=1,
        number_of_neighbours=5,
        batch_size=16,
        layers=[(3 / 4, 'sigmoid'), (2 / 3, 'relu')],
):
    start_time = datetime.now()
    test_tensorflow()

                                                                            # vgg16, vgg19, xception, resnet50
    Image.load_images(
        anomalies=anomalies,
        max_images=None,
        pca_dimensions=pca_dimensions,
        feature_model_name=feature_model_name,
        base_as=base_as,
        filtering=filtering,
        # {
        #     'plane': 0.1,
        #     'person': 0.1,
        #     'both': 0.1,
        #     'other': 0.1,
        # }
    )

    stats = {}
    for _ in Image.get_training_data(size=crossvalidation):
        five_percent = int(5.0 * len(Image.get_images()) / 100)

        run_lof(
            number_of_neighbours=number_of_neighbours,
            max_number_of_outliers=five_percent
        )
        run_autoencoder(
            batch_size=batch_size,
            number_of_outliers=five_percent,
            layers=layers,
        )
        compute_zscore(
            max_number_of_outliers=five_percent,
        )

        # for anomaly_detector in Image.anomaly_detectors:
        for anomaly_detector in ['z-score', 'lof', 'autoencoder']:
            tp = tn = fp = fn = total = float(0)
            for image in Image.current_validation_data:
                if image.detected_by(anomaly_detector) and image.is_in_base():
                    tp += 1
                if image.detected_by(anomaly_detector) and not image.is_in_base():
                    fp += 1
                if not image.detected_by(anomaly_detector) and image.is_in_base():
                    fn += 1
                if not image.detected_by(anomaly_detector) and not image.is_in_base():
                    tn += 1
                total += 1

            f1_score = (2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0)
            accuracy = ((tp + tn) / total if total > 0 else 0)
            if anomaly_detector not in stats: stats[anomaly_detector] = {'f1':[], 'acc':[]}
            stats[anomaly_detector]['f1'].append(f1_score)
            stats[anomaly_detector]['acc'].append(accuracy)

    info = ""
    info += "anomalies:            %s\n" % anomalies
    info += "pca_dimensions:       %s\n" % pca_dimensions
    info += "feature_model_name:   %s\n" % feature_model_name
    info += "base_as:              %s\n" % base_as
    info += "filtering:            %s\n" % filtering
    info += "crossvalidation:      %s\n" % crossvalidation
    info += "number_of_neighbours: %s\n" % number_of_neighbours
    info += "batch_size:           %s\n" % batch_size
    info += "layers:               %s\n" % str(layers)
    for anomaly_detector in ['z-score', 'lof', 'autoencoder']:
        # https://en.wikipedia.org/wiki/Confusion_matrix
        info += "Anomaly detector %s:\n" % anomaly_detector
        info += "accuracy:              %20f (how many percent is right)\n" % np.mean(stats[anomaly_detector]['acc'], axis=0)
        # print "precision:             %20f (tp/(tp + fp))" % (tp / (tp + fp) if tp + fp > 0 else 0)
        # print "recall:                %20f (probability of detection)" % (tp / (tp + fn) if tp + fn > 0 else 0)
        # print "false negative rate:   %20f (miss rate)" % (fn / (tp + fn) if tp + fn > 0 else 1)
        # print "fall-out:              %20f (false alarm detection)" % (fp / (fp + tn) if fp + tn > 0 else 1)
        info += "f1 score:              %20f (0 - worst, 1 - best)\n" % np.mean(stats[anomaly_detector]['f1'], axis=0)

    print "execution took %d seconds" % (datetime.now() - start_time).total_seconds()

    # after all computation, reduce dims to 2 and plot result
    Image.perform_pca(2)
    Image.create_plot(info=info)


if __name__ == "__main__":
    run()
