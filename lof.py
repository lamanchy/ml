from image import Image
from libraries.pylof.lof import outliers, LOF
import numpy as np


def run_lof(number_of_neighbours=5, max_number_of_outliers=None):
    print "computing LOF"
    lof = outliers(number_of_neighbours, [tuple(image.features) for image in Image.current_training_data])
    last = (lof[max_number_of_outliers] if len(lof) >= max_number_of_outliers else lof[-1])['lof']

    lof = LOF([tuple(image.features) for image in Image.current_training_data])
    for test in Image.current_validation_data:
        value = lof.local_outlier_factor(number_of_neighbours, tuple(test.features))
        if value > last:
            test.set_as_anomaly('lof')
