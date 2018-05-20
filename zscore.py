import numpy as np
from image import Image


def compute_zscore():
    threshold = 2.5

    images = Image.get_images()
    image_feat = []
    for image in images:
        image_feat.append(image.features)

    mean = np.mean(image_feat)
    print "mean: ", mean
    stdev = np.std(image_feat)
    print "stdev: ", stdev
    z_scores = [(image - mean) / stdev for image in image_feat]
    # print "z_scores", np.abs(z_scores)

    anomalies = np.where(np.abs(z_scores) > threshold)
    print "anomalies: ", anomalies

    for i, outlier in enumerate(anomalies):
        for image in Image.get_images():
            # if anomalies()[0] == image:
            # # if list(outlier["instance"]) == image[i]:
            #     image.set_as_anomaly('zscore', 'outlier')
            break

