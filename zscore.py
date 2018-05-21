import numpy as np
from image import Image


def compute_zscore(threshold, max_number_of_outliers):
    images = Image.get_images()
    features = np.array([i.features for i in images])

    z_scores = np.abs(features - features.mean(axis=0) / features.std(axis=0))
    outliers = [(images[i], max(z_score)) for i, z_score in enumerate(z_scores)]
    outliers.sort(key=lambda x: x[1], reverse=True)

    i = 0
    for image, z_score in outliers:
        if z_score > threshold and i < max_number_of_outliers:
            image.set_as_anomaly('z-score')
            i += 1


if __name__ == "__main__":
    Image.load_images()
    compute_zscore(2.1, 5)
