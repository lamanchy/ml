import numpy as np
from image import Image


def compute_zscore(threshold):
    images = Image.get_images()
    features = np.array([i.features for i in images])

    z_scores = np.abs(features - features.mean(axis=0) / features.std(axis=0))
    for i, z_score in enumerate(z_scores):
        if (z_score > threshold).any():
            images[i].set_as_anomaly('z-score')


if __name__ == "__main__":
    Image.load_images()
    compute_zscore(2.1)
