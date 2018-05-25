import numpy as np
from image import Image


def compute_zscore(max_number_of_outliers):
    images = Image.current_training_data
    features = np.array([i.features for i in images])

    means = features.mean(axis=0)
    stds = features.std(axis=0)
    z_scores = [max(f) for f in np.abs(features - means / stds)]
    z_scores.sort(reverse=True)

    threshold = z_scores[max_number_of_outliers]

    for image in Image.current_validation_data:
        value = max(np.abs((np.array(image.features) - means) / stds))
        if value > threshold:
            image.set_as_anomaly('z-score')


if __name__ == "__main__":
    Image.load_images()
    compute_zscore(5)
