from image import Image
from libraries.pylof.lof import outliers


def compute_lof(number_of_neighbours=5, max_number_of_outliers=None):
    print "computing LOF"
    lof = outliers(number_of_neighbours, [tuple(image.features) for image in Image.get_images()], normalize=True)

    for i, outlier in enumerate(lof):
        for image in Image.get_images():
            if list(outlier["instance"]) == image.features:
                image.set_as_anomaly('lof', '%d. outlier' % i)
                break

        if max_number_of_outliers is not None and i+1 >= max_number_of_outliers:
            break
