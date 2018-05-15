from image import Image
from libraries.pylof.lof import outliers


def compute_lof():
    print "computing LOF"
    lof = outliers(1, [tuple(image.features) for image in Image.get_images()], normalize=True)

    i = 1
    for outlier in lof[:20]:
        for image in Image.get_images():
            if list(outlier["instance"]) == image.features:
                image.set_as_anomaly('lof', '%d. outlier' % i)
                i += 1
                break