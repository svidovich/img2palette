import argparse
import numpy
import os
import time

from numpy import ndarray
from PIL import Image
from sklearn.cluster import KMeans
from typing import List, Tuple

def read_image(path: str):

    image = Image.open(path)
    # show image with image.show()

    size = image.size
    width = size[0]
    height = size[1]

    print(f'Image mode is {image.mode}; image size is {image.size}')
    pixels: List[Tuple] = list()
    for x in range(width):
        for y in range(height):
            pixels.append(image.getpixel((x, y)))
    print(f'Acquired {len(pixels)} pixels!')
    print('Packing pixels...')
    numpy_pixel_data: ndarray = numpy.array(pixels)


    print('Initiating model...')
    # Estimate clustering structure from vector array.
    # OPTICS (Ordering Points To Identify the Clustering Structure), closely related to DBSCAN, 
    # finds core sample of high density and expands clusters from them [1]_. Unlike DBSCAN, keeps 
    # cluster hierarchy for a variable neighborhood radius. Better suited for usage on large datasets 
    # than the current sklearn implementation of DBSCAN.
    # Author's note:
    # In K Means, we choose the number of clusters we want. I like that.
    model = KMeans(n_clusters = 9)
    print('Fitting ( this may take some time )...')
    t0 = time.time()
    output_dataset: ndarray = model.fit_predict(numpy_pixel_data)
    t1 = time.time()
    print(f'Fit data in {t1-t0} seconds.')
    print(f'Found {len(set(model.labels_))} clusters; labels are of type {type(model.labels_)}')
    print('Cluster centers:')
    for center in model.cluster_centers_:
        print(center)

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True, help='The image to read')
    args = parser.parse_args()
    input_file = args.input_file
    if not os.path.isfile(input_file):
        print(f'No such file {input_file}, exiting.')
        return

    read_image(args.input_file)


if __name__ == '__main__':
    main()