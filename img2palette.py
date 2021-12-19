import argparse
import numpy
import os
import time

from math import ceil, floor
from numpy import ndarray
from PIL import Image
from numpy.lib.function_base import place
from sklearn.cluster import KMeans
from typing import List, Tuple

def mkpalette(colors: ndarray) -> Image:
    width = 200
    height = 900
    row_height = 100
    image_size = (width, height)
    palette_image = Image.new('RGB', image_size)
    for index, color in enumerate(colors):
        color_tuple = tuple(floor(component) for component in color)
        for x in range(width):
            for y in range(index * row_height, (index + 1) * row_height):
                palette_image.putpixel((x, y), color_tuple)
    
    return palette_image

def read_image(path: str):

    image = Image.open(path)

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
    # Author's note:
    # In K Means, we choose the number of clusters we want. I like that.
    model = KMeans(n_clusters = 9)
    print('Fitting ( this may take some time )...')
    t0 = time.time()
    output_dataset: ndarray = model.fit_predict(numpy_pixel_data)
    t1 = time.time()
    print(f'Fit data in {ceil(t1-t0)} seconds.')

    print('Making palette...')
    palette_image = mkpalette(model.cluster_centers_)
    image.show()
    palette_image.show()    

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