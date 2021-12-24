import argparse
import math
import numpy
import os
import time

from math import ceil, floor
from numpy import ndarray
from PIL import Image
from numpy.lib.function_base import place
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple

KEYHOLE_DISTANCE = 2.5

def simple_mkpalette(model: KMeans, **kwargs) -> Image:
    width: int = kwargs['width']
    height: int = kwargs['height']
    row_height: int = kwargs['row_height']

    image_size = (width, height)
    palette_image = Image.new('RGB', image_size)

    color_centers  = model.cluster_centers_
    rounded_colors = numpy.floor(color_centers)
    for index, color in enumerate(rounded_colors):
        color_tuple = tuple(int(ordinate) for ordinate in color)
        for x in range(width):
            for y in range(index * row_height, (index + 1) * row_height):
                palette_image.putpixel((x, y), color_tuple)
    return palette_image


def keyhole_mkpalette(model: KMeans, **kwargs) -> Image:
    pixel_data: ndarray = kwargs['pixel_data']
    width: int = kwargs['width']
    height: int = kwargs['height']
    row_height: int = kwargs['row_height']

    image_size = (width, height)
    palette_image = Image.new('RGB', image_size)
    color_centers  = model.cluster_centers_
    rounded_colors = numpy.floor(color_centers)
    final_colors = dict()
    for rounded_color in rounded_colors:
        print(rounded_color)
        # exit()
        center_r: float = rounded_color[0]
        center_g: float = rounded_color[1]
        center_b: float = rounded_color[2]

        center_colors = [center_r, center_g, center_b]
        color_key = f'{center_r},{center_g},{center_b}'
        final_colors[color_key] = list()

        for rgb_color in pixel_data:
            current_color = [
                rgb_color[0],  # red, hopefully
                rgb_color[1],  # green, hopefully
                rgb_color[2],  # blue, hopefully
            ]
            if math.dist(center_colors, current_color) < KEYHOLE_DISTANCE:
                final_colors[color_key].append(rgb_color)



def mkpalette(model: KMeans, **kwargs) -> Image:
    width = 200
    height = 900
    row_height = 100
    pixel_data = kwargs['pixel_data']
    use_keyhole = kwargs.get('keyhole')
    if use_keyhole:
        return keyhole_mkpalette(model, width=width, height=height, row_height=row_height, pixel_data=pixel_data)
    else:
        return simple_mkpalette(model, width=width, height=height, row_height=row_height)


def read_image(path: str, **kwargs):
    express: bool = kwargs.get('express', False)
    keyhole: bool = kwargs.get('keyhole', False)
    image = Image.open(path)
    size = image.size
    width = size[0]
    height = size[1]

    print(f'Image mode is {image.mode}; image size is {image.size}')
    pixels: List[Tuple] = list()
    for x in range(width):
        for y in range(height):
            if express:
                if x % 2 == 0 and y % 2 == 0:
                    pixels.append(image.getpixel((x, y)))
            else:
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
    palette_image = mkpalette(model, keyhole=keyhole, pixel_data=numpy_pixel_data)
    image.show()
    palette_image.show()    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True, help='The image to read')
    parser.add_argument('-x', '--express', action='store_true', required=False, help='Don\'t use the entire image for palette generation')
    parser.add_argument('-k', '--keyhole', action='store_true', required=False, help='Use keyhole style averaging to compute palettes')
    args = parser.parse_args()
    input_file = args.input_file
    if not os.path.isfile(input_file):
        print(f'No such file {input_file}, exiting.')
        return

    read_image(args.input_file, express=args.express, keyhole=args.keyhole)


if __name__ == '__main__':
    main()