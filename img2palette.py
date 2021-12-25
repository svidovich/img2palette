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
from typing import List, Tuple

KEYHOLE_DISTANCE = 2.5

def average_3d(color_lists: List[List]) -> List:
    output_averages = list()
    for color_list in color_lists:
        point_count = len(color_list)
        total_x: int = sum([coordinate[0] for coordinate in color_list])
        total_y: int = sum([coordinate[1] for coordinate in color_list])
        total_z: int = sum([coordinate[2] for coordinate in color_list])
        try:
            output_averages.append([
                floor(total_x / point_count or 1),
                floor(total_y / point_count or 1),
                floor(total_z / point_count or 1),
            ])
        except ZeroDivisionError:
            output_averages.append([0, 0, 0])

    return output_averages

def simple_mkpalette(model: KMeans, **kwargs) -> Image:
    width: int = kwargs['width']
    height: int = kwargs['height']
    row_height: int = kwargs['row_height']

    image_size = (width, height)
    palette_image = Image.new('RGB', image_size)

    color_centers  = model.cluster_centers_
    rounded_colors = numpy.floor(color_centers)

    print('Building palette!')
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
    print('Building globes...')
    # TODO: This is a prime candidate for multiprocessing
    for index, rounded_color in enumerate(rounded_colors):
        print(f'Working on color {index}, {rounded_color}.')
        center_r: float = rounded_color[0]
        center_g: float = rounded_color[1]
        center_b: float = rounded_color[2]

        center_colors = [center_r, center_g, center_b]
        color_key = f'{center_r},{center_g},{center_b}'
        final_colors[color_key] = list()
        for pixel_index, rgb_color in enumerate(pixel_data):
            if index % 1000 == 0:
                print(f'Working on pixel {pixel_index} / {len(pixel_data)}\r', end='' if index != len(pixel_data) else '\n')
            current_color = [
                rgb_color[0],  # red, hopefully
                rgb_color[1],  # green, hopefully
                rgb_color[2],  # blue, hopefully
            ]

            if numpy.linalg.norm(
                numpy.array(center_colors) - numpy.array(current_color)
                    ) < KEYHOLE_DISTANCE:
                final_colors[color_key].append(rgb_color)

    print(f'Color keys: {[x for x in final_colors.keys()]}')

    print('Obtaining averages....')
    averaged_colors: list = average_3d(final_colors.values())

    print(f'Color averages: {averaged_colors}')

    print('Building palette!')
    for index, color in enumerate(averaged_colors):
        color_tuple = tuple(int(ordinate) for ordinate in color)
        for x in range(width):
            for y in range(index * row_height, (index + 1) * row_height):
                palette_image.putpixel((x, y), color_tuple)
    return palette_image

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