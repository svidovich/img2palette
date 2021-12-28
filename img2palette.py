import argparse
import math
import numpy
import os
import time

from functools import partial
from math import ceil, floor
from multiprocessing import cpu_count, Pool, Queue
from numpy import ndarray
from PIL import Image
from numpy.lib.function_base import place
from sklearn.cluster import KMeans
from typing import List, Tuple

PALETTE_MAX = 20
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

final_colors = dict()

def keyhole_globe_generator(**kwargs):
    rounded_color: ndarray = kwargs['rounded_color']
    pixel_data: ndarray = kwargs['pixel_data']
    queue: Queue = kwargs['queue']

    center_r: float = rounded_color[0]
    center_g: float = rounded_color[1]
    center_b: float = rounded_color[2]

    center_colors = [center_r, center_g, center_b]
    color_key = f'{center_r},{center_g},{center_b}'

    color_list = list()
    for rgb_color in pixel_data:
        current_color = [
            rgb_color[0],  # red, hopefully
            rgb_color[1],  # green, hopefully
            rgb_color[2],  # blue, hopefully
        ]

        if numpy.linalg.norm(
            numpy.array(center_colors) - numpy.array(current_color)
                ) < KEYHOLE_DISTANCE:
            color_list.append(rgb_color)

    colors = queue.get()
    colors[color_key] = color_list


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

    processing_pool = Pool(cpu_count())
    processing_queue = Queue()
    # Make the final colors dictionary available for processes to use
    # between them.
    processing_queue.put(final_colors)

    processing_pool.map(
        keyhole_globe_generator,
        [
            {
            'rounded_color': rounded_color,
            'pixel_data': pixel_data,
            'queue': queue,
            } for rounded_color in rounded_colors
        ]
    )

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
    color_count: int = kwargs.get('color_count', 9)
    width = 200
    height = 100 * color_count
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
    color_count: int = kwargs.get('color_count', 9)
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
    model = KMeans(n_clusters = color_count)
    print('Fitting ( this may take some time )...')
    t0 = time.time()
    output_dataset: ndarray = model.fit_predict(numpy_pixel_data)
    t1 = time.time()
    print(f'Fit data in {ceil(t1-t0)} seconds.')

    print('Making palette...')
    palette_image = mkpalette(model, keyhole=keyhole, pixel_data=numpy_pixel_data, color_count=color_count)
    image.show()
    palette_image.show()    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True, help='The image to read')
    parser.add_argument('-x', '--express', action='store_true', required=False, help='Don\'t use the entire image for palette generation')
    parser.add_argument('-k', '--keyhole', action='store_true', required=False, help='Use keyhole style averaging to compute palettes')
    parser.add_argument('-c', '--color-count', required=False, default=9, type=int, help='Number of colors in the final palette. Defaults to 9.')
    args = parser.parse_args()

    color_count = args.color_count
    if color_count > PALETTE_MAX:
        print(f'Whoa there cowboy, {color_count} is too many colors. We will be here all night. I will default to my maximum, {PALETTE_MAX}.')
        color_count = PALETTE_MAX
    input_file = args.input_file
    if not os.path.isfile(input_file):
        print(f'No such file {input_file}, exiting.')
        return

    read_image(args.input_file, express=args.express, keyhole=args.keyhole, color_count=args.color_count)


if __name__ == '__main__':
    main()