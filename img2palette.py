import argparse
import os

from PIL import Image


def read_image(path: str):

    image = Image.open(path)
    # show image with image.show()

    size = image.size
    width = size[0]
    height = size[1]

    print(f'Image mode is {image.mode}; image size is {image.size}')
    pixels = list()
    for x in range(width):
        for y in range(height):
            pixels.append(image.getpixel((x, y)))
    print(f'Acquired {len(pixels)} pixels!')

    

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