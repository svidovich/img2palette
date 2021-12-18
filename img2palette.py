import argparse

from PIL import Image


def read_image(path: str):
    image = Image.open(path)
    image.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True, help='The image to read')
    args = parser.parse_args()

    read_image(args.input_file)


if __name__ == '__main__':
    main()