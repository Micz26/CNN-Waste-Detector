from cnnwastedetector.utils import (
    resize_and_duplicate_images,
)
import fire


def main():
    data_path = '../data'
    nothing_path = data_path + '/nothing'

    resize_and_duplicate_images(nothing_path)


if __name__ == '__main__':
    fire.Fire(main)
