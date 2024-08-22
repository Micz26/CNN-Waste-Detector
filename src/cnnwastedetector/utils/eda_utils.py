import os
import random as r
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils.data import DataLoader


def dataset_histogram(data_path: str) -> None:
    description = {}

    for subdir in os.listdir(data_path):
        subdir_path = os.path.join(data_path, subdir)

        if os.path.isdir(subdir_path):
            count = len(
                [
                    name
                    for name in os.listdir(subdir_path)
                    if os.path.isfile(os.path.join(subdir_path, name))
                ]
            )
            description[subdir] = count

    plt.figure(figsize=(10, 6))
    plt.bar(description.keys(), description.values(), color='skyblue')
    plt.xlabel('Trash Type')
    plt.ylabel('Count')
    plt.title('Distribution of Trash Types in Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_images_pixels(images_path: str) -> None:
    subdirs = [
        d
        for d in os.listdir(images_path)
        if os.path.isdir(os.path.join(images_path, d))
    ]

    num_rows = 2
    num_cols = 3

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    axes = axes.flatten()

    for i, subdir in enumerate(subdirs):
        images_width = []
        images_height = []

        subdir_path = os.path.join(images_path, subdir)
        images = os.listdir(subdir_path)

        for image_file in images:
            image_path = os.path.join(subdir_path, image_file)
            image = Image.open(image_path)
            images_width.append(image.size[0])
            images_height.append(image.size[1])

        axes[i].scatter(images_width, images_height, c='blue', alpha=0.5)
        axes[i].set_title(f'{subdir} ({len(images)} images)')
        axes[i].set_xlabel('Image Width (pixels)')
        axes[i].set_ylabel('Image Height (pixels)')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_rgb_histograms(images_path: str) -> None:
    images = os.listdir(images_path)
    r.shuffle(images)

    if not os.path.isdir(images_path):
        raise FileNotFoundError(f'The directory {images_path} does not exist.')

    r_values = []
    g_values = []
    b_values = []

    for image_file in images[:200]:
        image_path = os.path.join(images_path, image_file)
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f'Could not read image {image_path}')

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            B, G, R = cv2.split(image_rgb)

            r_values.extend(R.flatten())
            g_values.extend(G.flatten())
            b_values.extend(B.flatten())
        except Exception as e:
            print(f'Could not process {image_file}: {e}')

    R_histo = cv2.calcHist([np.array(r_values)], [0], None, [256], [0, 256])
    G_histo = cv2.calcHist([np.array(g_values)], [0], None, [256], [0, 256])
    B_histo = cv2.calcHist([np.array(b_values)], [0], None, [256], [0, 256])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(R_histo, color='red')
    axes[0].set_title('Red Channel')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Frequency')

    axes[1].plot(G_histo, color='green')
    axes[1].set_title('Green Channel')
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Frequency')

    axes[2].plot(B_histo, color='blue')
    axes[2].set_title('Blue Channel')
    axes[2].set_xlabel('Pixel Value')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_sample_images(images_path: str) -> None:
    images = os.listdir(images_path)
    r.shuffle(images)

    fig, axs = plt.subplots(1, 4, figsize=(15, 15))
    for i, image_file in enumerate(images[:4]):
        image_path = os.path.join(images_path, image_file)
        im = Image.open(image_path)

        # Plot image using axs
        axs[i].imshow(im)
        axs[i].axis('off')  # Hide axes
        axs[i].set_title(str(image_file))

    plt.tight_layout()
    plt.show()


def calculate_mean_std(loader: DataLoader) -> tuple[list[int], list[int]]:
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += images.size(0)

    mean /= total_images_count
    std /= total_images_count

    return mean, std
