import os
from PIL import Image
from torchvision import transforms


def resize_and_duplicate_images(
    folder_path: str, new_size: tuple[int, int] = (384, 512)
) -> None:
    transform = transforms.Compose([transforms.Resize(new_size), transforms.ToTensor()])

    image_counter = 1

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, filename)

            img = Image.open(img_path)
            img_resized = transform(img)

            img_resized = transforms.ToPILImage()(img_resized)

            for i in range(30):
                new_filename = f'nothing{image_counter}.jpg'
                new_img_path = os.path.join(folder_path, new_filename)
                img_resized.save(new_img_path)
                print(f'Saved: {new_filename}')
                image_counter += 1
