from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import os


data_path = './data' if os.path.exists('./data') else '../data'

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6528, 0.6192, 0.5828], std=[0.1669, 0.1669, 0.1757]
        ),
    ]
)

DATASET = datasets.ImageFolder(root=data_path, transform=transform)


def visualize_batch(batch, predicted_labels):
    class_names = ['cardboard', 'glass', 'metal', 'nothing', 'paper', 'plastic']
    images, true_labels = batch
    images = images.cpu()
    true_labels = true_labels.cpu()
    predicted_labels = predicted_labels.cpu()

    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    axs = axs.flatten()
    mean = torch.tensor([0.6615, 0.6287, 0.5936])
    std = torch.tensor([0.1744, 0.1741, 0.1835])
    transform = transforms.Normalize(-mean / std, 1 / std)

    for i in range(8):
        img = transform(images[i])
        img = img.permute(1, 2, 0)
        axs[i].imshow(img)
        axs[i].axis('off')

        predicted_class = class_names[predicted_labels[i]]
        true_class = class_names[true_labels[i]]
        axs[i].set_title(f'Pred: {predicted_class}\nTrue: {true_class}')
    plt.tight_layout()
    plt.show()
