import torch
import torch.nn as nn
import torchvision.models as models
import pickle
from typing import Optional
import fire
import torch.optim as optim

from cnnwastedetector import Trainer


def main(
    epochs: int = 5,
    on_save: bool = False,
    save_path: Optional[str] = None,
):
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using {device} device')

    vgg16 = models.vgg16(pretrained=True)

    for param in vgg16.features.parameters():
        param.requires_grad = False

    num_classes = 6
    vgg16.classifier[6] = nn.Linear(4096, num_classes)

    vgg16 = vgg16.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.001, weight_decay=1e-4)

    vgg16_trainer = Trainer(vgg16, loss, optimizer)
    vgg16_trainer(epochs)

    if on_save:
        with open(save_path, 'wb') as file:
            pickle.dump(vgg16_trainer.get_model().state_dict(), file)

        print(f'Model saved to {save_path}')


if __name__ == '__main__':
    fire.Fire(main)
