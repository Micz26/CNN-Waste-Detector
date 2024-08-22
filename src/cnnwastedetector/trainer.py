from typing import Any
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from cnnwastedetector.utils import DATASET


class Trainer:
    def __init__(
        self, model_: nn.Module, loss: nn.Module, optimizer: optim.Optimizer
    ) -> None:
        self.model_ = model_
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_.to(self.device)
        self.loss = loss
        self.optimizer = optimizer
        self.load_data()

    def __call__(self, epochs) -> Any:
        for epoch in range(epochs):
            self.model_.train()
            running_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = (
                    images.to(self.device),
                    labels.to(self.device),
                )
                self.optimizer.zero_grad()
                outputs = self.model_(images)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(
                f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(self.train_loader)}'
            )

            self.model_.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = (
                        images.to(self.device),
                        labels.to(self.device),
                    )
                    outputs = self.model_(images)
                    loss = self.loss(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(
                f'Validation Loss: {val_loss / len(self.val_loader)}, Accuracy: {100 * correct / total}%'
            )

        print('Finished Training')

    def load_data(self):
        train_set, val_set, test_set = torch.utils.data.random_split(
            DATASET, [2176, 512, 32]
        )
        self.train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=8, shuffle=True)

    def get_model(self):
        return self.model_
