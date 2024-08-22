import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from typing import Optional, Literal
import fire

from cnnwastedetector import ModelRegularized, BaseModel


def main(
    model: Literal['base', 'regularized'] = 'regularized',
    epochs: int = 10,
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

    model_ = ModelRegularized() if model == 'regularized' else BaseModel()

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_.model.parameters(), lr=0.001, weight_decay=1e-4)

    model_trainer = model_.as_trainer(loss, optimizer)
    model_trainer(epochs)

    if on_save:
        with open(save_path, 'wb') as file:
            pickle.dump(model_trainer.get_model().state_dict(), file)

        print(f'Model saved to {save_path}')


if __name__ == '__main__':
    fire.Fire(main)
