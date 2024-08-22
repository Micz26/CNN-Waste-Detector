# imports
import cv2
import torch
from cnnwastedetector import ModelRegularized, BaseModel
import pickle
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Literal
import fire


def main(
    model: Literal['base', 'regularized', 'vgg16'] = 'vgg16',
    model_path: str = './pickle/vgg16_model.pkl',
):
    if model == 'vgg16':
        model_ = models.vgg16(pretrained=False)
        model_.classifier[6] = nn.Linear(4096, 6)
    elif model == 'regularized':
        model_ = ModelRegularized()
    elif model == 'base':
        model_ = BaseModel()

    with open(model_path, 'rb') as file:
        state_dict = pickle.load(file)

    model_.load_state_dict(state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ = model_.to(device)
    model_.eval()
    print('Model loaded successfully.')

    font = cv2.FONT_HERSHEY_SIMPLEX
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('live_predictions.avi', fourcc, 20.0, (640, 480))
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Error: Could not open camera.')
    else:
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((384, 512)),
                transforms.Normalize(
                    mean=[0.6528, 0.6192, 0.5828], std=[0.1669, 0.1669, 0.1757]
                ),
            ]
        )
        while True:
            ret, frame = cap.read()

            if not ret:
                print('Error: Could not read frame.')
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(frame_rgb)
            input_tensor = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model_(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                max_val_index = probabilities.argmax().item()
                probability_value = probabilities[max_val_index].item()

            classes = ['Cardboard', 'Glass', 'Metal', 'Nothing', 'Paper', 'Plastic']
            obj = classes[max_val_index]

            cv2.putText(frame, obj, (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                str(round(probability_value * 100, 2)) + '%',
                (180, 75),
                font,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            out.write(frame)

            cv2.imshow('Result', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fire.Fire(main)
