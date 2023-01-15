import streamlit as st
import yaml
import os
import torch
import numpy as np
import base64

from PIL import Image

from dataset.transform import TestFocusingTransform, TrainFocusingTransform
from models.mobilenet import MobileNetV3Large

model_config_path = "/home/dkrivenkov/program/autofocusing/config/model/mobilenet.yaml"
dataset_config_path = "/home/dkrivenkov/program/autofocusing/config/dataset/rgb_dataset.yaml"

runs_path = "/home/dkrivenkov/program/autofocusing/experiments/mobilenet/runs/2023-01-15_18-47-11"
config_path = os.path.join(runs_path, ".hydra", "config.yaml")
weight_path = os.path.join(runs_path, "weight", "epoch=40-step=32964.ckpt")


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_state_dict(checkpoint):
    correct_state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        new_name = ".".join(k.split(".")[1:])
        correct_state_dict[new_name] = v
    return correct_state_dict


class ModelConfig:
    def __init__(self, configuration):
        self.set_parameters(configuration)

    def set_parameters(self, configuration):
        for k, v in configuration["config"].items():
            setattr(self, k, v)


def predict(model, image):
    with torch.no_grad():
        return model(image)


with open(config_path, "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

transform_mean = config["dataset"]["test_transform"]["mean"]
transform_std = config["dataset"]["test_transform"]["std"]
transform_crop_size = config["dataset"]["test_transform"]["crop_size"]

config_class = ModelConfig(config["model"])
checkpoint = torch.load(weight_path, map_location="cpu")
state_dict = get_state_dict(checkpoint)

if __name__ == "__main__":
    upload = st.file_uploader(
        "Insert image for classification",
        type=['png', 'jpg']
    )
    model = MobileNetV3Large(config_class)
    model.load_state_dict(state_dict)
    model.eval()

    c1, c2 = st.columns(2)
    if upload is not None:
        image = Image.open(upload)
        image = np.asarray(image)
        c1.header('Input Image')
        c1.image(image)

        predictions = []
        transform = TestFocusingTransform(
            transform_mean,
            transform_std,
            False,
            transform_crop_size
        )
        images = transform(image)
        image_count = len(images)
        for i in range(image_count):
            predictions.append(
                predict(model, images[i].unsqueeze(0))
            )

        predictions = torch.cat(predictions, dim=1)
        pred_focus = torch.median(predictions, dim=1, keepdim=True)[0]
        c2.header("Prediction")
        c2.write(pred_focus.item())
