import streamlit as st
import yaml
import os
import torch
import numpy as np
import base64

from PIL import Image

from dataset.utils import TestFocusingTransform
from models.mobilenet import MobileNetV3Large

model_config_path = "/home/dkrivenkov/program/autofocusing/config/model/mobilenet.yaml"
dataset_config_path = "/home/dkrivenkov/program/autofocusing/config/dataset/rgb_dataset.yaml"

runs_path = "/home/dkrivenkov/program/autofocusing/experiments/mobilenetv3/runs/2022-12-18_19-45-27"
config_path = os.path.join(runs_path, ".hydra", "config.yaml")
weight_path = os.path.join(runs_path, "weight", "epoch=17-step=14472.ckpt")

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


def img_to_patch(image, patch_size, flatten_channels=True):
    B, C, H, W = image.shape
    image = image.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    image = image.permute(0, 2, 4, 1, 3, 5)
    return image.flatten(1,2)

def predict(model, image):
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
    upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
    model = MobileNetV3Large(config_class)
    model.load_state_dict(state_dict)

    c1, c2= st.columns(2)
    if upload is not None:
        image = Image.open(upload)
        image = np.asarray(image)
        c1.header('Input Image')
        c1.image(image)

        predictions = []
        if image.shape[1] == 224:
            transform = TestFocusingTransform(transform_mean, transform_std, (224, 224))
            image_tensor = transform(image)["image"]
            image_tensor = image_tensor.unsqueeze(0)
            predictions.append(predict(model, image_tensor))
        else:
            transform = TestFocusingTransform(transform_mean, transform_std, transform_crop_size)
            image_tensor = transform(image)["image"]
            images = img_to_patch(image_tensor.unsqueeze(0), 224)
            image_count = images.shape[1]
            for i in range(image_count):
                predictions.append(predict(model, images[:, i]))

        predictions = torch.cat(predictions, dim=1)
        pred_focus = torch.median(predictions, dim=1, keepdim=True)[0]
        c2.header(f"Prediction")
        c2.write(pred_focus.item())
