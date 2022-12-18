import streamlit as st
import yaml
import numpy as np

from PIL import Image
from io import BytesIO

from dataset.utils import ValFocusingTransform
from models.mobilenet import MobileNetV3Large
from hydra.utils import instantiate

model_config_path = "/home/dkrivenkov/program/autofocusing/config/model/mobilenet.yaml"
dataset_config_path = "/home/dkrivenkov/program/autofocusing/config/dataset/rgb_dataset.yaml"



if __name__ == "__main__":
    

