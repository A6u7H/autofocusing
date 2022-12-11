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


def convert_image(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def extract_focus_distance(image, model, transform):
    image = Image.open(image)
    image = transform(np.array(image))
    focus = model(image)
    return focus

with open(model_config_path, "r") as model_file:
    model_config = yaml.safe_load(model_file)

with open(dataset_config_path, "r") as dataset_file:
    dataset_config = yaml.safe_load(dataset_file)

transform = ValFocusingTransform(
    dataset_config["val_transform"]["mean"], 
    dataset_config["val_transform"]["std"]
)

model = instantiate(model_config["config"])

st.set_page_config(layout="wide", page_title="Microscope image focus info")
st.write("## Get focus distance from your image")
st.sidebar.write("## Upload and download :gear:")

columns = st.columns(2)
image = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# st.sidebar.download_button("Download fixed image", convert_image(image), "fixed.png", "image/png")

focus = extract_focus_distance(image, model, transform)

