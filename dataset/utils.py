import numpy as np
import torch
import cv2

from torch import Tensor


def split_dataset(data, train_ratio: float = 0.8):
    train_size = int(len(data) * train_ratio)
    data = np.random.permutation(data)
    return data[:train_size], data[train_size:]


def get_fourier_channel(image: Tensor):
    img_numpy = image.numpy().transpose(1, 2, 0)
    img_gray = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2GRAY)
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)
    eps = 1e-9
    magnitude_spectrum = 20 * np.log(cv2.magnitude(
        dft_shift[..., 0],
        dft_shift[..., 1]) + eps
    )
    magnitude_spectrum_tensor = torch.tensor(magnitude_spectrum)
    return magnitude_spectrum_tensor
