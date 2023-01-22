import numpy as np
import torch
import cv2

from torch import Tensor


def split_dataset(data, train_ratio: float = 0.8, smart_split: bool = True):
    if smart_split:
        train_size = int(len(data) * train_ratio)
        train_data = []
        val_data = []
        for i, (_, images) in enumerate(data.items()):
            if i <= train_size:
                train_data.extend(images)
            else:
                val_data.extend(images)
    else:
        all_data = []
        for v in data.values():
            all_data.extend(v)
        all_data = np.array(all_data)
        train_size = int(len(all_data) * train_ratio)

        indices = np.random.permutation(len(all_data))
        train_idx, val_idx = indices[:train_size], indices[train_size:]
        train_data, val_data = all_data[train_idx], all_data[val_idx]
    return train_data, val_data


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
