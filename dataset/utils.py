import numpy as np
import torch
import cv2

from torch import Tensor


def split_dataset(
    data,
    train_ratio: float = 0.8,
    smart_split: bool = True,
    two_image_pipeline: bool = False
):
    if smart_split:
        if two_image_pipeline:
            train_size = int(len(data) * train_ratio)
            train_data = []
            val_data = []
            for i, (_, images) in enumerate(data.items()):
                defocus = list(map(
                    lambda x: int(x.split("defocus")[1][:-4]),
                    images
                ))
                defocus2id = dict(zip(defocus, np.arange(len(defocus))))
                if i <= train_size:
                    for defocus, idx in defocus2id.items():
                        delta_defocus = defocus + 2000
                        if delta_defocus in defocus2id:
                            new_idx = defocus2id[delta_defocus]
                            train_data.append((images[idx], images[new_idx]))
                else:
                    for defocus, idx in defocus2id.items():
                        delta_defocus = defocus + 2000
                        if delta_defocus in defocus2id:
                            new_idx = defocus2id[delta_defocus]
                            val_data.append((images[idx], images[new_idx]))
        else:
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


def get_fourier_channel(image: np.ndarray):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)
    eps = 1e-9
    magnitude_spectrum = 20 * np.log(cv2.magnitude(
        dft_shift[..., 0],
        dft_shift[..., 1]) + eps
    )
    magnitude_spectrum_tensor = torch.tensor(magnitude_spectrum)
    return magnitude_spectrum_tensor
