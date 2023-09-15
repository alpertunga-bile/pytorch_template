import numpy as np
from torch import Tensor
from typing import Tuple
from PIL import Image

"""
* Reference
* https://github.com/Lornatang/FSRCNN-PyTorch/tree/master
"""


def get_image_and_numpy_arrays(
    image_path: str,
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    y_image, cb_image, cr_image = Image.open(image_path).convert("YCbCr").split()

    cb_array = np.array(cb_image, dtype=np.float32) / 255.0
    cr_array = np.array(cr_image, dtype=np.float32) / 255.0

    return y_image, cb_array, cr_array


def numpy2pil(image: np.ndarray) -> Image.Image:
    image *= 255.0

    return Image.fromarray(image)


def get_yvalue_as_array(output: Tensor) -> np.ndarray:
    y_value = (
        output.detach()
        .squeeze(0)
        .permute(1, 2, 0)
        .mul(255)
        .clamp(0, 255)
        .cpu()
        .numpy()
        .astype("uint8")
    ).astype(np.float32) / 255.0

    return y_value


def ycbcr2rgb(image: np.ndarray) -> np.ndarray:
    image_dtype = image.dtype

    image *= 255.0

    image = np.matmul(
        image,
        [
            [0.00456621, 0.00456621, 0.00456621],
            [0, -0.00153632, 0.00791071],
            [0.00625893, -0.00318811, 0],
        ],
    ) * 255.0 + [-222.921, 135.576, -276.836]

    image /= 255.0
    image = image.astype(image_dtype)

    return image


def ycbcr2bgr(image: np.ndarray) -> np.ndarray:
    image_dtype = image.dtype

    image *= 255.0

    image = np.matmul(
        image,
        [
            [0.00456621, 0.00456621, 0.00456621],
            [0.00791071, -0.00153632, 0],
            [0, -0.00318811, 0.00625893],
        ],
    ) * 255.0 + [-276.836, 135.576, -222.921]

    image /= 255.0
    image = image.astype(image_dtype)

    return image
