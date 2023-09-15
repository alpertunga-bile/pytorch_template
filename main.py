import torch
from torchvision import transforms
from torch_utilities.consts import available_device
from torch_utilities.model.fsrcnn.fsrcnn import FSRCNN
from torch_utilities.model.fsrcnn.fsrcnn_utility import *
from cv2 import merge, imwrite


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    set_seed()

    D_VALUE = 56
    S_VALUE = 12
    M_VALUE = 4
    EPOCHS = 5
    HALF_IMAGE_SIZE = 300

    model = FSRCNN(D_VALUE, S_VALUE, M_VALUE).to(available_device)

    model.load_state_dict(torch.load("models/FSRCNN.pth"))
    model.eval()

    test_image, cb_array, cr_array = get_image_and_numpy_arrays(
        "dataset/hard_210_1100.jpg"
    )

    test_image_tensor = (
        transforms.ToTensor()(test_image.resize((HALF_IMAGE_SIZE, HALF_IMAGE_SIZE)))
        .unsqueeze(0)
        .to(available_device)
    )

    output_tensor = model(test_image_tensor).clamp_(0.0, 1.0)

    y_value = get_yvalue_as_array(output_tensor)

    ycbcr_image = merge([y_value, cb_array, cr_array])
    output_image = ycbcr2bgr(ycbcr_image) * 255.0

    imwrite("test.png", output_image)
