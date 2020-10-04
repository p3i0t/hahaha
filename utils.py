import torch
import numpy as np
from PIL import Image
import cv2
from torch.nn.functional import interpolate


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def array_to_image(image):
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)  # Convert to PIL.Image
    return image


def tensor_to_image(image):
    return array_to_image(image.numpy())


def crop_resize_back(img_origin, img_crop, box, box_size):
    assert type(img_origin) == type(img_crop)
    img = img_origin.copy()

    if isinstance(img_crop, np.ndarray):
        out = cv2.resize(
            img_crop,
            box_size,
            interpolation=cv2.INTER_AREA
        ).copy()
        img[box[1]:box[3], box[0]:box[2]] = out
    elif isinstance(img_crop, torch.Tensor):
        out = imresample(
            img_crop.permute(2, 0, 1).unsqueeze(0).float(),
            box_size
        ).byte().squeeze(0).permute(1, 2, 0)
        img[box[1]:box[3], box[0]:box[2], :] = out
    else:
        out = img_crop.resize(box_size, Image.BILINEAR)

        img = np.asarray(img).copy()
        out = np.asarray(out)
        # print(img.flags)
        # img.setflags(write=1)
        # print(out.shape)
        # print(img.shape)
        # print(box)
        img[box[1]:box[3], box[0]:box[2]] = out
        img = array_to_image(img)
    return img


def compute_dist(img_a, img_b):
    """Compute the distance of two images.
    Args:
        img_a: 3d array in [h, w, c].
        img_a: 3d array in [h, w, c].
    Return:
        Distance.
    """
    diff = img_a - img_b
    epsilon = 20.4
    diff = np.clip(diff, -epsilon, epsilon)

    pixel_diff_norms = np.linalg.norm(diff, ord=2, axis=2)
    # print(np.sum(pixel_diff_norms))
    return np.mean(pixel_diff_norms)

