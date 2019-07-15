import numpy as np
import torch
import util.util as util
import video_utils
from PIL import Image

## BASIC TRANSFORMS (zoom, translate, rotate)

def zoom_in(input_tensor, zoom_level=4):
    """Return a zoomed-in tensor"""
    with torch.no_grad():
        img_nda = util.tensor2im(input_tensor.data[0])
        img_pil = Image.fromarray(img_nda)
        img_pil = img_pil.transform(
            img_pil.size,
            Image.EXTENT,
            data=(zoom_level, zoom_level, img_pil.width-zoom_level, img_pil.height-zoom_level),
            resample=Image.BILINEAR
        )
        return video_utils.im2tensor(img_pil)

def translate(input_tensor, translation_tuple):
    """Apply the translation to tensor"""
    with torch.no_grad():
        img_nda = util.tensor2im(input_tensor.data[0])
        img_pil = Image.fromarray(img_nda)
        img_pil = img_pil.transform(
            img_pil.size, 
            Image.AFFINE, 
            data = translation_tuple,
            resample=Image.BILINEAR
        )
        return video_utils.im2tensor(img_pil)


def rotate(input_tensor, rotation_level=4):
    """Return a rotated tensor"""
    with torch.no_grad():
        img_nda = util.tensor2im(input_tensor.data[0])
        img_pil = Image.fromarray(img_nda)
        img_pil = img_pil.rotate(rotation_level, resample=Image.BILINEAR)
        return video_utils.im2tensor(img_pil)

def concatenate(left_tensor, right_tensor):

    left_nda = util.tensor2im(left_tensor.data[0])
    left_pil = Image.fromarray(left_nda)

    right_nda = util.tensor2im(right_tensor.data[0])
    right_pil = Image.fromarray(right_nda)

    width, height = left_pil.width*2, right_pil.height

    new_im = Image.new('RGB', (width, height))
    new_im.paste(left_pil, (0,0))
    new_im.paste(right_pil, (left_pil.width,0))

    return video_utils.im2tensor(new_im)

def flip_left_right(input_tensor):
    with torch.no_grad():
        img_nda = util.tensor2im(input_tensor.data[0])
        img_pil = Image.fromarray(img_nda)
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        return video_utils.im2tensor(img_pil)


## HEAT-SEEKING
SMOOTHING_WINDOW_SIZE = 30
IDENTITY_TRANSFORM = (1, 0, 0, 0, 1, 0)
transform_history = [IDENTITY_TRANSFORM for i in range(SMOOTHING_WINDOW_SIZE)]

def get_homing_direction(tensor):
    """Return the most "intense" corner of the tensor"""
    with torch.no_grad():
        # torch image tensors are (?, 3, H, W)
        mid_height = tensor.shape[-2] // 2
        mid_width = tensor.shape[-1] // 2

        corner_intensity = {
            "top_left": tensor.data[0][:,:mid_height,:mid_width].mean().item(),
            "top_right": tensor.data[0][:,:mid_height,mid_width:].mean().item(),
            "bottom_left": tensor.data[0][:,mid_height:,:mid_width].mean().item(),
            "bottom_right": tensor.data[0][:,mid_height:,mid_width:].mean().item(),
        }
        itense_corner = sorted(corner_intensity, key=corner_intensity.get, reverse=True)[0]
        return itense_corner

def get_homing_translation(tensor, translation_level=4):
    """Return the required homing PIL translation tuple"""
    # Note: PIL affine transform format is (1, 0, left/right, 0, 1, up/down)
    translation_strategies = {
        "top_left": (1, 0, -translation_level, 0, 1, -translation_level),
        "top_right": (1, 0, translation_level, 0, 1, -translation_level),
        "bottom_left": (1, 0, -translation_level, 0, 1, translation_level),
        "bottom_right": (1, 0, translation_level, 0, 1, translation_level),
    }
    return translation_strategies[
        get_homing_direction(tensor)
    ]

def heat_seeking(input_tensor, translation_level=4, zoom_level=4):
    """Return an image tensor slightly and smoothly shifted (and zoomed) towards the most intense corner"""
    global transform_history
    required_translation = get_homing_translation(input_tensor, translation_level=translation_level)
    transform_history.append(required_translation)
    transform_history = transform_history[-SMOOTHING_WINDOW_SIZE:]   
    smoothed_translation = tuple(
        np.average(
            np.array(transform_history), # uniform moving average, could be something else
            axis=0
        )
    )
    tranformed_tensor = translate(input_tensor, smoothed_translation)
    tranformed_tensor = zoom_in(tranformed_tensor, zoom_level=zoom_level)
    
    return tranformed_tensor
