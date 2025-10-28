import torch
import PIL.Image as Image
import numpy as np
from typing import Union, List, Tuple, Any, Optional

"""
TODO: This file aims to provide preprocessed sketch image for training and generating.

1) We might acquire identical sketch images for each object, which means the rest part of the sketch should be ignored.
We can (a) mask rest part with gray / black / white color. (b) just zoom in the object. This operation will create different
sketches for different objects. Note that you might need multiple sketches by 2 as DiT generation process use ?CFG?.

2) To implement Spatial Gating Attention for sketch fusion. We need to prepare gating weight for sketches. Exactly, during 
cross attention computing, a gating weight K should be added to the attention score KV / sqrt(n). The weight K is big as
this patch (token) contains edges while small if this patch contains no lines.  
"""

import cv2
import numpy as np

def split_rgb_mask(rgb_image, seg_image):
    if isinstance(rgb_image, str):
        rgb_image = Image.open(rgb_image)
    if isinstance(seg_image, str):
        seg_image = Image.open(seg_image)
    rgb_image = rgb_image.convert("RGB")
    seg_image = seg_image.convert("L")

    rgb_array = np.array(rgb_image)
    seg_array = np.array(seg_image)

    label_ids = np.unique(seg_array)
    label_ids = label_ids[label_ids > 0]

    instance_rgbs, instance_masks, scene_rgbs = [], [], []

    for segment_id in sorted(label_ids):
        # Here we set the background to white
        white_background = np.ones_like(rgb_array) * 255

        mask = np.zeros_like(seg_array, dtype=np.uint8)
        mask[seg_array == segment_id] = 255
        segment_rgb = white_background.copy()
        segment_rgb[mask == 255] = rgb_array[mask == 255]

        segment_rgb_image = Image.fromarray(segment_rgb)
        segment_mask_image = Image.fromarray(mask)
        instance_rgbs.append(segment_rgb_image)
        instance_masks.append(segment_mask_image)
        scene_rgbs.append(rgb_image)

    return instance_rgbs, instance_masks, scene_rgbs

def expand_mask_cv2(mask_array: np.ndarray, iterations: int = 5, kernel_size: int = 3) -> np.ndarray:
    """
    膨胀mask，用来挖出草图中的物体部分，之所以要扩展几个单位，是因为草图的线条就在边缘，扩展两个单位避免把边缘的线条去掉了。

    """
    # @TODO: Need Test
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    expanded_mask = cv2.dilate(mask_array, kernel, iterations=iterations)

    return expanded_mask

def get_min_bounding_box(mask_array: np.ndarray) -> tuple:
    """
        Calculate boundary position of mask
    """
    y_coords, x_coords = np.where(mask_array == 1)
    if y_coords.size == 0:
        return None

    y_min = y_coords.min()
    y_max = y_coords.max()
    x_min = x_coords.min()
    x_max = x_coords.max()
    return y_min, y_max, x_min, x_max

def preprocess_input_for_prepare_sketch_images(seg_images):
    if isinstance(seg_images, list) and isinstance(seg_images[0], Image.Image):
        seg_images = [np.array(image).astype(np.float32) / 255.0 for image in seg_images]

    elif isinstance(seg_images, list) and isinstance(seg_images[0], np.ndarray):
        seg_images = [(image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image) for image in seg_images]
    elif isinstance(seg_images, list) and isinstance(seg_images[0], torch.Tensor):
        seg_images = [image.cpu().numpy() for image in seg_images]

    return seg_images

@torch.no_grad()
def prepare_sketch_images(
        sketch_images: Union[List[Image.Image]],
        num_object_per_scene: Union[int, List[int]],
        batch_size: int,
        seg_images: List[List]= None,
        cfg: bool = False,
        mode:str="no"
):
    """
    输入每个物体的mask，以及这个场景的草图，得到各个物体对应的草图特征。分为三种模式。注意，必须加上最外层的batch（即使batch是1）。
    Use split_rgb_mask() first to get seg_images input.
    Test done (main function test three different mode)
    :return: List[List[Image]].
    """
    # @TODO: 目前没搞懂数据的形状，不知道形状就不太好理解 Multi-Instance Attention计算逻辑。后续再改。这里prepare_sketch_images返回的
    # @TODO: 形状是 [B, I]，每一个元素都是PIL.Image，如果cfg == True，I *= 2。这里不是特别理解。
    assert mode in ["no", "mask", "zoom"], f"Unsupported sketch mode: {mode}. Current support mode includes: no, mask, zoom"
    seg_images = [preprocess_input_for_prepare_sketch_images(seg_image_list) for seg_image_list in seg_images]
    if isinstance(sketch_images,list):
        assert len(sketch_images) == batch_size, f"No match batch size: {batch_size} and {sketch_images}"
    if isinstance(num_object_per_scene,int):
        num_object_per_scene = [num_object_per_scene] * batch_size
    if mode == "no":
        if cfg:
            sketch_image_list = [[sketch_images[i]] * (2 *num_object_per_scene[i]) for i in range(batch_size)]
        else:
            sketch_image_list = [[sketch_images[i]] * num_object_per_scene[i] for i in range(batch_size)]
    elif mode == "mask":
        assert seg_images is not None, f"No seg images."
        sketch_image_list = []
        for idx,(seg_image, sketch_image,) in enumerate(zip(seg_images, sketch_images)):
            sketch_image_curr_list = []
            for j in range(num_object_per_scene[idx]):
                seg_image_normalized = seg_image[j]
                dilated_mask = expand_mask_cv2(seg_image_normalized)
                sketch_image_np = np.array(sketch_image)
                sketch_image_curr_list.append(Image.fromarray(((dilated_mask[...,None] * sketch_image_np) * 255.0).astype(np.uint8)))
            sketch_image_list.append(sketch_image_curr_list)
    elif mode == "zoom":
        assert seg_images is not None, f"No seg images."
        sketch_image_list = []
        for idx,(seg_image, sketch_image,) in enumerate(zip(seg_images, sketch_images)):
            sketch_image_curr_list = []
            for j in range(num_object_per_scene[idx]):
                seg_image_normalized = seg_image[j]
                y_min, y_max, x_min, x_max = get_min_bounding_box(seg_image_normalized)
                bbox = (
                    x_min - 5,  # left
                    y_min - 5,  # top
                    x_max + 10,  # right
                    y_max + 5  # bottom
                )
                crop_img = sketch_image.crop(bbox)

                sketch_image_curr_list.append(crop_img)
            sketch_image_list.append(sketch_image_curr_list)
    else:
        return None
    return sketch_image_list



if __name__ == "__main__":
    rgb_image = "assets/example_data/Realistic-Style/00_rgb.png"
    seg_image = "assets/example_data/Realistic-Style/00_seg.png"
    instance_rgbs, instance_masks, scene_rgbs = split_rgb_mask(rgb_image, seg_image)
    print(instance_rgbs , "", instance_masks, ' ', scene_rgbs)
    sketch_image = Image.open("assets/example_data/Realistic-Style/00_sketch.png")
    sketch_image_list = prepare_sketch_images(
        [sketch_image],
        5,
        1,
        seg_images=[instance_masks],
        mode="mask"
    )
    print(sketch_image_list)
    for i in sketch_image_list[0]:
        i.show()
