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
            print('FUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKFUCKv')
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
            if cfg:
                sketch_image_list.append(sketch_image_curr_list + sketch_image_curr_list)
            else:
                sketch_image_list.append(sketch_image_curr_list)

    elif mode == "zoom":
        # Zoom method has updated. (1) Resize the bounding box into 正方形. This is because ViT will resize and clip edge parts.
        # (2) Use mask mode before zoom. To remove other objects in sketch image.
        FILL_PAD = 10
        assert seg_images is not None, f"No seg images."
        sketch_image_list = []

        for idx, (seg_image, sketch_image) in enumerate(zip(seg_images, sketch_images)):
            sketch_image_curr_list = []

            for j in range(num_object_per_scene[idx]):
                seg_image_normalized = seg_image[j]
                dilated_mask = expand_mask_cv2(seg_image_normalized)
                sketch_image_np = np.array(sketch_image)
                sketch_image_pil = Image.fromarray(((dilated_mask[...,None] * sketch_image_np) * 255.0).astype(np.uint8))
                # 1. 获取原始最小 Bounding Box (ymin, ymax, xmin, xmax)
                y_min_raw, y_max_raw, x_min_raw, x_max_raw = get_min_bounding_box(seg_image_normalized)

                # 2. 计算原始宽度和高度
                width_raw = x_max_raw - x_min_raw
                height_raw = y_max_raw - y_min_raw

                # 3. 确定新正方形的边长
                # 边长 = max(原始长宽) + 填充 (FILL_PAD)
                side = max(width_raw, height_raw) + FILL_PAD

                # 4. 计算原始 BBox 的中心点
                center_x = (x_min_raw + x_max_raw) / 2
                center_y = (y_min_raw + y_max_raw) / 2

                # 5. 计算新的正方形 BBox 坐标

                # 新的左上角 (左/上)
                x_min_square = int(center_x - side / 2)
                y_min_square = int(center_y - side / 2)

                # 新的右下角 (右/下)
                x_max_square = int(center_x + side / 2)
                y_max_square = int(center_y + side / 2)

                # 6. 构建 PIL crop 所需的 bbox (left, top, right, bottom)
                # PIL 使用 (左上角X, 左上角Y, 右下角X, 右下角Y)
                bbox_square = (
                    x_min_square,  # left (x_min)
                    y_min_square,  # top (y_min)
                    x_max_square,  # right (x_max)
                    y_max_square  # bottom (y_max)
                )

                # 7. 裁剪图像
                # @TODO: 有可能我们在crop后，要先手动插值提高分辨率，因为有可能物体的size不足224。
                crop_img = sketch_image_pil.crop(bbox_square)

                sketch_image_curr_list.append(crop_img)
            if cfg:
                sketch_image_list.append(sketch_image_curr_list + sketch_image_curr_list)
            else:
                sketch_image_list.append(sketch_image_curr_list)

            # sketch_image_list.append(sketch_image_curr_list)
    else:
        return None
    return sketch_image_list

"""
Spatial Gating Attention may not work. But we still need to try.
"""


def get_single_sketch_gating_map(
        sketch_image: Image.Image,
        patch_size: int = 32,
        threshold: float = 0.4,
        device: str = "cuda"
) -> torch.Tensor:
    """
    calculate single gating map.
    :return:  Tensor [patch_size, patch_size]
    """
    # 1. 预处理：PIL -> 灰度 -> 归一化 Tensor

    # 转换为灰度图 (L 模式)，确保通道为 1
    sketch_gray = sketch_image.convert('L')

    # 转为 NumPy 数组 (H, W)，并归一化到 0.0 ~ 1.0
    sketch_np = np.array(sketch_gray).astype(np.float32) / 255.0

    # 转为 PyTorch Tensor (1, H, W)，并移到 GPU (如果可用)
    sketch_tensor = torch.from_numpy(sketch_np).unsqueeze(0).unsqueeze(0).to(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    # 形状: (1, 1, H, W)

    # 2. 图像调整大小以适应分块 (可选但推荐)
    # 如果图像大小不是 patch_size 的整数倍，应先调整大小或裁剪。
    H, W = sketch_tensor.shape[2], sketch_tensor.shape[3]

    target_H = (H // patch_size) * patch_size
    target_W = (W // patch_size) * patch_size
    if target_H != H or target_W != W:
        sketch_tensor = sketch_tensor[:, :, :target_H, :target_W]
    with torch.no_grad():
        max_pooled_tensor = torch.nn.functional.max_pool2d(
            sketch_tensor,
            kernel_size=patch_size,
            stride=patch_size
        )
    # 形状: (1, 1, P_num_H, P_num_W)
    patch_map = (max_pooled_tensor >= threshold).float()
    # 6. 最终形状调整
    # 将形状从 (1, 1, P_num_H, P_num_W) 调整为 (P_num_H, P_num_W)
    # print(f"SHAPE {patch_map.squeeze().shape}")
    return patch_map.squeeze().flatten().to(device)

def get_sketch_spatial_gating_map(
        sketch_image_per_instance: List[List[Image.Image]],
        patch_size:int,
        device:str = "cuda",
        concat:bool = False
) -> Union[torch.Tensor,List[torch.Tensor]]:
    """
    Test done.
    计算Gating map，即哪些patch有线条，哪些没线条。
    Calculate spatial gating map. We firstly try the simplest way: the patches containing lines are 1, others are 0.
    :param sketch_image_per_instance: A batch of sketch images. [[instance_A_sketch, instance_B_sketch, instance_C_sketch],[instance_A_sketch, instance_B_sketch, instance_C_sketch]]
    :param patch_size: must be consistent with the sketch vision tower's patch size
    :param concat: 是否把gating map合成为一个大的tensor，如果False，则返回的是List[Tensor]，每一个元素是一个场景的，形状为[I * H * W], H和W分别表示行数和列数。
    """
    res = []

    for sketch_image_this_scene_list in sketch_image_per_instance:
        print(sketch_image_this_scene_list)
        gating_map_this_scene = [get_single_sketch_gating_map(i, patch_size, device=device) for i in sketch_image_this_scene_list]
        res.append(torch.stack(gating_map_this_scene))
    if concat:
        res = torch.vstack(res)
    return res


def visualize_patch_map_on_image(
        original_image: Image.Image,
        patch_map: torch.Tensor,
        opacity: float = 0.5,
):
    """
        tool function. 用来可视化gating map效果，这个可以忽略。
    """
    W, H = original_image.size

    if patch_map.is_cuda:
        patch_map = patch_map.cpu()

    patch_map = patch_map.float()
    patch_tensor = patch_map.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        # print(patch_tensor.shape)
        upsampled_mask = torch.nn.functional.interpolate(
            patch_tensor, size=(H, W), mode='nearest'
        ).squeeze()
    alpha_value_opaque = int(255 * opacity)
    rgb_channels = np.ones((H, W, 3), dtype=np.uint8) * 255
    alpha_channel = (upsampled_mask.numpy() * alpha_value_opaque).astype(np.uint8)
    alpha_3d = alpha_channel[:, :, np.newaxis]
    overlay_array = np.concatenate([rgb_channels, alpha_3d], axis=2)
    overlay_image = Image.fromarray(overlay_array, mode='RGBA')
    base_image_rgba = original_image.convert("RGBA")
    final_image = Image.alpha_composite(base_image_rgba, overlay_image)
    final_image.show()


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def tensor_to_pil_list(tensor: torch.Tensor) -> list[Image.Image]:
    """
    将 CLIP Processor 输出的 Tensor 转换回 PIL Image 列表。
    - 如果输入形状为 (C, H, W)，返回包含 1 个 Image 的列表。
    - 如果输入形状为 (B, C, H, W)，返回包含 B 个 Image 的列表。
    """
    print('本来应该',type(tensor))
    if tensor.ndim == 3:
        # 如果是 (C, H, W)，添加一个 Batch 维度使其成为 (1, C, H, W)
        tensor = tensor.unsqueeze(0)

    # 检查维度是否正确，确保是 (B, C, H, W)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(
            f"Input tensor must be 3D (C, H, W) or 4D (B, C, H, W) with 3 channels, but got shape {tensor.shape}")

    # 使用 torch.unbind(dim=0) 将 Batch 维度拆分成一个包含 B 个 (C, H, W) tensor 的元组
    list_of_tensors = list(torch.unbind(tensor, dim=0))

    output_images = []

    # 遍历每个 (C, H, W) 图像 tensor
    for single_image_tensor in list_of_tensors:
        # 确保 MEAN 和 STD 在同一设备上
        current_mean = CLIP_MEAN.to(single_image_tensor.device)
        current_std = CLIP_STD.to(single_image_tensor.device)

        # 1. 反归一化： tensor = tensor * std + mean
        restored_tensor = single_image_tensor * current_std + current_mean

        # 2. 裁剪到 0-1 范围，防止溢出
        restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)

        # 3. 转换维度： (C, H, W) -> (H, W, C)
        #    并转换为 NumPy 数组 (0-255 整数)
        image_numpy = (restored_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # 4. 转换为 PIL Image
        output_images.append(Image.fromarray(image_numpy))

    return output_images

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
        mode="zoom"
    )
    print(sketch_image_list)
    from transformers import CLIPImageProcessor
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(images=sketch_image_list[0], return_tensors="pt")

    # 获取 PyTorch Tensor
    tensor_sketch_image = inputs['pixel_values']

    preprocessed_sketch_image = tensor_to_pil_list(tensor_sketch_image)
    for idx,i in enumerate(preprocessed_sketch_image):
        i.save(str(idx) + ".png", format="png")
    # for i in sketch_image_list[0]:
    #     i.show()
    # gating_map = get_sketch_spatial_gating_map([preprocessed_sketch_image], 14,device="cpu")
    # print(gating_map)
    # for i, j in zip(gating_map[0], sketch_image_list[0]):
    #     visualize_patch_map_on_image(j, i.reshape(16, 16))


