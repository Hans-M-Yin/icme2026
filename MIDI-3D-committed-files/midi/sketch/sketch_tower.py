import torch
import torch.nn as nn
from typing import Union, List, Tuple, Callable
from PIL.Image import Image
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, AutoModel, AutoProcessor
import dataclasses
import PIL
import numpy as np
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
import os
"""
Vision encoder to compress sketch's information into latents. 
For vision tower backbone, here are candidates:
1. CLIP ViT: classic and popular.
2. DINO v3: strong performance. Dino-v3-base contains 600MB parameters.  
"""



# [ISSUE]: I don't know what input type is needed for SketchVisionTower.vision_tower, might be tensor or Images. The input type determines return type of this function.
def resize_image(images: Union[List,np.array,Image], size: Tuple[int, int], smart_resize: bool = False) -> List[Image]:
    """
    Input an image and resize it before put in ViT. Smart resizing will automatically resize the image into proper size ( This is used for Arbitrary Resolution ViT)
    :param image: Image to be resized
    :param size: Target size
    :param smart_resize: Bool, whether to use smart resize
    :return:
    """
    # @TODO: Note the channels' ORDER.
    def resize_single_image(image: Union[Image, np.array], size: Tuple[int,int], smart_resize: bool = False):
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = PIL.Image.fromarray(image)
        if smart_resize:
            # @TODO: Implement smart resize.
            raise NotImplementedError
        else:
            # @TODO: We try classic ViT first. classic ViT requires 224 * 224 hard.
            image = image.resize(size)
        return image

    resize_result = []
    if isinstance(images, list):
        for image in images:
            resize_result.append(resize_single_image(image,size,smart_resize))
    else:
        resize_result.append(resize_single_image(images,size,smart_resize))
    return resize_result

@dataclasses.dataclass
class SketchVisionTowerConfig(ConfigMixin):

    select_feature_type: str = None

    arbitrary_input_size: bool = False

    input_size: Tuple[int, int] = (224,224)

    select_layer: Union[str, int, list] = "all"
    # Not Use
    image_preprocessor: Callable = resize_image

    device: str = "cuda"

    pretrained_model_name_or_path: str = None

    model_type: str = "sketch_vision_tower"


# @TODO: Need implementation for parallel training methods. model loaded from huggingface is simply on GPU:0. Although
# @TODO: ViT is such a small module that it's not necessary for parallel, we should consider it's parallel to boost training
# @TODO: and inference process.
class SketchVisionTower(nn.Module, ModelMixin):
    def __init__(self, config: SketchVisionTowerConfig):
        super().__init__()

        self.config = config
        self.vision_tower_model_name = config.vision_tower_model
        self.arbitrary_input_size = config.arbitrary_input_size
        self.input_size = config.input_size
        self.vision_tower = None
        self.preprocessor = None

        self.select_layer = config.select_layer
        self.select_feature_type = config.select_feature_type
        # [ISSUE]: I'm not sure whether all ViT models contain preprocessor. Preprocessor resize images into proper size needed for the vision tower.
        # Preprocessor accept different types of input. tensor / np.ndarray / Image
        super().__init__()

        # 1. 获取模型标识符（可能是本地路径或 Hub ID）
        model_id_or_path = config.get("pretrained_model_name_or_path", None)

        # 2. 定义回退的官方/默认模型 ID
        # 假设这是你在找不到本地权重时希望使用的官方地址
        DEFAULT_CLIP_ID = "openai/clip-vit-base-patch32"

        load_path = model_id_or_path

        # 3. 检查提供的路径/ID是否是有效的本地目录（包含权重文件）
        # 如果是本地路径，我们检查它是否包含权重文件。
        if model_id_or_path is not None:
            is_local_path = os.path.isdir(model_id_or_path)

        # 4. 确定最终的加载源
            if is_local_path:
                # 这是一个本地目录，我们检查它是否包含权重文件 (例如 pytorch_model.bin 或 model.safetensors)
                has_weights = (
                        os.path.exists(os.path.join(model_id_or_path, 'pytorch_model.bin')) or
                        os.path.exists(os.path.join(model_id_or_path, 'model.safetensors'))
                    # 也可以检查其他可能的权重文件名
                )

                if not has_weights:
                    # 本地路径存在，但没有权重文件，回退到官方地址
                    print(
                        f"Local path '{model_id_or_path}' found but no weights. Falling back to official ID: {DEFAULT_CLIP_ID}")
                    load_path = DEFAULT_CLIP_ID
                # else: load_path 保持为 model_id_or_path

            elif not is_local_path and "/" not in model_id_or_path:
                # 既不是目录，又不是 Hub ID 格式 (例如 "my_model" 而不是 "user/my_model")
                # 这种情况下，如果 from_pretrained 失败，也会回退
                # 但更安全的方式是依赖 from_pretrained 的内部机制，
                # 这里的 load_path 保持 model_id_or_path，如果 from_pretrained 失败，则打印错误
                # 我们可以添加一个更严格的回退：
                print(
                    f"Model ID/Path '{model_id_or_path}' is not a directory or a standard HuggingFace Hub ID. Attempting to load, but may fallback.")
        else:
            load_path = DEFAULT_CLIP_ID
            # 由于 CLIPVisionModel.from_pretrained 内部会处理 Hub ID，这里主要解决本地路径问题。

        # 5. 加载 CLIP 模型
        try:
            # from_pretrained 能够处理本地路径和 Hub ID
            self.vision_model = CLIPVisionModel.from_pretrained(load_path)
            print(f"Successfully loaded CLIPVisionModel from: {load_path}")

        except Exception as e:
            # 6. 最终回退（如果 Hub 加载也失败了，例如网络问题或 ID 错误）
            if load_path != DEFAULT_CLIP_ID:
                print(
                    f"Failed to load CLIPVisionModel from {load_path}. Attempting final fallback to default ID: {DEFAULT_CLIP_ID}. Error: {e}")

                try:
                    self.vision_model = CLIPVisionModel.from_pretrained(DEFAULT_CLIP_ID)
                    print(f"Successfully loaded CLIPVisionModel from default official ID: {DEFAULT_CLIP_ID}")
                except Exception as final_e:
                    # 7. 如果最终回退也失败了，只能随机初始化结构
                    print(
                        f"FATAL: Failed to load CLIPVisionModel even from official source. Initializing with random weights. Error: {final_e}")
                    config = CLIPVisionConfig.from_pretrained(DEFAULT_CLIP_ID)
                    self.vision_model = CLIPVisionModel(config)

            else:
                # load_path 已经是 DEFAULT_CLIP_ID，但加载失败了，直接随机初始化
                print(
                    f"FATAL: Failed to load CLIPVisionModel from official source {DEFAULT_CLIP_ID}. Initializing with random weights. Error: {e}")
                config = CLIPVisionConfig.from_pretrained(DEFAULT_CLIP_ID)
                self.vision_model = CLIPVisionModel(config)

    def feature_select(self, image_forward_outs) -> List[torch.Tensor]:
        if self.select_layer == "all":
            features = image_forward_outs.hidden_states
        else:
            if isinstance(self.select_layer, int):
                features = (image_forward_outs.hidden_states[self.select_layer],)
            elif isinstance(self.select_layer, list):
                features = [image_forward_outs.hidden_states[i] for i in self.select_layer]
            else:
                features = image_forward_outs.hidden_states[self.select_layer]
                if not isinstance(features, tuple):
                    features = (features,)

        processed = []
        for layer_feat in features:
            if self.select_feature_type == 'patch':
                processed.append(layer_feat[:, 1:])
            else:
                processed.append(layer_feat)
        return processed


    def forward(self, images: Union[Image, List[Image]]) -> List[torch.Tensor]:
        """
        Get layer-level latent tokens of input images.
        :param self:
        :param images: input images. Note that the images must be preprocessed before calling this function. Determined by the tower's type, maybe you should
        firstly resize the images into a fixed sizes (CLIP needs 224 * 224). We should try to find suitable models for arbitrary resolution inputs.
        :return: image features result
        """
        inputs = self.preprocessor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device=self.device, dtype=self.dtype)
        image_forward_outs = self.vision_tower(
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        image_features = self.feature_select(image_forward_outs)

        # Note that this include initial features.
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device


    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


if __name__ == "__main__":
    """
    Test sketch encoder
    CLIP ViT is tested.
    """
    config = SketchVisionTowerConfig(
        vision_tower_model="openai/clip-vit-base-patch32",
        select_feature_type="cls_patch"
    )
    sketch_vision_tower = SketchVisionTower(config)
    test_image_list = [
        PIL.Image.open("./test/sketch1.png"),
        PIL.Image.open("./test/sketch2.png")
    ]
    features = sketch_vision_tower(test_image_list)

    print("-" * 30)
    print("Sketch Vision Tower Test Results:")
    print(f"Type: {type(features)}")  # 应该输出 <class 'list'>
    print(f"Number of layers/features returned: {len(features)}")

    for feature in features:
        print(f"""
    ViT Name: {config.vision_tower_model}
    Type:{type(feature)}
    Shape:{feature.shape}
        """)