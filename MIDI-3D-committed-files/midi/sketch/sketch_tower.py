import torch
import torch.nn as nn
from typing import Union, List, Tuple, Callable
from PIL.Image import Image
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, AutoModel, AutoProcessor
import dataclasses
import PIL
import numpy as np

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
class SketchVisionTowerConfig:
    vision_tower_model: str

    select_feature_type: str

    arbitrary_input_size: bool = False

    input_size: Tuple[int, int] = (224,224)

    select_layer: Union[str, int, list] = "all"
    # Not Use
    image_preprocessor: Callable = resize_image

    device: str = "cuda"



# @TODO: Need implementation for parallel training methods. model loaded from huggingface is simply on GPU:0. Although
# @TODO: ViT is such a small module that it's not necessary for parallel, we should consider it's parallel to boost training
# @TODO: and inference process.
class SketchVisionTower(nn.Module):
    def __init__(self, config: SketchVisionTowerConfig):
        """
        :param args: Configuration needed for building ViT tower.
        :param delay_load: Whether to load the model with delay or not
        """
        # @TODO: initialize vision tower.
        super().__init__()
        self.vision_tower_model_name = config.vision_tower_model
        self.arbitrary_input_size = config.arbitrary_input_size
        self.input_size = config.input_size
        self.vision_tower = None
        if "clip" in self.vision_tower_model_name:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_model_name).to(config.device)
            self.model_config = self.vision_tower.config
            print(f"Model already loaded.\n"
                f"Model configs: {self.model_config}")
        elif "dino" in self.vision_tower_model_name:
            self.vision_tower = AutoModel.from_pretrained(self.vision_tower_model_name).to(config.device)
            self.model_config = self.vision_tower.config
            print(f"Model already loaded.\n"
                f"Model configs: {self.model_config}")
        self.select_layer = config.select_layer
        self.select_feature_type = config.select_feature_type
        # [ISSUE]: I'm not sure whether all ViT models contain preprocessor. Preprocessor resize images into proper size needed for the vision tower.
        # Preprocessor accept different types of input. tensor / np.ndarray / Image
        self.preprocessor = AutoProcessor.from_pretrained(self.vision_tower_model_name)

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
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

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