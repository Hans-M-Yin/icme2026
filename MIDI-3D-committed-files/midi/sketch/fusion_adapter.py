"""
After getting features from ViT, we need to project them for fusion with DiT latents.
In original implementation of MIDI, projectors are employed just after CLIP ViT and Dino ViT. Note that in theory two
sequences that will be fused by cross attention are not required for the same dimension, while it's enough if Q,K,V vectors
is of the same length.
But in Diffusion setting, simple projection might be insufficient because, timestep T plays a significant role in modeling
de-noising stage. This is to say, when model extract information from images (sketches), it tries to find geometry info
firstly (or in a minor T step), and as T increases with de-noising process goes on, the model gradually focus on other
info such as semantics and so on.

TL;DR: timestep T should be considered into feature fusion to help the model know in this step, what kind of info it needs.
Simply, as T gradually increase, the model should pay less attention on sketches. NOTICE: this point of view is focus on
inference process as during training process, the model denoise the object just in one step.

Now we dive deeper: in MIDI's implementation, CLIP latents and Dino latents are fused into DiT latents through cross attn.
We have analysed the dimension mis-alignment of image latents and DiT latents: image latents are only required to have
Q,K,V vectors with the same dimension, which explain why CLIP and Dino use projectors -- the projectors are used not for
dimension alignment between latent, but exactly for calculating K vectors. (这是寒哥的分析，不一定对啊，有可能就是普通的cross attention，projector就是为了把image latent投影到和DiT latent相同长度）

We just want timestep T to be an input to control sketch information, where T should be inserted into projectors. If
hange's analysis is correct, we edit the projectors equals to we design a new attention method.
And the new attention method are novelty we need.

我确认了一下，MIDI并没有直接用projector的结果作为K,V矩阵啊，MIDI是用CLIP预训练好的projector，得到最终的features，即：最后一层的hidden_states(latents)->投影到features
这个投影的维度是固定的，为了使用CLIP预训练好的projector权重。
模型用AttentionProcessor对两个序列计算Q和KV，并结算Attention结果。上面的brain storm只是化简了Attention计算过程。
"""
import dataclasses
import os
from typing import List, Tuple, Union
import torch.nn as nn
from .sketch_tower import SketchVisionTower, SketchVisionTowerConfig
from diffusers.models.modeling_utils import ModelMixin
import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
import torch.nn.functional as F
# @TODO: Use projector or not is not determined. For implementation, the difference between using projectors or not is
# @TODO: minor, we only need to consider dimensions. In MIDI's implementation, it seems that the cross attention module
# @TODO: doesn't need same dimension between two sequence as well.

# @TODO: We need implement something about **device setting**.

@dataclasses.dataclass
class FusionAdapterConfig(ConfigMixin):

    num_dit_layers: int = None
    num_vision_layers: int = None
    dim_dit_latent: int = None
    dim_vision_latent: int = None

    # Fusion method
    fusion_mode: str = "specific"
    # specific: manually assign two sequences, DiT latent in layer {dit_layer_seqs[i]} will be fused with vision latent in layer {vision_layer_seqs[i]}
    # Other method...
    dit_layer_seqs: List[int] = None
    vision_layer_seqs: List[int] = None

    # Projector
    enable_projector: bool = False

    # layer-wise: each vision tower layer has a own projector.
    # all: all layers use the same projector
    projector_mode: str = "layer_wise"

    # linear: nn.Linear as projector
    # mlp: nn.Linear + nn.Linear as projector. hidden_dim is required
    projector_type: str = "linear"
    projector_hidden_dim: int = None

    dim_projected_latent: int = 512

    model_type: str = "sketch_vision_tower"



from diffusers.loaders import PeftAdapterMixin


class SketchFusionAdapter(nn.Module, ModelMixin, PeftAdapterMixin):
    """
    FusionAdapter collects layer-wise image latents and decide which layer does each of them will be fused to. This is
    because SketchVisionTower layer num (CLIP is 15) is not equal to DiT layer num. So layer by layer is not available.
    Manually order specific layers to fuse is the simplest way.
    """
    def __init__(self, config: FusionAdapterConfig, sketch_tower_config: SketchVisionTowerConfig, **kwargs):

        super(SketchFusionAdapter, self).__init__()
        self.config = config
        self.num_dit_layers = config.num_dit_layers
        self.num_vision_layers = config.num_vision_layers
        self.dim_dit_latent = config.dim_dit_latent
        self.dim_vision_latent = config.dim_vision_latent

        self.fusion_mode = config.fusion_mode
        self.dit_layer_seqs = config.dit_layer_seqs
        self.vision_layer_seqs = config.vision_layer_seqs

        self.enable_projector = config.enable_projector
        self.dim_projected_latent = config.dim_projected_latent

        self.check_layer_fusion()

        self.projectors = self.init_projectors()

        """
        SketchVisionTower is included as a attribute. the forward function requires Image input and firstly adopt 
        vision_tower.forward() to get valina latents, and then uses projectors to get projected latents. projected latents
        and 
        """
        self.sketch_tower = SketchVisionTower(sketch_tower_config)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            **kwargs
    ):
        # 先看看有没有已有权重。
        try:
            config, kwargs = cls.load_config(pretrained_model_name_or_path, **kwargs)

            if not isinstance(config, FusionAdapterConfig):
                config = FusionAdapterConfig.from_dict(config)

            # 尝试从配置或 kwargs 中获取 SketchVisionTowerConfig
            sketch_tower_config = kwargs.pop("sketch_tower_config", None)
            if sketch_tower_config is None and hasattr(config, "sketch_tower_config") and config.sketch_tower_config:
                sketch_tower_config = SketchVisionTowerConfig.from_dict(config.sketch_tower_config)

            is_config_loaded = True

        except Exception as e:
            # 🚨 配置加载失败！这是你需要手动初始化的场景。
            print(f"Failed to load FusionAdapter config from {pretrained_model_name_or_path}. Error: {e}")
            print("Attempting to initialize model using provided/default configuration.")

            is_config_loaded = False

            # 2. 手动初始化：从 kwargs 或默认值获取配置

            # 从 kwargs 中获取 FusionAdapterConfig 的参数，并创建 config
            sketch_fusion_adapter_config = kwargs.pop("sketch_fusion_adapter_config", None)
            if sketch_fusion_adapter_config is None:

                config_data = {k: kwargs.pop(k) for k in list(kwargs.keys()) if
                               k in FusionAdapterConfig.__dataclass_fields__}
                config = FusionAdapterConfig(**config_data)
            else:
                if isinstance(sketch_fusion_adapter_config, dict):
                    config = FusionAdapterConfig.from_dict(sketch_fusion_adapter_config)
                else:
                    assert isinstance(sketch_fusion_adapter_config, SketchVisionTowerConfig), "Invalid sketch_fusion_adapter_config."
                    config = sketch_fusion_adapter_config
            sketch_tower_config = kwargs.pop("sketch_tower_config", None)
            if sketch_tower_config is None:
                raise ValueError("No SketchVisionTowerConfig is provided for initializing FusionAdapter.")
            if isinstance(sketch_tower_config, dict):
                sketch_tower_config = SketchVisionTowerConfig.from_dict(sketch_tower_config)
        model = cls(config=config, sketch_tower_config=sketch_tower_config, **kwargs)

        # 加载权重 (只有在成功加载配置时才尝试加载权重)。上一行先对SketchFusionAdapter做了默认初始化，即各个参数已经被随机初始化了，假如pretrained path里面有权重
        # 则将随机初始化的参数替换为训练的结果。
        if is_config_loaded:
            try:
                # 尝试加载完整的 state_dict
                # 注意：load_state_dict 是一个类方法，用于将文件内容读取到字典中
                state_dict = cls.load_state_dict(pretrained_model_name_or_path)

                # 加载权重到实例化的模型中
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

                # Projector 权重的控制：Projector 权重如果缺失，会被报告在 missing_keys 中，但会保持随机初始化
                print(
                    f"Missing keys (Projectors or others): {missing_keys}. These modules will be initialized randomly.")
                print(f"Unexpected keys: {unexpected_keys}.")

            except Exception as e:
                print(
                    f"Error loading state dict from {pretrained_model_name_or_path}: {e}. Model will keep its current initialization.")
                pass

        else:
            # 如果配置加载失败，默认也不尝试加载权重，模型保持随机初始化
            print("Model initialized from scratch with default/provided configs. No weights loaded.")

        return model
    def create_projector(self, projector_type: str, input_dim, output_dim, hidden_dim: int = None) -> nn.Module:
        """
        Create a projector based on condition.
        :param projector_type: linear / mlp(mlp is 2-layer linear)
        :return: projector
        """
        # @TODO: we need to implementation for load projectors from checkpoints.
        if projector_type == 'linear':
            linear_projector = nn.Linear(input_dim, output_dim)
            return linear_projector
        elif projector_type == 'mlp':
            assert hidden_dim is not None, "Hidden dimension is required for build MLP projector"
            mlp_projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            return mlp_projector
        else:
            raise ValueError(f'Unknown projector type: {projector_type}')

    def init_projectors(self):
        if not self.enable_projector:
            return None
        else:
            if self.projector_mode == 'layer_wise':
                projectors = [self.create_projector(self.projector_type,self.dim_vision_latent,
                                                    self.dim_projected_latent,
                                                    self.dim_hidden_latent)
                              for _ in range(len(self.vision_layer_seqs))]

            elif self.projector_mode == 'all':
                projectors = self.create_projector(self.projector_type,self.dim_vision_latent,
                                                    self.dim_projected_latent,
                                                    self.dim_hidden_latent)
            else:
                raise ValueError(f'Unknown projector method:{self.projector_mode}')
        return projectors
    def check_layer_fusion(self) -> bool:
        """
        Check if layer to be fused is qualified. For example, for layer-wise fusion, self.dit_layer_seqs should equals
        self.vision_layer_seqs. The verifying method is decided by self.fusion_mode.
        :return: bool, True if layer to be fused is qualified.
        """
        if self.fusion_mode == "specific":
            return len(self.dit_layer_seqs) == len(self.vision_layer_seqs)

        elif self.fusion_mode == "":
            pass

        return False

    def forward(self, images) -> List[List[torch.Tensor]]:
        """
        Input raw sketch images and firstly encode them through self.sketch_tower and then project them through projector.
        :param images: raw sketch images, supporting PIL.Image.Image, numpy.ndarray and torch.Tensor
        :return: projected latent sequences used for DiT fusion
        """
        # valina_latents: List[torch.tensor] [[B, L, H]], each element is one layer's latent.
        valina_latents = self.sketch_tower(images)
        selected_latents = valina_latents[self.vision_layer_seqs]
        selected_projected_latents = [self.projectors[i](selected_latents[i]) for i in range(len(selected_latents))]
        return selected_projected_latents



