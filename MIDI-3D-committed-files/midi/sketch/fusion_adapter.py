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
from typing import List, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
# @TODO: Use projector or not is not determined. For implementation, the difference between using projectors or not is
# @TODO: minor, we only need to consider dimensions. In MIDI's implementation, it seems that the cross attention module
# @TODO: doesn't need same dimension between two sequence as well.



@dataclasses.dataclass
class FusionAdapterConfig:

    num_dit_layers: int
    num_vision_layers: int
    dim_dit_latent: int
    dim_vision_latent: int

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



class SketchFusionAdapter(nn.Module):
    """
    FusionAdapter collects layer-wise image latents and decide which layer does each of them will be fused to. This is
    because SketchVisionTower layer num (CLIP is 15) is not equal to DiT layer num. So layer by layer is not available.
    Manually order specific layers to fuse is the simplest way.
    """
    def __init__(self, config: FusionAdapterConfig):

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

    # def forward(self, ):

