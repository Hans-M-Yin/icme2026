import torch
import torch.nn as nn
import torch.nn.functional as F


class SketchGatingIntensityMLP(nn.Module):
    """
    Predict gating intensity scalar from given time_embedding (from main model)
    and the current layer index.

    Output in (0, 1): smaller value → stronger suppression.
    """

    def __init__(
            self,
            time_embed_dim: int,  # must match external time_embedding dim
            num_layers: int,  # max layer count
            layer_embed_dim: int = 64,
            hidden_dim: int = 128,
            init_value: float = 0.2,
    ):
        super().__init__()
        print("昨天晚上做了个梦",time_embed_dim)
        time_embed_dim = 2048
        # Learnable layer embeddings (each layer has its own vector)
        self.layer_embedding = nn.Embedding(num_layers, layer_embed_dim)

        # MLP to combine time_emb + layer_emb → scalar
        self.fc = nn.Sequential(
            nn.Linear(time_embed_dim + layer_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # constrain output to (0, 1)
        )

        # ---- Initialization ----
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

        # Initialize final bias toward desired starting intensity (≈init_value). We hope that gating intensity over different
        # layers is initialized with init_value. And different layers will learn different gating intensity on their own.
        with torch.no_grad():
            self.fc[-2].bias.fill_(torch.logit(torch.tensor(init_value)))

    def forward(self, time_emb: torch.Tensor, layer_id: int) -> torch.Tensor:
        """
        The gating intensity is computed with time embedding and layer id. This is intuitive.
        :param time_emb: Time embedding provided from outer model. MIDI or DiT has trained one, we use this trained one for simplicity.
        :param layer_id: Which layer it is now. Different layer will focus these no-drawing-line area differently.
        :return: a scalar tensor representing the gating intensity
        """
        # @TODO: ISSUE We need to think about the time when we compute gating intensity. because the same intensity will be
        #       reused for multiple times.
        B = time_emb.shape[0]
        # print(time_emb.shape)
        if not torch.is_tensor(layer_id):
            layer_id = torch.tensor([layer_id], device=time_emb.device, dtype=torch.long)
        if layer_id.ndim == 1:
            # print("HHHHLL")
            layer_id = layer_id.expand(B,-1)  # [B]
            layer_id.squeeze()
        layer_feat = self.layer_embedding(layer_id)  # [B, layer_embed_dim]

        # print(layer_feat.shape, ' ', time_emb.shape)
        layer_feat = layer_feat.squeeze()
        feat = torch.cat([time_emb, layer_feat], dim=-1)
        gating_intensity = self.fc(feat)  # [B,1]
        return gating_intensity