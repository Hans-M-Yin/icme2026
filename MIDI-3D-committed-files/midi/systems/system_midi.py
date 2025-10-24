import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange, repeat
from peft import LoraConfig, get_peft_model_state_dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
from skimage import measure
from transformers import CLIPVisionModelWithProjection, Dinov2Model

from ..models.autoencoders import TripoSGVAEModel
from ..models.transformers import TripoSGDiTModel
from ..pipelines.pipeline_midi import MIDIPipeline
from ..schedulers import (
    RectifiedFlowScheduler,
    compute_density_for_timestep_sampling,
    compute_loss_weighting,
)
from ..utils.typing import *
from .base import BaseSystem
from .model_utils import (
    preprocess_image_for_clip,
    preprocess_image_for_dinov2,
    to_pil_image,
)

from ..sketch.fusion_adapter import FusionAdapterConfig,SketchFusionAdapter
from ..sketch.sketch_tower import SketchVisionTowerConfig,SketchVisionTower


class MIDISystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):

        # Model / Adapter
        pretrained_model_name_or_path: str = ""

        ## Image encoder 2
        pretrained_image_encoder_2_processor_config: Optional[Dict[str, Any]] = None
        image_encoder_2_input_channels: int = 7
        image_encoder_2_init_projection_method: str = "clone"


        ## Attention processor
        set_self_attn_module_names: Optional[List[str]] = None

        ## LoRA
        transformer_lora_config: Optional[Dict[str, Any]] = None
        image_encoder_1_lora_config: Optional[Dict[str, Any]] = None
        image_encoder_2_lora_config: Optional[Dict[str, Any]] = None

        # Training
        image_drop_prob: float = 0.1
        new_cond_size: int = 512

        vae_slicing_length: Optional[int] = None
        gradient_checkpointing: bool = False

        # Noise sampler
        noise_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
        weighting_scheme: str = (
            "logit_normal_dist"  # ["sigma_sqrt", "logit_normal", "logit_normal_dist", "mode", "cosmap"]
        )

        # Evaluation
        eval_seed: int = 42
        eval_num_inference_steps: int = 50
        eval_guidance_scale: float = 7.0

        # Others
        visual_resolution: int = 512

    cfg: Config

    def configure(self):
        super().configure()

        # Some parameters
        self.train_transformer_lora = self.cfg.transformer_lora_config is not None
        self.train_image_encoder_1_lora = (
            self.cfg.image_encoder_1_lora_config is not None
        )
        self.train_image_encoder_2_lora = (
            self.cfg.image_encoder_2_lora_config is not None
        )
        self.sketch_image_encoder_lora = (
            # @TODO: Whether apply lora into sketch image encoder?
        )
        # Prepare pre-trained pipeline
        pipeline: MIDIPipeline = MIDIPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path
        )

        # Initialize custom adapter: adapted Dinov2 and attn_processor
        pipeline.init_custom_adapter(
            self.cfg.set_self_attn_module_names,
            self.cfg.pretrained_image_encoder_2_processor_config,
            self.cfg.image_encoder_2_input_channels,
            self.cfg.image_encoder_2_init_projection_method,
            self.cfg.transformer_lora_config,
            self.cfg.image_encoder_1_lora_config,
            self.cfg.image_encoder_2_lora_config,
            # @TODO
        )

        noise_scheduler = RectifiedFlowScheduler.from_config(
            pipeline.scheduler.config, **self.cfg.noise_scheduler_kwargs
        )
        pipeline.scheduler = noise_scheduler

        # Initialize the system
        self.pipeline: MIDIPipeline = pipeline
        self.noise_scheduler: RectifiedFlowScheduler = self.pipeline.scheduler
        self.vae: TripoSGVAEModel = self.pipeline.vae
        self.transformer: TripoSGDiTModel = self.pipeline.transformer
        self.image_encoder_1: CLIPVisionModelWithProjection = (
            self.pipeline.image_encoder_1
        )
        self.image_encoder_2: Dinov2Model = self.pipeline.image_encoder_2

        self.sketch_fusion_adapter : SketchFusionAdapter = self.pipeline.sketch_fusion_adapter

        self.vae.requires_grad_(False)

        # Others
        if self.cfg.vae_slicing_length is not None:
            self.vae.enable_slicing(self.cfg.vae_slicing_length)

        # Prepare gradient checkpointing
        if self.cfg.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            if self.train_image_encoder_1_lora:
                self.image_encoder_1.gradient_checkpointing_enable()
            if self.train_image_encoder_2_lora:
                self.image_encoder_2.gradient_checkpointing_enable()
            # @TODO add lora for sketch vision tower


    def on_fit_start(self):
        pass

    def on_train_start(self):
        # Set the model to train mode for some specific design like dropout, gradient checkpointing, etc.
        if self.train_transformer_lora:
            self.transformer.train()
        if self.train_image_encoder_1_lora:
            self.image_encoder_1.train()
        if self.train_image_encoder_2_lora:
            self.image_encoder_2.train()

    def forward(
        self,
        noisy_latents: torch.Tensor,
        conditioning_pixel_values_one: torch.Tensor,
        conditioning_pixel_values_two: torch.Tensor,
        conditioning_sketch_images: Union[torch.Tensor, List[PIL.Image.Image]],
        timesteps: torch.Tensor,
        num_instances: Union[torch.IntTensor, List[int]],
        num_instances_per_batch: int,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        bsz = noisy_latents.shape[0]
        num_batches = bsz // num_instances_per_batch

        # Prepare image conditions
        image_drop_mask = (
            torch.rand(num_batches, device=noisy_latents.device)
            < self.cfg.image_drop_prob
        )
        image_drop_mask = image_drop_mask.repeat_interleave(num_instances_per_batch)

        image_1 = preprocess_image_for_clip(conditioning_pixel_values_one)
        image_1 = self.image_encoder_1(image_1).image_embeds.unsqueeze(1)
        image_1[image_drop_mask] = 0.0

        image_2 = []
        for i in range(0, conditioning_pixel_values_two.shape[1], 3):
            pixel_values_ = conditioning_pixel_values_two[:, i : i + 3]
            if pixel_values_.shape[1] == 3:
                pixel_values_ = preprocess_image_for_dinov2(
                    pixel_values_, size=self.cfg.new_cond_size
                )
            elif pixel_values_.shape[1] == 1:
                pixel_values_ = F.interpolate(
                    pixel_values_, size=self.cfg.new_cond_size, mode="nearest"
                )
            else:
                raise ValueError(f"Invalid pixel_values shape: {pixel_values_.shape}")
            image_2.append(pixel_values_)
        image_2 = torch.cat(image_2, dim=1)
        image_2 = self.image_encoder_2(image_2).last_hidden_state
        image_2[image_drop_mask] = 0.0

        # process sketch images and get latents
        sketch_latents = self.sketch_fusion_adapter(sketch_images)
        # [ISSUES]: add dropout experimentally. But should check shape firstly.
        sketch_latents[image_drop_mask] = 0.0

        # Model prediction
        model_pred = self.transformer(
            noisy_latents,
            timesteps,
            encoder_hidden_states=image_1,
            encoder_hidden_states_2=image_2,
            attention_kwargs={
                "num_instances": num_instances,
                "num_instances_per_batch": num_instances_per_batch,
            },
            sketch_hidden_states=sketch_latents
        ).sample

        return {"model_pred": model_pred}

    def training_step(self, batch, batch_idx):
        """
        Training step for the pl module.

        Args:
            batch: The batch of data. Each batch data includes keys `num_instances`, `surface`, `rgb`,
            `mask`, `rgb_scene`.
            batch_idx: The index of the batch.
        """
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            latents = self.vae.encode(batch["surface"]).latent_dist.sample()

        bsz = latents.shape[0]
        num_instances_per_batch = batch["num_instances_per_batch"]
        num_batches = bsz // num_instances_per_batch

        # Sample random timesteps for each batch sample, but keep the same for each sample in the batch
        sigmas = compute_density_for_timestep_sampling(
            self.cfg.weighting_scheme,
            num_batches,
            logit_mean=0.0,
            logit_std=1.0,
        ).to(latents.device)
        sigmas = self.noise_scheduler.time_shift(sigmas)
        sigmas = sigmas.repeat_interleave(num_instances_per_batch)
        timesteps = self.noise_scheduler._sigma_to_t(sigmas)

        # Add noise to latents according to flow matching
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.scale_noise(latents, noise, timesteps)

        # Predict the noise residuals
        conditioning_pixel_values_one = batch["rgb"]
        conditioning_pixel_values_two = torch.cat(
            [batch["rgb"], batch["rgb_scene"], batch["mask"]], dim=1
        )
        conditioning_sketch_images = batch['sketch']


        model_pred: Tensor = self(
            noisy_latents,
            conditioning_pixel_values_one,
            conditioning_pixel_values_two,
            conditioning_sketch_images,
            timesteps,
            **batch,
        )["model_pred"]

        # Flow matching loss
        target = latents - noise
        weighting = compute_loss_weighting(self.cfg.weighting_scheme, sigmas)[
            :, None, None
        ]
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        loss = loss.mean()

        self.log("train/loss", loss, prog_bar=True)

        # will execute self.on_check_train every self.cfg.check_train_every_n_steps steps
        self.check_train(batch)

        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def get_input_visualizations(self, batch):
        images = [
            {
                "type": "rgb",
                "img": rearrange(batch["rgb"], "B C H W -> (B H) W C"),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(batch["rgb_scene"], "B C H W -> (B H) W C"),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(
                    batch["mask"].repeat(1, 3, 1, 1), "B C H W -> (B H) W C"
                ),
                "kwargs": {"data_format": "HWC"},
            },
        ]
        return images

    def get_output_visualizations(self, batch, outputs):
        num_instances = batch["num_instances"][0]  # hard-coded
        images = [
            {
                "type": "rgb",
                "img": rearrange(batch["rgb"][:num_instances], "B C H W -> (B H) W C"),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(
                    batch["rgb_scene"][:num_instances], "B C H W -> (B H) W C"
                ),
                "kwargs": {"data_format": "HWC"},
            },
            {
                "type": "rgb",
                "img": rearrange(
                    batch["mask"][:num_instances].repeat(1, 3, 1, 1),
                    "B C H W -> (B H) W C",
                ),
                "kwargs": {"data_format": "HWC"},
            },
        ]
        return images

    def on_check_train(self, batch):
        self.save_image_grid(
            f"it{self.true_global_step}-train.jpg",
            self.get_input_visualizations(batch),
            name="train_step_input",
            step=self.true_global_step,
            log_to_wandb=False,
        )

    @rank_zero_only
    def save_model_weights(self):
        if (
            self.train_transformer_lora
            or self.train_image_encoder_1_lora
            or self.train_image_encoder_2_lora
            # @TODO: add sketch vision tower lora
        ):
            save_dir = os.path.join(
                os.path.dirname(self.get_save_dir()), "custom_adapter"
            )
            os.makedirs(save_dir, exist_ok=True)
            self.pipeline.save_custom_adapter(
                save_dir,
                f"custom_adapter_e{self.current_epoch}_it{self.true_global_step}.safetensors",
                safe_serialization=True,
                include_keys=["lora"],
            )

    @torch.no_grad()
    def generate_samples(
        self, batch, return_dict: bool = True, **kwargs
    ) -> Tuple[List[trimesh.Trimesh], torch.Tensor, torch.Tensor]:
        # Inference pipeline
        output = self.pipeline(
            image=to_pil_image(batch["rgb"]),
            mask=to_pil_image(batch["mask"]),
            image_scene=to_pil_image(batch["rgb_scene"]),
            sketch_image=to_pil_image(batch["sketch"]),
            num_inference_steps=self.cfg.eval_num_inference_steps,
            guidance_scale=self.cfg.eval_guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(
                self.cfg.eval_seed
            ),
            attention_kwargs={
                "num_instances": batch["num_instances"],
                "num_instances_per_batch": batch["num_instances_per_batch"],
            },
            decode_progressive=True,
            return_dict=False,
        )

        # marching cubes
        vertices_list, faces_list, trimeshes = [], [], []
        for logits_, grid_size, bbox_size, bbox_min, bbox_max in zip(*output):
            grid_logits = logits_.view(grid_size).float().cpu().numpy()
            vertices, faces, normals, _ = measure.marching_cubes(
                grid_logits, 0, method="lewiner"
            )
            vertices = vertices / grid_size * bbox_size + bbox_min

            # Trimesh
            mesh = trimesh.Trimesh(
                vertices.astype(np.float32), np.ascontiguousarray(faces)
            )
            trimeshes.append(mesh)

            # Vertices and faces
            vertices = torch.from_numpy(vertices.copy()).to(self.device, torch.float32)
            faces = torch.from_numpy(faces.copy()).to(self.device, torch.float32)
            vertices_list.append(vertices)
            faces_list.append(faces)

        if not return_dict:
            return (trimeshes, vertices_list, faces_list)

        return {"trimeshes": trimeshes, "vertices": vertices_list, "faces": faces_list}

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def compute_metrics(self, batch, outputs, use_icp: bool = True):
        from ..utils.metrics import (
            compute_chamfer_distance,
            compute_fscore,
            compute_volume_iou,
            icp,
        )

        def normalize_(tensor):
            min_vals = tensor.min(dim=1, keepdim=True)[0]
            max_vals = tensor.max(dim=1, keepdim=True)[0]

            ranges = max_vals - min_vals
            ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)

            normalized_tensor = 1.9 * (tensor - min_vals) / ranges - 0.95

            return normalized_tensor

        cd_scene, cd_scene_1, cd_scene_2, fscore_scene = [], [], [], []
        cd_object, fscore_object, iou_bbox = [], [], []

        start_idx = 0
        for num in batch["num_instances"]:
            # preprocess: fps sampling
            vertices_pred = [
                torch.from_numpy(trimesh.sample.sample_surface(mesh, 20480)[0])
                for mesh in outputs["trimeshes"]
            ]
            vertices_pred = torch.stack(vertices_pred).float().to(self.device)
            vertices_gt = batch["surface"][..., :3].float()

            # 1. scene
            vertices_scene_pred = rearrange(
                vertices_pred[start_idx : start_idx + num], "B N C -> (B N) C"
            ).unsqueeze(0)
            vertices_scene_gt = rearrange(
                vertices_gt[start_idx : start_idx + num], "B N C -> (B N) C"
            ).unsqueeze(0)

            ## 1.1 icp
            if use_icp:
                vertices_scene_pred, R, t = icp(
                    vertices_scene_pred, vertices_scene_gt, max_iterations=50
                )

            ## 1.2 metrics
            cds = compute_chamfer_distance(vertices_scene_pred, vertices_scene_gt)
            cd_scene.append(cds[0])
            cd_scene_1.append(cds[1])
            cd_scene_2.append(cds[2])
            fscore_scene.append(compute_fscore(vertices_scene_pred, vertices_scene_gt))

            # 2. object
            vertices_object_pred = vertices_pred[start_idx : start_idx + num]
            vertices_object_gt = vertices_gt[start_idx : start_idx + num]

            # 2.1. object iou in global scene
            iou_bbox.append(
                compute_volume_iou(
                    vertices_object_pred, vertices_object_gt, mode="bbox"
                )
            )

            # 2.2 object quality
            vertices_object_pred = normalize_(vertices_object_pred)
            vertices_object_gt = normalize_(vertices_object_gt)

            ## 2.2.1 icp
            if use_icp:
                vertices_object_pred, _, _ = icp(
                    vertices_object_pred, vertices_object_gt, max_iterations=50
                )

            ## 2.2.2 metrics
            cd_object.append(
                compute_chamfer_distance(vertices_object_pred, vertices_object_gt)[0]
            )
            fscore_object.append(
                compute_fscore(vertices_object_pred, vertices_object_gt)
            )

            if batch["num_instances_per_batch"] is not None:
                start_idx += batch["num_instances_per_batch"]
            else:
                start_idx += num

        for item in [
            cd_scene,
            cd_scene_1,
            cd_scene_2,
            fscore_scene,
            cd_object,
            fscore_object,
            iou_bbox,
        ]:
            if len(item) == 0:
                return None

        mean_acc = lambda x: torch.cat(x).mean()

        return {
            "scene_cd": mean_acc(cd_scene),
            "scene_cd_1": mean_acc(cd_scene_1),
            "scene_cd_2": mean_acc(cd_scene_2),
            "scene_fscore": mean_acc(fscore_scene),
            "object_cd": mean_acc(cd_object),
            "object_fscore": mean_acc(fscore_object),
            "iou_bbox": mean_acc(iou_bbox),
        }

    def validation_step(self, batch, batch_idx):
        try:
            outputs = self.generate_samples(batch)
        except Exception as e:
            rank_zero_warn(f"Error in validation step: {e}")
            return

        if (
            self.cfg.check_val_limit_rank > 0
            and self.global_rank < self.cfg.check_val_limit_rank
        ):
            self.save_image_grid(
                f"it{self.true_global_step}-validation-{self.global_rank}_{batch_idx}.jpg",
                self.get_output_visualizations(batch, outputs),
                name=f"validation_step_output_{self.global_rank}_{batch_idx}",
                step=self.true_global_step,
                log_to_wandb=False,
            )

            # Compose the output meshes and save them
            start_idx = 0
            for num in batch["num_instances"]:
                trimeshes = outputs["trimeshes"][start_idx : start_idx + num]
                composed_mesh = trimesh.util.concatenate(trimeshes)
                composed_mesh.export(
                    os.path.join(
                        self.get_save_dir(),
                        f"it{self.true_global_step}-validation-{self.global_rank}_{batch_idx}.glb",
                    )
                )
                if batch["num_instances_per_batch"] is not None:  # no padding
                    start_idx += batch["num_instances_per_batch"]
                else:
                    start_idx += num

    def on_validation_epoch_start(self):
        self.transformer.eval()
        self.image_encoder_1.eval()
        self.image_encoder_2.eval()

    def on_validation_epoch_end(self):
        self.save_model_weights()
        torch.cuda.empty_cache()

        if self.train_transformer_lora:
            self.transformer.train()
        if self.train_image_encoder_1_lora:
            self.image_encoder_1.train()
        if self.train_image_encoder_2_lora:
            self.image_encoder_2.train()

    def test_step(self, batch, batch_idx):
        try:
            outputs = self.generate_samples(batch)

            # log metrics
            metrics = self.compute_metrics(batch, outputs)
            if metrics is not None:
                for k, v in metrics.items():
                    self.log(
                        f"test/{k}",
                        v,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=True,
                    )
        except Exception as e:
            rank_zero_warn(f"Error in test step: {e}")
            return

        self.save_image_grid(
            f"it{self.true_global_step}-test-{self.global_rank}_{batch_idx}.jpg",
            self.get_output_visualizations(batch, outputs),
            name=f"test_step_output_{self.global_rank}_{batch_idx}",
            step=self.true_global_step,
            log_to_wandb=False,
        )

        # Compose the output meshes and save them
        start_idx = 0
        for scene_idx, num in enumerate(batch["num_instances"]):
            scene = trimesh.Scene()
            for i, mesh in enumerate(outputs["trimeshes"][start_idx : start_idx + num]):
                mesh.export(
                    os.path.join(
                        self.get_save_dir(),
                        f"it{self.true_global_step}-test-{self.global_rank}_{batch_idx}_{i}-{batch['id'][scene_idx]}-{i}.glb",
                    )
                )
                scene.add_geometry(mesh)

            scene.export(
                os.path.join(
                    self.get_save_dir(),
                    f"it{self.true_global_step}-test-{self.global_rank}_{batch_idx}-{batch['id'][scene_idx]}_scene.glb",
                )
            )
            if batch["num_instances_per_batch"] is not None:
                start_idx += batch["num_instances_per_batch"]
            else:
                start_idx += num

    def on_test_end(self):
        self.save_model_weights()

        import pandas as pd

        metric_keys = [
            "test/scene_cd",
            "test/scene_cd_1",
            "test/scene_cd_2",
            "test/scene_fscore",
            "test/object_cd",
            "test/object_fscore",
            "test/iou_bbox",
        ]
        metrics = {k: [self.trainer.callback_metrics[k].item()] for k in metric_keys}
        pd.DataFrame(metrics).to_csv(
            os.path.join(self.get_save_dir(), "test_metrics.csv"), index=False
        )
