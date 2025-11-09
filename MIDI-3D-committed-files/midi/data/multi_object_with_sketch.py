import json
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset

from ..utils.config import parse_structured
from ..utils.typing import *
from .multi_object import *
from transformers import AutoProcessor
from ..sketch.sketch_utils import get_sketch_spatial_gating_map, prepare_sketch_images, tensor_to_pil_list

@dataclass
class MultiObjectDataWithSketchConfig(MultiObjectDataModuleConfig):
    sketch_vision_tower_patch_size: int = 32
    sketch_vision_tower_path: Optional[str] = None
class MultiObjectWithSketchDataset(MultiObjectDataset):
    def __init__(self, cfg: Any, sketch_vision_tower_path: str, split: str = "train") -> None:
        super().__init__(cfg,split)
        self.cfg: MultiObjectDataWithSketchConfig = cfg

        if sketch_vision_tower_path is None:
            sketch_vision_tower_path = self.cfg.sketch_vision_tower_path
        self.sketch_processor = AutoProcessor.from_pretrained(sketch_vision_tower_path)
    
    def load_image(
        self,
        path,
        height,
        width,
        background_color,
        rescale: bool = False,
        return_mask: bool = False,
        remove_bg: bool = False,
        idmap_path: Optional[str] = None,
        return_sketch: bool = False,
    ):
        image_pil = Image.open(path).resize((width, height))
        image = torch.from_numpy(np.array(image_pil)).float() / 255.0

        if image_pil.mode == "RGBA":
            image_bg = image[:, :, :3] * image[:, :, 3:4] + background_color * (
                1 - image[:, :, 3:4]
            )
            mask = (image[:, :, 3] > 0.5).float()
        elif remove_bg and idmap_path is not None:
            id_map = torch.from_numpy(
                np.array(Image.open(idmap_path).resize((width, height), Image.NEAREST))
            )
            mask = (id_map > 0).float()
            mask_ = mask.unsqueeze(-1).repeat(1, 1, 3)
            image_bg = image * mask_ + background_color * (1 - mask_)
        else:
            image_bg = image
            mask = torch.ones_like(image[:, :, 0]).float()

        if rescale:
            image_bg = image_bg * 2.0 - 1.0

        if return_mask:
            return image_bg, mask
        return image_bg

    
    def load_parts(
        self,
        rgb_path: str,
        idmap_path: str,
        indexes: List[int],
        height: int,
        width: int,
        background_color: torch.Tensor,
        skip_small_object: bool = False,
        small_image_proportion: float = 0.005,
        morph_perturb: bool = False,  # Whether to apply morphological perturbation
        max_kernel_size: int = 5,
        p_dilation: float = 0.5,
        return_sketch: bool = True,
        return_gating_maps: bool = True,
        sketch_mode: Optional[str] = "zoom"
    ):
        """
        (1) generate sketch image from rgb image
        (2) process the sketch image and get each object's own sketch images
        (3) Using sketch images, calculate gating map if needed
        (4) return sketch images list and gating maps along with RGB and mask

        It‘s important that to get gating maps, we need to resize sketch images into 224 * 224. So we need image
         processor of model.sketch_adapter.vision_tower.

        :returns: image: torch.Tensor, mask: torch.Tensor, sketch_image: **tensor.Tensor**
        """
        rgb_image = self.load_image(rgb_path, height, width, background_color)

        # @TODO: use informative_drawings API to get sketch image of scene. Check rgb whether it's the original image of
        #   the scene. If yes, use this image as input to get sketch images.
        # print("#########################")
        sketch_image = Image.new('RGB', (rgb_image.shape[0], rgb_image.shape[1]), color=(255, 255, 255))

        id_map = torch.from_numpy(
            np.array(Image.open(idmap_path).resize((width, height), Image.NEAREST))
        )
        height, width, _ = rgb_image.shape
        rgb_list, mask_list, success_list = [], [], []
        for idx in indexes:
            mask = (id_map == idx).float()

            if (
                skip_small_object
                and mask.sum() <= small_image_proportion * height * width
            ):
                success_list.append(False)
                continue

            if morph_perturb:
                mask = random_morphological_transform(
                    mask, max_kernel_size=max_kernel_size, p_dilation=p_dilation
                )

            mask_3c = mask.unsqueeze(-1).repeat(1, 1, 3)
            part_rgb = rgb_image * mask_3c + background_color * (1 - mask_3c)
            rgb_list.append(part_rgb)
            mask_list.append(mask)
            success_list.append(True)

        # get sketch images of each object and then get their own gating map.
        # @TODO: 这里可能提前转换成torch.Tensor会有更高的效率，避免以PIL.Image的方式在不同位置进行传输。
        if len(mask_list) > 0:
            sketch_image_list = prepare_sketch_images(
                [sketch_image],
                len(mask_list), # num_instance
                1,
                seg_images=[mask_list],
                mode="zoom",
                cfg=False,
            )
        else:
            raise ValueError("No valid sketch images found.")
        processed_sketch_image_tensor = self.sketch_processor(images=sketch_image_list, return_tensors="pt")['pixel_values']
        if return_gating_maps:
            process_sketch_image = tensor_to_pil_list(processed_sketch_image_tensor)
            patch_size = self.cfg.sketch_vision_tower_patch_size
            gating_map = get_sketch_spatial_gating_map([process_sketch_image], patch_size, str(rgb_list[0].device), concat=True)


            # gating_map 和 sketch_image的类型，还要看最后train的代码怎么写的进行相应调整。
        #
            return rgb_list, mask_list, success_list, processed_sketch_image_tensor, gating_map
        return rgb_list, mask_list, success_list, processed_sketch_image_tensor

    
    def _getitem_scene(self, index):
        background_color = torch.as_tensor(self.get_bg_color(self.cfg.background_color))

        # Surface
        scene = self.all_scenes[index]
        scene_objects = self.all_objects[index]
        surfaces = []
        for scene_object in scene_objects:
            surface_path = os.path.join(
                scene, f"{scene_object}.{self.cfg.surface_suffix}"
            )
            surface = self.load_surface(
                surface_path, self.cfg.num_surface_samples_per_object
            )
            surfaces.append(surface)
        surfaces = torch.stack(surfaces)  # (num_instances, num_points, 6)

        num_instances = surfaces.shape[0]

        # Image
        image_dir = self.all_images[index]
        image_name = (
            random.choice(self.cfg.image_names)
            if self.split == "train"
            else self.cfg.image_names[0]
        )
        image_prefix = (
            [self.cfg.image_prefix]
            if isinstance(self.cfg.image_prefix, str)
            else self.cfg.image_prefix
        )
        image_prefix = (
            random.choice(image_prefix) if self.split == "train" else image_prefix[0]
        )
        image_path = os.path.join(
            image_dir, f"{image_prefix}_{image_name}.{self.cfg.image_suffix}"
        )
        idmap_path = (
            os.path.join(
                image_dir,
                f"{self.cfg.idmap_prefix}_{image_name}.{self.cfg.idmap_suffix}",
            )
            .replace("_controlnet", "")
            .replace("_inpaint", "")
        )

        # Load image and parts
        rgb_scene = (
            self.load_image(
                image_path,
                height=self.cfg.height,
                width=self.cfg.width,
                background_color=background_color,
                remove_bg=self.cfg.remove_scene_bg,
                idmap_path=idmap_path,
            )
            .unsqueeze(0)
            .repeat(num_instances, 1, 1, 1)
            .permute(0, 3, 1, 2)
        )

        # @TODO: 暂时sketch_image_list就用PIL.Image，避免自己乱搞搞得效果不太好。
        rgbs, masks, success_list, sketch_image_list, gating_map = self.load_parts(
            image_path,
            idmap_path,
            list(range(1, num_instances + 1)),
            self.cfg.height,
            self.cfg.width,
            background_color,
            skip_small_object=self.cfg.skip_small_object,
            small_image_proportion=self.cfg.small_image_proportion,
            morph_perturb=self.cfg.morph_perturb,
            max_kernel_size=self.cfg.max_kernel_size,
            p_dilation=self.cfg.p_dilation,
        )

        #

        if len(rgbs) == 0:
            return self._getitem(random.randint(0, self.__len__() - 1))

        rgb = torch.stack(rgbs).permute(0, 3, 1, 2)
        mask = torch.stack(masks).unsqueeze(1)

        # Update `surfaces`, `num_instances`, `rgb_scene` according to `success_list`
        success_list = torch.tensor(success_list, dtype=torch.bool)
        surfaces = surfaces[success_list]
        num_instances = surfaces.shape[0]
        rgb_scene = rgb_scene[success_list]

        if self.cfg.max_num_instances is not None:
            if num_instances > self.cfg.max_num_instances:
                indices = torch.randperm(num_instances)[: self.cfg.max_num_instances]
                surfaces = surfaces[indices]
                rgb = rgb[indices]
                mask = mask[indices]
                rgb_scene = rgb_scene[indices]

                sketch_image_list = sketch_image_list[indices]
                gating_map = gating_map[indices]

                num_instances = self.cfg.max_num_instances

        # Scene id
        scene_id = "-".join(image_dir.split("/")[-2:])

        rv = {
            "id": scene_id,
            "num_instances": num_instances,
            "surface": surfaces,
            "rgb": rgb,
            "mask": mask,
            "sketch": sketch_image_list,
            "gating_map": gating_map,
            "rgb_scene": (
                rgb_scene if self.cfg.use_scene_image else torch.zeros_like(rgb_scene)
            ),
        }

        if self.cfg.return_scene:
            surface_scene = surfaces.view(-1, *surfaces.shape[2:])
            rv.update({"surface_scene": surface_scene})

        if self.cfg.return_crop_padded:
            cropped_rgbs, cropped_masks = self.crop_and_pad(
                rgbs, masks, self.cfg.height_crop_padded, self.cfg.width_crop_padded
            )
            cropped_rgb = torch.stack(cropped_rgbs)
            cropped_mask = torch.stack(cropped_masks).unsqueeze(1)

            rv.update(
                {"rgb_crop_padded": cropped_rgb, "mask_crop_padded": cropped_mask}
            )

        if self.cfg.padding:
            keys = [
                "surface",
                "rgb",
                "mask",
                "rgb_scene",
                "rgb_crop_padded",
                "mask_crop_padded",
                "sketch",
                "gating_map"
            ]
            # print(f"BEFORE, rgb: {rv['rgb'].shape} | gating: {rv['gating_map'].shape}")
            if num_instances < self.cfg.num_instances_per_batch:
                pad = self.cfg.num_instances_per_batch - num_instances
                indices = torch.randint(
                    0, num_instances, (pad,), device=surfaces.device
                )
                updated_dict = {
                    k: torch.cat([v, v[indices]]) if k in keys else v
                    for k, v in rv.items()
                }
                # print(f"AFTER, rgb: {updated_dict['rgb'].shape} | gating: {updated_dict['gating_map'].shape}")

            else:
                indices = torch.randperm(num_instances, device=surfaces.device)
                indices = indices[: self.cfg.num_instances_per_batch]
                updated_dict = {
                    k: v[indices] if k in keys else v for k, v in rv.items()
                }
                updated_dict.update({"num_instances": self.cfg.num_instances_per_batch})
            rv = updated_dict

        return rv

    
    def _getitem_mix(self, index):


        background_color = torch.as_tensor(self.get_bg_color(self.cfg.background_color))

        surfaces, rgbs, masks, sketch, gating_map = [], [], [], [], []

        indexes = torch.randint(
            0, len(self.mix_all_scenes), (self.cfg.num_instances_per_batch,)
        )
        for i in indexes:
            scene = self.mix_all_scenes[i]

            # Surface
            surface_path = f"{scene}.{self.cfg.mix_surface_suffix}"
            surface = self.load_surface(
                surface_path, self.cfg.num_surface_samples_per_object
            )
            surfaces.append(surface)

            # Image
            image_dir = self.mix_all_images[i]
            image_name = (
                random.choice(self.cfg.mix_image_names)
                if self.split == "train"
                else self.cfg.mix_image_names[0]
            )
            image_path = os.path.join(
                image_dir,
                f"{self.cfg.mix_image_prefix}_{image_name}.{self.cfg.mix_image_suffix}",
            )

            rgb, mask = self.load_image(
                image_path,
                height=self.cfg.height,
                width=self.cfg.width,
                background_color=background_color,
                return_mask=True,
            )
            # @TODO: informative_drawings API. Result should be a list[PIL.Image], each element is a sketch image of a scene.
            #   Might you will check if rgb is corresponding original image. If yes, use rgb as original input and get sketch.
            sketch_image = Image.new('RGB', (self.cfg.width, self.cfg.height), color=(255, 255, 255))

            if mask is not None:
                sketch_image_list = prepare_sketch_images(
                    [sketch_image],
                    mask.shape[0],  # num_instance
                    1,
                    seg_images=[mask],
                    mode="zoom",
                    cfg=False,
                )
            else:
                raise ValueError("No valid sketch images found.")
            processed_sketch_image_tensor = self.sketch_processor(images=sketch_image_list, return_tensors="pt")[
                'pixel_values']
            process_sketch_image = tensor_to_pil_list(processed_sketch_image_tensor)
            patch_size = self.cfg.sketch_vision_tower_patch_size
            _gating_map = get_sketch_spatial_gating_map(
                [process_sketch_image],
                patch_size,
                str(rgb.device),
                concat=True)

            rgbs.append(rgb)
            masks.append(mask)
            sketch.append(processed_sketch_image_tensor)
            gating_map.append(_gating_map)
        # print(f"#########################44num instance: 1")
        surfaces = torch.stack(surfaces)  # (num_instances, num_points, 6)
        rgb = torch.stack(rgbs).permute(0, 3, 1, 2)
        mask = torch.stack(masks).unsqueeze(1)
        sketch = torch.stack(sketch)
        gating_map = torch.stack(gating_map)

        # Scene ID
        scene_id = self.mix_all_images[indexes[0]].split("/")[-1]

        rv = {
            "id": scene_id,
            "num_instances": 1,
            "surface": surfaces,
            "rgb": rgb,
            "mask": mask,
            "sketch": sketch,
            "gating_map": gating_map,
            "rgb_scene": (
                rgb if self.cfg.use_scene_image else torch.zeros_like(rgb)
            ),  # here single object is the scene
        }

        return rv

    
    def collate(self, batch):
        # for idx,i in enumerate(batch):
        #     for k,v in i.items():
        #         if type(v) == torch.Tensor:
        #             print(f"{idx} key: {k} value: {v.shape}")
        #         else:
        #             print(f"{idx} key: {k} value: {v}")
        #     # print("####################")
        batch = torch.utils.data.default_collate(batch)
        pack = lambda t: t.view(-1, *t.shape[2:])
        for k in batch.keys():
            if k in [
                "surface",
                "rgb",
                "mask",
                "rgb_scene",
                "sketch",
                "gating_map",
                "rgb_crop_padded",
                "mask_crop_padded",
            ]:
                batch[k] = pack(batch[k])

        batch["num_instances_per_batch"] = self.cfg.num_instances_per_batch

        return batch

class MultiObjectWithSketchDataModule(pl.LightningDataModule):
    cfg: MultiObjectDataWithSketchConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiObjectDataWithSketchConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            # @TODO: change back to train
            self.train_dataset = MultiObjectWithSketchDataset(self.cfg, "openai/clip-vit-base-patch32", "test")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiObjectWithSketchDataset(self.cfg, "openai/clip-vit-base-patch32", "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MultiObjectWithSketchDataset(self.cfg, "openai/clip-vit-base-patch32", "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        # print("#########3")
        # print(len(self.train_dataset))
        # print("#########4")

        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=self.train_dataset.collate,
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.val_dataset.collate,
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.test_dataset.collate,
            persistent_workers=True
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()