import argparse
import os
from glob import glob
from typing import Any, List, Union

import gradio as gr
import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from skimage import measure

from midi.pipelines.pipeline_midi import MIDIPipeline
from midi.utils.smoothing import smooth_gpu


def prepare_pipeline(device, dtype):
    local_dir = "pretrained_weights/MIDI-3D"
    snapshot_download(repo_id="VAST-AI/MIDI-3D", local_dir=local_dir)
    pipe: MIDIPipeline = MIDIPipeline.from_pretrained(local_dir).to(device, dtype)
    pipe.init_custom_adapter(
        set_self_attn_module_names=[
            "blocks.8",
            "blocks.9",
            "blocks.10",
            "blocks.11",
            "blocks.12",
        ]
    )
    return pipe


def preprocess_image(rgb_image, seg_image):
    if isinstance(rgb_image, str):
        rgb_image = Image.open(rgb_image)
    if isinstance(seg_image, str):
        seg_image = Image.open(seg_image)
    rgb_image = rgb_image.convert("RGB")
    seg_image = seg_image.convert("L")

    width, height = rgb_image.size

    seg_np = np.array(seg_image)
    rows, cols = np.where(seg_np > 0)
    if rows.size == 0 or cols.size == 0:
        return rgb_image, seg_image

    # compute the bounding box of combined instances
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    L = max(
        max(abs(max_row - width // 2), abs(min_row - width // 2)) * 2,
        max(abs(max_col - height // 2), abs(min_col - height // 2)) * 2,
    )

    # pad the image
    if L > width * 0.8:
        width = int(L / 4 * 5)
    if L > height * 0.8:
        height = int(L / 4 * 5)
    rgb_new = Image.new("RGB", (width, height), (255, 255, 255))
    seg_new = Image.new("L", (width, height), 0)
    x_offset = (width - rgb_image.size[0]) // 2
    y_offset = (height - rgb_image.size[1]) // 2
    rgb_new.paste(rgb_image, (x_offset, y_offset))
    seg_new.paste(seg_image, (x_offset, y_offset))

    # pad to the square
    max_dim = max(width, height)
    rgb_new = ImageOps.expand(
        rgb_new, border=(0, 0, max_dim - width, max_dim - height), fill="white"
    )
    seg_new = ImageOps.expand(
        seg_new, border=(0, 0, max_dim - width, max_dim - height), fill=0
    )

    return rgb_new, seg_new


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


@torch.no_grad()
def run_midi(
    pipe: Any,
    rgb_image: Union[str, Image.Image],
    seg_image: Union[str, Image.Image],
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    do_image_padding: bool = False,
) -> trimesh.Scene:
    if do_image_padding:
        rgb_image, seg_image = preprocess_image(rgb_image, seg_image)
    instance_rgbs, instance_masks, scene_rgbs = split_rgb_mask(rgb_image, seg_image)

    pipe_kwargs = {}
    if seed != -1 and isinstance(seed, int):
        pipe_kwargs["generator"] = torch.Generator(device=pipe.device).manual_seed(seed)

    num_instances = len(instance_rgbs)
    outputs = pipe(
        image=instance_rgbs,
        mask=instance_masks,
        image_scene=scene_rgbs,
        attention_kwargs={"num_instances": num_instances},
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        decode_progressive=True,
        return_dict=False,
        **pipe_kwargs,
    )

    # marching cubes
    trimeshes = []
    for _, (logits_, grid_size, bbox_size, bbox_min, bbox_max) in enumerate(
        zip(*outputs)
    ):
        grid_logits = logits_.view(grid_size)
        grid_logits = smooth_gpu(grid_logits, method="gaussian", sigma=1)
        torch.cuda.empty_cache()
        vertices, faces, normals, _ = measure.marching_cubes(
            grid_logits.float().cpu().numpy(), 0, method="lewiner"
        )
        vertices = vertices / grid_size * bbox_size + bbox_min

        # Trimesh
        mesh = trimesh.Trimesh(vertices.astype(np.float32), np.ascontiguousarray(faces))
        trimeshes.append(mesh)

    # compose the output meshes
    scene = trimesh.Scene(trimeshes)

    return scene


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.bfloat16

    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, required=True)
    parser.add_argument("--seg", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--do-image-padding", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./")
    args = parser.parse_args()

    pipe = prepare_pipeline(device, dtype)
    run_midi(
        pipe,
        rgb_image=args.rgb,
        seg_image=args.seg,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        do_image_padding=args.do_image_padding,
    ).export(os.path.join(args.output_dir, "output.glb"))
