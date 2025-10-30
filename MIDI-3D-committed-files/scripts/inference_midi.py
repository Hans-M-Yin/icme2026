import argparse
import os
from collections import defaultdict
from glob import glob
from typing import Any, List, Union

import gradio as gr
import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps
from skimage import measure
import sys
from pathlib import Path
current_dir = Path(__file__).parent
# 获取项目根目录（即midi和scripts的父目录）
root_dir = current_dir.parent
# 将根目录添加到sys.path
sys.path.append(str(root_dir))

from midi.pipelines.pipeline_midi import MIDIPipeline
from midi.utils.smoothing import smooth_gpu
from midi.sketch.fusion_adapter import FusionAdapterConfig,SketchFusionAdapter
from midi.sketch.sketch_tower import SketchVisionTowerConfig,SketchVisionTower
from midi.models.transformers.triposg_transformer import TripoSGDiTModel
import torch.nn as nn

def check_meta_tensors(model, verbose: bool = True) -> dict:
    """
    检查模型中所有张量的设备类型，特别关注 meta device 上的张量

    参数:
        model: 要检查的 PyTorch 模型
        verbose: 是否打印详细结果

    返回:
        包含所有张量设备信息的字典
    """
    device_stats = defaultdict(list)

    # 递归遍历所有模块
    for name, module in model.named_modules():
        # 检查模块参数
        for param_name, param in module.named_parameters(recurse=False):
            device_type = param.device.type
            device_stats[device_type].append((f"{name}.{param_name}", param.shape))

            if device_type == "meta" and verbose:
                print(f"⚠️ Meta tensor found: {name}.{param_name} | Shape: {tuple(param.shape)}")

        # 检查模块缓冲区
        for buffer_name, buffer in module.named_buffers(recurse=False):
            device_type = buffer.device.type
            device_stats[device_type].append((f"{name}.{buffer_name}", buffer.shape))

            if device_type == "meta" and verbose:
                print(f"⚠️ Meta buffer found: {name}.{buffer_name} | Shape: {tuple(buffer.shape)}")

    # 打印汇总统计
    if verbose:
        print("\n📊 Device Distribution Summary:")
        for device, tensors in device_stats.items():
            print(f"• {device.upper()} device: {len(tensors)} tensors")

            # 显示前5个meta张量
            if device == "meta" and tensors:
                print("  Top Meta Tensors:")
                for tensor_name, shape in tensors[:5]:
                    print(f"    - {tensor_name} | Shape: {shape}")

    return device_stats


def check_pipeline_meta_tensors(pipe):
    """返回 (组件统计, 总meta张量数, 所有meta张量列表)"""
    component_stats = {}
    all_meta_tensors = []
    total_meta = 0

    components = ['unet', 'vae', 'text_encoder', 'scheduler', 'adapter_layers']
    for comp_name in components:
        if hasattr(pipe, comp_name):
            comp = getattr(pipe, comp_name)
            if isinstance(comp, nn.Module):
                comp_stats = check_meta_tensors(comp, verbose=False)
                component_stats[comp_name] = comp_stats

                comp_meta = comp_stats.get('meta', [])
                total_meta += len(comp_meta)
                all_meta_tensors.extend(comp_meta)

    # 修正报告输出
    print("\n" + "=" * 50)
    print("MIDI Pipeline Meta Tensor Report")
    print("=" * 50)

    for comp, stats in component_stats.items():
        meta_count = len(stats.get('meta', []))
        print(f"{comp.upper()}: {meta_count} meta tensors")

    print("\n" + "=" * 50)
    print(f"TOTAL META TENSORS: {total_meta}")

    if total_meta > 0:
        print("🔴 META TENSORS FOUND!")
    else:
        print("✅ No tensors found on meta device!")
    print("=" * 50)

    return component_stats, total_meta, all_meta_tensors


def materialize_meta_tensors(module: nn.Module, device: str):
    """将元设备张量转移到实际设备"""
    for name, param in module.named_parameters():
        if param.device.type == 'meta':
            # 创建空张量（不初始化）
            new_param = nn.Parameter(
                torch.empty_like(param, device=device),
                requires_grad=param.requires_grad
            )
            setattr(module, name, new_param)

    for name, buf in module.named_buffers():
        if buf.device.type == 'meta':
            new_buf = torch.empty_like(buf, device=device)
            setattr(module, name, new_buf)

    # 递归处理子模块
    for child in module.children():
        materialize_meta_tensors(child, device)

    return module
def prepare_pipeline(device, dtype):
    local_dir = "pretrained_weights/MIDI-3D"
    snapshot_download(repo_id="VAST-AI/MIDI-3D", local_dir=local_dir)

    # Original implementation
    # pipe: MIDIPipeline = MIDIPipeline.from_pretrained(local_dir).to(device, dtype)

    # Sketch Test
    sketch_fusion_adapter_config = FusionAdapterConfig(
        num_dit_layers=21,
        num_vision_layers=12,
        dim_dit_latent=2048,
        dim_vision_latent=768,
        fusion_mode="specific",
        dit_layer_seqs=[0,2,4,5,6,8,9,10,12,14,16,18,20],
        vision_layer_seqs=[0,1,2,3,4,5,6,7,8,9,10,11,12],
        enable_projector=False,
        dim_projected_latent=768,
    )
    sketch_vision_tower_config = SketchVisionTowerConfig(
        select_feature_type="patch",
        arbitrary_input_size=False,
        input_size=(224,224),
        select_layer="all"
    )
    transformer = TripoSGDiTModel.from_pretrained(os.path.join(local_dir, "./transformer"),
                                                  strict=False,
                                                  ignore_mismatched_sizes=True,
                                                  low_cpu_mem_usage=False)
    sketch_fusion_adapter = SketchFusionAdapter(sketch_fusion_adapter_config,
                                                sketch_vision_tower_config)
    # for name, param in transformer.named_parameters():
    #     print('FIRST', name, '| ', param.device)

    pipe: MIDIPipeline = MIDIPipeline.from_pretrained(
        local_dir,
        tranformer=transformer,
        sketch_fusion_adapter=sketch_fusion_adapter,
    )

    # for name, param in transformer.named_parameters():
    #     print('THIRD', name, '| ', param.device)


    pipe.transformer = transformer
    pipe = pipe.to(device)
    comp_stats, total_meta, all_meta_tensors = check_pipeline_meta_tensors(pipe)
    # print(pipe.vae.encoder.block)
    # print(type(pipe.vae))
    # for name, param in pipe.vae.named_parameters():
    #     print(name, '| ', param.device)
    # for name, param in transformer.named_parameters():
    #     print(name, '| ', param.device)

    if total_meta > 0:
        print(f"\nFound {total_meta} tensors on meta device:")
        for tensor_info in all_meta_tensors[:5]:  # 只显示前5个
            name, shape = tensor_info
            print(f"  - {name} | Shape: {shape}")
        if total_meta > 5:
            print(f"  ... and {total_meta - 5} more.")

        # 实际处理meta张量
        print("\nMaterializing meta tensors...")
        pipe = materialize_meta_tensors(pipe, device=device)

        # 验证处理结果
        _, post_meta, _ = check_pipeline_meta_tensors(pipe)
        if post_meta > 0:
            raise RuntimeError("Failed to materialize all meta tensors!")
    else:
        # 直接转移到目标设备
        pipe = pipe.to(device)

    pipe.init_custom_adapter(
        # @TODO: 由于SketchFusionAdapter暂时就是普通cross attention，所以这里还不用设置。
        set_self_attn_module_names=[
            "blocks.8",
            "blocks.9",
            "blocks.10",
            "blocks.11",
            "blocks.12",
        ],
        set_sketch_attn_module_names=[
            "blocks.0",
            "blocks.1",
            "blocks.2",
            "blocks.3",
            "blocks.4",
            "blocks.5",
            "blocks.6",
            "blocks.7",
            "blocks.8",
            "blocks.9",
            "blocks.10",
            "blocks.11",
            "blocks.12",
            "blocks.13",
            "blocks.14",
            "blocks.15",
            "blocks.16",
            "blocks.17",
            "blocks.18",
            "blocks.19",
            "blocks.20",
            "blocks.21",

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
    sketch_image: Union[str, Image.Image],
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
        sketch_image=sketch_image,
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
    parser.add_argument("--sketch", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--do-image-padding", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./")
    args = parser.parse_args()


    sketch_image = Image.open(args.sketch)
    sketch_image_list = [sketch_image,sketch_image,sketch_image,sketch_image,sketch_image] * 2
    pipe = prepare_pipeline(device, dtype)
    run_midi(
        pipe,
        rgb_image=args.rgb,
        seg_image=args.seg,
        sketch_image=sketch_image_list,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        do_image_padding=args.do_image_padding,
    ).export(os.path.join(args.output_dir, "output.glb"))
