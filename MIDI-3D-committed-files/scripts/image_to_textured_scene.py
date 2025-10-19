import argparse
import os
from typing import Any, Union

import torch
import trimesh
from huggingface_hub import snapshot_download
from mvadapter.pipelines.pipeline_texture import ModProcessConfig, TexturePipeline
from PIL import Image
from tqdm import tqdm

import scripts.inference_midi as midi_infer
import scripts.mvadapter_ig2mv as ig2mv_infer
from midi.pipelines.pipeline_midi import MIDIPipeline
from midi.utils.mesh_process import process_raw


def prepare_midi_pipeline(device, dtype):
    return midi_infer.prepare_pipeline(device, dtype)


def prepare_ig2mv_pipeline(device, dtype):
    return ig2mv_infer.prepare_pipeline(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        unet_model=None,
        lora_model=None,
        adapter_path="huanngzh/mv-adapter",
        scheduler=None,
        num_views=6,
        device=device,
        dtype=dtype,
    )


def prepare_texture_pipeline(device, dtype):
    os.makedirs("checkpoints", exist_ok=True)
    lama_path = "checkpoints/big-lama.pt"
    esrgan_path = "checkpoints/RealESRGAN_x2plus.pth"

    if not os.path.exists(lama_path):
        os.system(
            f"wget https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt -O {lama_path}"
        )
    if not os.path.exists(esrgan_path):
        os.system(
            f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -O {esrgan_path}"
        )

    texture_pipe = TexturePipeline(
        upscaler_ckpt_path=esrgan_path,
        inpaint_ckpt_path=lama_path,
        device=device,
    )
    return texture_pipe


def run_midi(
    midi_pipe: MIDIPipeline,
    rgb_image: Union[str, Image.Image],
    seg_image: Union[str, Image.Image],
    seed: int,
) -> trimesh.Scene:
    scene = midi_infer.run_midi(
        midi_pipe, rgb_image, seg_image, seed, num_inference_steps=30
    )
    torch.cuda.empty_cache()
    return scene


def run_i2tex(
    ig2mv_pipe: Any,
    texture_pipe: Any,
    scene: trimesh.Scene,
    rgb_image: Union[str, Image.Image],
    seg_image: Union[str, Image.Image],
    seed: int,
    output_dir: str,
) -> trimesh.Scene:
    os.makedirs(output_dir, exist_ok=True)

    scene.export(f"{output_dir}/scene_notextured.glb")
    print(f"Total meshes in scene: {len(scene.geometry)}")

    instance_rgbs, instance_masks, _ = midi_infer.split_rgb_mask(rgb_image, seg_image)

    textured_scene = trimesh.Scene()
    for i, (mesh, rgb, mask) in tqdm(
        enumerate(zip(scene.geometry.values(), instance_rgbs, instance_masks)),
        total=len(instance_rgbs),
    ):
        tmp_path = f"{output_dir}/mesh_{i}.glb"
        mesh.export(tmp_path)

        # preprocess mesh
        tmp_path_new = f"{output_dir}/mesh_{i}_preprocessed.glb"
        process_raw(tmp_path, tmp_path_new, preprocess=True)
        tmp_path = tmp_path_new

        # prepare mvadapter input
        rgba = rgb.convert("RGBA")
        rgba.putalpha(mask)
        rgba_path = f"{output_dir}/rgba_{i}.png"
        rgba.save(rgba_path)

        # run mvadapter
        mv_images, _, _, _, _ = ig2mv_infer.run_pipeline(
            ig2mv_pipe,
            tmp_path,
            num_views=6,
            text=f"high quality",
            image=rgba,
            height=768,
            width=768,
            num_inference_steps=35,
            guidance_scale=3.0,
            reference_conditioning_scale=0.7,
            seed=seed,
        )
        mv_path = f"{output_dir}/mv_{i}.png"
        ig2mv_infer.make_image_grid(mv_images, rows=1).save(mv_path)

        # run texture generation
        texture_out = texture_pipe(
            mesh_path=tmp_path,
            save_dir=output_dir,
            save_name=f"mesh_{i}",
            move_to_center=True,
            uv_unwarp=False,
            preprocess_mesh=False,
            uv_size=4096,
            rgb_path=mv_path,
            rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
            camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        )
        textured_obj_path = texture_out.shaded_model_save_path

        textured_mesh = trimesh.load(textured_obj_path, process=False)
        textured_scene.add_geometry(textured_mesh)

        torch.cuda.empty_cache()

    return textured_scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_image", type=str, required=True)
    parser.add_argument("--seg_image", type=str, required=True)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()

    midi_pipe = prepare_midi_pipeline(device="cuda", dtype=torch.float16)
    ig2mv_pipe = prepare_ig2mv_pipeline(device="cuda", dtype=torch.float16)
    texture_pipe = prepare_texture_pipeline(device="cuda", dtype=torch.float16)

    print(f"Running MIDI...")
    scene = run_midi(midi_pipe, args.rgb_image, args.seg_image, args.seed)
    print(f"Run MV-Adapter for texture generation...")
    scene = run_i2tex(
        ig2mv_pipe,
        texture_pipe,
        scene,
        args.rgb_image,
        args.seg_image,
        args.seed,
        output_dir=args.output,
    )

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "textured_scene.glb")
    scene.export(output_path)
    print(f"Textured scene saved to {output_path}")
