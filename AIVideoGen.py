import os
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import subprocess
import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler

main_prompt = input("Main prompt: ")
project_name = input("Project name: ").strip() or "video"
frame_count = int(input("How many frames to generate: ").strip() or "24")
fps = int(input("Frame speed (fps) for MP4: ").strip() or "8")

system_prompt = f"""
You are an animation planner.
Split the motion into sequential frames.
Keep character and camera identical.
Only change pose slightly each frame.
Return one line per frame.

Prompt: {main_prompt}
"""

print("\n[Ollama] Generating frame prompts...\n")

frames = []
while len(frames) < frame_count:
    result = subprocess.run(
        ["ollama", "run", "dolphin-phi"],
        input=system_prompt.encode(),
        stdout=subprocess.PIPE
    )
    new_frames = [l.strip() for l in result.stdout.decode().split("\n") if l.strip()]
    frames.extend(new_frames)
frames = frames[:frame_count]

subprocess.run(["ollama", "stop", "dolphin-phi"])
print(f"[Ollama] Generated {len(frames)} frames")

out_dir = f"{project_name}_frames"
os.makedirs(out_dir, exist_ok=True)

height = 768
width = 768
steps = 30
seed = 42

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.enable_attention_slicing()
pipe.vae.enable_slicing()

for i, frame_prompt in enumerate(frames, start=1):
    print(f"[SDXL] Frame {i}/{len(frames)}")
    generator = torch.Generator(device="cuda").manual_seed(seed + i)

    result = pipe(
        prompt=frame_prompt,
        negative_prompt="blurry, low quality, bad anatomy, malformed, fused anatomy",
        guidance_scale=7.0,
        height=height,
        width=width,
        num_inference_steps=steps,
        generator=generator
    )

    image = result.images[0]
    image.save(os.path.join(out_dir, f"frame{i}.png"))

    del result, image, generator
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

video_name = f"{project_name}.mp4"
subprocess.run([
    "ffmpeg",
    "-y",
    "-framerate", str(fps),
    "-i", os.path.join(out_dir, "frame%d.png"),
    "-pix_fmt", "yuv420p",
    video_name
])

print(f"\nVideo saved: {video_name} at {fps} fps")
