# from stable_diffusion_videos.stable_diffusion_walk import walk
from stable_diffusion_videos import StableDiffusionWalkPipeline
import torch
from walk import use_prompt_n_seed, prompt_n_seed

pipeline = StableDiffusionWalkPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    revision="fp16",
).to("cuda")

def dummy(images, **kwargs):
    return images, False
pipeline.safety_checker = dummy

# total len = 212
num_lines = len(prompt_n_seed.keys())


# Seconds in the song.
audio_offsets = [i * 212 // num_lines for i in range(num_lines)]
print(audio_offsets)
fps = 60  # Use lower values for testing (5 or 10), higher values for better quality (30 or 60)

# Convert seconds to frames
num_interpolation_steps = [(b-a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]

video_path = pipeline.walk(
    prompts=list(use_prompt_n_seed.keys()),
    seeds=list((use_prompt_n_seed.values())),
    num_interpolation_steps=num_interpolation_steps,
    audio_filepath='audio/storm.mp3',
    audio_start_sec=audio_offsets[0],
    fps=fps,
    height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
    output_dir='gravity_sequestered',        # Where images/videos will be saved
    name='take_7',        # Subdirectory of output_dir where images/videos will be saved
    guidance_scale=7.5,         # Higher adheres to prompt more, lower lets model take the wheel
    num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
)


