# from stable_diffusion_videos.stable_diffusion_walk import walk
from stable_diffusion_videos import StableDiffusionWalkPipeline
import torch

pipeline = StableDiffusionWalkPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    revision="fp16",
).to("cuda")


prompt_n_seed = {
    "a man-eating dustbunny that will soon roam the neighborhood": 743,
    "A cute gecko eating a huge oreo": 140,
    "A rubber duck on a chess board": 40,
    "A  beautiful painting of butterflies flying in golden light": 560,
    "A  beautiful painting of a rooster reading an algebra textbook": 996,
    "A caterpillar doing math homework": 283,
    "A museum skeleton of a T-Rex": 116,
    "Fish playing soccer at the world cup": 277,
    "Flowers playing violin": 277,
    "A close-up of a violin in a bed of flowers": 277,
}

video_path = pipeline.walk(
    prompts=list(prompt_n_seed.keys()),
    seeds=(prompt_n_seed.values()),
    num_interpolation_steps=3,
    height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
    width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
    output_dir='dreams',        # Where images/videos will be saved
    name='animals_test',        # Subdirectory of output_dir where images/videos will be saved
    guidance_scale=8.5,         # Higher adheres to prompt more, lower lets model take the wheel
    num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
    upscale=True,               # Upscale images to 1024x1024 before saving
)



# walk(prompts=list(prompt_n_seed.keys()), 
#      seeds=list(prompt_n_seed.values()),
#      make_video=True, 
#      name="gravity_sequestered",
#      num_steps=200,
#      use_lerp_for_text=True,
#      upscale=True
#      )