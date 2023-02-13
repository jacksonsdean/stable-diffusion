# from stable_diffusion_videos.stable_diffusion_walk import walk
from stable_diffusion_videos import StableDiffusionWalkPipeline
import torch

prompt_n_seed = {
    "a cute, beautiful painting of a ball of hair that will soon roam the neighborhood riding a unicycle": 743,
    "A cute gecko eating a huge oreo": 140,
    "A rubber duck on a chess board": 40,
    "A  beautiful painting of butterflies flying in golden light": 560,
    "A  beautiful painting of a rooster reading an algebra textbook": 996,
    "A caterpillar doing math homework": 283,
    "A sparse black and white painting of a museum skeleton of a T-Rex. The ground is wet and reflective.": 116,
    "Fish playing soccer at the world cup": 277,
    "Flowers playing violin": 277,
    "A close-up of a violin in a bed of flowers": 277,
}


use_prompt_n_seed = {
    "a cute, beautiful painting of a ball of hair that will soon roam the neighborhood riding a unicycle": 4765, # 4765
    "A cute gecko eating a huge oreo": 345, # 345
    "A rubber duck on a chess board": 9660,  # 9660
    "A  beautiful painting of butterflies flying in golden light": 359, # 359
    "A  beautiful painting of a rooster reading an algebra textbook": 9819, #9819,  4023, 2379
    "A  beautiful painting of a rooster reading an algebra textbook": 4023, #9819,  4023, 2379
    "A  beautiful painting of a rooster reading an algebra textbook": 2379, #9819,  4023, 2379
    "A  beautiful painting of a rooster reading an algebra textbook": 7683, #9819,  4023, 2379, 7683
    "A caterpillar doing math homework": 7626, # 7626, 4118
    "A caterpillar doing math homework": 4118, # 7626, 4118
    "A sparse black and white painting of a museum skeleton of a T-Rex. The ground is wet and reflective.": 2631, # 2631
    "A sparse black and white painting of a museum skeleton of a T-Rex. The ground is wet and reflective.": 4662, # 4662
    "A sparse black and white painting of a museum skeleton of a T-Rex. The ground is wet and reflective.": 8889, # 8889
    "Fish playing soccer at the world cup": 2717, # 2717, 6803
    "Fish playing soccer at the world cup": 6803, # 2717, 6803
    "Fish playing soccer at the world cup": 9881, # 2717, 6803, 9881
    "Flowers playing violin": 2148,
    "A close-up of a violin in a bed of flowers": 277,
    "A close-up of a violin in a bed of flowers": 264,
    "A close-up of a violin in a bed of flowers": 4878,
}

if __name__ == '__main__':

    pipeline = StableDiffusionWalkPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        revision="fp16",
    ).to("cuda")

    video_path = pipeline.walk(
        prompts=list(use_prompt_n_seed.keys()),
        seeds=list((use_prompt_n_seed.values())),
        num_interpolation_steps=50,
        height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
        width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
        output_dir='gravity_sequestered',        # Where images/videos will be saved
        name='take_2',        # Subdirectory of output_dir where images/videos will be saved
        guidance_scale=7.5,         # Higher adheres to prompt more, lower lets model take the wheel
        num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
    )





