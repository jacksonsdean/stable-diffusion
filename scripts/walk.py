from stable_diffusion_videos import StableDiffusionWalkPipeline
import torch


prompt_n_seed = {
        # "Many yellow leaves showering from an aspen tree in a forest of aspens, bright yellow sunlight": None,
        # "A photorealistic closeup texture of a detailed rough stone with many colors, volumetric lighting": None,
        # "A psychedelic, patterned bird with spread wings, in a pastel color scheme": None,
        # "A blizzard of white birds over an icy lake": None,
        # "A photo of a distant flock of black birds flying on a white sky over a snow-covered neighborhood with big puffs of billowing steam from the roofs 8K": None,
        # "A high-quality 8k photo of a distant murmuration of black birds flying on a white sky over snow-covered homes with big puffs of billowing steam from the chimneys": None,
        "Big puffs of steam coming from snow-covered homes and a murmuration of black birds. 8k photo.": None,
        # "A photo of an iridescent vibrant rooster standing on one foot on top of a large book of math equations in a sunny park": None,
        # "A photo of two colorful parakeets nuzzling in a lush rainforest, volumetric lighting": None,
        # "An otherworldly vast dark desert with footprints in the sky": None,
        # "Footprints across the vast dark blue sky": None,
        # # "Glowing moon-like lanterns in the darkness on a winding path": None,
        # "An oil painting with a blue color scheme of a mysterious underwater satellite with bubbles and refracted light": None,
        # "A vibrant, psychedelic moon in the style of Alex Grey": None,
        # "A tall, cute, pastel apartment building with a balcony high in the air with a cat sitting on the balcony looking out": None,
        # "Wind-chimes hanging in a rainy window": None,
        # "A detailed dollhouse in beautiful, colorful coral reef made with unreal engine": None,
        # "A flock of butterflies in a dark foggy city with volumetric lighting": None,
        # "A foggy city street with a ghostly flock of birds": None,
        # "A painting the interior of a bus with light through the windows, in the style of edward hopper, orange color scheme": None,
        # "A colorful, photorealistic piñata breaking open with a gray office in the background": None,
        # "Pastel summer flowers growing out of a peaceful banjo under a vast sky": None,
        # "A photorealistic Tyrannosaurus Rex skeleton looking at itself reflected in a rain puddle. Volumetric lighting.”": None,
        # "A bight huge topographical map in the sky over a small red bicycle fallen in the street": None,
        # "A bright huge topographical map in the sky over a small red bicycle fallen in the street": None,
        # "A shiny trumpet in a peaceful, overgrown garden": None,
        # "A sparse black and white painting of a museum skeleton of a T-Rex. The ground is wet and reflective.": None,

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





