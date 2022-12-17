import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import random
import matplotlib.pyplot as plt
import os


prompts = [
          "a man-eating dustbunny that will soon roam the neighborhood",
"A cute gecko eating a huge oreo",
"A rubber duck on a chess board",
"A  beautiful painting of butterflies flying in golden light",
"A  beautiful painting of a rooster reading an algebra textbook",
"A caterpillar doing math homework",
"A museum skeleton of a T-Rex",
"Fish playing soccer at the world cup",
"Flowers playing violin",
"A close-up of a violin in a bed of flowers"                           ,                       
]


# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", 
                                                revision="fp16",
                                                torch_dtype=torch.float16, 
                                                use_auth_token=True)  
pipe = pipe.to("cuda")

def infer(prompt, num_inference_steps=50, 
            samples=5, seed=1024, guidance_scale=7.5, 
            width=512, height=512):
    generator = torch.Generator("cuda").manual_seed(seed)
    w = width//8*8
    h = height//8*8
    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5, 
                    generator=generator, width=w, height=h, 
                    num_inference_steps=num_inference_steps)["sample"][0]
    return image


for p in prompts:

    prompt_orig = p 
    if not os.path.exists("imagery"):
        os.mkdir("imagery")

    if not os.path.exists(f"imagery/{prompt_orig}"):
        os.mkdir(f"imagery/{prompt_orig}/")

    HM = 200
    for i in range(HM):
        print(f"{i+1}/{HM}")
        prompt_to_use = prompt_orig
        seed = random.randint(0, 10000)
        print(seed)
        image = infer(prompt_to_use, num_inference_steps=75, 
                      seed=seed, width=512, height=512)
        image.save(f"imagery/{prompt_to_use}/{seed}.png")