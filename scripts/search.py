import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import random
import matplotlib.pyplot as plt
import os
from walk import prompt_n_seed

prompts = list(prompt_n_seed.keys())

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
        output = pipe(prompt, guidance_scale=guidance_scale, 
                    generator=generator, width=w, height=h, 
                    num_inference_steps=num_inference_steps)
        image = output.images[0]
    return image


for p in prompts:

    prompt_orig = p 
    if not os.path.exists("imagery"):
        os.mkdir("imagery")

    if not os.path.exists(f"imagery/{prompt_orig}"):
        os.mkdir(f"imagery/{prompt_orig}/")

    HM = 10
    for i in range(HM):
        print(f"{i+1}/{HM}")
        prompt_to_use = prompt_orig
        seed = random.randint(0, 10000)
        print("seed:", seed)
        image = infer(prompt_to_use, num_inference_steps=50, 
                      seed=seed, width=512, height=512)
        image.save(f"imagery/{prompt_to_use}/{seed}.png")