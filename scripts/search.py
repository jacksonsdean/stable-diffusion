import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import random
import matplotlib.pyplot as plt
import os
from walk import prompt_n_seed
from diffusers.models.cross_attention import AttnProcessor2_0

prompts = list(prompt_n_seed.keys())
model = "CompVis/stable-diffusion-v1-4"
# model = "stabilityai/stable-diffusion-2-1"
torch.set_float32_matmul_precision('high')
pipe = StableDiffusionPipeline.from_pretrained(
                                                model, 
                                                # revision="fp16",
                                                # torch_dtype=torch.float16,
                                                safety_checker=None,
                                                use_auth_token=True)  
pipe = pipe.to("cuda")

pipe.unet.set_attn_processor(AttnProcessor2_0())
# pipe.unet = torch.compile(pipe.unet)
pipe.enable_xformers_memory_efficient_attention()


def infer(prompt, num_inference_steps=50, 
            samples=5, seed=1024, guidance_scale=7.5, 
            width=512, height=512):
    generator = torch.Generator("cuda").manual_seed(seed)
    w = width//8*8
    h = height//8*8
    output = pipe(prompt, guidance_scale=guidance_scale, 
                generator=generator, 
                width=w, height=h, 
                num_inference_steps=num_inference_steps)
    image = output.images[0]
    return image


for p in prompts:

    prompt_orig = p 
    if not os.path.exists(f"imagery/{model}"):
        os.makedirs(f"imagery/{model}")

    if not os.path.exists(f"imagery/{model}/{prompt_orig}"):
        os.makedirs(f"imagery/{model}/{prompt_orig}/")

    per_prompt = 100
    print(f"prompt: {prompt_orig} ({per_prompt} images)")
    for i in range(per_prompt):
        print(f"{i+1}/{per_prompt}")
        prompt_to_use = prompt_orig
        seed = random.randint(0, 10000)
        print("seed:", seed)
        image = infer(prompt_to_use, num_inference_steps=50, 
                      seed=seed, width=1024, height=1024, guidance_scale=7.5)
        image.save(f"imagery/{model}/{prompt_to_use}/{seed}.png")