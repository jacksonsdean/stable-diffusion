# from stable_diffusion_videos.stable_diffusion_walk import walk
from evolution_pipeline import EvolutionPipeline
from diffusers import StableDiffusionPipeline

from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from util import save, embed


def generation(prompt, pipeline, seed, num_samples=3):
    imgs = []
    # for i in range(iters):
    g = torch.Generator(device=pipeline.device).manual_seed(seed)
    outputs = pipeline(
        prompt=prompt,
        # prompt_embeds = embeddings[i].unsqueeze(0),
        height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
        width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
        guidance_scale=7.5,         # Higher adheres to prompt more, lower lets model take the wheel
        num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
        generator=g,        # PyTorch Generator for reproducibility
        num_images_per_prompt = num_samples
    )
    
    pil_images = outputs[0]
    print(pil_images)
    imgs.extend(pil_images)
    torch_images = []
    for i, image in enumerate(imgs):
        image = ToTensor()(image)
        torch_images.append(image)
    # grid = make_grid(torch_images)

        
    save(torch_images, "outputs/current.png",text=True)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # initial_population_variance = 0.1
    mutation_rate = .25
    num_samples = 3
    history = []
    seed = 42
    torch.manual_seed(seed)

    pipeline = EvolutionPipeline.from_pretrained(
    # pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        # "stabilityai/stable-diffusion-2-1",
        # torch_dtype=torch.float16,
        # revision="fp16",
        safety_checker=None,
        requires_safety_checker = False,
        mutation_rate = mutation_rate,
        samples_per_gen = num_samples,
    ).to(device)

    # FOR LOW MEMORY GPUs
    pipeline.enable_attention_slicing() 
    pipeline.enable_vae_slicing()
    # pipeline.enable_sequential_cpu_offload()
    
    

    # population = torch.randn((num_samples, 77, 1024)).to(device) * initial_population_variance
    prompt = "love is golden yellow sunshine, abstract digital painting. Trending on artstation."
    # prompt = "a dog"
    inp = ""
    # embeddings = embed(prompt, pipeline)
    # embeddings = embeddings.repeat(population.shape[0], 1, 1)
    try:
        while inp != "q":
            # embeddings = embeddings + population
            generation(prompt, pipeline, seed, num_samples)
            
            print(f"Done generation, choose one to mutate in {list(range(num_samples))} or q to quit")
            inp = input()
            if inp == "q":
                break
            while inp not in list(range(num_samples)):
                try:
                    inp = int(inp)
                except:
                    print("Invalid input")
                    inp = input()
                    continue
            print(f"Mutating {inp}")
            pipeline.select(inp)
            pipeline.mutate()
            
    except KeyboardInterrupt:
        pass
    history = torch.stack(pipeline.history)
    torch.save(history, f"outputs/history_{prompt}.pt")
        

   


        