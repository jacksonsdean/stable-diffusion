# from stable_diffusion_videos.stable_diffusion_walk import walk
from diffusers import StableDiffusionPipeline as Stab
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from util import save, embed

def generation(embeddings, prompt, pipeline, generator, iters=1):
    imgs = []
    for i in range(iters):
        g = torch.Generator(device=device).manual_seed(42)
        
        outputs = pipeline(
            # prompt=None,
            prompt_embeds = embeddings[i].unsqueeze(0),
            # seed=42,
            height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
            width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
            guidance_scale=7.5,         # Higher adheres to prompt more, lower lets model take the wheel
            num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
            generator=g,        # PyTorch Generator for reproducibility

        )
        
        pil_images = outputs[0]
        imgs.extend(pil_images)
    torch_images = []
    for i, image in enumerate(imgs):
        image = ToTensor()(image)
        torch_images.append(image)
    grid = make_grid(torch_images)
    save(grid, "outputs/current.png")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    initial_population_variance = 0.1
    mutation_rate = 0.01
    num_samples = 3
    history = []
    seed = 42
    torch.manual_seed(seed)

    pipeline = Stab.from_pretrained(
        # "CompVis/stable-diffusion-v1-4",
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        revision="fp16",
    ).to(device)


    def dummy(images, device, d, **kwargs):
        return images, False
    pipeline.run_safety_checker = dummy

    # FOR LOW MEMORY GPUs
    # pipe.enable_attention_slicing() 
    

    population = torch.randn((num_samples, 77, 1024)).to(device) * initial_population_variance
    prompt = "abstract digital painting. Trending on artstation."
    inp = ""
    embeddings = embed(prompt, pipeline)
    embeddings = embeddings.repeat(population.shape[0], 1, 1)
    try:
        while inp != "q":
            embeddings = embeddings + population
            g = torch.Generator(device=device).manual_seed(seed)
            generator = [g]*num_samples
            generation(embeddings, prompt, pipeline, generator, iters=num_samples)
            # evolution:
            print(f"Done generation, choose one to mutate in {list(range(num_samples))} or q to quit")
            inp = input()
            if inp == "q":
                break
            while inp not in list(range(num_samples)):
                try:
                    inp = int(inp)
                except:
                    print("Invalid input")
                    continue
            selected_genome = population[inp]
            history.append(embeddings[inp].detach().cpu())
            population = selected_genome.repeat(num_samples, 1, 1)
            mut = torch.randn((num_samples, 77, 1024)).to(device) * mutation_rate
            population += mut
    except KeyboardInterrupt:
        pass
    history = torch.stack(history)
    torch.save(history, f"outputs/history_{prompt}.pt")
        

   


        