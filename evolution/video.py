# from stable_diffusion_videos.stable_diffusion_walk import walk
from diffusers import StableDiffusionPipeline as Stab
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from util import save, embed
import imageio as iio


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    torch.manual_seed(seed)


    pipeline = Stab.from_pretrained(
        # "CompVis/stable-diffusion-v1-4",
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        revision="fp16",
    ).to(device)


    
    prompt = "a perfectly round sphere bathed in golden yellow rays of beautiful sunshine, abstract digital painting. Trending on artstation."
    
    emb_history = torch.load( f"outputs/history_{prompt}.pt")
    
    hist = [e for e in emb_history]
    images = []
    
    interps_per_image = 10
    
    interps = []
    for i in range(len(hist)-1):
        interps.append(torch.tensor(np.linspace(hist[i].cpu().numpy(), hist[i+1].cpu().numpy(), interps_per_image),device=device))
    interps = torch.cat(interps)

    
    for emb in interps:
        outputs = pipeline(
            # prompt=None,
            prompt_embeds = emb.unsqueeze(0),
            # seed=42,
            height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
            width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
            guidance_scale=7.5,         # Higher adheres to prompt more, lower lets model take the wheel
            num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
            generator=torch.Generator(device=device).manual_seed(seed),        # PyTorch Generator for reproducibility
        )
        images.append(outputs[0][0])
    iio.mimsave(f"outputs/{prompt}.gif", images)