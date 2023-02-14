# from stable_diffusion_videos.stable_diffusion_walk import walk
from evolution_pipeline import EvolutionPipeline
import numpy as np
import torch
import imageio as iio
from tqdm import tqdm

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    torch.manual_seed(seed)


    pipeline = EvolutionPipeline.from_pretrained(
    # pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        safety_checker=None,
        requires_safety_checker = False,
    ).to(device)

    
    # prompt = "a perfectly round sphere bathed in golden yellow rays of beautiful sunshine, abstract digital painting. Trending on artstation."
    prompt = "love is golden yellow sunshine, abstract digital painting. Trending on artstation."
    
    
    latent_history = torch.load( f"outputs/history_{prompt}.pt")
    
    hist = [e for e in latent_history]
    images = []
    
    interps_per_image = 10
    smoothing_inters = 10
    
    interps = []
    for i in range(len(hist)-1):
        interps.append(torch.tensor(np.linspace(hist[i].cpu().numpy(), hist[i+1].cpu().numpy(), interps_per_image)))
    # interps = torch.cat(interps)

    pipeline.replay_mode()
    pipeline.enable_attention_slicing(slice_size="max") 
    pipeline.enable_vae_slicing()
    
    batch_size = 1
    batch_indx = 0
    images = []
    pbar = tqdm(total=len(interps))
    
    
    
    pipeline.history = [interps[0], interps[1]]

    outputs = pipeline(
        prompt=prompt,
        # prompt_embeds = embeddings[i].unsqueeze(0),
        height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
        width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
        guidance_scale=7.5,         
        num_inference_steps=50,     
        generator=None,      
        num_images_per_prompt = 1
    )

    
    
    
    
    while len(images) < len(interps):
        pipeline.history = interps[batch_indx:min(batch_indx+batch_size, len(images)-1)]
        g = torch.Generator(device=pipeline.device).manual_seed(seed)
        outputs = pipeline(
            prompt=prompt,
            # prompt_embeds = embeddings[i].unsqueeze(0),
            height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
            width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
            guidance_scale=7.5,         
            num_inference_steps=50,     
            generator=g,      
            num_images_per_prompt = 1
        )
        images.extend(outputs[0])
        pbar.update(len(outputs[0]))
        batch_indx += batch_size
    # images.append(outputs[0][0])
    
    images_interps = []
    for image in images:
        # interpolate between images
        images_interps.append(torch.tensor(np.linspace(image.cpu().numpy(), images[images.index(image)+1].cpu().numpy(), smoothing_inters),device=device))
    
        

    iio.mimsave(f"outputs/{prompt}.gif", images_interps)