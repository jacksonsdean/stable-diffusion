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
    # prompt = "love is golden yellow sunshine, abstract digital painting. Trending on artstation."
    prompt = "a psychedelic patterned bird with spread wings. A pastel color scheme in the style of Alex Grey."
    
    latent_history = torch.load( f"outputs/history_{prompt}.pt")
    
    
    # latent_history = torch.load( f"outputs/hist.pt")
    
    hist = [e for e in latent_history][:-1]
    print(len(hist))
    images = []
    
    interps_per_image = 4
    smoothing_inters = 3
    
    interps = []
    for i in range(len(hist)-1):
        # interps.append(torch.tensor(np.linspace(hist[i].cpu().numpy(), hist[i+1].cpu().numpy(), interps_per_image)))
        intp = torch.tensor(np.linspace(hist[i].cpu().numpy(), hist[i+1].cpu().numpy(), interps_per_image))
        for j in range(interps_per_image):
            interps.append(intp[j])
    
    # interps = hist
    # interps = torch.cat(interps)
    pipeline.replay_mode()
    # pipeline.enable_attention_slicing(slice_size="max") 
    # pipeline.enable_vae_slicing()
    
    batch_size = 8
    batch_indx = 0
    images = []
    pbar = tqdm(total=len(interps))
    
    
    while len(images) < len(interps):
        # pipeline.history = interps[batch_indx:min(batch_indx+batch_size, len(images)-1)]
        pipeline.history = torch.stack(interps[batch_indx:min(batch_indx+batch_size, len(interps))], dim=0)
        # print(pipeline.history.shape)
        # print([h.shape for h in pipeline.history])
        g = torch.Generator(device=pipeline.device).manual_seed(seed)
        outputs = pipeline(
            prompt=prompt,
            guidance_scale=7.5,         
            num_inference_steps=50,     
            # generator=g,      
            num_images_per_prompt = pipeline.history.shape[0],
            output_type="numpy"
            
        )
        images.extend(outputs[0])
        pbar.update(len(outputs[0]))
        batch_indx += batch_size
    # images.append(outputs[0][0])
    
    images_interps = []
    for i in range(1, len(images)-1):
        # interpolate between images
        # intp = np.linspace(images[i-1], images[i], smoothing_inters//2)
        # images_interps.extend(pipeline.numpy_to_pil(intp))
        # intp = np.linspace(images[i], images[i+1], smoothing_inters//2)
        # intp = np.linspace(images[i-1], images[i+1], smoothing_inters)
        intp = np.linspace(images[i], images[i+1], smoothing_inters//2)
        images_interps.extend(pipeline.numpy_to_pil(intp))
    
        

    iio.mimsave(f"outputs/{prompt}.gif", images_interps)