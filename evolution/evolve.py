# from stable_diffusion_videos.stable_diffusion_walk import walk
import traceback
from evolution_pipeline import EvolutionPipeline
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import xformers
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from util import save, embed
from PIL import Image
torch.set_float32_matmul_precision('high')
from diffusers.models.cross_attention import AttnProcessor2_0

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def make_imgs(prompt, neg_prompt, pipeline, seed, num_samples=3, width=512, height=512, num_inference_steps=50, guidance_scale=7.5):
    g = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    w = width//8*8
    h = height//8*8
        
    print(f"Generating {num_samples} images at {w}x{h} with {num_inference_steps} inference steps")
    
    outputs = pipeline(
        prompt=[prompt] * num_samples,
        negative_prompt=[neg_prompt] * num_samples,
        height=h,
        width=w, 
        guidance_scale=guidance_scale,         
        num_inference_steps=num_inference_steps,  
        generator=g,        
    )
    return outputs
    
def generation(prompt, pipeline, seed, num_samples=3, width=512, height=512, num_inference_steps=50, guidance_scale=7.5):
    outputs = make_imgs(prompt, pipeline, seed, num_samples, width, height, num_inference_steps, guidance_scale)
    grid = image_grid(outputs.images, rows=2, cols=num_samples//2)
    grid.save("outputs/current.png")
    return outputs.images

def print_help():
    print("h: help")
    print("q: quit")
    print("sX: save image X")
    print("b: go back one generation")


def setup():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # initial_population_variance = 0.1
    mutation_rate = .08
    num_samples = 8
    history = []
    seed = 1608
    torch.manual_seed(seed)
    pipeline = EvolutionPipeline.from_pretrained(
    # pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", # I might actually like this better
        # "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
        revision="fp16",
        safety_checker=None,
        requires_safety_checker = False,
        mutation_rate = mutation_rate,
        samples_per_gen = num_samples,
        # device_map="auto"
    ).to(device)
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    # pipeline.unet = torch.compile(pipeline.unet)
    pipeline.enable_xformers_memory_efficient_attention()
    
    # FOR LOW MEMORY GPUs
    # pipeline.enable_attention_slicing() 
    # pipeline.enable_vae_slicing()
    # pipeline.enable_sequential_cpu_offload(1)
    
    return device, mutation_rate, num_samples, history, seed, pipeline


if __name__ == "__main__":
    device, mutation_rate, num_samples, history, seed, pipeline = setup()
    gen = 0
    # generation_compiled = torch.compile(generation)
    
    prompt = "abstract crisp 8k photo closeup of jupiter."
    inp = ""
    skip_mutate = False
    try:
        while inp != "q":
            imgs = generation(prompt, pipeline, seed, num_samples)
            print(f"Done generation, choose one to mutate in {list(range(num_samples))} or q to quit")
            inp = input()
            valid_input = False
            while not valid_input:
                if inp == "q":
                    exit()
                if inp == "h":
                    print_help()
                if inp =='b':
                    pipeline.back()
                    valid_input = True
                    skip_mutate = True
                    gen-=1
                    continue
                    
                if inp.lower().startswith("s"):
                    try:
                        inp = int(inp[1:])
                        assert inp in list(range(num_samples))
                        valid_input = True
                        imgs[inp].save(f"outputs/{prompt[:20]}_{gen}_{inp}.png")
                        inp = input("Saved, choose another:")
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        inp = input("Invalid input, try again:")
                        continue
                else:
                    try:
                        inp = int(inp)
                        assert inp in list(range(num_samples))
                        valid_input = True
                    except:
                        inp = input("Invalid input, try again:")
                        continue
            if not skip_mutate:
                print(f"Mutating {inp}")
                pipeline.select(inp)
                pipeline.mutate()
                gen+=1
            skip_mutate = False
            
    except KeyboardInterrupt:
        pass
    history = torch.stack(pipeline.history)
    torch.save(history, f"outputs/history_{prompt}.pt")
        

   


        