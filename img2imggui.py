import torch
from diffusers import StableDiffusionImg2ImgPipeline

import random
import os
from  diffusers.models import attention_processor
import argparse
import gradio as gr
from PIL import Image
from diffusers.schedulers import  DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler

device = f"cuda:{torch.cuda.device_count()-1}" if torch.cuda.is_available() else "cpu"

DIFFUSION_2_MODELS = ["stabilityai/stable-diffusion-2-base"]
DIFFUSION_1_MODELS = ["CompVis/stable-diffusion-v1-4", "CompVis/stable-diffusion-v1-3"]
SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "PNDM": PNDMScheduler,
    "LMS": LMSDiscreteScheduler,
    "Euler": EulerDiscreteScheduler,
    "EulerAncestral": EulerAncestralDiscreteScheduler
}
models = DIFFUSION_2_MODELS + DIFFUSION_1_MODELS
                      
def image_grid(imgs, rows, cols):

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    current_pipe = None
    
    def get_pipe_module(model):
        return StableDiffusionPipeline if model in DIFFUSION_1_MODELS else DiffusionPipeline
        # return StableDiffusionPipeline if model in DIFFUSION_1_MODELS else PipelineParallel
    
    def get_pipe(model, scheduler="DDIM"):
        global current_pipe
        scheduler = SCHEDULERS[scheduler]
        if current_pipe is not None and current_pipe.current_model == model and isinstance(current_pipe.scheduler, scheduler):
            return current_pipe
        print("loading model")
        scheduler = scheduler.from_pretrained(model, subfolder="scheduler")
        pipe = get_pipe_module(model).from_pretrained(
                                                        model, 
                                                        revision="fp16",
                                                        torch_dtype=torch.float16,
                                                        safety_checker=None,
                                                        use_auth_token=True,
                                                        scheduler=scheduler)  
        
        pipe.current_model = model
        pipe = pipe.to(device)
        pipe.unet.set_attn_processor(attention_processor.AttnProcessor2_0())
        # pipe.unet = torch.compile(pipe.unet)
        pipe.enable_xformers_memory_efficient_attention()
        return pipe
    
    current_pipe = get_pipe(DIFFUSION_2_MODELS[0])
    
    def infer(prompt, num_inference_steps=50, 
                samples=5, seed=1024, guidance_scale=7.5, 
                width=512, height=512, model=DIFFUSION_2_MODELS[0], scheduler="DDIM"):
        generator = torch.Generator(device).manual_seed(seed)
        w = width//8*8
        h = height//8*8
        output = get_pipe(model, scheduler)(prompt, guidance_scale=guidance_scale, 
                    generator=generator, 
                    width=w, height=h, 
                    num_inference_steps=num_inference_steps, num_images_per_prompt=samples)
        return output.images

    def generate_image(prompt, seed, width, height, guidance_scale, num_inference_steps,model, samples, scheduler):
        seed = int(seed)
        print(model)
        images = infer(prompt, num_inference_steps=num_inference_steps, 
                seed=seed, width=width, height=height, guidance_scale=guidance_scale, model=model,samples=samples,scheduler=scheduler)
        if samples == 1:
            return images[0]
        else:
            return image_grid(images, samples, 1)
    
    seed = random.randint(0, 10000)
    seed_input = gr.inputs.Slider(minimum=0, maximum=10000, step=1, default=seed)
    inference_steps_input = gr.inputs.Slider(minimum=1, maximum=200, step=1, default=50)
    width_input = gr.inputs.Slider(minimum=256, maximum=1024, step=8, default=512)
    height_input = gr.inputs.Slider(minimum=256, maximum=1024, step=8, default=512)
    guidance_scale_input = gr.inputs.Slider(minimum=0.1, maximum=15, step=0.1, default=7.5) 
    model_input = gr.inputs.Dropdown(models, default=DIFFUSION_2_MODELS[0])
    prompt_input = gr.inputs.Textbox(lines=2, placeholder="Enter a prompt")
    samples_input = gr.inputs.Slider(minimum=1, maximum=10, step=1, default=1)
    scheduler_input = gr.inputs.Dropdown(list(SCHEDULERS.keys()), default="DDIM")
    
    gr.Interface(generate_image, 
                 [
                    prompt_input,
                    seed_input,
                    width_input,
                    height_input,
                    guidance_scale_input,
                    inference_steps_input,
                    model_input,
                    samples_input,
                    scheduler_input
                ], 
                "image").launch(share=True)
    exit()
