import math
import random
from diffusers.schedulers import  DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
import gradio as gr
from evolve import *
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
from  diffusers.models import attention_processor
# device = f"cuda:{torch.cuda.device_count()-1}" if torch.cuda.is_available() else "cpu"
device = f"cuda:0" if torch.cuda.is_available() else "cpu"

GEN = 0
image_history = []
selected_history = []
max_outputs = 12

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

def generation(prompt,neg_prompt, pipeline, seed, num_samples=3, width=512, height=512, num_inference_steps=50, guidance_scale=7.5):
    outputs = make_imgs(prompt, neg_prompt, pipeline, seed, num_samples, width, height, num_inference_steps, guidance_scale)
    return outputs.images

def variable_outputs(k):
    k = int(k)
    return [gr.Image.update(visible=True)]*k + [gr.Image.update(visible=False)]*(max_outputs-k)

def l_to_int(label):
    return int(label.split(" ")[-1])


if __name__ == "__main__":
    current_pipe = None
    seed = random.randint(0, 10000)
    
    def get_pipe_module(model):
        return EvolutionPipeline
    
    def get_pipe(model, scheduler="DDIM", force_reload=False):
        global current_pipe
        scheduler = SCHEDULERS[scheduler]
        if current_pipe is not None and not force_reload and current_pipe is not None and current_pipe.current_model == model and isinstance(current_pipe.scheduler, scheduler):
            return current_pipe
        print("loading model")
        scheduler = scheduler.from_pretrained(model, subfolder="scheduler")
        pipe = get_pipe_module(model).from_pretrained(model, 
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
    
    current_pipe = get_pipe(models[1])
    
    
    # Buttons
    
    def previous_gen(*args):
        global GEN
        if GEN<=0:
            return
        GEN-=1
        image_history.pop()
        selected_history.pop()
        images = image_history[-1]
        current_pipe.back()
        return  [f"Generation {GEN}"] + images + [None]*(max_outputs-len(images))
    
    def retry_gen(prompt,
        seed,
        width,
        height,
        guidance_scale,
        num_inference_steps,
        model,
        samples,
        scheduler,
        mutation_rate):
        global GEN, current_pipe
        if GEN==1:
            return [f"Press start on setup tab to restart from Gen 0"] + image_history[-1] + [None]*(max_outputs-len(image_history[-1]))
        GEN-=1
        image_history.pop()
        sel = selected_history.pop()
        current_pipe.back()
        return  next_gen(prompt, seed,width,height,guidance_scale,num_inference_steps, model, samples, scheduler,mutation_rate)
    
    selected = []
    def select_genome(si):
        # find button with label si
        found = None
        for btn in select_inputs:
            if btn.label == si:
                found = btn
                break   
        btn = found
        if btn is None:
            raise ValueError(f"Button with label {si} not found")
        
        if btn.label.startswith("Select"):
            if len(selected) >= 2:
                # TODO
                print("Only 2 images can be selected for now")
                return btn.label             
            selected.append(si)
            btn.label = f"Unselect {l_to_int(btn.label)}"
        else:
            selected.remove(si)
            btn.label = f"Select {l_to_int(btn.label)}"
        return btn.label
    
    def next_gen(
            prompt,
            neg_prompt,
            seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            model,
            samples,
            scheduler,
            mutation_rate,
            ):
        global GEN
        
        # switch to evolve tab
       
        
        selected_ints = [l_to_int(s) for s in selected]
        pipe = get_pipe(model, scheduler)
        
        
        if GEN > 0:
            if len(selected) < 0:
                # restart
                pipe = get_pipe(model, scheduler, True)
                pipe.mutation_rate = mutation_rate
                pipe.samples_per_gen = samples
                pipe.restart()
                GEN = 0
            else:
                pipe.mutation_rate = mutation_rate
                pipe.samples_per_gen = samples
                pipe.select(selected_ints)
                pipe.mutate()
        GEN+=1
        
            
        seed = int(seed)

        images = generation(prompt, neg_prompt, pipe, seed, num_samples=samples, width=width, height=height, 
                            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        image_history.append(images)
        selected_history.append(selected)
        return [f"Generation {GEN}:\n\"{prompt}\""] + images + [None]*(max_outputs-len(images))


    with gr.Blocks() as app:
        with gr.Tabs(titles=["Setup", "Evolve"]) as tabs:
            with gr.Tab("Setup"):
                model_input = gr.inputs.Dropdown(models, default=models[1], label="Model")
                prompt_input = gr.inputs.Textbox(lines=2, placeholder="Enter a prompt", label="Prompt")
                neg_prompt_input = gr.inputs.Textbox(lines=2, placeholder="Enter a negative prompt", label="Negative Prompt", default=None)
                start_button = gr.Button("Start")
                scheduler_input = gr.inputs.Dropdown(list(SCHEDULERS.keys()), default="DDIM", label="Scheduler")
                seed_input = gr.inputs.Slider(minimum=0, maximum=10000, step=1, default=seed, label="Seed")
                width_input = gr.inputs.Slider(minimum=256, maximum=1024, step=8, default=512, label="Width")
                height_input = gr.inputs.Slider(minimum=256, maximum=1024, step=8, default=512, label="Height")
                guidance_scale_input = gr.inputs.Slider(minimum=0.1, maximum=15, step=0.1, default=7.5, label="Guidance Scale") 
                samples_input = gr.inputs.Slider(minimum=1, maximum=max_outputs, step=1, default=max_outputs, label="Samples")
                inference_steps_input = gr.inputs.Slider(minimum=1, maximum=200, step=1, default=50, label="Inference Steps")

            
            callback = gr.CSVLogger()
            
            
            with gr.Tab("Evolve", id="Evolve"):
                gen_label = gr.Label("0", label="Generation")
                image_outputs = []
                select_inputs = []
                save_inputs = []
                outputs_per_row = 4 
                
                n_rows = math.ceil(samples_input.value/outputs_per_row)
                for row_i in range(n_rows):
                    with gr.Row(variant='panel').style(equal_height=True):
                        for i in range(outputs_per_row):
                            idx = row_i*outputs_per_row + i
                            image_outputs.append(gr.Image(visible=idx<samples_input.value, label=f"Output {idx}", interactive=False))
                    
                    with gr.Row().style(equal_height=True):
                        for i in range(outputs_per_row):
                            idx = row_i*outputs_per_row + i
                            select_inputs.append(gr.Button(visible = idx<samples_input.value, label=f"Select {idx}", value=f"Select {idx}", variant='secondary', size='small'))
                    
                    with gr.Row().style(equal_height=True):
                        for i in range(outputs_per_row):
                            idx = row_i*outputs_per_row + i
                            save_inputs.append(gr.Button(visible = idx<samples_input.value, label=f"Save {idx}",value=f"Save {idx}", variant='success', size='small'))
                
                
                samples_input.change(variable_outputs, samples_input, image_outputs)
                samples_input.change(variable_outputs, samples_input, select_inputs)
                samples_input.change(variable_outputs, samples_input, save_inputs)
                
                
                mutation_rate_input = gr.inputs.Slider(minimum=0, maximum=1.0, step=0.001, default=.08, label="Mutation Rate")
                
                next_gen_button = gr.Button("Next Generation")
                
                
                with gr.Row(equal_height=True):
                    retry_button = gr.Button("Retry")
                    back_button = gr.Button("Back")
                back_button.click(previous_gen, inputs=[], outputs=[gen_label]+image_outputs)
                retry_button.click(retry_gen, inputs=[
                            prompt_input,
                            neg_prompt_input,
                            seed_input,
                            width_input,
                            height_input,
                            guidance_scale_input,
                            inference_steps_input,
                            model_input,
                            samples_input,
                            scheduler_input,
                            mutation_rate_input
                        ], outputs=[gen_label]+image_outputs)
                
            for index, si in enumerate(select_inputs):
                si.click(select_genome, inputs=[si], outputs=[si])
            
            next_gen_button.click(next_gen, inputs=[
                            prompt_input,
                            neg_prompt_input,
                            seed_input,
                            width_input,
                            height_input,
                            guidance_scale_input,
                            inference_steps_input,
                            model_input,
                            samples_input,
                            scheduler_input,
                            mutation_rate_input,
                        ], outputs=[gen_label]+ image_outputs
            )
            
            callback.setup([prompt_input,
                            seed_input,
                            width_input,
                            height_input,
                            guidance_scale_input,
                            inference_steps_input,
                            model_input,
                            samples_input,
                            scheduler_input,
                            mutation_rate_input]+[gen_label]+image_outputs, "evolution/saved")
            for si in save_inputs:
                si.click(lambda *args: callback.flag(args), [
                            prompt_input,
                            neg_prompt_input,
                            seed_input,
                            width_input,
                            height_input,
                            guidance_scale_input,
                            inference_steps_input,
                            model_input,
                            samples_input,
                            scheduler_input,
                            mutation_rate_input,
                            gen_label,
                            image_outputs[l_to_int(si.label)]
                            ], None, preprocess=False)
            
            
            
            start_button.click(next_gen, inputs=[
                            prompt_input,
                            neg_prompt_input,
                            seed_input,
                            width_input,
                            height_input,
                            guidance_scale_input,
                            inference_steps_input,
                            model_input,
                            samples_input,
                            scheduler_input,
                            mutation_rate_input,
                            # gr.Textbox(value="s -1", visible=False) # restart
                        ], outputs=[gen_label]+image_outputs)
    app.launch(share=True)
        