
import os
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers import StableDiffusionPipeline
from diffusers.utils import deprecate, is_accelerate_available, logging, randn_tensor, replace_example_docstring
import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from itertools import tee

class EvolutionPipeline(StableDiffusionPipeline):
        def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPFeatureExtractor,
            requires_safety_checker: bool = True,
            mutation_rate: float = 0.1,
            samples_per_gen = 3
        ):
            super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker)
            self.mutation_rate = mutation_rate
            self.samples_per_gen = samples_per_gen
            self.population = None
            self.selected_latents = None
            self.history = []
            self.replay = None
        
        def replay_mode(self,):
            # self.history = history
            self.replay = 0
        
        def crossover(self, idx_ints):
            # TODO: only 2 parents for now
            # find the average of the selected latents
            
            idx_a, idx_b = idx_ints[0], idx_ints[1]
            
            
            # find the average of the parents
            latents_a = self.selected_latents[idx_a]
            latents_b = self.selected_latents[idx_b]
            
            latents_size = latents_a.shape[0]
            
            latent_child = torch.clone(latents_a)
            latent_child[:latents_size//2] = latents_a[:latents_size//2]
            latent_child[latents_size//2:] = latents_b[latents_size//2:]
            # latent_child = (latents_a + latents_b) / 2
            
            parent_a = self.population[idx_a:idx_a+2].cuda() # (2, 77, emb_size)
            parent_b = self.population[idx_b:idx_b+2].cuda() # (2, 77, emb_size)
            
            # mean in dim 2
            child = torch.clone(parent_a)
            child[:, :, :parent_a.shape[2]//2] = parent_a[:, :, :parent_a.shape[2]//2]
            child[:, :, parent_a.shape[2]//2:] = parent_b[:, :, parent_a.shape[2]//2:]
            
            # child = (parent_a + parent_b) / 2 # LOL bad crossover, BAD
            
            # repeat to match the shape of the population
            population = child.repeat(self.samples_per_gen, *[1]*(len(child.shape)-1))
            selected_latents = latent_child.repeat(self.samples_per_gen, *[1]*(len(self.selected_latents.shape)-1))
            
            return population, selected_latents
        
            
        
        def select(self, idxs):
            idx_ints = [int(idx) for idx in idxs]
            self.history.append(self.population)
            print("Selected: ", idx_ints)
            print(self.population.shape)
            
            
            do_crossover = len(idx_ints) > 1
            if do_crossover:
                self.population, self.selected_latents = self.crossover(idx_ints)
            else:
                # TODO: multiple
                idx = idx_ints[0]
                self.population = self.population[idx:idx+2].cuda() # (2, 77, emb_size)
           
                # print("Selected: ", self.population.shape) 
                # repeat the selected population and latents
                self.population = self.population.repeat(self.samples_per_gen, *[1]*(len(self.population.shape)-1))
                self.selected_latents = self.selected_latents[idx].repeat(self.samples_per_gen, *[1]*(len(self.selected_latents.shape)-1))
            
            # TODO:
            # mutate the population such that the selected image is the most likely
            # covariance = torch.eye(self.population.shape[1]).to(self.population.device)
            # selected = self.population[idx:idx+2]
            
            # self.population = torch.distributions.MultivariateNormal(selected, covariance).sample((self.samples_per_gen,))


        def mutate(self):
            # TODO:
            # use the covariance of the population to mutate
            mut = torch.randn_like(self.population).cuda() * self.mutation_rate
            self.population += mut

        def back(self):
            self.population = self.history.pop()
            
        
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            if self.replay is not None:
                out = self.history.to(device)
                return out
            if self.selected_latents is not None:
                return self.selected_latents.to(device)
            shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device)
            # latents = self.population.to(device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            if self.selected_latents is None:
                self.selected_latents = latents
            # if self.population is None:
                # self.population = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # latents+=self.population
            # print("Latents shape: ", latents.shape)
            return latents.cuda()
        
        def restart(self):
            self.population = None
            self.selected_latents = None
            self.history = []
            self.replay = None
            
        def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ):
            emb = super()._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds, negative_prompt_embeds)
            if self.population is None:
                self.population = torch.randn_like(emb) * .01
            print("Embedding shape: ", emb.shape)
            print("Range of population: ", self.population.min(), self.population.max())
            print("Range of embedding: ", emb.min(), emb.max())
            return torch.add(emb, self.population).cuda()
