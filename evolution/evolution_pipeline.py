
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
            self.history = []
            self.replay = None
        
        def replay_mode(self,):
            # self.history = history
            self.replay = 0
        
        def select(self, idx):
            self.population = self.population[idx]
            self.history.append(self.population)
            self.population = self.population.repeat(self.samples_per_gen, 1, 1, 1)
            
            
        def mutate(self):
            mut = torch.randn_like(self.population) * self.mutation_rate
            self.population += mut
        
        def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            # return super().prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents)
            if self.replay is not None:
                # out = torch.stack(self.history, dim=0).to(device)
                out = self.history.to(device)
                return out
                # out = self.history[self.replay].to(device)
                # self.replay += 1
                # return out
            if self.population is not None:
                return self.population.to(device)
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
            self.population = latents
            # print("Latents shape: ", latents.shape)
            return latents
        
        # def from_pretained(
        #    cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs
        # ):
        #     return EvolutionPipeline(super().from_pretrained(pretrained_model_name_or_path, **kwargs))