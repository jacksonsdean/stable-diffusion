
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F

def save(imgs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(path, bbox_inches="tight", pad_inches=0)


def embed(prompt, pipe):
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = pipe.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids
    ):
        removed_text = pipe.tokenizer.batch_decode(
            untruncated_ids[:, pipe.tokenizer.model_max_length - 1 : -1]
        )
        print(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {pipe.tokenizer.model_max_length} tokens: {removed_text}"
        )

    if hasattr(pipe.text_encoder.config, "use_attention_mask") and pipe.text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(pipe.device)
    else:
        attention_mask = None

    prompt_embeds = pipe.text_encoder(
        text_input_ids.to(pipe.device),
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]
   
    return prompt_embeds
