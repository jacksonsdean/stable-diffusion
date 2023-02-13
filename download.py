from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="runwayml/stable-diffusion-inpainting", filename="sd-v1-5-inpainting.ckpt", cache_dir=".")
# hf_hub_download(repo_id="CompVis/stable-diffusion-v-1-4-original", filename="sd-v1-4.ckpt", cache_dir=".")