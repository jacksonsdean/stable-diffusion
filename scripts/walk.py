from stable_diffusion_videos.stable_diffusion_walk import walk

prompt_n_seed = {
    "1965 Porsche 911": 743,
    "1975 Porsche 911": 140,
    "1985 Porsche 911": 40,
    "1995 Porsche 911": 560,
    "2005 Porsche 911 directly facing camera": 996,
    "2015 Porsche 911": 283,
    "2020 Porsche 911": 116,
    "2020 Porsche 911 GT3 RS": 277,
}


walk(prompts=list(prompt_n_seed.keys()), 
     seeds=list(prompt_n_seed.values()),
     make_video=True, 
     name="porschevolution",
     num_steps=200,
     use_lerp_for_text=True,
     upscale=True
     )