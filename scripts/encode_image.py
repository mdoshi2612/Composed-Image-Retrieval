from clip_encoder import load_clip_encoder
from text2image import load_sd_model
import torch
from os.path import dirname

encoder = load_clip_encoder()
sd_model = load_sd_model("CompVis/stable-diffusion-v1-4")

prompt = "a photo of an astronaut riding a horse on mars"
image = sd_model(prompt).images

print(image)
