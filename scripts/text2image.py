from torch import autocast
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch
load_dotenv()


def load_sd_model(repo_id = "./stable-diffusion-v1-4"):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# HUGGING_FACE_CLI_KEY = os.getenv('HUGGING_FACE_CLI_KEY')
	# login(HUGGING_FACE_CLI_KEY)

	pipe = DiffusionPipeline.from_pretrained(
		repo_id, 
		use_safetensors = True
	).to(device)

	return pipe
	
	prompt = "a photo of an astronaut riding a horse on mars"
	image = pipe(prompt).images

	plt.imshow(image[0])
	print(image)
