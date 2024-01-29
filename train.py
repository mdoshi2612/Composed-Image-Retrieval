from builtins import RuntimeWarning
from builtins import NotImplementedError
import os, sys
import numpy as np
import PIL
from PIL import Image
import warnings
import random
import pickle

import torch
import json
import torch.utils.data
import torchvision
from torchvision import transforms as tfms
import torch, logging
import torch.nn as nn

import matplotlib.pyplot as plt
logging.disable(logging.WARNING)
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
# %matplotlib inline

from IPython.display import display
import shutil
import os

## For video display
from IPython.display import HTML
from base64 import b64encode

PIL.Image.MAX_IMAGE_PIXELS = 1000000000
PIL.Image.warnings.simplefilter('error', PIL.Image.DecompressionBombWarning)

from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler

from dataset import CIRR
from pipeline import CIRPipeline
from loss import LossFunction

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    ## Import the CLIP artifacts  
    # Initialize device
    args = parse_args()
    device = args.device
    # print("Using", device)
    
    ## Initiating tokenizer and encoder.
    tokenizer = CLIPTokenizer.from_pretrained("clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("clip-vit-large-patch14").to(device)
    print("Text encoder and tokenzier loaded")
    
    ## Initiating the VAE
    vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-4", subfolder="vae").to(device)
    print("VAE loaded")

    ## Initializing a scheduler and Setting number of sampling steps
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(50)
    
    ## Initializing the U-Net model
    unet = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-4", subfolder="unet").to(device)
    print("UNet loaded")
    
    # CLIPImage Processor
    processor = CLIPProcessor.from_pretrained("clip-vit-large-patch14")
    image_encoder = CLIPModel.from_pretrained("clip-vit-large-patch14")

    print("All CLIP Artifacts loaded to device", device)

    dataset = CIRR('./data/cirr', 
                   processor = processor,
                   image_encoder = image_encoder,
                   tokenizer = tokenizer,
                   text_encoder = text_encoder, 
		   device = device)

    print("Dataset loaded")

    pipeline = CIRPipeline(tokenizer, text_encoder, vae, scheduler, unet, processor, image_encoder, device)
    print("Model Loaded")

    # pipeline.load_state_dict(torch.load("model.pt"))

    criterion = LossFunction(margin = 0.8, device = device)
    optimizer = torch.optim.Adam(pipeline.parameters(), lr=0.0005, weight_decay=0.0005)
    print("Loss function and optimizers loaded")

    batch_size = args.batch_size
    dataloader = dataset.get_loader(batch_size)

    NUM_EPOCHS = args.num_epochs

    print(f"Number of epochs is {NUM_EPOCHS}")
    print(f"Batch size is {batch_size}")

    for epoch in range(NUM_EPOCHS):
        print(100*"=")
        print(f"Starting epoch {epoch+1}")
        train_loss = 0
        for i, data in enumerate(dataloader):
    
            # Get data
            source_data = torch.stack([data[i][0] for i in range(len(data))], dim = 0).to(device).requires_grad_()
            target_data = torch.stack([data[i][1] for i in range(len(data))], dim = 0).to(device).requires_grad_().reshape(-1, 768)
            caption_data = torch.stack([data[i][2] for i in range(len(data))], dim = 0).to(device).requires_grad_()
    
            # Reset optimizer
            optimizer.zero_grad()
    
            # Get image embedding
            target_embedding = pipeline(source_data, caption_data, steps = args.steps).requires_grad_().reshape(-1, 768)
            # print(target_embedding.size(), target_data.size())
          
            # Get loss and do one optimization step
            loss = criterion(target_embedding, target_data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
        # Write logs
        text = f"Epoch {epoch} loss is " + str(train_loss)
        with open('logs.txt', 'a') as file:
          file.write(text + "\n")
    
        # Save model to path
        torch.save(pipeline.state_dict(), "cosine_model.pt")
    
        # Print losses every epoch
        print(f"Training Loss is {train_loss}")
