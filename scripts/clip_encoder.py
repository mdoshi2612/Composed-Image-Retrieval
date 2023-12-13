import numpy as np
import torch
from pkg_resources import packaging
import clip

# Load up the CLIP Encoder Module

def load_clip_encoder(model_name = "ViT-B/16"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device = device)
    print("Model loaded")
    model.eval()
    return model

