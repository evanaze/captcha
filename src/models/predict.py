""" Predicts for a processed image
"""
import os
import argparse
from skimage import io
import torch
from .model import Net
from ..features.preprocess import preprocess
from .. import config


def predict(data, model_loc):
    "Predict on an image from a dataloader"
    # load the model
    model = Net()
    model.load_state_dict(torch.load(model_loc))
    # set the model to eval state
    model.eval()
    # get the output of the model
    output = model(data.to("cpu")) 
    # choosing the model's result
    res = output.argmax(dim=1, keepdim=True).numpy()[0][0] 
    return res

