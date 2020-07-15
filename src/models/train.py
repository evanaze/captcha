""" Runs the engine to implement model training.

    Only has one method: run() to perform the model training
"""
from __future__ import print_function, absolute_import
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from .data_loader import CaptchaDataset
from .model import Net
from .engine import train_fn, eval_fn
from .. import config


def run():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Captcha')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='fnumber of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help=f'learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    # if we are using cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # set the seet
    torch.manual_seed(config.RAND_STATE)
    # detect if we have a cuda device
    device = torch.device("cuda" if use_cuda else "cpu")
    # kwargs for training
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # where we store the training logs
    logs, log_name = {}, f"{datetime.now().isoformat()}.json"
    # the name of the model and where we store the models
    models, model_name = {}, "captcha_dcnn"
    # add the model name to logs and the number of folds
    logs["model"], logs["n_folds"] = model_name, config.N_FOLDS
    # perform K-fold CV
    for fold in range(config.N_FOLDS):
        print(f"Fold: {fold}")
        # split data into folds
        train = pd.read_csv("data/train_proc.csv")
        train[train.kfold != fold].to_csv("data/train_temp.csv", index=False)
        train[train.kfold == fold].to_csv("data/valid_temp.csv", index=False)
        # we just need the simplest transform
        transform = transforms.ToTensor()
        # the train and val loaders for this fold
        train_ds = CaptchaDataset(
            csv_file=config.TRAIN_DATA, 
            root_dir=config.PROC_DIR, 
            transform=transform
        )
        valid_ds = CaptchaDataset(
            csv_file=config.VALID_DATA, 
            root_dir=config.PROC_DIR, 
            transform=transform
        )
        train_loader = DataLoader(
            train_ds, 
            batch_size=config.TRAIN_BATCH_SIZE, 
            shuffle=True, 
            **kwargs
        )
        valid_loader = DataLoader(
            valid_ds, 
            batch_size=config.VALID_BATCH_SIZE, 
            shuffle=True, 
            **kwargs
        )
        # load the model onto the device
        model = Net().to(device)
        # initialize Ada optimizer
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        # initialize the scheduler
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        # loop through epochs
        for epoch in range(1, args.epochs + 1):
            # perform the training passthrough
            train_loss = train_fn(model, device, train_loader, optimizer, epoch)
            # record the test loss
            test_loss = eval_fn(model, device, valid_loader)
            # scheduler step
            scheduler.step()
            # record the epoch result
            logs[str(datetime.now().time())] = {
                "fold": fold,
                "epoch": epoch,
                "val_err": test_loss
            }
        # save the final model state
        if args.save_model:
            torch.save(model.state_dict(), f"models/kFoldModels/{model_name}_{fold}.pt")
    # select the best model from the cross validation
    scores = np.array([(log['fold'], log['val_err']) for t, log in logs.items() if log['epoch'] == args.epochs])
    # select the best model by final val score
    best_model = np.argmax(scores[1])
    # calculate the average validation error and add to log
    avg_val_err = np.mean(scores[1])
    logs["avg_val_err"] = avg_val_err
    # move model
    os.rename(f"models/kFoldModels/{model_name}_{best_model}.pt", f"models/{model_name}.pt")
    # delete the old model files
    for root, dirs, files in os.walk("models/kFoldModels"):
        for file in files:
            if file != ".gitkeep":
                os.remove(os.path.join(root, file))
    with open(f"logs/{log_name}", "w") as f:
        json.dump(logs, f)


if __name__ == '__main__':
    run()