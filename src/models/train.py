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

class Train:
    def __init__(self, args):
        self.args = args
        # choosing device
        self.use_cuda = not self.args.no_cuda and torch.cuda.is_available()
        # set the seed
        torch.manual_seed(config.RAND_STATE)
        # detect if we have a cuda device
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # kwargs for training
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        # the name of the model and where we store the models
        self.model_name = "captcha_dcnn"
        # we just need the simplest transform
        self.transform = transforms.ToTensor()
        self.model = Net().to(device)
        # initialize Ada optimizer
        self.optimizer = optim.Adadelta(model.parameters(), lr=self.args.lr)
        # initialize the scheduler
        self.scheduler = StepLR(optimizer, step_size=1, gamma=self.args.gamma)


    def full_model(self):
        "Trains the model on the full training dataset"
        # train on the full training data
        train = pd.read_csv("data/train_proc.csv")
        # the train and val loaders for this fold
        train_ds = CaptchaDataset(
            csv_file=config.TRAIN_ALL, 
            root_dir=config.PROC_DIR, 
            transform=self.transform
        )
        train_loader = DataLoader(
            train_ds, 
            batch_size=config.TRAIN_BATCH_SIZE, 
            shuffle=True, 
            **self.kwargs
        )
        # loop through epochs
        for epoch in range(1, self.args.epochs + 1):
            # perform the training passthrough
            train_loss = train_fn(self.model, self.device, train_loader, self.optimizer, epoch)
            # scheduler step
            self.scheduler.step()
        # save the final model state
        torch.save(self.model.state_dict(), f"models/{self.model_name}.pt")


    def kfold_cv(self):
        "Performs KFold CV to estimate the generalization score"
        # where we store the training logs
        logs, log_name = {}, f"{datetime.now().isoformat()}.json"
        # add the model name to logs and the number of folds
        logs["model"], logs["n_folds"] = self.model_name, config.N_FOLDS
        # perform K-fold CV
        for fold in range(config.N_FOLDS):
            print(f"Fold: {fold}")
            # split data into folds
            train = pd.read_csv("data/train_proc.csv")
            train[train.kfold != fold].to_csv("data/train_temp.csv", index=False)
            train[train.kfold == fold].to_csv("data/valid_temp.csv", index=False)
            # the train and val loaders for this fold
            train_ds = CaptchaDataset(
                csv_file=config.TRAIN_DATA, 
                root_dir=config.PROC_DIR, 
                transform=self.transform
            )
            valid_ds = CaptchaDataset(
                csv_file=config.VALID_DATA, 
                root_dir=config.PROC_DIR, 
                transform=self.transform
            )
            train_loader = DataLoader(
                train_ds, 
                batch_size=config.TRAIN_BATCH_SIZE, 
                shuffle=True, 
                **self.kwargs
            )
            valid_loader = DataLoader(
                valid_ds, 
                batch_size=config.VALID_BATCH_SIZE, 
                shuffle=True, 
                **self.kwargs
            )
            # loop through epochs
            for epoch in range(1, self.args.epochs + 1):
                # perform the training passthrough
                train_loss = train_fn(self.model, self.device, train_loader, self.optimizer, epoch)
                # record the test loss
                test_loss = eval_fn(self.model, self.device, valid_loader)
                # scheduler step
                self.scheduler.step()
                # record the epoch result
                logs[str(datetime.now().time())] = {
                    "fold": fold,
                    "epoch": epoch,
                    "val_err": test_loss
                }
        # select the best model from the cross validation
        scores = np.array([(log['fold'], log['val_err']) for t, log 
        in logs.items() if log['epoch'] == self.args.epochs])
        # calculate the average validation error and add to log
        avg_val_err = np.mean(scores[1])
        logs["avg_val_err"] = avg_val_err
        print("Average Validation Error:", avg_val_err)
        # delete the old model files
        with open(f"logs/{log_name}", "w") as f:
            json.dump(logs, f)


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
    parser.add_argument('--full-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    # the train object
    t = Train(args)
    # whether to train on the full training data
    if args.full_model:
        t.full_model()
    else:
        t.kfold_cv()


if __name__ == '__main__':
    run()