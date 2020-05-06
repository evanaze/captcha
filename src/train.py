from __future__ import print_function, absolute_import
import argparse
import pandas as pd
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from .dataset import CaptchaDataset
from .model import Net
from . import config
from .utils import get_dataset_stats
from .engine import train_fn, eval_fn

""" Runs the engine to implement model training.

    Only has one method: run() to perform the model training
"""


def run():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Captcha')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(config.RAND_STATE)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    for fold in range(config.N_FOLDS):
        # split data into folds
        train = pd.read_csv("input/syn_train.csv")
        train[train.kfold != fold].to_csv("input/train_temp.csv", index=False)
        train[train.kfold == fold].to_csv("input/valid_temp.csv", index=False)
        transform = transforms.ToTensor()
        train_ds = CaptchaDataset(
            csv_file=config.SYN_TRAIN_DATA, 
            root_dir=config.SYN_DIR, 
            transform=transform
        )
        valid_ds = CaptchaDataset(
            csv_file=config.SYN_VALID_DATA, 
            root_dir=config.SYN_DIR, 
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

        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train_fn(model, device, train_loader, optimizer, epoch)
            eval_fn(model, device, valid_loader)
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), f"models/captcha_cnn_f{fold}.pt")


if __name__ == '__main__':
    run()