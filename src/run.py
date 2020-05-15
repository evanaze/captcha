""" ## Main Script
    The main script to run the model.
    ### Usage
    To use this script to predict on a new image, run with the `-i` flag:  
    ```python -m src.run -i new_image.png```
    Alternatively, to run by retraining on new data:  
    ```python -m src.run --retrain```
"""

import argparse
import click

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