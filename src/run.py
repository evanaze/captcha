""" ## Main Script
    The main script to run the model.
    ### Usage
    To use this script to predict on a new image, run with the `-i` flag:  
    ```python -m src.run -i new_image.png```
    Alternatively, to run by retraining on new data:  
    ```python -m src.run --retrain```
"""
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='PyTorch Captcha')
    parser.add_argument("-i", "--image", type=str, 
                        help="The new image to predict on")
    parser.add_argument("-r", "--retrain", action="store_true", 
                        help="A flag for if we should retrain")
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--save-model', action='store_true',
                        help='For saving the current Model')
    args = parser.parse_args()
    
    if args.retrain:
        cmd = f"python -m src.models.train --epochs {args.epochs} --lr {args.lr} --gamma {args.gamma}"
        if args.no_cuda:
            cmd += " --no-cuda"
        if args.save_model:
            cmd += " --save-model"
        os.system(cmd)

    if args.image:
        os.system(f"python -m src.models.predict --file-name {args.image}")


if __name__ == "__main__":
    main()