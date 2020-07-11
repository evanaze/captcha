""" ## Main Script
    The main script to run the model.
    ### Usage
    To use this script to predict on a new image, run with the `-i` flag:  
    ```python -m src.run -i new_image.png```
    Alternatively, to run by retraining on new data:  
    ```python -m src.run --retrain```
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description='PyTorch Captcha')
    parser.add_argument("-i", "--image", type=str, 
                        help="The new image to predict on")
    parser.add_argument("--retrain", action="store_true", 
                        help="A flag for if we should retrain")
    args = parser.parse_args()
    print(args)
    if args.retrain:
        from .models import train
        train.run()
    if args.image:
        from .models import predict
        predict.predict(args.image)


if __name__ == "__main__":
    main()