import os

import pandas as pd
from torchvision import transforms

from .utils import get_dataset_stats
from .dataset import CaptchaDataset
from .train import Train
from . import config

ds_mean, ds_std = get_dataset_stats()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=ds_mean, 
        std=ds_std)
])
logger = {
    "fold": [i for i in range(config.N_FOLDS)],
    "val_score": []
}

if __name__ == "__main__":
    for fold in range(config.N_FOLDS):
        df_train = pd.read_csv(config.TRAIN_DATA)
        df_train[df_train["kfold"] != fold].to_csv("input/temp_train.csv", index=False)
        df_train[df_train["kfold"] == fold].to_csv("input/temp_val.csv", index=False)
        train_ds = CaptchaDataset(csv_file="input/temp_train.csv", root_dir=config.DATA_DIR, transform=transform)
        valid_ds = CaptchaDataset(csv_file="input/temp_val.csv", root_dir=config.DATA_DIR, transform=transform)
        t = Train(train_ds, valid_ds, fold)
        t.main()
        logger["val_score"].append(t.test_loss)
    os.remove("input/temp_train.csv"); os.remove("input/temp_val.csv")
    
