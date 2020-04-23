import os

from skimage import io
import torch
import torchvision.transforms as transforms

from .dataset import CaptchaDataset
from . import config

def get_dataset_stats():
    transform = transforms.ToTensor()
    dataset = CaptchaDataset(
        csv_file='input/all.csv',
        root_dir=config.DATA_DIR,
        transform=transform
    )
    n = len(dataset)
    tensors = []
    for i in range(n):
        ts = dataset[0]
        tensors.append(ts[0])
    outp=torch.stack(tensors)
    return torch.mean(outp, [0,2,3]).numpy(), torch.std(outp, [0,2,3]).numpy()

def label():
    files = sorted(os.listdir(config.DATA_DIR))
    tot = len(files)
    y = []
    for i, f in enumerate(files): 
        file_path = os.path.join(data_path, f)
        io.imshow(file_path)
        inp = input(f"{i} of {tot}. Number of squares in image: ")
        try:
            n = int(inp)
            y.append(n)
        except Exception:
            raise Exception
    df_out = pd.DataFrame({"filenames": files, "target": y})
    df_out.to_csv("input/new_data.csv", index=False)
