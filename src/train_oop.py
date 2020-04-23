from __future__ import print_function, absolute_import
import argparse
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


class Train:
    def __init__(self, train_ds, test_ds, fold):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.fold = fold
        self.params =  {
            "batch_size": 1,
            "epochs": 100,
            "lr": 0.1,
            "gamma": 0.7,
            "log_interval": 100,
            "seed": 1212,
        }
        self.device = "cpu"
        kwargs = {}
        self.model = Net().to(self.device)
        self.train_loader = DataLoader(self.train_ds, batch_size=self.params["batch_size"], shuffle=True, **kwargs)
        self.val_loader = DataLoader(self.test_ds, batch_size=self.params["batch_size"], shuffle=True, **kwargs)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.params["lr"])
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.params["gamma"])
        self.test_loss = 0

    def train(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
        if self.epoch % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                self.epoch, batch_idx * len(data), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), loss.item()))

    def test(self):
        self.model.eval()
        self.test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                self.test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        self.test_loss /= len(self.val_loader.dataset)
        if self.epoch % 5 == 0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                self.test_loss, correct, len(self.val_loader.dataset),
                100. * correct / len(self.val_loader.dataset)))

    def main(self):
        for self.epoch in range(self.params["epochs"]):
            self.train()
            self.test()
            self.scheduler.step()

        torch.save(self.model.state_dict(), f"models/captcha_cnn_{self.fold}.pt")

if __name__ == "__main__":
    ds_mean, ds_std = get_dataset_stats()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=ds_mean, 
            std=ds_std)
    ])
    train_ds = CaptchaDataset(csv_file=config.TRAIN_DATA, root_dir=config.DATA_DIR, transform=transform)
    valid_ds = CaptchaDataset(csv_file=config.TEST_DATA, root_dir=config.DATA_DIR, transform=transform)
    Train(train_ds, valid_ds, 0).main()