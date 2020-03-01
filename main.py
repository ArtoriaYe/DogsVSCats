from models.ResNet import ResNet
from engine.trainer import do_train
from data.dataset import train_loader, eval_loader
import torchvision as tv
import torch.nn as nn

if __name__ == '__main__':
    model = tv.models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 2)
    do_train(model, train_loader, eval_loader)
