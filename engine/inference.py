import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision as tv
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint

from config import opt
from utils.metric_utils import TestResult
from data.dataset import test_loader
from models.ResNet import ResNet


def create_supervised_evaluator(model, metrics, device="cuda"):
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    def _inference(engine, batch):
        with torch.no_grad():
            data, label = batch
            data = data.to(device).float() if torch.cuda.device_count()>=1 else data
            output = model(data)
            output = nn.functional.softmax(output, dim=1)
            return output[:, 1], label

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine


def inference(test_loader, metircs):
    if os.listdir(opt.checkpoint_dir):
        for root, dirs, files in os.walk(opt.checkpoint_dir):
            checkpoint_file = root + '\\' + files[-1]
        checkpoint = torch.load(checkpoint_file)
        model = tv.models.resnet50()
        model.fc = nn.Linear(2048, 2)
        object_to_checkpoint = {'model': model}
        Checkpoint.load_objects(to_load=object_to_checkpoint, checkpoint=checkpoint)
    test = create_supervised_evaluator(model, metrics)

    @test.on(Events.COMPLETED)
    def get_result(engine):
        preds, labels = test.state.metrics['test_result']
        preds = preds.reshape((-1, 1))
        labels = labels.reshape((-1, 1))
        preds = np.clip(preds, 0.005, 0.995)
        result = np.concatenate((labels, preds), axis=1)
        result = pd.DataFrame(result, columns=['id', 'label'])
        print(result)
        result.to_csv("..\\result.csv", index=None)
        return result

    test.run(test_loader)

if __name__ == "__main__":
    metrics = {"test_result": TestResult()}
    inference(test_loader, metrics)