from ignite.metrics import Metric
import torch
import numpy as np

class CatDogAcc(Metric):
    def __init__(self):
        super(CatDogAcc, self).__init__()

    def reset(self):
        self.preds = []
        self.labels = []

    def update(self, output):
        pred, label = output
        self.preds.extend(pred)
        self.labels.extend(label)

    def compute(self):
        true_num = sum((torch.tensor(self.preds) == torch.tensor(self.labels)).int())
        acc = 1.*true_num/len(self.preds)
        return acc


class TestResult(Metric):
    def __init__(self):
        super(TestResult, self).__init__()

    def reset(self):
        self.preds = []
        self.labels = []

    def update(self, output):
        pred, label = output
        pred = pred.cpu().numpy()
        self.preds  = np.append(self.preds, pred)
        self.labels = np.append(self.labels, label)

    def compute(self):
        return self.preds, self.labels