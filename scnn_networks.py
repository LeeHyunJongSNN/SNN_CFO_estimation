import os
import gc
import numpy as np

from spikingjelly.activation_based import surrogate, neuron, functional

from scipy.signal import detrend

import argparse

import torch
import torch.nn as nn
import torch.optim as optim


# define model
class OneDCNN_Conv(nn.Module):
    def __init__(self):
        super(OneDCNN_Conv, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 512)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):

        r = 0
        x = x.unsqueeze(1)
        for steps in range(2):
            nif = self.conv(x)
            nif = self.pool(nif)
            nif = self.if1(self.flatten(nif))
            r += nif

        s = r / 2

        x = self.fc1(s)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)

        return x

class OneDCNN_FC1(nn.Module):
    def __init__(self):
        super(OneDCNN_FC1, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool1d(2, 2)  # MaxPool2d or AvgPool2d
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 512)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = nn.functional.relu(x)

        r = 0
        for steps in range(2):
            nif = self.fc1(self.flatten(x))
            nif = self.if1(nif)
            r += nif

        s = r / 2
        x = self.fc2(s)
        x = nn.functional.relu(x)
        x = self.fc3(x)

        return x

class OneDCNN_FC2(nn.Module):
    def __init__(self):
        super(OneDCNN_FC2, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 512)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = nn.functional.relu(x)
        x = self.fc1(self.flatten(x))
        x = nn.functional.relu(x)

        r = 0
        for steps in range(2):
            nif = self.fc2(x)
            nif = self.if1(nif)
            r += nif

        s = r / 2
        x = self.fc3(s)

        return x

class OneDCNN_ConvFC1(nn.Module):
    def __init__(self):
        super(OneDCNN_ConvFC1, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 512)
        self.if2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)

        r = 0
        for steps in range(2):
            nif = self.conv(x)
            nif = self.pool(nif)
            nif = self.if1(self.flatten(nif))
            nif = self.fc1(nif)
            nif = self.if2(nif)

            r += nif

        s = r / 2
        x = self.fc2(s)
        x = nn.functional.relu(x)
        x = self.fc3(x)

        return x


class OneDCNN_ConvFC2(nn.Module):
    def __init__(self):
        super(OneDCNN_ConvFC2, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 512)
        self.if2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)

        r = 0
        for steps in range(2):
            nif = self.conv(x)
            nif = self.pool(nif)
            nif = self.if1(self.flatten(nif))

            r += nif

        s = r / 2
        x = self.fc1(s)
        x = nn.functional.relu(x)

        r = 0
        for steps in range(2):
            nif = self.fc2(x)
            nif = self.if2(nif)
            r += nif

        s = r / 2
        x = self.fc3(s)

        return x

class OneDCNN_FC1FC2(nn.Module):
    def __init__(self):
        super(OneDCNN_FC1FC2, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 512)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.if2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = nn.functional.relu(x)

        r = 0
        for steps in range(2):
            nif = self.fc1(self.flatten(x))
            nif = self.if1(nif)
            nif = self.fc2(nif)
            nif = self.if2(nif)
            r += nif

        s = r / 2
        x = self.fc3(s)

        return x

class OneDCNN_ConvFC1FC2(nn.Module):
    def __init__(self):
        super(OneDCNN_ConvFC1FC2, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv1d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 512)
        self.if2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.if3 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)

        r = 0
        for steps in range(2):
            nif = self.conv(x)
            nif = self.pool(nif)
            nif = self.if1(self.flatten(nif))
            nif = self.fc1(nif)
            nif = self.if2(nif)
            nif = self.fc2(nif)
            nif = self.if3(nif)
            r += nif

        s = r / 2
        x = self.fc3(s)

        return x


class TwoDCNN_Conv(nn.Module):
    def __init__(self):
        super(TwoDCNN_Conv, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv2d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(640, 512)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        r = 0
        x = x.unsqueeze(1)
        for steps in range(2):
            nif = self.conv(x)
            nif = self.pool(nif)
            nif = self.if1(self.flatten(nif))
            r += nif

        s = r / 2

        x = self.fc1(s)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)

        return x


class TwoDCNN_FC1(nn.Module):
    def __init__(self):
        super(TwoDCNN_FC1, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv2d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.AvgPool2d(2, 2)  # MaxPool2d or AvgPool2d
        self.batch_norm = nn.BatchNorm2d(8)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(640, 512)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        # x = nn.functional.dropout(x, 0.25)
        x = self.conv(x)
        x = self.pool(x)
        x = self.batch_norm(x)
        x = nn.functional.relu(x)
        # x = nn.functional.dropout(x, 0.25)

        r = 0
        for steps in range(2):
            nif = self.fc1(self.flatten(x))
            nif = self.if1(nif)
            # nif = nn.functional.dropout(nif, 0.25)
            r += nif

        s = r / 2
        x = self.fc2(s)
        x = nn.functional.relu(x)
        # x = nn.functional.dropout(x, 0.25)
        x = self.fc3(x)

        return x


class TwoDCNN_FC2(nn.Module):
    def __init__(self):
        super(TwoDCNN_FC2, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv2d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(640, 512)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = nn.functional.relu(x)
        x = self.fc1(self.flatten(x))
        x = nn.functional.relu(x)

        r = 0
        for steps in range(2):
            nif = self.fc2(x)
            nif = self.if1(nif)
            r += nif

        s = r / 2
        x = self.fc3(s)

        return x


class TwoDCNN_ConvFC1(nn.Module):
    def __init__(self):
        super(TwoDCNN_ConvFC1, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv2d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(640, 512)
        self.if2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)

        r = 0
        for steps in range(2):
            nif = self.conv(x)
            nif = self.pool(nif)
            nif = self.if1(self.flatten(nif))
            nif = self.fc1(nif)
            nif = self.if2(nif)

            r += nif

        s = r / 2
        x = self.fc2(s)
        x = nn.functional.relu(x)
        x = self.fc3(x)

        return x


class TwoDCNN_ConvFC2(nn.Module):
    def __init__(self):
        super(TwoDCNN_ConvFC2, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv2d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(640, 512)
        self.if2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)

        r = 0
        for steps in range(2):
            nif = self.conv(x)
            nif = self.pool(nif)
            nif = self.if1(self.flatten(nif))

            r += nif

        s = r / 2
        x = self.fc1(s)
        x = nn.functional.relu(x)

        r = 0
        for steps in range(2):
            nif = self.fc2(x)
            nif = self.if2(nif)
            r += nif

        s = r / 2
        x = self.fc3(s)

        return x


class TwoDCNN_FC1FC2(nn.Module):
    def __init__(self):
        super(TwoDCNN_FC1FC2, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv2d(1, 8, (3, 3), padding=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)  # MaxPool2d or AvgPool2d
        self.batch_norm = nn.BatchNorm2d(8)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(640, 512)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.if2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        # x = nn.functional.dropout(x, 0.25)
        x = self.conv(x)
        x = self.pool(x)
        # x = self.batch_norm(x)
        x = nn.functional.relu(x)
        # x = nn.functional.dropout(x, 0.25)

        r = 0
        for steps in range(2):
            nif = self.fc1(self.flatten(x))
            nif = self.if1(nif)
            # nif = nn.functional.dropout(nif, 0.25)
            nif = self.fc2(nif)
            nif = self.if2(nif)
            # nif = nn.functional.dropout(nif, 0.25)
            r += nif

        s = r / 2
        x = self.fc3(s)

        return x


class TwoDCNN_ConvFC1FC2(nn.Module):
    def __init__(self):
        super(TwoDCNN_ConvFC1FC2, self).__init__()
        functional.set_backend(self, backend='cupy')
        self.conv = nn.Conv2d(1, 8, 3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(640, 512)
        self.if2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(512, 128)
        self.if3 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)

        r = 0
        for steps in range(2):
            nif = self.conv(x)
            nif = self.pool(nif)
            nif = self.if1(self.flatten(nif))
            nif = self.fc1(nif)
            nif = self.if2(nif)
            nif = self.fc2(nif)
            nif = self.if3(nif)
            r += nif

        s = r / 2
        x = self.fc3(s)

        return x


def network_choose(network):
    if network == 'OneDCNN_Conv':
        return OneDCNN_Conv
    if network == 'OneDCNN_FC1':
        return OneDCNN_FC1
    if network == 'OneDCNN_FC2':
        return OneDCNN_FC2
    if network == 'OneDCNN_ConvFC1':
        return OneDCNN_ConvFC1
    if network == 'OneDCNN_ConvFC2':
        return OneDCNN_ConvFC2
    if network == 'OneDCNN_FC1FC2':
        return OneDCNN_FC1FC2
    if network == 'OneDCNN_ConvFC1FC2':
        return OneDCNN_ConvFC1FC2
    if network == 'TwoDCNN_Conv':
        return TwoDCNN_Conv
    if network == 'TwoDCNN_FC1':
        return TwoDCNN_FC1
    if network == 'TwoDCNN_FC2':
        return TwoDCNN_FC2
    if network == 'TwoDCNN_ConvFC1':
        return TwoDCNN_ConvFC1
    if network == 'TwoDCNN_ConvFC2':
        return TwoDCNN_ConvFC2
    if network == 'TwoDCNN_FC1FC2':
        return TwoDCNN_FC1FC2
    if network == 'TwoDCNN_ConvFC1FC2':
        return TwoDCNN_ConvFC1FC2

    else:
        raise ValueError('Invalid network name')
