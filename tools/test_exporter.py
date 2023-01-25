import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import struct 
from torchinfo import summary

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.conv1 = nn.Conv2d(3,32,(3,3),stride=(2,2),padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        return x;

net = model()
model