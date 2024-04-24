# %% utilkaggle
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TTF
import cv2 as cv
import dataclasses
import functools
import itertools
import matplotlib.pyplot as plt
import numpy as np
import openpyxl as opx
import os
import platform
import time
import torch
import torchvision
import typing
import enum
import os
import sys
def installLib(installpath, gitpath):
    projPath = os.path.join(installpath, os.path.splitext(os.path.basename(gitpath))[0])
    if not os.path.exists(projPath):
        os.system(rf"git clone {gitpath} {projPath}")
    else:
        %cd {projPath}
        os.system(rf"git pull")
    if projPath not in sys.path:
        sys.path.append(projPath)