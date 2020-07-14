import numpy as np
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import glob
import os
from os import listdir
from os.path import isfile, join
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from syntectic import syntectic
from make_background import make_background

coco_path = '/Users/kwon/Documents/dataset/coco/images/'
object_path = '/Users/kwon/Downloads/alpha_png/alpha_png/'


data_transform = transforms.Compose([transforms.ToTensor()])

backgroundDataset = torchvision.datasets.ImageFolder(coco_path, transform=data_transform)
object_filename = [f for f in listdir(object_path) if isfile(join(object_path, f))]

try:
    object_filename.remove('.DS_Store')
except:
    pass


print(object_filename)


syntectic(backgroundDataset, object_filename, object_path)


