import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time

def backward_warping_background(source, target, displace, theta, frames, index, scale):
    # target is PIL Image
    # source is numpy array
    temp1 = target.copy()
    temp1 = np.asarray(temp1)

    h, w, _ = temp1.shape

    new_length = int(np.floor(scale*h))

    temp2 = Image.fromarray(np.uint8(source*255))

    target = target.resize([new_length, new_length], resample=3)

    temp2.paste(target, box=[100, 100])
    temp2 = np.asarray(temp2, dtype=np.float32)/255
    print(temp2[100,100,:])
    h, w, _ = temp2.shape
    output_image = np.zeros([h, w, 3], dtype=np.float32)

    backwardX = - displace * np.cos(theta) * index / frames
    backwardY = - displace * np.sin(theta) * index / frames

    for y in range(100, h-100):
        for x in range(100, w-100):
            targetY = y + backwardY
            targetX = x + backwardX

            x_lo = int(np.floor(targetX))
            x_hi = int(np.ceil(targetX))
            y_lo = int(np.floor(targetY))
            y_hi = int(np.ceil(targetY))

            output_image[y, x, :] = (y_hi - targetY)*((x_hi - targetX)*temp2[y_lo, x_lo, :] + (targetX - x_lo)*temp2[y_lo, x_hi, :]) + (targetY-y_lo)*((x_hi-targetX)*temp2[y_hi, x_lo, :] + (targetX - x_lo)*temp2[y_hi, x_hi, :])

    output_image = Image.fromarray(np.uint8(output_image*255))
    return output_image







def backward_warping_object(source, target, displace, theta, frames, index, scale, position_x, position_y):
    #target is numpy array
    #source is PIL Image

    target = target/255

    h, w, _ = target.shape
    target = Image.fromarray(np.uint8(target*255))



    # target = Image.fromarray(np.uint8(target*255))
    target = target.resize([int(np.floor(scale*w)), int(np.floor(scale*h))], resample=3)

    temp1 = source.copy()
    temp1 = np.asarray(temp1)
    h_, w_, _ = temp1.shape

    temp2 = np.zeros([h_, w_, 3])
    temp2 = Image.fromarray(np.uint8(temp2*255), 'RGB')
    temp2.putalpha(0)

    temp2.paste(target, box=[100+position_y, 100+position_x], mask=target)

    output_image = np.zeros([h_, w_, 4], dtype=np.float32)

    temp2 = np.asarray(temp2, dtype=np.float32)/255

    backwardX = - displace * np.cos(theta) * index / frames
    backwardY = - displace * np.sin(theta) * index / frames

    for y in range(100, h_-100):
        for x in range(100, w_-100):
            targetY = y + backwardY
            targetX = x + backwardX

            x_lo = int(np.floor(targetX))
            x_hi = int(np.ceil(targetX))
            y_lo = int(np.floor(targetY))
            y_hi = int(np.ceil(targetY))

            output_image[y, x, :] = (y_hi - targetY)*((x_hi - targetX)*temp2[y_lo, x_lo, :] + (targetX - x_lo)*temp2[y_lo, x_hi, :]) + (targetY-y_lo)*((x_hi-targetX)*temp2[y_hi, x_lo, :] + (targetX - x_lo)*temp2[y_hi, x_hi, :])


    output_image = Image.fromarray(np.uint8(output_image*255), 'RGBA')
    source.paste(output_image, box = [0, 0], mask=output_image)

    return source








def make_background(croped_back, displace, random_theta, frames, index, scale):


    h, w, c = croped_back.shape
    h = int(np.floor(h*scale))
    w = int(np.floor(w*scale))

    base_imag = np.zeros([h+20, w+20, 3], dtype=np.float32)
    croped_back = Image.fromarray(croped_back, 'RGB')





    return grid_image


def make_foreground(object_image, seg_image, displace, theta, frames, index, scale, position_x, position_y):
    object_image = torch.tensor(object_image, dtype=torch.float32)
    seg_image = torch.tensor(seg_image, dtype=torch.float32)

    object_image = torch.unsqueeze(object_image, 0)
    seg_image = torch.unsqueeze(seg_image, 0)

    bn, c, h, w = object_image.shape

    # h = int(np.floor(h * scale))
    # w = int(np.floor(w * scale))

    row = torch.linspace(-1, 1, w)
    col = torch.linspace(-1, 1, h)
    grid = torch.zeros(bn, h, w, 2)

    for n in range(bn):
        for i in range(h):
            grid[n, i, :, 0] = row
        for i in range(w):
            grid[n, :, i, 1] = col

    displace = (displace * 2 / h) * (index / frames)

    grid[:, :, :, 0] = grid[:, :, :, 0] + displace * np.cos(theta) + position_x
    grid[:, :, :, 1] = grid[:, :, :, 1] + displace * np.sin(theta) + position_y

    grid_image = F.grid_sample(object_image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    grid_seg = F.grid_sample(seg_image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    grid_image = grid_image.permute(2, 3, 1, 0)
    grid_seg = grid_seg.permute(2, 3, 1, 0)

    grid_image = grid_image.numpy()
    grid_seg = grid_seg.numpy()

    grid_image = np.squeeze(grid_image)
    grid_seg = np.squeeze(grid_seg)

    return grid_image, grid_seg