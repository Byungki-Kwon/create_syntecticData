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
from make_background import *


def syntectic(backgroundDataset, object_filename, object_path):


    backIndex = np.random.randint(0, len(backgroundDataset) - 1)

    background = backgroundDataset.__getitem__(backIndex)[0]

    c, h, w = background.shape

    # image를 정사각형으로 만듦
    length  =  np.min([h, w])
    background = background[:, 0:length, 0:length]
    background = np.array(background)
    background = np.transpose(background, [1, 2, 0])

    objectNumber = np.random.randint(7, 15)
    object_index = np.random.randint(0, len(object_filename)-1, objectNumber)

    temp_object = []

    #잘나옴
    for f in object_index:
        object_root = object_path + object_filename[f]
        temp = Image.open(object_root)
        temp = np.asarray(temp)
        # temp = temp/255
        temp_object.append(temp)

    object_image = []

    for i in range(len(temp_object)):
        h, w, c = temp_object[i].shape
        if np.max([h, w]) < length - 20:
            object_image.append(temp_object[i])

    objectNumber = len(object_image)


    #### 여기까지 배경을 선정하고, 배경크기에 맞는 오브젝트를 선별하는 과정이었음

    #random Value들 생성

    scale = np.random.uniform(1, 2, objectNumber + 1)
    displace = np.random.uniform(0, 10, objectNumber + 1) # 물체 + background
    displace = scale[0] * displace
    random_theta = np.random.uniform(0, 2*np.pi, objectNumber + 1)  # 물체 + background
    frames = np.random.uniform(5, 15, objectNumber + 1) + 1

    new_length = int(np.floor(scale[0]*length))

    # object의 initial position을 정해줌
    position_x = np.random.randint(0, scale[0]*length, objectNumber)

    position_y = np.random.randint(0, scale[0]*length, objectNumber)

    # frame 진동을 위해 설정해둔 인덱스
    index = np.zeros([objectNumber + 1])

    # base_img 처음에 비어있는 이미지
    base_img = np.zeros([new_length+200, new_length+200, 3], dtype=np.float32)


    background = Image.fromarray(np.uint8(background*255), 'RGB')
    # background = np.asarray(background)


    for i in range(60):

        for idx in range(len(index)):
            index[idx] = index[idx] + (-1)**(i//frames[idx])

        output_image = backward_warping_background(base_img, background, displace[0], random_theta[0], frames[0], index[0], scale[0])

        for i in range(objectNumber):
            output_image = backward_warping_object(output_image, object_image[i], displace[i+1], random_theta[i+1], frames[i+1], index[i+1], scale[i+1], position_x[i], position_y[i])

        plt.imshow(np.asarray(output_image))
        plt.show()
